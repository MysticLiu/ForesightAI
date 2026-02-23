#!/usr/bin/env python3
"""
Production daily brief generator for ForesightAI.

This replaces manual notebook execution with a deterministic CLI workflow:
1) Fetch processed events from the worker API.
2) Cluster events with embeddings + UMAP + HDBSCAN.
3) Generate final brief text with LLM.
4) Generate title + continuity TLDR.
5) Publish the report through /reports/report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any
from urllib.parse import urlparse

import hdbscan
import numpy as np
import requests
import umap
from dotenv import load_dotenv
from json_repair import repair_json
from sentence_transformers import SentenceTransformer

from src.events import Event, Source, get_events
from src.llm import call_llm


LOGGER = logging.getLogger("daily-brief-generator")


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_FINAL_MODEL = "gemini-2.5-pro-exp-03-25"
DEFAULT_FINAL_MODEL_FALLBACK = "gemini-2.0-flash"
DEFAULT_TITLE_MODEL = "gemini-2.0-flash"
DEFAULT_TLDR_MODEL = "gemini-2.0-flash"


@dataclass(frozen=True)
class EventRecord:
    id: int
    source_id: int
    source_name: str
    title: str
    url: str
    publish_date: datetime | None
    summary: str
    content: str


@dataclass(frozen=True)
class ClusterRecord:
    article_indices: list[int]
    score: float
    source_diversity: int
    latest_publish_date: datetime | None


@dataclass(frozen=True)
class ClusterParams:
    umap_n_neighbors: int
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_epsilon: float


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and publish the daily intelligence brief.")
    parser.add_argument(
        "--date",
        dest="target_date",
        type=str,
        default=None,
        help="Target day in YYYY-MM-DD (UTC). Defaults to current UTC day.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force generation even if a report already exists for the target day.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end generation but skip publishing.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=int(os.environ.get("MAX_CLUSTERS", "18")),
        help="Maximum number of clusters included in LLM input.",
    )
    parser.add_argument(
        "--max-articles-per-cluster",
        type=int,
        default=int(os.environ.get("MAX_ARTICLES_PER_CLUSTER", "8")),
        help="Maximum number of articles included per cluster in LLM input.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args()


def parse_target_date(raw: str | None) -> date:
    if raw is None or raw.strip() == "":
        return datetime.now(timezone.utc).date()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid --date '{raw}'. Expected YYYY-MM-DD.") from exc


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ValueError(f"{name} environment variable is required.")
    return value.strip()


def ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_last_report(base_url: str, secret: str, timeout_seconds: int = 60) -> dict[str, Any] | None:
    response = requests.get(
        f"{base_url}/reports/last-report",
        headers={"Authorization": f"Bearer {secret}"},
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def report_date_utc(report: dict[str, Any] | None) -> date | None:
    if report is None:
        return None
    created_at = report.get("createdAt")
    if not created_at:
        return None
    parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    return ensure_utc(parsed).date()


def extract_event_summary(raw_summary: str | None, fallback_title: str) -> str:
    if raw_summary is None or raw_summary.strip() == "":
        return fallback_title.strip()

    summary = raw_summary.strip()
    event_match = re.search(r"EVENT:\s*(.*?)(?:\n[A-Z_]+:|$)", summary, flags=re.IGNORECASE | re.DOTALL)
    context_match = re.search(r"CONTEXT:\s*(.*?)(?:\n[A-Z_]+:|$)", summary, flags=re.IGNORECASE | re.DOTALL)

    parts: list[str] = []
    if event_match:
        parts.append(event_match.group(1).strip())
    if context_match:
        parts.append(context_match.group(1).strip())

    if parts:
        return " ".join(parts).strip()

    # Fallback for non-structured summaries.
    return re.sub(r"\s+", " ", summary).strip()


def source_name_map(sources: list[Source]) -> dict[int, str]:
    return {source.id: source.name for source in sources}


def build_event_records(sources: list[Source], events: list[Event]) -> list[EventRecord]:
    source_map = source_name_map(sources)
    records: list[EventRecord] = []
    for event in events:
        publish_date = ensure_utc(event.publishDate)
        summary = extract_event_summary(event.summary, event.title)
        content = (event.content or "").strip()
        records.append(
            EventRecord(
                id=event.id,
                source_id=event.sourceId,
                source_name=source_map.get(event.sourceId, f"source-{event.sourceId}"),
                title=(event.title or "").strip(),
                url=(event.url or "").strip(),
                publish_date=publish_date,
                summary=summary,
                content=content,
            )
        )
    return records


def embedding_inputs(records: list[EventRecord]) -> list[str]:
    payload: list[str] = []
    for record in records:
        text = record.summary if record.summary else record.title
        if not text:
            text = f"article-{record.id}"
        payload.append(f"query: {text}")
    return payload


def generate_embeddings(records: list[EventRecord], model_name: str) -> np.ndarray:
    inputs = embedding_inputs(records)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return np.asarray(embeddings)


def cluster_score(cluster_records: list[EventRecord], now_utc: datetime) -> tuple[float, int, datetime | None]:
    source_diversity = len({record.source_name for record in cluster_records})
    latest_publish_date = max(
        (record.publish_date for record in cluster_records if record.publish_date is not None),
        default=None,
    )
    recency_bonus = 0.0
    if latest_publish_date is not None:
        age_hours = max(0.0, (now_utc - latest_publish_date).total_seconds() / 3600.0)
        recency_bonus = max(0.0, 1.0 - (age_hours / 72.0))

    score = (len(cluster_records) * 0.65) + (source_diversity * 0.25) + (recency_bonus * 0.10)
    return score, source_diversity, latest_publish_date


def fallback_singletons(records: list[EventRecord], now_utc: datetime) -> tuple[list[ClusterRecord], ClusterParams]:
    clusters: list[ClusterRecord] = []
    for idx, record in enumerate(records):
        score, source_diversity, latest_publish_date = cluster_score([record], now_utc)
        clusters.append(
            ClusterRecord(
                article_indices=[idx],
                score=score,
                source_diversity=source_diversity,
                latest_publish_date=latest_publish_date,
            )
        )
    params = ClusterParams(
        umap_n_neighbors=1,
        hdbscan_min_cluster_size=1,
        hdbscan_min_samples=1,
        hdbscan_epsilon=0.0,
    )
    return clusters, params


def cluster_events(records: list[EventRecord], embeddings: np.ndarray) -> tuple[list[ClusterRecord], ClusterParams]:
    now_utc = datetime.now(timezone.utc)
    n = len(records)

    if n == 0:
        return [], ClusterParams(1, 1, 1, 0.0)
    if n < 6:
        return fallback_singletons(records, now_utc)

    umap_n_neighbors = min(20, max(5, n // 6))
    hdbscan_min_cluster_size = min(12, max(3, n // 10 + 2))
    hdbscan_min_samples = max(2, min(5, hdbscan_min_cluster_size // 2))
    hdbscan_epsilon = 0.15

    try:
        reduced = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=min(10, max(2, n - 1)),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_epsilon,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced)
    except Exception as exc:
        LOGGER.warning("Clustering failed; falling back to singleton clusters: %s", exc)
        return fallback_singletons(records, now_utc)

    grouped: dict[int, list[int]] = defaultdict(list)
    noise_indices: list[int] = []
    for idx, label in enumerate(labels.tolist()):
        if label == -1:
            noise_indices.append(idx)
        else:
            grouped[int(label)].append(idx)

    clusters: list[ClusterRecord] = []
    for indices in grouped.values():
        subset = [records[idx] for idx in indices]
        score, source_diversity, latest_publish_date = cluster_score(subset, now_utc)
        clusters.append(
            ClusterRecord(
                article_indices=sorted(indices, key=lambda i: records[i].publish_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True),
                score=score,
                source_diversity=source_diversity,
                latest_publish_date=latest_publish_date,
            )
        )

    # Keep noise events as singleton clusters so niche but important stories are not dropped.
    for idx in noise_indices:
        subset = [records[idx]]
        score, source_diversity, latest_publish_date = cluster_score(subset, now_utc)
        clusters.append(
            ClusterRecord(
                article_indices=[idx],
                score=score,
                source_diversity=source_diversity,
                latest_publish_date=latest_publish_date,
            )
        )

    clusters.sort(key=lambda c: c.score, reverse=True)
    params = ClusterParams(
        umap_n_neighbors=umap_n_neighbors,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_epsilon=hdbscan_epsilon,
    )
    return clusters, params


def compact_text(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def format_clusters_for_llm(
    clusters: list[ClusterRecord],
    records: list[EventRecord],
    max_clusters: int,
    max_articles_per_cluster: int,
) -> tuple[str, list[int]]:
    lines: list[str] = []
    included_article_ids: list[int] = []

    for cluster_idx, cluster in enumerate(clusters[:max_clusters], start=1):
        article_indices = cluster.article_indices[:max_articles_per_cluster]
        cluster_records = [records[i] for i in article_indices]
        included_article_ids.extend(record.id for record in cluster_records)

        lines.append(f"## Cluster {cluster_idx}")
        lines.append(f"- Signal score: {cluster.score:.3f}")
        lines.append(f"- Article count in cluster: {len(cluster.article_indices)}")
        lines.append(f"- Source diversity: {cluster.source_diversity}")
        if cluster.latest_publish_date is not None:
            lines.append(f"- Latest publish time (UTC): {cluster.latest_publish_date.isoformat()}")
        lines.append("- Articles:")

        for record in cluster_records:
            publish_time = record.publish_date.isoformat() if record.publish_date else "unknown"
            summary = compact_text(record.summary, 320)
            if not summary:
                summary = compact_text(record.content, 320)
            lines.append(
                f"  - [#{record.id}] {record.title} | {record.source_name} | {publish_time}\n"
                f"    URL: {record.url}\n"
                f"    Summary: {summary}"
            )
        lines.append("")

    return "\n".join(lines).strip(), sorted(set(included_article_ids))


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = strip_code_fences(raw)
    repaired = repair_json(cleaned)
    parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def extract_tagged_section(text: str, start_tag: str, end_tag: str) -> str:
    if start_tag in text and end_tag in text:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return text.strip()


def run_llm_with_fallback(
    primary_model: str,
    fallback_model: str | None,
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, str]:
    try:
        answer, _usage = call_llm(model=primary_model, messages=messages, temperature=temperature)
        return answer, primary_model
    except Exception as primary_exc:
        if not fallback_model:
            raise
        LOGGER.warning(
            "Primary model '%s' failed, trying fallback '%s': %s",
            primary_model,
            fallback_model,
            primary_exc,
        )
        answer, _usage = call_llm(model=fallback_model, messages=messages, temperature=temperature)
        return answer, fallback_model


def previous_report_context(last_report: dict[str, Any] | None, target_day: date) -> str:
    if not last_report:
        return ""
    created = report_date_utc(last_report)
    if created is None or created >= target_day:
        return ""

    title = str(last_report.get("title") or "").strip()
    tldr = str(last_report.get("tldr") or "").strip()
    created_at = str(last_report.get("createdAt") or "")
    if not title or not tldr or not created_at:
        return ""

    day = created_at.split("T", 1)[0]
    return (
        f"## Previous Day Coverage Context ({day})\n\n"
        f"### {title}\n\n"
        f"{tldr}\n"
    )


def build_brief_prompt(cluster_markdown: str, previous_context: str, target_day: date) -> str:
    return f"""
You are preparing a private daily intelligence brief.

Date (UTC): {target_day.isoformat()}

Your job:
1) Read all cluster data.
2) Identify the highest-signal stories.
3) Synthesize what changed, why it matters, and likely implications.
4) Keep writing grounded in provided data only.
5) Prefer substance over length.

If previous context is provided, use it only for continuity and focus on what changed today.

{previous_context}

<cluster_data>
{cluster_markdown}
</cluster_data>

Return only the brief content wrapped in tags:
<final_brief>
## what matters now
...
## france focus
...
## global landscape
### power & politics
...
### china monitor
...
### economic currents
...
## tech & science developments
...
## noteworthy & under-reported
...
## positive developments
...
</final_brief>

Style requirements:
- use lowercase by default
- clear, direct, analytical voice
- include concrete entities and numbers when available
- no fluff, no generic hedging
""".strip()


def generate_final_brief(
    cluster_markdown: str,
    previous_context: str,
    target_day: date,
    final_model: str,
    fallback_model: str | None,
) -> tuple[str, str]:
    prompt = build_brief_prompt(cluster_markdown, previous_context, target_day)
    answer, model_used = run_llm_with_fallback(
        primary_model=final_model,
        fallback_model=fallback_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    brief = extract_tagged_section(answer, "<final_brief>", "</final_brief>")
    if brief == "":
        raise ValueError("Final brief generation returned empty content.")
    return brief, model_used


def generate_brief_title(brief_text: str, title_model: str) -> str:
    prompt = f"""
<brief>
{brief_text}
</brief>

Create a concise factual title for this brief.
Requirements:
- lowercase
- no colon
- non-clickbait
- represent major themes/entities

Return only JSON:
{{
  "title": "string"
}}
""".strip()

    answer, _usage = call_llm(
        model=title_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    parsed = parse_json_object(answer)
    title = str(parsed.get("title") or "").strip()
    if not title:
        raise ValueError("Title model returned empty title.")
    return title


def generate_tldr_context(brief_title: str, brief_text: str, tldr_model: str) -> str:
    prompt = f"""
You are generating a compressed continuity context for tomorrow.

Input brief:
<final_brief>
# {brief_title}

{brief_text}
</final_brief>

Output only lines in this exact format:
[Story Identifier] | [Inferred Status] | [Key Entities] | [Core Issue Snippet]

Rules:
- one line per major story
- no extra commentary
- concise and factual
""".strip()

    answer, _usage = call_llm(
        model=tldr_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    tldr = strip_code_fences(answer)
    return tldr.strip()


def source_domains_for_articles(records: list[EventRecord], included_article_ids: list[int]) -> list[str]:
    id_set = set(included_article_ids)
    domains: set[str] = set()
    for record in records:
        if record.id not in id_set:
            continue
        parsed = urlparse(record.url)
        domain = parsed.netloc.strip().lower()
        if domain:
            domains.add(domain)
    return sorted(domains)


def publish_report(
    base_url: str,
    secret: str,
    title: str,
    brief_text: str,
    tldr: str,
    records: list[EventRecord],
    sources: list[Source],
    included_article_ids: list[int],
    model_author: str,
    cluster_params: ClusterParams,
    report_created_at: datetime,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    body = {
        "title": title,
        "content": brief_text,
        "totalArticles": len(records),
        "totalSources": len(sources),
        "usedArticles": len(set(included_article_ids)),
        "usedSources": len(source_domains_for_articles(records, included_article_ids)),
        "tldr": tldr,
        "model_author": model_author,
        "createdAt": ensure_utc(report_created_at).isoformat(),
        "clustering_params": {
            "umap": {"n_neighbors": cluster_params.umap_n_neighbors},
            "hdbscan": {
                "min_cluster_size": cluster_params.hdbscan_min_cluster_size,
                "min_samples": cluster_params.hdbscan_min_samples,
                "epsilon": cluster_params.hdbscan_epsilon,
            },
        },
    }

    response = requests.post(
        f"{base_url}/reports/report",
        headers={"Authorization": f"Bearer {secret}"},
        json=body,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_level)

    try:
        target_day = parse_target_date(args.target_date)
        base_url = require_env("MERIDIAN_API_URL")
        secret = require_env("MERIDIAN_SECRET_KEY")

        embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
        final_model = os.environ.get("FINAL_BRIEF_MODEL", DEFAULT_FINAL_MODEL)
        fallback_model = os.environ.get("BRIEF_FALLBACK_MODEL", DEFAULT_FINAL_MODEL_FALLBACK)
        title_model = os.environ.get("TITLE_MODEL", DEFAULT_TITLE_MODEL)
        tldr_model = os.environ.get("TLDR_MODEL", DEFAULT_TLDR_MODEL)

        LOGGER.info("Target day (UTC): %s", target_day.isoformat())
        LOGGER.info("Fetching previous report state for idempotency checks...")
        last_report = fetch_last_report(base_url=base_url, secret=secret)
        last_report_day = report_date_utc(last_report)
        if last_report_day is not None:
            LOGGER.info("Latest published report day (UTC): %s", last_report_day.isoformat())

        if not args.force and last_report_day == target_day:
            LOGGER.info(
                "A report already exists for %s (UTC). Skipping generation. Use --force to override.",
                target_day.isoformat(),
            )
            return 0

        LOGGER.info("Fetching processed events from worker API...")
        sources, events = get_events(date=target_day.isoformat(), timeout_seconds=120)
        if len(events) == 0:
            raise RuntimeError(f"No events returned for target day {target_day.isoformat()}.")
        LOGGER.info("Fetched %s events from %s sources.", len(events), len(sources))

        records = build_event_records(sources, events)
        LOGGER.info("Generating embeddings using model: %s", embedding_model)
        embeddings = generate_embeddings(records, embedding_model)

        LOGGER.info("Clustering events...")
        clusters, cluster_params = cluster_events(records, embeddings)
        if len(clusters) == 0:
            raise RuntimeError("No clusters were produced.")
        LOGGER.info("Built %s clusters.", len(clusters))

        cluster_markdown, included_article_ids = format_clusters_for_llm(
            clusters=clusters,
            records=records,
            max_clusters=max(1, args.max_clusters),
            max_articles_per_cluster=max(1, args.max_articles_per_cluster),
        )
        if not included_article_ids:
            raise RuntimeError("No article IDs selected for final brief generation.")
        LOGGER.info(
            "Prepared LLM input with %s clusters and %s article references.",
            min(len(clusters), max(1, args.max_clusters)),
            len(included_article_ids),
        )

        continuity_context = previous_report_context(last_report, target_day)
        LOGGER.info("Generating final brief...")
        brief_text, brief_model_used = generate_final_brief(
            cluster_markdown=cluster_markdown,
            previous_context=continuity_context,
            target_day=target_day,
            final_model=final_model,
            fallback_model=fallback_model,
        )

        LOGGER.info("Generating title...")
        brief_title = generate_brief_title(brief_text=brief_text, title_model=title_model)
        LOGGER.info("Generating continuity TLDR context...")
        brief_tldr = generate_tldr_context(brief_title=brief_title, brief_text=brief_text, tldr_model=tldr_model)
        if brief_tldr == "":
            raise RuntimeError("Generated TLDR context is empty.")

        LOGGER.info("Generation complete.")
        LOGGER.info("Title: %s", brief_title)
        LOGGER.info("Model author: %s", brief_model_used)
        LOGGER.info("Used articles: %s", len(set(included_article_ids)))

        if args.dry_run:
            LOGGER.info("Dry run enabled; skipping publish.")
            print("\n===== BRIEF TITLE =====")
            print(brief_title)
            print("\n===== BRIEF PREVIEW (first 1200 chars) =====")
            print(brief_text[:1200])
            print("\n===== TLDR PREVIEW =====")
            print(brief_tldr)
            return 0

        # Re-check idempotency right before publish.
        final_last_report = fetch_last_report(base_url=base_url, secret=secret)
        final_last_day = report_date_utc(final_last_report)
        if not args.force and final_last_day == target_day:
            LOGGER.warning(
                "Another run already published a report for %s before publish step. Skipping publish.",
                target_day.isoformat(),
            )
            return 0

        LOGGER.info("Publishing report...")
        # Use the target reporting day for createdAt to keep date-slug retrieval stable.
        report_created_at = datetime.combine(target_day, time(7, 0, tzinfo=timezone.utc))
        publish_response = publish_report(
            base_url=base_url,
            secret=secret,
            title=brief_title,
            brief_text=brief_text,
            tldr=brief_tldr,
            records=records,
            sources=sources,
            included_article_ids=included_article_ids,
            model_author=brief_model_used,
            cluster_params=cluster_params,
            report_created_at=report_created_at,
        )
        LOGGER.info("Publish response: %s", publish_response)
        LOGGER.info("Daily report generation finished successfully.")
        return 0

    except Exception as exc:
        LOGGER.exception("Daily report generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
