# ForesightAI: Your Personal Intelligence Agency

[![Build Status](https://img.shields.io/github/actions/workflow/status/MysticLiu/ForesightAI/deploy-services.yaml?branch=main)](https://github.com/MysticLiu/ForesightAI/actions/workflows/deploy-services.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Presidential-level intelligence briefings, built with AI, tailored for you.**

ForesightAI cuts through news noise by scraping hundreds of sources, analyzing stories with AI, and delivering concise, personalized daily briefs.

## Why It Exists

Presidents get tailored daily intelligence briefs. Now with AI, you can too. ForesightAI delivers:

- Key global events filtered by relevance
- Context and underlying drivers
- Analysis of implications
- Open-source transparency

Built for the curious who want depth beyond headlines without the time sink.

## Key Features

- **Source Coverage**: Hundreds of diverse news sources
- **AI Analysis**: Multi-stage LLM processing (Gemini) for article and cluster analysis
- **Smart Clustering**: Embeddings + UMAP + HDBSCAN to group related articles
- **Personalized Briefing**: Daily brief with analytical voice and continuity tracking
- **Web Interface**: Clean Nuxt 3 frontend

## How It Works

1. **Scraping**: Cloudflare Workers fetch RSS feeds, store metadata
2. **Processing**: Extract text, analyze with Gemini for relevance and structure
3. **Brief Generation**: Cluster articles, generate analysis, synthesize final brief
4. **Frontend**: Display briefs via Nuxt/Vercel

## Tech Stack

- **Infrastructure**: Turborepo, Cloudflare (Workers, Workflows), Vercel
- **Backend**: Hono, TypeScript, PostgreSQL (Neon), Drizzle
- **AI/ML**: Gemini models, multilingual-e5-small embeddings, UMAP, HDBSCAN
- **Frontend**: Nuxt 3, Vue 3, Tailwind

## Setup

See [SETUP.md](./SETUP.md) for detailed setup instructions.

**Prerequisites**: Node.js v22+, pnpm v9.15+, Python 3.10+, PostgreSQL, Cloudflare account, Google AI API key

```bash
git clone https://github.com/MysticLiu/ForesightAI.git
cd ForesightAI
pnpm install
# Configure .env files (see SETUP.md)
# Add GitHub secrets and push to main to deploy
```

## Current Status (Updated: 2026-02-23)

ForesightAI is currently in a **working beta/prototype** state.

### What is working now

- **Automated ingestion**: RSS scraping runs on Cloudflare schedule.
- **Automated article processing**: extraction + AI analysis (relevance/completeness/location/summary) runs via workflows.
- **Automated brief generation**: production CLI (`apps/briefs/generate_daily_report.py`) creates + publishes daily reports non-interactively.
- **Data pipeline persistence**: sources, articles, reports, newsletter subscriptions are stored in PostgreSQL.
- **Report APIs + frontend rendering**: briefs list/detail/latest pages and server APIs are functional.
- **Core validation checks**: parser tests and TypeScript checks currently pass.

### What is not yet fully automated

- **Automated newsletter delivery of generated briefs** is not yet complete (subscription capture exists; full send pipeline still pending).
- **Quality gate/evaluation harness** for generated briefs is still pending for consistent editorial scoring.
- **Operational dashboards/SLO reporting** for report-generation reliability is still pending.

### Immediate reliability/correctness issues to fix

- CI deploy workflow still references `@meridian/frontend` in one step (workspace package is `@foresight-ai/frontend`).
- Brief slug/date lookup has timezone inconsistencies that can make some reports hard to resolve reliably.
- OG image paths are inconsistent on some pages (`/og/default` vs `/openGraph/default`).
- Scraper workflow has edge-case logic issues (tricky-domain browser-first flow and source `lastChecked` updates).
- Subscription flow should better decouple DB success from optional MailerLite failures.
- Markdown rendering should be explicitly hardened/sanitized for safety policy clarity.

## Executive Build-Next Plan (2026 H1)

### North Star

Deliver one high-quality intelligence brief **every day without manual intervention**, then reliably publish and distribute it to users (web + email), with continuity from prior days.

### Guiding priorities (in order)

1. **P0: Correctness + reliability hardening**
2. **P1: Full daily automation of final brief generation/publishing**
3. **P2: Quality and trust controls**
4. **P3: Delivery + personalization**
5. **P4: Production maturity and operations**

### Phase 0: Stabilize Foundations (2026-02-23 to 2026-03-13)

- Fix CI/package naming mismatches and deployment script drift.
- Fix report identity/slug/date handling for timezone-safe deterministic retrieval.
- Align OG endpoint paths everywhere.
- Harden scraper/workflow edge cases and `lastChecked` behavior.
- Harden subscription API behavior and provider-failure handling.
- Decide and implement markdown safety policy.
- Add regression tests for all above.

### Phase 1: Automate Daily Generation (2026-03-16 to 2026-04-10)

- Expand scheduler observability + alerts around the production generator.
- Improve idempotency and backfill controls for reruns and late data.
- Add artifact snapshots for auditability and reproducibility per run.
- Define publish runbook and fast recovery path for failed days.

### Phase 2: Quality + Trust System (2026-04-13 to 2026-05-08)

- Implement a measurable brief quality rubric (relevance, coherence, novelty, continuity, grounding).
- Strengthen reliability weighting in cluster synthesis and contradiction handling.
- Add category/coverage checks so briefs do not miss major signal when available.
- Improve previous-day continuity handoff quality.
- Optional temporary human-approval draft gate while automation stabilizes.

### Phase 3: Delivery + Product Value (2026-05-11 to 2026-06-05)

- Build full daily email delivery pipeline (not just subscription capture).
- Add subscriber lifecycle features (unsubscribe/suppression/error handling).
- Add preference-aware personalization (topic/region weighting).
- Improve archive usability (filter/search/navigation by metadata).

### Phase 4: Production Maturity (2026-06-08 to 2026-06-26)

- Add SLO dashboards (pipeline success, latency, publish timeliness, delivery reliability).
- Add cost/token guardrails and model fallback policies.
- Add incident playbooks (missed brief, API failure, model outage, DB issues).
- Add change governance for prompt/model/schema updates.

### Success metrics

- Daily brief generation success rate
- Time-to-publish from ingestion start
- Article processing success ratio
- Brief quality score trend
- Duplicate report rate
- Email delivery success/bounce/unsubscribe rates
- Daily compute/token cost envelope

## License

MIT License - See [LICENSE](./LICENSE) file for details.

---

*Forked from [iliane5/meridian](https://github.com/iliane5/meridian) - Original concept by Iliane Amadou*
