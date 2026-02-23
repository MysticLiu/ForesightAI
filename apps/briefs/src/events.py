import os
from datetime import datetime

import requests
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()


class Source(BaseModel):
    id: int
    name: str


class Event(BaseModel):
    id: int
    sourceId: int
    url: str
    title: str
    publishDate: datetime  # changed from date to datetime
    content: str
    location: str
    relevance: str
    completeness: str
    summary: str

    @field_validator("publishDate", mode="before")
    @classmethod
    def parse_date(cls, value):
        if value is None:
            return None

        # Handle ISO format with timezone info
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # For older Python versions or non-standard formats
            # you might need dateutil
            from dateutil import parser

            return parser.parse(value)


def get_events(date: str | None = None, timeout_seconds: int = 60):
    # MERIDIAN_API_URL is required - set it to your Cloudflare Worker URL
    base_url = os.environ.get("MERIDIAN_API_URL")
    if not base_url:
        raise ValueError("MERIDIAN_API_URL environment variable is required. Set it to your Cloudflare Worker URL.")
    url = f"{base_url}/events"

    if date:
        url += f"?date={date}"

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('MERIDIAN_SECRET_KEY')}"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()

    if "sources" not in data or "events" not in data:
        raise ValueError("Invalid /events response payload: expected 'sources' and 'events' keys.")

    sources = [Source(**source) for source in data["sources"]]
    events = [Event(**event) for event in data["events"]]

    return sources, events
