
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from app.config import SEMANTIC_SCHOLAR_API_KEY


BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 20
REQUEST_TIMEOUT_SECONDS = 30

FIELDS = (
    "title,abstract,authors,year,url,venue,openAccessPdf,"
    "externalIds,citationCount,publicationTypes"
)


def get_retry_delay(response, attempt: int) -> float:
    """Determine a safe delay using Retry-After or exponential backoff."""
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return min(
                max(float(retry_after), 0),
                MAX_BACKOFF_SECONDS,
            )
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)

                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)

                seconds = (
                    retry_time - datetime.now(timezone.utc)
                ).total_seconds()

                return min(
                    max(seconds, 0),
                    MAX_BACKOFF_SECONDS,
                )
            except (TypeError, ValueError, OverflowError):
                pass

    delay = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
    return min(delay, MAX_BACKOFF_SECONDS)


def search_papers(query: str, limit: int = 10) -> list[dict]:
    """
    Search Semantic Scholar with retries for temporary failures.

    Returns the paper records in the same format as the original
    function. Raises an exception if all retries fail.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    headers = {}

    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    params = {
        "query": query.strip(),
        "limit": min(limit, 100),
        "fields": FIELDS,
        "openAccessPdf": "",
    }

    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                BASE,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= MAX_RETRIES:
                raise

            delay = min(
                INITIAL_BACKOFF_SECONDS * (2 ** attempt),
                MAX_BACKOFF_SECONDS,
            )

            print(
                "Semantic Scholar connection issue. "
                f"Retrying in {delay} seconds "
                f"(attempt {attempt + 1}/{MAX_RETRIES})."
            )

            time.sleep(delay)
            continue

        if response.status_code in retryable_statuses:
            if attempt >= MAX_RETRIES:
                response.raise_for_status()

            delay = get_retry_delay(response, attempt)

            print(
                f"Semantic Scholar returned HTTP "
                f"{response.status_code}. "
                f"Retrying in {delay:g} seconds "
                f"(attempt {attempt + 1}/{MAX_RETRIES})."
            )

            response.close()
            time.sleep(delay)
            continue

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(
                "Semantic Scholar returned an unexpected response format."
            )

        papers = data.get("data", [])

        if not isinstance(papers, list):
            raise ValueError(
                "Semantic Scholar returned an invalid paper list."
            )

        return papers

    return []
