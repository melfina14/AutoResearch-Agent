
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote


ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, limit: int = 5) -> list[dict]:
    """
    Search arXiv for academic papers.

    Returns paper metadata in a format suitable for the
    AutoResearch Agent retrieval pipeline.
    """

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(limit, 20),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "AutoResearchAgent/1.0 (academic project)"
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall("atom:entry", ATOM_NS):
        title = " ".join(
            (entry.findtext("atom:title", default="", namespaces=ATOM_NS)).split()
        )
        abstract = " ".join(
            (entry.findtext("atom:summary", default="", namespaces=ATOM_NS)).split()
        )

        paper_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        published = entry.findtext(
            "atom:published", default="", namespaces=ATOM_NS
        )

        authors = [
            author.findtext("atom:name", default="", namespaces=ATOM_NS)
            for author in entry.findall("atom:author", ATOM_NS)
        ]

        arxiv_id = paper_url.rstrip("/").split("/")[-1]

        papers.append({
            "paperId": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "url": paper_url,
            "source": "arXiv",
        })

    return papers
