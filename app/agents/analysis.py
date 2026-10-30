
import json
from typing import Any

from app.llm import generate


NOT_SPECIFIED = "Not specified in the abstract"

EXPECTED_STRING_FIELDS = (
    "objective",
    "methodology",
    "dataset",
    "research_domain",
)

EXPECTED_LIST_FIELDS = (
    "key_results",
    "limitations",
    "keywords",
)

ALLOWED_QUALITY_LABELS = {
    "missing",
    "short",
    "review",
    "unchecked",
}


def _to_text(value: Any, default: str = NOT_SPECIFIED) -> str:
    """Convert a valid scalar value into a clean string."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)

    return default


def _to_string_list(value: Any) -> list[str]:
    """Normalize an LLM-generated list while rejecting nested structures."""
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    cleaned = []

    for item in value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            cleaned.append(str(item))

    return cleaned


def _parse_json_response(response: str) -> dict:
    """Parse a JSON object, including responses wrapped in code fences."""
    cleaned = response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # First, try parsing the complete response.
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover a JSON object if the model added surrounding prose.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("The response does not contain a JSON object.")

        result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the LLM.")

    return result


def _normalize_analysis(
    analysis: dict,
    abstract_quality: str,
    abstract_warning: str,
) -> dict:
    """Validate and normalize the structured analysis fields."""
    normalized = {}

    for field in EXPECTED_STRING_FIELDS:
        normalized[field] = _to_text(analysis.get(field))

    for field in EXPECTED_LIST_FIELDS:
        normalized[field] = _to_string_list(analysis.get(field))

    # Preserve the model's limitations while ensuring quality warnings
    # remain visible whenever the abstract has been flagged.
    if abstract_quality in {"short", "review"}:
        warning = (
            abstract_warning
            or "Abstract requires manual quality review."
        )

        if warning not in normalized["limitations"]:
            normalized["limitations"].append(warning)

    normalized["evidence_basis"] = "Abstract only"
    normalized["abstract_quality"] = abstract_quality
    normalized["abstract_warning"] = abstract_warning

    return normalized


def analysis_agent(paper: dict) -> dict:
    """Analyze a scientific paper using abstract-level evidence only."""

    if not isinstance(paper, dict):
        return {
            "paper": "Unknown title",
            "year": "Unknown",
            "url": "",
            "analysis": {
                "error": "Invalid paper record; expected a dictionary.",
                "evidence_basis": "No abstract available",
            },
            "abstract_quality": "missing",
            "abstract_warning": "Invalid paper record.",
        }

    title = _to_text(paper.get("title"), "Unknown title")
    year = paper.get("year", "Unknown")
    url = paper.get("url", "")

    if not isinstance(url, str):
        url = ""

    abstract = paper.get("abstract", "")

    abstract_quality = paper.get("abstract_quality", "unchecked")
    abstract_warning = paper.get("abstract_warning", "")

    if abstract_quality not in ALLOWED_QUALITY_LABELS:
        abstract_quality = "unchecked"

    if not isinstance(abstract_warning, str):
        abstract_warning = ""

    # Missing abstracts cannot support a reliable paper analysis.
    if not isinstance(abstract, str) or not abstract.strip():
        return {
            "paper": title,
            "year": year,
            "url": url,
            "analysis": {
                "error": "No abstract available for analysis.",
                "objective": NOT_SPECIFIED,
                "methodology": NOT_SPECIFIED,
                "dataset": NOT_SPECIFIED,
                "key_results": [],
                "limitations": [
                    "No abstract is available; inspect the original paper."
                ],
                "research_domain": NOT_SPECIFIED,
                "keywords": [],
                "evidence_basis": "No abstract available",
                "abstract_quality": "missing",
                "abstract_warning": "Abstract is missing or empty.",
            },
            "abstract_quality": "missing",
            "abstract_warning": "Abstract is missing or empty.",
        }

    prompt = f"""
You are the Paper Analysis Agent in a multi-agent academic research system.

Analyze the paper using ONLY the title, year, and abstract supplied below.
The abstract may omit important details. Do not fill these gaps using guesses
or general knowledge.

PAPER METADATA
Title: {title}
Year: {year}

ABSTRACT QUALITY METADATA
Quality label: {abstract_quality}
Warning: {abstract_warning or "No automatic quality warning."}

PAPER ABSTRACT
<abstract>
{abstract}
</abstract>

Return exactly one valid JSON object with these fields:
{{
  "objective": "Research problem or objective",
  "methodology": "Methods or techniques explicitly described",
  "dataset": "Dataset or data source explicitly identified",
  "key_results": ["Findings explicitly reported"],
  "limitations": ["Limitations explicitly reported or evidence gaps"],
  "research_domain": "Research domain supported by the paper",
  "keywords": ["Relevant terms grounded in the title or abstract"]
}}

RULES
1. Use only information supported by the supplied title and abstract.
2. For an unspecified string field, use "{NOT_SPECIFIED}".
3. For a field with no supported list items, return an empty list.
4. Do not invent datasets, sample sizes, metrics, numerical results,
   algorithms, causal claims, or limitations.
5. If the abstract does not report limitations, state in the limitations
   list that limitations are not specified in the abstract.
6. Distinguish reported findings from proposed future work.
7. If quality is "short" or "review", include the supplied warning in
   limitations and be cautious about the strength of the evidence.
8. A quality warning is not proof that the abstract is inaccurate.
9. "unchecked" means basic automated checks passed; it does not mean
   the abstract has been independently verified.
10. Do not claim to have read the full paper, checked citations, or verified
    results independently.
11. Return JSON only. Do not include Markdown fences or surrounding prose.
"""

    response = ""

    try:
        response = generate(prompt)

        if not isinstance(response, str) or not response.strip():
            raise ValueError("The LLM returned an empty response.")

        parsed = _parse_json_response(response)

        analysis = _normalize_analysis(
            parsed,
            abstract_quality,
            abstract_warning,
        )

    except Exception as exc:
        # Keep the pipeline's output predictable even if generation or
        # JSON parsing fails. Do not fabricate an analysis.
        analysis = {
            "error": f"Paper analysis failed: {exc}",
            "objective": NOT_SPECIFIED,
            "methodology": NOT_SPECIFIED,
            "dataset": NOT_SPECIFIED,
            "key_results": [],
            "limitations": [
                "Structured analysis could not be completed reliably."
            ],
            "research_domain": NOT_SPECIFIED,
            "keywords": [],
            "evidence_basis": "Abstract supplied; analysis unavailable",
            "abstract_quality": abstract_quality,
            "abstract_warning": abstract_warning,
        }

        if abstract_quality in {"short", "review"} and abstract_warning:
            analysis["limitations"].append(abstract_warning)

    return {
        "paper": title,
        "year": year,
        "url": url,
        "analysis": analysis,
        "abstract_quality": abstract_quality,
        "abstract_warning": abstract_warning,
    }
