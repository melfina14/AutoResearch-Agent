
import json
import re
from app.llm import generate


NOT_SPECIFIED = "Not specified in the provided analysis"

LIST_FIELDS = (
    "common_research_themes",
    "methodology_comparison",
    "dataset_comparison",
    "key_findings",
    "similarities",
    "differences",
    "research_conflicts",
)

EXPECTED_FIELDS = LIST_FIELDS + ("overall_summary",)


def _clean_text(value) -> str:
    """Convert a value into clean text without inventing information."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    return ""


def _prepare_analyses(analyses: list) -> list[dict]:
    """Keep usable paper analyses and remove empty entries."""
    prepared = []

    for item in analyses:
        if not isinstance(item, dict):
            continue

        paper_title = _clean_text(
            item.get("paper") or item.get("title")
        )

        analysis_data = item.get("analysis")

        # Support both wrapped analysis-agent outputs and direct analyses.
        if isinstance(analysis_data, dict):
            analysis_data = analysis_data.copy()
        else:
            analysis_data = {
                key: value
                for key, value in item.items()
                if key not in ("paper", "title", "year", "url")
            }

        if not analysis_data:
            continue

        paper_record = {
            "paper": paper_title or "Title not provided",
            "year": item.get("year"),
            "url": _clean_text(item.get("url")),
            "analysis": analysis_data,
        }

        # Remove unavailable metadata rather than implying it is known.
        paper_record = {
            key: value
            for key, value in paper_record.items()
            if value is not None and value != ""
        }

        prepared.append(paper_record)

    return prepared


def _extract_json(response: str) -> dict:
    """Parse a JSON object, including one enclosed in Markdown fences."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The LLM returned an empty response.")

    cleaned = response.strip()

    # Remove optional Markdown code fences.
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, count=1)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to extract a JSON object if extra text surrounds it.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("No valid JSON object was found.")

        result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the LLM.")

    return result


def _normalize_comparison(comparison: dict) -> dict:
    """Ensure the comparison output follows the expected schema."""
    normalized = {}

    for field in LIST_FIELDS:
        value = comparison.get(field)

        if value is None:
            normalized[field] = [NOT_SPECIFIED]
        elif isinstance(value, str):
            normalized[field] = [value.strip()] if value.strip() else [
                NOT_SPECIFIED
            ]
        elif isinstance(value, list):
            items = [
                text
                for item in value
                if (text := _clean_text(item))
            ]
            normalized[field] = items or [NOT_SPECIFIED]
        else:
            normalized[field] = [NOT_SPECIFIED]

    summary = comparison.get("overall_summary")
    if isinstance(summary, str) and summary.strip():
        normalized["overall_summary"] = summary.strip()
    else:
        normalized["overall_summary"] = NOT_SPECIFIED

    return normalized


def comparison_agent(analyses: list[dict]) -> dict:
    """Compare analyzed research papers using only supplied evidence."""

    if not isinstance(analyses, list):
        return {
            "error": "Input must be a list of paper analyses."
        }

    valid_analyses = _prepare_analyses(analyses)

    if len(valid_analyses) < 2:
        return {
            "error": (
                "At least two usable paper analyses are required "
                "for comparison."
            ),
            "papers_compared": len(valid_analyses),
        }

    paper_data = json.dumps(
        valid_analyses,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are the Comparison Agent in a domain-independent academic
research system.

Compare the supplied paper analyses carefully and objectively.

EVIDENCE RULES:
1. Use only the information in the supplied paper analyses.
2. Do not invent methods, datasets, findings, metrics, limitations,
   citations, or publication details.
3. The supplied information may be based on abstracts only.
   Do not claim to have read full papers.
4. Clearly distinguish reported information from missing information.
5. Missing information does not mean that a method, dataset, or
   evaluation was not used.
6. Report a research conflict only when the supplied evidence shows
   genuinely inconsistent claims or results about comparable questions.
   Different methods or results alone do not necessarily constitute
   a conflict.
7. Do not infer that papers are duplicates from similar titles,
   overlapping keywords, or similar summaries.
8. Compare numerical results only when the supplied analyses provide
   them. Do not assume results are directly comparable when evaluation
   settings or datasets differ or are unknown.
9. Avoid unsupported claims about which method is best or superior.
10. Make comparisons relevant to the research domain, whatever it is.
11. If information is unavailable, write:
    "{NOT_SPECIFIED}"
12. If no research conflict is supported, explicitly state that no
    conflict is established by the supplied analyses.

PAPER ANALYSES:
{paper_data}

Return ONLY a valid JSON object with exactly these fields:
{{
  "common_research_themes": [],
  "methodology_comparison": [],
  "dataset_comparison": [],
  "key_findings": [],
  "similarities": [],
  "differences": [],
  "research_conflicts": [],
  "overall_summary": ""
}}

All fields except "overall_summary" must be lists of concise strings.
"overall_summary" must be a string.
Do not add Markdown fences or text outside the JSON object.
"""

    try:
        response = generate(prompt)
        comparison = _extract_json(response)
        normalized = _normalize_comparison(comparison)

        return {
            "papers_compared": len(valid_analyses),
            "comparison": normalized,
        }

    except Exception as exc:
        return {
            "papers_compared": len(valid_analyses),
            "error": f"Comparison failed: {exc}",
            "comparison": None,
        }
