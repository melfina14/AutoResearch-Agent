
import json
import re

from app.llm import generate


NOT_SPECIFIED = "Not specified in the provided comparison"

ALLOWED_GAP_TYPES = {
    "methodological",
    "dataset",
    "evaluation",
    "application",
    "theoretical",
}

ALLOWED_CONFIDENCE = {"Low", "Medium", "High"}

GAP_FIELDS = (
    "gap",
    "supporting_evidence",
    "gap_type",
    "importance",
    "confidence",
    "validation_needed",
)


def _clean_text(value, default=NOT_SPECIFIED) -> str:
    """Return clean text without inventing missing information."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    return default


def _extract_comparison(comparison):
    """Accept either raw comparison data or the comparison-agent wrapper."""
    if not isinstance(comparison, dict):
        return comparison

    if comparison.get("error"):
        raise ValueError(
            f"Comparison agent reported an error: {comparison['error']}"
        )

    nested = comparison.get("comparison")

    if isinstance(nested, dict):
        if nested.get("error"):
            raise ValueError(
                f"Comparison data contains an error: {nested['error']}"
            )
        return nested

    return comparison


def _parse_json_response(response: str) -> dict:
    """Parse a JSON object, including responses wrapped in code fences."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The LLM returned an empty response.")

    cleaned = response.strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, count=1)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("No valid JSON object was found.")

        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the LLM.")

    return parsed


def _normalize_gap(gap: dict) -> dict:
    """Normalize a gap while restricting categorical values."""
    normalized = {
        "gap": _clean_text(gap.get("gap")),
        "supporting_evidence": _clean_text(
            gap.get("supporting_evidence")
        ),
        "gap_type": _clean_text(gap.get("gap_type"), "Not specified"),
        "importance": _clean_text(gap.get("importance")),
        "confidence": _clean_text(gap.get("confidence"), "Low").title(),
        "validation_needed": _clean_text(
            gap.get("validation_needed"),
            "Validate against additional academic literature",
        ),
    }

    if normalized["gap_type"].lower() not in ALLOWED_GAP_TYPES:
        normalized["gap_type"] = "Not specified"

    if normalized["confidence"] not in ALLOWED_CONFIDENCE:
        normalized["confidence"] = "Low"

    return normalized


def gap_agent(comparison) -> dict:
    """Identify preliminary, evidence-based research gap candidates."""

    if comparison is None or comparison == {} or comparison == []:
        return {
            "agent": "Research Gap Detection Agent",
            "gap_analysis": {
                "error": "No paper comparison data provided."
            },
        }

    try:
        comparison = _extract_comparison(comparison)

        if isinstance(comparison, (dict, list)):
            comparison_data = json.dumps(
                comparison,
                indent=2,
                ensure_ascii=False,
            )
        elif isinstance(comparison, str) and comparison.strip():
            comparison_data = comparison.strip()
        else:
            raise ValueError(
                "Comparison data must be a non-empty object, list, or string."
            )

        prompt = f"""
You are the Research Gap Detection Agent in a domain-independent
academic research system.

Identify potential research gaps supported by the supplied
cross-paper comparison.

EVIDENCE RULES:
1. Base every proposed gap on specific evidence in the comparison.
2. Treat gaps as preliminary candidates, not confirmed facts about
   the entire research field.
3. Do not claim that a topic has never been studied unless the
   supplied evidence establishes this, which is unlikely.
4. Missing information is not proof that a research gap exists.
5. Do not infer that an experiment, dataset, evaluation, or method
   was absent merely because the comparison does not mention it.
6. Do not invent citations, studies, results, numerical values,
   limitations, or claims.
7. Distinguish the observed evidence from your interpretation.
8. Similar titles or similar findings alone do not establish a gap.
9. Do not claim that the comparison represents all published literature.
10. If only abstract-level evidence is available, state that the
    proposed gaps require validation using full papers and additional
    academic literature.
11. Do not force 3-5 gaps. Return fewer when evidence is limited.
12. If no specific gap is sufficiently supported, return an empty
    research_gaps list and explain why.
13. Use the research domain indicated by the comparison; do not assume
    the topic is healthcare or machine learning.

For each candidate, provide:
- gap: A concise description of the potential gap.
- supporting_evidence: The specific information in the comparison
  supporting this candidate. Do not fabricate quotations.
- gap_type: One of Methodological, Dataset, Evaluation, Application,
  or Theoretical.
- importance: Why investigating this candidate could matter.
- confidence: Low, Medium, or High, based on the supplied evidence.
- validation_needed: What additional evidence is needed to assess it.

COMPARISON DATA:
{comparison_data}

Return ONLY a valid JSON object in this structure:
{{
  "research_gaps": [
    {{
      "gap": "Description of the potential gap",
      "supporting_evidence": "Evidence from the supplied comparison",
      "gap_type": "Methodological",
      "importance": "Why this candidate may matter",
      "confidence": "Low",
      "validation_needed": "Evidence needed to validate this candidate"
    }}
  ],
  "overall_summary": "Evidence-based summary and limitations"
}}

The research_gaps value must be a list of objects.
The overall_summary value must be a string.
Do not add Markdown fences or text outside the JSON.
"""

        response = generate(prompt)
        result = _parse_json_response(response)

        raw_gaps = result.get("research_gaps", [])

        if not isinstance(raw_gaps, list):
            raise ValueError(
                "The research_gaps field must be a list."
            )

        normalized_gaps = [
            _normalize_gap(gap)
            for gap in raw_gaps
            if isinstance(gap, dict)
        ]

        summary = result.get("overall_summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = (
                "Potential gaps are preliminary and require validation "
                "against additional academic literature."
            )

        return {
            "agent": "Research Gap Detection Agent",
            "gap_analysis": {
                "research_gaps": normalized_gaps,
                "overall_summary": summary.strip(),
            },
        }

    except Exception as exc:
        return {
            "agent": "Research Gap Detection Agent",
            "gap_analysis": {
                "error": f"Research gap detection failed: {exc}"
            },
        }
