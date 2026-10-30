
import json
import re

from app.llm import generate


NOT_SPECIFIED = "Not specified in the provided research gaps"

HYPOTHESIS_FIELDS = (
    "hypothesis",
    "research_question",
    "related_research_gap",
    "independent_variable",
    "dependent_variable",
    "expected_relationship",
    "testing_method",
    "evaluation_metrics",
    "rationale",
    "assumptions",
)


def _clean_text(value, default=NOT_SPECIFIED) -> str:
    """Return clean text while preserving missing information."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    return default


def _normalize_string_list(value) -> list[str]:
    """Normalize a field expected to contain a list of strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, list):
        return [
            text
            for item in value
            if (text := _clean_text(item, "")).strip()
        ]

    return []


def _extract_gaps(gaps) -> list[dict]:
    """Accept direct gap lists or complete gap-agent output."""
    if isinstance(gaps, dict):
        if gaps.get("error"):
            raise ValueError(
                f"Gap detection reported an error: {gaps['error']}"
            )

        if "gap_analysis" in gaps:
            gaps = gaps["gap_analysis"]

        if isinstance(gaps, dict):
            if gaps.get("error"):
                raise ValueError(
                    f"Gap detection reported an error: {gaps['error']}"
                )

            gaps = gaps.get("research_gaps", [])

    if not isinstance(gaps, list):
        return []

    valid_gaps = []

    for gap in gaps:
        if not isinstance(gap, dict):
            continue

        if gap.get("error"):
            continue

        description = gap.get("gap")

        if not isinstance(description, str) or not description.strip():
            continue

        valid_gaps.append(gap)

    return valid_gaps


def _parse_json_response(response: str) -> dict:
    """Parse a JSON object, including an optional Markdown code fence."""
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
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("No valid JSON object was found.")

        result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")

    return result


def _normalize_hypothesis(hypothesis: dict) -> dict:
    """Ensure a generated hypothesis follows the required schema."""
    normalized = {}

    for field in HYPOTHESIS_FIELDS:
        value = hypothesis.get(field)

        if field in ("evaluation_metrics", "assumptions"):
            normalized[field] = _normalize_string_list(value)
        else:
            normalized[field] = _clean_text(value)

    return normalized


def hypothesis_agent(gaps) -> dict:
    """Generate preliminary, testable hypotheses from research gaps."""

    agent_name = "Hypothesis Generation Agent"

    try:
        valid_gaps = _extract_gaps(gaps)

        if not valid_gaps:
            return {
                "agent": agent_name,
                "hypothesis_analysis": {
                    "hypotheses": [],
                    "overall_summary": (
                        "No usable research gaps were provided. "
                        "Validate the literature comparison and research "
                        "gaps before generating hypotheses."
                    ),
                },
            }

        gaps_data = json.dumps(
            valid_gaps,
            indent=2,
            ensure_ascii=False,
        )

        prompt = f"""
You are the Hypothesis Generation Agent in a domain-independent
academic research system.

Generate up to 3 candidate research hypotheses based only on the
research gaps supplied below.

EVIDENCE AND REASONING RULES:
1. Each hypothesis must be specific, measurable, and testable.
2. Each hypothesis must be linked to a supplied research gap.
3. Do not present hypotheses, expected outcomes, or predictions
   as established findings.
4. Do not invent citations, datasets, experimental results, or
   numerical performance improvements.
5. Distinguish supplied evidence from assumptions and predictions.
6. Do not assume that a dataset, tool, or resource is available
   unless the supplied information establishes its availability.
7. Suggest realistic testing methods appropriate to the domain.
8. Choose evaluation metrics appropriate to the research question.
   Do not force classification metrics onto non-classification tasks.
9. If a numerical target or baseline is not supplied, do not invent
   one. Explain what baseline or threshold must be established.
10. If a gap is too vague to support a testable hypothesis, explain
    what additional information is needed.
11. Do not force three hypotheses when the evidence supports fewer.
12. If no testable hypothesis can be responsibly proposed, return
    an empty hypotheses list and explain why.
13. Treat all generated hypotheses as candidates requiring empirical
    testing, not conclusions.

For each hypothesis, provide:
- hypothesis: A clear, testable statement.
- research_question: The question being investigated.
- related_research_gap: The supplied gap motivating the hypothesis.
- independent_variable: The variable or condition changed or compared.
- dependent_variable: The outcome to measure.
- expected_relationship: The predicted relationship, clearly marked
  as an expectation rather than a known result.
- testing_method: A suggested experiment or evaluation procedure.
- evaluation_metrics: Appropriate measurable evaluation criteria.
- rationale: How the supplied gap motivates this hypothesis.
- assumptions: Assumptions or prerequisites that need verification.

IDENTIFIED RESEARCH GAPS:
{gaps_data}

Return ONLY a valid JSON object with this structure:
{{
  "hypotheses": [
    {{
      "hypothesis": "A clear, testable candidate statement",
      "research_question": "The question being investigated",
      "related_research_gap": "The supplied research gap",
      "independent_variable": "Variable being changed or compared",
      "dependent_variable": "Outcome being measured",
      "expected_relationship": "Predicted relationship, not a proven result",
      "testing_method": "Suggested experiment or evaluation",
      "evaluation_metrics": ["Appropriate metric"],
      "rationale": "Evidence-based reason for proposing this hypothesis",
      "assumptions": ["Assumption requiring verification"]
    }}
  ],
  "overall_summary": "Summary of the candidate hypotheses and limitations"
}}

The hypotheses field must be a list of objects.
The overall_summary field must be a string.
Do not include Markdown fences or text outside the JSON.
"""

        response = generate(prompt)
        result = _parse_json_response(response)

        raw_hypotheses = result.get("hypotheses")

        if not isinstance(raw_hypotheses, list):
            raise ValueError(
                "The response must contain a hypotheses list."
            )

        normalized_hypotheses = [
            _normalize_hypothesis(item)
            for item in raw_hypotheses
            if isinstance(item, dict)
            and isinstance(item.get("hypothesis"), str)
            and item["hypothesis"].strip()
        ][:3]

        summary = result.get("overall_summary")

        if not isinstance(summary, str) or not summary.strip():
            summary = (
                "These hypotheses are preliminary candidates and require "
                "validation through appropriate experiments."
            )

        return {
            "agent": agent_name,
            "hypothesis_analysis": {
                "hypotheses": normalized_hypotheses,
                "overall_summary": summary.strip(),
            },
        }

    except Exception as exc:
        return {
            "agent": agent_name,
            "hypothesis_analysis": {
                "hypotheses": [],
                "error": f"Hypothesis generation failed: {exc}",
                "overall_summary": (
                    "Hypothesis generation could not complete successfully. "
                    "Check the supplied research gaps and model response."
                ),
            },
        }
