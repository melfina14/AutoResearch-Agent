
import json
import re

from app.llm import generate


NOT_SPECIFIED = "Not specified in the supplied information"

CRITERIA = (
    "clarity",
    "testability",
    "measurability",
    "feasibility",
    "evidence_support",
)

DECISIONS = {
    "promising",
    "needs revision",
    "insufficient information",
}


def _clean_text(value, default=NOT_SPECIFIED):
    """Convert a value to a safe, readable string."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()
        return value if value else default

    if isinstance(value, (int, float, bool)):
        return str(value)

    return default


def _parse_json_response(response):
    """Parse JSON, including responses wrapped in Markdown fences."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The model returned an empty response.")

    cleaned = response.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, count=1)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover a JSON object if explanatory text surrounds it.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise

        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The evaluation response must be a JSON object.")

    return parsed


def _extract_hypotheses(value):
    """Support common wrappers and direct hypothesis lists."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    # Unwrap common agent-output structures.
    for _ in range(4):
        if isinstance(value, dict):
            if "hypothesis_analysis" in value:
                value = value["hypothesis_analysis"]
            elif "hypotheses" in value:
                value = value["hypotheses"]
            elif "hypothesis" in value:
                value = [value]
            else:
                return []
        else:
            break

    if not isinstance(value, list):
        return []

    # Keep only identifiable hypotheses.
    identified = []

    for item in value:
        if isinstance(item, str) and item.strip():
            identified.append({"hypothesis": item.strip()})
        elif isinstance(item, dict):
            statement = item.get("hypothesis")

            if isinstance(statement, str) and statement.strip():
                identified.append(item)

    return identified


def _normalize_score(value):
    """Return a valid integer score from 1 to 5, or None."""
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)) and value.is_integer() if isinstance(value, float) else isinstance(value, int):
        score = int(value)
        if 1 <= score <= 5:
            return score

    return None


def _normalize_evaluation(evaluation, hypotheses):
    """Validate the model output and normalize scores and decisions."""
    raw_evaluations = evaluation.get("evaluated_hypotheses", [])

    if not isinstance(raw_evaluations, list):
        raw_evaluations = []

    normalized = []

    for index, original in enumerate(hypotheses):
        # Match results by hypothesis text; use position as a fallback.
        original_text = original["hypothesis"]
        matched = next(
            (
                item for item in raw_evaluations
                if isinstance(item, dict)
                and item.get("hypothesis") == original_text
            ),
            None,
        )

        if matched is None and index < len(raw_evaluations):
            candidate = raw_evaluations[index]
            if isinstance(candidate, dict):
                matched = candidate

        if not isinstance(matched, dict):
            matched = {}

        criteria_result = {}
        valid_scores = []

        for criterion in CRITERIA:
            raw = matched.get(criterion, {})
            if not isinstance(raw, dict):
                raw = {}

            score = _normalize_score(raw.get("score"))

            if score is not None:
                valid_scores.append(score)

            criteria_result[criterion] = {
                "score": score,
                "reason": _clean_text(raw.get("reason")),
            }

        if len(valid_scores) == len(CRITERIA):
            overall_score = round(sum(valid_scores) / len(CRITERIA), 2)
        else:
            overall_score = None

        decision = _clean_text(matched.get("decision"), "Insufficient information")
        decision_key = decision.strip().lower()

        if decision_key not in DECISIONS:
            decision = "Insufficient information"
        else:
            decision = {
                "promising": "Promising",
                "needs revision": "Needs revision",
                "insufficient information": "Insufficient information",
            }[decision_key]

        def string_list(key):
            value = matched.get(key, [])
            if not isinstance(value, list):
                return []

            return [
                str(item).strip()
                for item in value
                if isinstance(item, (str, int, float))
                and str(item).strip()
            ]

        normalized.append(
            {
                "hypothesis": original_text,
                **criteria_result,
                "overall_score": overall_score,
                "decision": decision,
                "strengths": string_list("strengths"),
                "weaknesses": string_list("weaknesses"),
                "suggested_improvements": string_list(
                    "suggested_improvements"
                ),
                "missing_information": string_list("missing_information"),
            }
        )

    return {
        "evaluated_hypotheses": normalized,
        "evaluation_scale": "Scores range from 1 (poor) to 5 (excellent)",
        "overall_summary": _clean_text(
            evaluation.get("overall_summary"),
            "Evaluation completed, but the summary was not provided.",
        ),
        "recommended_next_steps": [
            str(item).strip()
            for item in evaluation.get("recommended_next_steps", [])
            if isinstance(item, str) and item.strip()
        ]
        if isinstance(evaluation.get("recommended_next_steps", []), list)
        else [],
    }


def hypothesis_evaluation_agent(hypotheses) -> dict:
    """
    Evaluate research hypotheses for clarity, testability,
    measurability, feasibility, and evidence support.
    """

    candidate_hypotheses = _extract_hypotheses(hypotheses)

    if not candidate_hypotheses:
        return {
            "agent": "Hypothesis Evaluation Agent",
            "evaluation": {
                "status": "Unable to evaluate",
                "evaluated_hypotheses": [],
                "evaluation_scale": (
                    "Scores range from 1 (poor) to 5 (excellent)"
                ),
                "overall_summary": (
                    "No identifiable hypotheses were supplied. "
                    "Provide a hypothesis statement or a list of "
                    "objects containing a 'hypothesis' field."
                ),
                "recommended_next_steps": [
                    "Run the Hypothesis Generation Agent or provide "
                    "a candidate hypothesis."
                ],
            },
        }

    hypotheses_data = json.dumps(
        candidate_hypotheses,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are the Hypothesis Evaluation Agent in an academic research system.

Evaluate EVERY candidate hypothesis supplied below.

Score each criterion using an integer from 1 (poor) to 5 (excellent):
1. Clarity: Is the hypothesis clear and specific?
2. Testability: Can it be investigated using a research method?
3. Measurability: Are variables and outcomes measurable?
4. Feasibility: Is the proposed test practical with the information given?
5. Evidence support: Is the hypothesis reasonably motivated by the
   supplied research gap and supporting evidence?

Scientific safeguards:
- Use only the supplied information.
- Do not invent results, citations, datasets, or evidence.
- Do not treat a missing detail in an abstract as proof that a method
  has never been studied.
- Identify assumptions and missing information.
- Do not penalize a hypothesis simply because it has not yet been tested.
- Distinguish a plausible candidate from a proven result.
- If feasibility cannot be determined from the available information,
  explain what resources or details are missing.
- Preserve each original hypothesis statement exactly.
- Do not silently omit any supplied hypothesis.
- Avoid overly high scores when evidence is limited.

Decision guidance:
- "Promising": generally clear, testable, and sufficiently supported.
- "Needs revision": potentially useful but needs clearer variables,
  methods, scope, or evidence.
- "Insufficient information": the supplied material is not enough
  to evaluate the hypothesis responsibly.

Calculate overall_score as the arithmetic mean of the five criterion
scores, rounded to two decimal places.

CANDIDATE HYPOTHESES:
{hypotheses_data}

Return ONLY valid JSON with this structure:
{{
  "evaluated_hypotheses": [
    {{
      "hypothesis": "Exact original hypothesis statement",
      "clarity": {{"score": 1, "reason": "Explanation"}},
      "testability": {{"score": 1, "reason": "Explanation"}},
      "measurability": {{"score": 1, "reason": "Explanation"}},
      "feasibility": {{"score": 1, "reason": "Explanation"}},
      "evidence_support": {{"score": 1, "reason": "Explanation"}},
      "overall_score": 1.0,
      "decision": "Promising, Needs revision, or Insufficient information",
      "strengths": [],
      "weaknesses": [],
      "suggested_improvements": [],
      "missing_information": []
    }}
  ],
  "evaluation_scale": "Scores range from 1 (poor) to 5 (excellent)",
  "overall_summary": "Evidence-aware summary of the evaluation",
  "recommended_next_steps": []
}}

Return one evaluation for every supplied hypothesis.
All five criterion scores must be integers from 1 to 5.
"""

    try:
        response = generate(prompt)
        evaluation = _parse_json_response(response)

        normalized = _normalize_evaluation(
            evaluation,
            candidate_hypotheses,
        )

        if len(normalized["evaluated_hypotheses"]) != len(
            candidate_hypotheses
        ):
            raise ValueError(
                "The number of evaluations does not match the hypotheses."
            )

        normalized["status"] = "Completed"

    except Exception as exc:
        normalized = {
            "status": "Evaluation failed",
            "evaluated_hypotheses": [],
            "evaluation_scale": (
                "Scores range from 1 (poor) to 5 (excellent)"
            ),
            "overall_summary": (
                "The hypotheses could not be evaluated successfully. "
                "No scores have been inferred."
            ),
            "recommended_next_steps": [
                "Check the model response and retry the evaluation."
            ],
            "error": str(exc),
        }

    return {
        "agent": "Hypothesis Evaluation Agent",
        "evaluation": normalized,
    }
