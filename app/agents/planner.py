
import json
import re
from typing import Any

from app.llm import generate


PLAN_LIST_FIELDS = (
    "research_objectives",
    "search_queries",
    "keywords",
    "subtopics",
    "inclusion_criteria",
    "exclusion_criteria",
)

PLAN_REQUIRED_FIELDS = (
    "research_topic",
    *PLAN_LIST_FIELDS,
    "research_scope",
)


def _extract_json(response: Any) -> dict:
    """Parse a JSON object from an LLM response."""

    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        raise ValueError("The planner response is not text or a JSON object.")

    text = response.strip()

    # Handle responses wrapped in Markdown code fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Recover a JSON object if the model added surrounding commentary.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        result = json.loads(text[start:end + 1])
        if isinstance(result, dict):
            return result

    raise ValueError("No valid JSON object was found in the response.")


def _validate_plan(plan: dict, question: str) -> dict:
    """Normalize and validate a research plan for any subject."""

    cleaned = {
        "research_topic": question.strip(),
        "research_objectives": [],
        "search_queries": [],
        "keywords": [],
        "subtopics": [],
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "research_scope": (
            "Scope to be refined using the research question and "
            "available evidence."
        ),
    }

    # Keep the expected output schema stable for downstream agents.
    topic = plan.get("research_topic")
    if isinstance(topic, str) and topic.strip():
        cleaned["research_topic"] = topic.strip()

    scope = plan.get("research_scope")
    if isinstance(scope, str) and scope.strip():
        cleaned["research_scope"] = scope.strip()

    for field in PLAN_LIST_FIELDS:
        value = plan.get(field, [])

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            value = []

        items = []
        seen = set()

        for item in value:
            if not isinstance(item, str):
                continue

            item = " ".join(item.split())
            normalized = item.casefold()

            if item and normalized not in seen:
                items.append(item)
                seen.add(normalized)

        cleaned[field] = items

    # Do not silently accept an almost-empty or unusable plan.
    minimum_counts = {
        "research_objectives": 3,
        "search_queries": 5,
        "keywords": 5,
        "subtopics": 3,
        "inclusion_criteria": 2,
        "exclusion_criteria": 2,
    }

    missing_or_short = [
        field
        for field, minimum in minimum_counts.items()
        if len(cleaned[field]) < minimum
    ]

    if missing_or_short:
        raise ValueError(
            "The research plan is incomplete. Fields needing more "
            "items: " + ", ".join(missing_or_short)
        )

    return cleaned


def planner_agent(question: str) -> dict:
    """
    Convert any research question into a structured research plan.

    The plan is topic-independent and preserves the output structure
    expected by the literature retrieval workflow.
    """

    if not isinstance(question, str) or not question.strip():
        return {
            "error": "A valid, non-empty research question is required.",
            "plan": {},
        }

    question = question.strip()

    prompt = f"""
You are the Research Planner Agent in a multi-agent scientific
literature review system.

Your task is to create a rigorous, focused, and useful research plan
from the user's research question.

IMPORTANT: This system must work for ANY research domain. Examples
include computer science, engineering, medicine, agriculture,
education, environmental science, economics, social sciences, and
other academic or technical subjects.

Do not assume that the question is about a particular field.
Do not insert unrelated topics, methods, datasets, or terminology.
Derive all objectives, queries, keywords, subtopics, and criteria
from the actual question and its context.

USER RESEARCH QUESTION:
{question}

Create these fields:

1. research_topic
   - A concise title that accurately represents the question.
   - Preserve the user's intended subject.

2. research_objectives
   - Provide 3 to 5 specific, answerable objectives.
   - Focus on the main problem, methods, evidence, comparison,
     evaluation, outcomes, or limitations as appropriate to the topic.
   - Do not introduce objectives unrelated to the question.

3. search_queries
   - Provide 5 to 8 distinct academic search queries.
   - Cover different relevant aspects of the same research question.
   - Include useful synonyms, alternate terminology, or related
     scientific expressions when appropriate.
   - Use Boolean operators such as AND, OR, and NOT when helpful.
   - Avoid duplicate queries and unrelated subfields.
   - Keep the queries suitable for academic search engines and
     scholarly databases.

4. keywords
   - Provide 8 to 15 relevant keywords or short phrases.
   - Include the main concepts, methods, subject terms, and useful
     synonyms specific to the question.
   - Avoid generic filler terms and unrelated terminology.

5. subtopics
   - Provide 4 to 8 meaningful subtopics.
   - Organize the major dimensions needed to answer the question.
   - Choose subtopics appropriate to the research domain.

6. inclusion_criteria
   - Provide 3 to 6 practical criteria for selecting relevant studies.
   - Consider topical relevance, study type, methods, population or
     setting when applicable, publication language, and evidence quality.
   - Adapt the criteria to the subject instead of applying fixed rules
     that only suit one discipline.
   - Do not require every study to use the same method or dataset
     unless the question explicitly requires it.
   - Abstracts may be used for preliminary screening when full text is
     unavailable. Clearly distinguish abstract-level evidence from
     evidence assessed using full text.
   - Do not claim that a paper is peer-reviewed unless that status
     can be established from available metadata or a trusted source.

7. exclusion_criteria
   - Provide 3 to 6 practical criteria for excluding irrelevant or
     unsuitable studies.
   - Exclude studies that do not address the research question,
     duplicate records, and records lacking enough information for
     the intended assessment.
   - Adapt other criteria to the topic and the inclusion criteria.
   - Do not automatically exclude useful studies solely because the
     full text is unavailable; flag the evidence limitation instead.
   - Ensure inclusion and exclusion criteria do not contradict each
     other.

8. research_scope
   - Write a concise description of the review's boundaries.
   - Explain the central subject, aspects to be covered, and relevant
     limits.
   - Reflect any population, location, period, technology, or method
     specified by the user.
   - Do not invent a date range, geographic restriction, or other
     constraint not supported by the question.
   - If the question is broad, establish a sensible focus without
     changing its intended meaning.

Quality checks before returning:
- Every field must be relevant to the user's actual question.
- Search queries, keywords, objectives, and subtopics must be
  consistent with one another.
- Avoid repeating the same idea using slightly different wording.
- Do not invent research findings, references, or evidence.
- Do not treat the proposed research plan as completed research.
- Return between 3 and 5 objectives, 5 and 8 queries, 8 and 15
  keywords, 4 and 8 subtopics, and 3 to 6 criteria in each of the
  inclusion and exclusion lists.

Return ONLY a valid JSON object. Do not include Markdown fences,
commentary, or text outside the JSON object.

Use exactly this structure:
{{
    "research_topic": "string",
    "research_objectives": ["string"],
    "search_queries": ["string"],
    "keywords": ["string"],
    "subtopics": ["string"],
    "inclusion_criteria": ["string"],
    "exclusion_criteria": ["string"],
    "research_scope": "string"
}}
"""

    try:
        response = generate(prompt)
        plan = _extract_json(response)
        validated_plan = _validate_plan(plan, question)

        return {"plan": validated_plan}

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "error": f"Research plan generation or validation failed: {exc}",
            "plan": {
                "research_topic": question,
                "research_objectives": [],
                "search_queries": [],
                "keywords": [],
                "subtopics": [],
                "inclusion_criteria": [],
                "exclusion_criteria": [],
                "research_scope": (
                    "The plan could not be validated. Please retry "
                    "generation before starting literature retrieval."
                ),
            },
        }
    except Exception as exc:
        return {
            "error": (
                "The research planner encountered an unexpected error: "
                f"{type(exc).__name__}: {exc}"
            ),
            "plan": {
                "research_topic": question,
                "research_objectives": [],
                "search_queries": [],
                "keywords": [],
                "subtopics": [],
                "inclusion_criteria": [],
                "exclusion_criteria": [],
                "research_scope": (
                    "Plan generation failed. Please check the LLM "
                    "configuration and retry."
                ),
            },
        }
