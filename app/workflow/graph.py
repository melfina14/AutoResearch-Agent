
import json
from pathlib import Path
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from app.agents.planner import planner_agent
from app.agents.retrieval import retrieval_agent
from app.agents.analysis import analysis_agent
from app.agents.comparison import comparison_agent
from app.agents.gap_detection import gap_agent
from app.agents.hypothesis import hypothesis_agent
from app.agents.hypothesis_evaluation import (
    hypothesis_evaluation_agent,
)
from app.agents.citation_verification import (
    citation_verification_agent,
)
from app.agents.report import report_agent


class ResearchState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    papers: list[dict[str, Any]]
    analyses: list[dict[str, Any]]
    comparison: dict[str, Any]
    gaps: dict[str, Any]
    hypotheses: dict[str, Any]
    hypothesis_evaluation: dict[str, Any]
    citation_verification: dict[str, Any]
    report: dict[str, Any]


def _error_result(agent, exc):
    """Create a consistent error result without fabricating findings."""
    return {
        "agent": agent,
        "error": str(exc),
    }


def _to_json(value):
    """Serialize agent output safely."""
    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def plan_node(state: ResearchState) -> dict:
    """Create a structured research plan."""
    try:
        question = state.get("question", "").strip()

        if not question:
            return {
                "plan": {
                    "error": "A research question is required."
                }
            }

        result = planner_agent(question)

        if isinstance(result, dict):
            plan = result.get("plan", result)
        else:
            plan = {"raw_response": str(result)}

        return {"plan": plan}

    except Exception as exc:
        return {"plan": _error_result("Research Planner", exc)}


def retrieve_node(state: ResearchState) -> dict:
    """Retrieve academic papers using the research plan."""
    plan = state.get("plan", {})

    if isinstance(plan, dict) and plan.get("error"):
        return {
            "papers": [],
            "retrieval_error": {
                "message": "Retrieval skipped because planning failed.",
                "details": plan["error"],
            },
        }

    try:
        papers = retrieval_agent(
            state["question"],
            plan,
            limit=8,
        )

        if not isinstance(papers, list):
            return {
                "papers": [],
                "retrieval_error": {
                    "message": "The retrieval agent did not return a list."
                },
            }

        return {"papers": papers}

    except Exception as exc:
        return {
            "papers": [],
            "retrieval_error": _error_result(
                "Literature Retrieval Agent",
                exc,
            ),
        }


def analyze_node(state: ResearchState) -> dict:
    """Analyze each retrieved paper independently."""
    analyses = []

    for index, paper in enumerate(state.get("papers", []), start=1):
        try:
            result = analysis_agent(paper)

            if isinstance(result, dict):
                analyses.append(result)
            else:
                analyses.append(
                    {
                        "paper": f"Paper {index}",
                        "error": "The analysis agent returned an invalid result.",
                    }
                )

        except Exception as exc:
            title = (
                paper.get("title", f"Paper {index}")
                if isinstance(paper, dict)
                else f"Paper {index}"
            )

            analyses.append(
                {
                    "paper": title,
                    "error": str(exc),
                }
            )

    return {"analyses": analyses}


def compare_node(state: ResearchState) -> dict:
    """Compare the analyzed papers."""
    analyses = state.get("analyses", [])

    usable_analyses = [
        item
        for item in analyses
        if isinstance(item, dict) and not item.get("error")
    ]

    if len(usable_analyses) < 2:
        return {
            "comparison": {
                "error": (
                    "At least two successfully analyzed papers "
                    "are required for comparison."
                ),
                "available_analysis_count": len(usable_analyses),
            }
        }

    try:
        result = comparison_agent(usable_analyses)

        if not isinstance(result, dict):
            return {
                "comparison": {
                    "error": "The comparison agent returned an invalid result."
                }
            }

        return {"comparison": result}

    except Exception as exc:
        return {
            "comparison": _error_result("Comparison Agent", exc)
        }


def gap_node(state: ResearchState) -> dict:
    """Identify potential research gaps."""
    comparison = state.get("comparison", {})

    if isinstance(comparison, dict) and comparison.get("error"):
        return {
            "gaps": {
                "error": "Gap detection skipped because comparison failed.",
                "details": comparison["error"],
            }
        }

    try:
        result = gap_agent(comparison)

        if not isinstance(result, dict):
            return {
                "gaps": {
                    "error": "The gap detection agent returned an invalid result."
                }
            }

        return {"gaps": result}

    except Exception as exc:
        return {"gaps": _error_result("Research Gap Detection Agent", exc)}


def hypothesis_node(state: ResearchState) -> dict:
    """Generate candidate hypotheses from research gaps."""
    gaps = state.get("gaps", {})

    if isinstance(gaps, dict) and gaps.get("error"):
        return {
            "hypotheses": {
                "error": "Hypothesis generation skipped because gap detection failed.",
                "details": gaps["error"],
            }
        }

    try:
        result = hypothesis_agent(gaps)

        if not isinstance(result, dict):
            return {
                "hypotheses": {
                    "error": "The hypothesis agent returned an invalid result."
                }
            }

        return {"hypotheses": result}

    except Exception as exc:
        return {
            "hypotheses": _error_result("Hypothesis Generation Agent", exc)
        }


def hypothesis_evaluation_node(state: ResearchState) -> dict:
    """Evaluate generated hypotheses."""
    try:
        result = hypothesis_evaluation_agent(
            state.get("hypotheses", {})
        )

        if not isinstance(result, dict):
            return {
                "hypothesis_evaluation": {
                    "error": "The evaluation agent returned an invalid result."
                }
            }

        return {"hypothesis_evaluation": result}

    except Exception as exc:
        return {
            "hypothesis_evaluation": _error_result(
                "Hypothesis Evaluation Agent",
                exc,
            )
        }


def report_node(state: ResearchState) -> dict:
    """Generate a preliminary report from available agent results."""
    try:
        result = report_agent(
            state.get("question", ""),
            state.get("papers", []),
            state.get("analyses", []),
            state.get("comparison", {}),
            state.get("gaps", {}),
            state.get("hypotheses", {}),
            hypothesis_evaluation=state.get(
                "hypothesis_evaluation", {}
            ),
            citation_verification={
                "status": "Pending",
                "note": (
                    "Citation verification will run after preliminary "
                    "report generation."
                ),
            },
        )

        if not isinstance(result, dict):
            return {
                "report": {
                    "error": "The report agent returned an invalid result."
                }
            }

        return {"report": result}

    except Exception as exc:
        return {"report": _error_result("Report Generator", exc)}


def citation_verification_node(state: ResearchState) -> dict:
    """Verify report claims against supplied paper records."""
    report_result = state.get("report", {})

    if (
        isinstance(report_result, dict)
        and report_result.get("error")
    ):
        return {
            "citation_verification": {
                "status": "Skipped",
                "error": (
                    "Citation verification skipped because "
                    "report generation failed."
                ),
                "report_error": report_result["error"],
            }
        }

    if isinstance(report_result, str):
        report_text = report_result
    elif isinstance(report_result, dict):
        report_text = report_result.get("report", "")
    else:
        report_text = ""

    if not isinstance(report_text, str) or not report_text.strip():
        return {
            "citation_verification": {
                "status": "Skipped",
                "error": "No report text was available to verify.",
            }
        }

    try:
        result = citation_verification_agent(
            state.get("papers", []),
            report_text=report_text,
        )

        if not isinstance(result, dict):
            return {
                "citation_verification": {
                    "status": "Error",
                    "error": (
                        "The citation verification agent returned "
                        "an invalid result."
                    ),
                }
            }

        return {"citation_verification": result}

    except Exception as exc:
        return {
            "citation_verification": _error_result(
                "Citation Verification Agent",
                exc,
            )
        }


def final_report_node(state: ResearchState) -> dict:
    """Insert final citation verification results into the report."""
    verification = state.get("citation_verification", {})
    report_result = state.get("report", {})

    if not isinstance(report_result, dict):
        return {
            "report": {
                "error": "The report result has an unexpected format."
            }
        }

    if report_result.get("error"):
        return {"report": report_result}

    report_text = report_result.get("report", "")

    if not isinstance(report_text, str) or not report_text.strip():
        return {
            "report": {
                **report_result,
                "error": "The generated report contains no report text.",
            }
        }

    section_heading = "## 9. Citation Verification"
    limitations_heading = "## 10. Limitations"

    verification_text = _to_json(verification)

    if section_heading in report_text:
        before, remaining = report_text.split(section_heading, 1)

        if limitations_heading in remaining:
            _, after = remaining.split(limitations_heading, 1)

            report_text = (
                before
                + section_heading
                + "\n\n"
                + verification_text
                + "\n\n"
                + limitations_heading
                + after
            )
        else:
            report_text = (
                before
                + section_heading
                + "\n\n"
                + verification_text
                + "\n"
            )
    else:
        report_text += (
            "\n\n## Citation Verification Results\n\n"
            + verification_text
            + "\n"
        )

    output_path = report_result.get("report_path")
    save_error = None

    if output_path:
        try:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report_text, encoding="utf-8")
        except OSError as exc:
            save_error = str(exc)

    updated_report = {
        **report_result,
        "report": report_text,
        "citation_verification": verification,
    }

    if save_error:
        updated_report["save_error"] = save_error
        updated_report["status"] = "error"
    else:
        updated_report["status"] = "success"

    return {"report": updated_report}


builder = StateGraph(ResearchState)

builder.add_node("planner", plan_node)
builder.add_node("retrieval", retrieve_node)
builder.add_node("analysis", analyze_node)
builder.add_node("comparison", compare_node)
builder.add_node("gap_detection", gap_node)
builder.add_node("hypothesis", hypothesis_node)
builder.add_node("hypothesis_evaluation", hypothesis_evaluation_node)
builder.add_node("report", report_node)
builder.add_node("citation_verification", citation_verification_node)
builder.add_node("final_report", final_report_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "retrieval")
builder.add_edge("retrieval", "analysis")
builder.add_edge("analysis", "comparison")
builder.add_edge("comparison", "gap_detection")
builder.add_edge("gap_detection", "hypothesis")
builder.add_edge("hypothesis", "hypothesis_evaluation")
builder.add_edge("hypothesis_evaluation", "report")
builder.add_edge("report", "citation_verification")
builder.add_edge("citation_verification", "final_report")
builder.add_edge("final_report", END)

research_graph = builder.compile()
