
import json
from datetime import datetime
from pathlib import Path


NOT_AVAILABLE = "Not available."


def _clean_text(value, default=NOT_AVAILABLE):
    """Convert a value into safe, readable text."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()
        return value if value else default

    if isinstance(value, (int, float, bool)):
        return str(value)

    return default


def _format_value(value, level=0):
    """Format nested agent output as readable Markdown."""
    if value is None:
        return NOT_AVAILABLE

    if isinstance(value, str):
        return value.strip() or NOT_AVAILABLE

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        if not value:
            return "Not specified."

        parts = []

        for item in value:
            if isinstance(item, dict):
                parts.append(_format_value(item, level + 1))
            elif isinstance(item, list):
                parts.append(_format_value(item, level + 1))
            else:
                parts.append(
                    f"{'  ' * level}- {_clean_text(item)}"
                )

        return "\n".join(parts)

    if isinstance(value, dict):
        if not value:
            return "Not specified."

        parts = []

        for key, item in value.items():
            label = str(key).replace("_", " ").strip().title()

            if isinstance(item, (dict, list)):
                parts.append(
                    f"{'#' * min(level + 4, 6)} {label}\n\n"
                    f"{_format_value(item, level + 1)}"
                )
            else:
                parts.append(
                    f"{'  ' * level}- **{label}:** "
                    f"{_clean_text(item)}"
                )

        return "\n\n".join(parts)

    return str(value)


def _unwrap_agent_output(value, wrapper_keys):
    """Unwrap common agent response containers."""
    current = value

    for _ in range(4):
        if not isinstance(current, dict):
            break

        matched_key = next(
            (key for key in wrapper_keys if key in current),
            None,
        )

        if matched_key is None:
            break

        current = current[matched_key]

    return current


def _paper_title(paper, index):
    """Return a paper title without assuming every record is complete."""
    if not isinstance(paper, dict):
        return f"Paper {index}"

    return _clean_text(
        paper.get("title"),
        f"Paper {index}",
    )


def _paper_reference(paper, index):
    """Build a reference entry from the retrieved paper metadata."""
    if not isinstance(paper, dict):
        return (
            f"### {index}. Paper {index}\n\n"
            "Paper metadata was not provided in a usable format."
        )

    title = _paper_title(paper, index)
    authors = paper.get("authors", [])
    year = _clean_text(paper.get("year"), "Not specified")
    venue = _clean_text(paper.get("venue"), "Not specified")
    url = _clean_text(paper.get("url"), "")
    abstract = _clean_text(paper.get("abstract"), "")

    external_ids = paper.get("externalIds", {})
    if not isinstance(external_ids, dict):
        external_ids = {}

    doi = _clean_text(external_ids.get("DOI"), "")

    if isinstance(authors, list):
        author_names = []

        for author in authors:
            if isinstance(author, dict):
                name = author.get("name", "")
            else:
                name = author

            if name is not None and str(name).strip():
                author_names.append(str(name).strip())

        author_text = ", ".join(author_names) or "Not specified"
    else:
        author_text = _clean_text(authors, "Not specified")

    lines = [
        f"### {index}. {title}",
        f"- **Authors:** {author_text}",
        f"- **Year:** {year}",
        f"- **Publication venue:** {venue}",
    ]

    if doi:
        doi = doi.removeprefix("https://doi.org/").strip()
        lines.append(f"- **DOI:** https://doi.org/{doi}")

    if url:
        lines.append(f"- **URL:** {url}")

    if abstract:
        lines.extend(["", "**Abstract:**", abstract])

    return "\n".join(lines)


def _analysis_sections(analyses, papers):
    """Associate analyses with titles while preserving their input order."""
    if not isinstance(analyses, list) or not analyses:
        return "No individual analyses were supplied."

    sections = []

    for index, analysis in enumerate(analyses, start=1):
        paper_index = index - 1

        if isinstance(analysis, dict):
            details = analysis.get("analysis", analysis)

            # Prefer the explicit title returned by the analysis agent.
            title = analysis.get("paper")

            if not title and paper_index < len(papers):
                title = _paper_title(papers[paper_index], index)

            title = _clean_text(title, f"Paper {index}")
        else:
            details = analysis
            title = (
                _paper_title(papers[paper_index], index)
                if paper_index < len(papers)
                else f"Paper {index}"
            )

        sections.append(
            f"### {index}. {title}\n\n{_format_value(details)}"
        )

    return "\n\n".join(sections)


def _write_report(output_path, report):
    """Save a report and return any filesystem error."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        return None
    except OSError as exc:
        return str(exc)


def report_agent(
    question,
    papers,
    analyses,
    comparison,
    gaps,
    hypotheses,
    hypothesis_evaluation=None,
    citation_verification=None,
    output_path=None,
) -> dict:
    """
    Generate and save a structured scientific literature review.

    Optional output_path allows a custom report destination.
    The default is data/reports/literature_review.md.
    """

    question = _clean_text(question, "")

    if not question:
        return {"error": "A research question is required."}

    if not isinstance(papers, list) or not papers:
        return {
            "error": "A non-empty list of research papers is required."
        }

    if not isinstance(analyses, list):
        analyses = []

    reference_sections = [
        _paper_reference(paper, index)
        for index, paper in enumerate(papers, start=1)
    ]

    comparison_content = _format_value(
        _unwrap_agent_output(
            comparison,
            ("comparison", "comparative_analysis"),
        )
    )

    gap_content = _format_value(
        _unwrap_agent_output(
            gaps,
            ("gap_analysis", "research_gaps"),
        )
    )

    hypothesis_content = _format_value(
        _unwrap_agent_output(
            hypotheses,
            ("hypothesis_analysis",),
        )
    )

    evaluation_content = _format_value(
        _unwrap_agent_output(
            hypothesis_evaluation,
            ("evaluation",),
        )
    )

    citation_content = _format_value(
        _unwrap_agent_output(
            citation_verification,
            ("verification", "citation_verification"),
        )
    )

    generated_on = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M %Z"
    )

    report = f"""# Scientific Literature Review

**Research question:** {question}

**Generated on:** {generated_on}

**Papers retrieved:** {len(papers)}

**Individual analyses supplied:** {len(analyses)}

## 1. Abstract

This report reviews {len(papers)} retrieved research papers related to
the research question. It presents available paper metadata and abstracts,
individual paper analyses, comparative findings, potential research gaps,
candidate hypotheses, hypothesis evaluation, and citation verification.

The conclusions depend on the retrieved literature and the information
supplied by the research agents. They should not be interpreted as
independently verified scientific findings.

## 2. Research Question and Objectives

### Research Question

{question}

### Review Objectives

- Summarize the retrieved literature relevant to the research question.
- Examine the objectives, methods, datasets, results, and limitations
  reported in the available paper information.
- Compare the supplied paper analyses.
- Identify potential research gaps and candidate hypotheses.
- Evaluate the candidate hypotheses and document verification limitations.

These are the objectives of this review process, not claims that every
objective has been fully achieved.

## 3. Literature Review and Retrieved Papers

{chr(10).join(reference_sections)}

## 4. Individual Paper Analysis

{_analysis_sections(analyses, papers)}

## 5. Comparative Analysis

{comparison_content}

## 6. Potential Research Gaps

{gap_content}

Potential gaps identified from abstracts or agent-generated summaries
should be treated as preliminary until checked against the original
publications and a sufficiently broad literature search.

## 7. Candidate Research Hypotheses

{hypothesis_content}

These hypotheses are candidates for further investigation. Their inclusion
does not establish novelty, correctness, or experimental validity.

## 8. Hypothesis Evaluation

{evaluation_content}

Scores and decisions represent an AI-assisted assessment based on the
information supplied. They are not evidence that a hypothesis is true.

## 9. Citation Verification

{citation_content}

Interpret verification results according to the checks actually performed.
A metadata consistency check alone does not establish that a publication
exists or that its contents support a claim.

## 10. Limitations

- The review depends on the papers retrieved by the system and may omit
  relevant publications.
- Abstract-level analysis may omit methods, datasets, limitations, and
  results available only in the full text.
- Retrieved paper relevance and possible duplicates require review.
- Research gaps inferred from incomplete information may be incorrect.
- Candidate hypotheses require appropriate study design and empirical
  investigation.
- Citation verification cannot be assumed to be independent unless an
  external publication source was actually consulted.
- Generated summaries may contain errors and should be checked against
  the original publications.
- This report does not independently validate experimental findings.

## 11. References

{chr(10).join(reference_sections)}
"""

    if output_path is None:
        output_path = (
            Path("data") / "reports" / "literature_review.md"
        )
    else:
        output_path = Path(output_path)

    write_error = _write_report(output_path, report)

    result = {
        "status": "success" if write_error is None else "error",
        "research_question": question,
        "paper_count": len(papers),
        "analysis_count": len(analyses),
        "report_path": str(output_path),
        "report": report,
    }

    if write_error:
        result["error"] = f"Could not save the report: {write_error}"

    return result
