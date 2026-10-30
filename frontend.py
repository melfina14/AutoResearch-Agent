
import streamlit as st
import traceback

from app.workflow.graph import research_graph


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AutoResearch Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.3px;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .section-description {
            color: #9CA3AF;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .paper-title {
            font-size: 1.05rem;
            font-weight: 600;
        }

        div[data-testid="stExpander"] {
            border-radius: 10px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 45px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def safe_text(value, default="Not available"):
    """Convert missing or unexpected values into readable text."""

    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()
        return value if value else default

    if isinstance(value, (dict, list, tuple)):
        return value

    return str(value)


def display_content(title, content, description=None):
    """Display an agent result in a readable section."""

    st.subheader(title)

    if description:
        st.caption(description)

    if content is None:
        st.info("No content was returned for this section.")
        return

    if isinstance(content, str):
        cleaned = content.strip()

        if cleaned:
            st.markdown(cleaned)
        else:
            st.info("No content was returned for this section.")

    elif isinstance(content, (dict, list)):
        st.json(content, expanded=False)

    else:
        st.write(content)


def display_papers(papers):
    """Display retrieved papers individually and clearly."""

    st.subheader("2. Retrieved Papers")

    if not isinstance(papers, list) or not papers:
        st.warning("No relevant papers were retrieved.")
        return

    st.caption(f"Retrieved {len(papers)} paper(s).")

    for index, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            continue

        title = safe_text(paper.get("title"), "Untitled paper")
        year = safe_text(paper.get("year"), "Year unavailable")
        source = safe_text(paper.get("source"), "Source unavailable")
        abstract = safe_text(paper.get("abstract"), "")

        with st.expander(
            f"{index}. {title} ({year})",
            expanded=False,
        ):
            st.markdown(f"**Title:** {title}")
            st.write(f"**Year:** {year}")
            st.write(f"**Source:** {source}")

            authors = paper.get("authors")

            if authors:
                if isinstance(authors, list):
                    authors = ", ".join(str(author) for author in authors)

                st.write(f"**Authors:** {authors}")

            doi = paper.get("doi")

            if doi:
                st.write(f"**DOI:** {doi}")

            url = paper.get("url")

            if url:
                st.markdown(f"[Open paper]({url})")

            if abstract:
                st.markdown("**Abstract**")
                st.write(abstract)
            else:
                st.info("An abstract was not provided by the source.")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔬 AutoResearch Agent")

st.markdown(
    '<div class="subtitle">'
    "Multi-Agent Scientific Literature Review and "
    "Hypothesis Generation Framework"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "Explore academic literature, compare research findings, "
    "identify potential research gaps, and generate candidate hypotheses."
)

st.divider()


# --------------------------------------------------
# Research input
# --------------------------------------------------
with st.container(border=True):
    st.subheader("Research Topic")

    question = st.text_area(
        "Enter your research question or topic",
        placeholder=(
            "Example: How are multi-agent LLM systems being used "
            "for scientific literature review?"
        ),
        height=110,
        key="research_question",
        help="Use a specific topic to improve the relevance of retrieved papers.",
    )

    run = st.button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Research workflow
# --------------------------------------------------
if run:
    if not question or not question.strip():
        st.warning("Please enter a research question before starting.")
        st.stop()

    try:
        with st.spinner(
            "Researching your topic... This may take a few minutes."
        ):
            result = research_graph.invoke(
                {"question": question.strip()}
            )

        if not isinstance(result, dict):
            st.error(
                "The workflow returned an unexpected result. "
                "Expected a dictionary of agent outputs."
            )
            st.stop()

        st.success("Research workflow completed.")

        # ------------------------------------------
        # 1. Research plan
        # ------------------------------------------
        display_content(
            "1. Research Plan",
            result.get("plan"),
            "The objectives, search strategy, and scope generated "
            "by the planner agent.",
        )

        st.divider()

        # ------------------------------------------
        # 2. Retrieved papers
        # ------------------------------------------
        display_papers(result.get("papers", []))

        st.divider()

        # ------------------------------------------
        # 3. Cross-paper comparison
        # ------------------------------------------
        display_content(
            "3. Cross-Paper Comparison",
            result.get("comparison"),
            "A synthesis of common themes, methodologies, findings, "
            "and limitations.",
        )

        st.divider()

        # ------------------------------------------
        # 4. Research gaps
        # ------------------------------------------
        display_content(
            "4. Research Gaps",
            result.get("gaps"),
            "Potential gaps identified from the retrieved literature.",
        )

        st.divider()

        # ------------------------------------------
        # 5. Candidate hypotheses
        # ------------------------------------------
        display_content(
            "5. Candidate Hypotheses",
            result.get("hypotheses"),
            "Proposed hypotheses that require validation against "
            "the full literature.",
        )

        st.divider()

        # ------------------------------------------
        # 6. Hypothesis evaluation
        # ------------------------------------------
        display_content(
            "6. Hypothesis Evaluation",
            result.get("hypothesis_evaluation"),
            "AI-assisted assessment of hypothesis clarity, testability, "
            "measurability, feasibility, and evidence support. Scores "
            "are preliminary assessments, not experimental validation.",
        )

        st.divider()

        # ------------------------------------------
        # 7. Citation verification
        # ------------------------------------------
        display_content(
            "7. Citation Verification",
            result.get("citation_verification"),
            "Checks supplied paper metadata and whether report claims "
            "are supported by the provided evidence. This does not "
            "independently verify online publication records.",
        )

        st.divider()

        # ------------------------------------------
        # 8. Final literature review report
        # ------------------------------------------
        st.subheader("8. Final Literature Review Report")

        report_result = result.get("report")
        report_text = ""

        if isinstance(report_result, str):
            report_text = report_result

        elif isinstance(report_result, dict):
            candidate_text = report_result.get("report", "")

            if isinstance(candidate_text, str):
                report_text = candidate_text

        if report_text.strip():
            with st.container(border=True):
                st.markdown(report_text)

            st.download_button(
                label="⬇️ Download Literature Review (.md)",
                data=report_text,
                file_name="literature_review.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.warning(
                "The workflow completed, but no readable final report "
                "was returned."
            )

            if report_result is not None:
                st.json(report_result, expanded=False)

        st.caption(
            "Note: Retrieved abstracts may not contain the full paper. "
            "Verify paper details, citations, and research gaps against "
            "the original publications before academic use."
        )

    except Exception:
        st.error("The research workflow could not be completed.")

        with st.expander(
            "Technical error details",
            expanded=True,
        ):
            st.code(traceback.format_exc())

        st.info(
            "Use the traceback to identify the file and line that caused "
            "the error. API quota errors, missing fields, retrieval failures, "
            "and report-generation errors require different fixes."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "AutoResearch Agent | Academic research assistance powered by "
    "multi-agent workflows and academic paper retrieval."
)
