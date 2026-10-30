
import re
import time
from typing import Any

from app.tools.semantic_scholar import search_papers
from app.tools.arxiv_search import search_arxiv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Starting thresholds; evaluate these using real retrieval results.
MIN_RELEVANCE_SCORE = 0.32
MIN_TITLE_RELEVANCE_SCORE = 0.24

MAX_SEARCH_QUERIES = 3
RESULTS_PER_QUERY = 5

_embedding_model = None
_model_load_attempted = False


# ---------------------------------------------------------
# Text processing
# ---------------------------------------------------------

def normalize_text(text: Any) -> str:
    """Normalize text for matching and deduplication."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_terms(text: str) -> set[str]:
    """Extract meaningful terms, retaining short terms such as AI and ML."""
    stop_words = {
        "the", "and", "for", "with", "from", "into", "using",
        "use", "what", "how", "why", "are", "is", "of", "in",
        "on", "to", "a", "an", "by", "at", "or", "as", "their",
        "this", "that", "these", "those", "be", "been", "being",
        "it", "its", "we", "our", "they", "them", "can", "could",
        "would", "should", "will", "may", "might", "do", "does",
        "did", "have", "has", "had", "not", "but", "about",
        "application", "applications", "impact", "role", "study",
        "studies", "based", "approach", "system", "systems",
        "review", "research", "analysis", "new", "different",
        "various", "related", "including",
    }

    return {
        word
        for word in normalize_text(text).split()
        if len(word) >= 2 and word not in stop_words
    }


def check_abstract_quality(abstract: Any, title: Any = "") -> dict:
    """
    Flag common abstract-quality problems without assuming a research domain.

    This is heuristic screening, not proof that an abstract is genuine.
    """
    if not isinstance(abstract, str) or not abstract.strip():
        return {
            "abstract_quality": "missing",
            "abstract_warning": "Abstract is missing or empty.",
        }

    text = re.sub(r"\s+", " ", abstract).strip()
    words = text.split()
    lower_text = text.lower()

    contamination_patterns = [
        r"identify applicable funding agency here",
        r"if none,\s*delete this",
        r"insert\s+(?:your\s+)?(?:text|abstract|figure|table)\s+here",
        r"lorem ipsum",
        r"this is a sample abstract",
        r"copyright\s+©?\s*\d{4}.*all rights reserved",
        r"references\s*:\s*(?:\[\d+\]|\d+\.)",
        r"^\s*(?:references|bibliography)\s*:",
    ]

    matched_patterns = [
        pattern
        for pattern in contamination_patterns
        if re.search(pattern, lower_text, flags=re.IGNORECASE)
    ]

    if matched_patterns:
        return {
            "abstract_quality": "corrupted",
            "abstract_warning": (
                "Abstract contains publisher-template, placeholder, or "
                "bibliography text; do not use it as research evidence."
            ),
        }

    if len(words) < 30:
        return {
            "abstract_quality": "short",
            "abstract_warning": (
                "Abstract is unusually short; treat findings cautiously."
            ),
        }

    if len(words) > 1200:
        return {
            "abstract_quality": "review",
            "abstract_warning": (
                "Abstract is unusually long; verify the original source."
            ),
        }

    normalized_words = [word.lower() for word in words]

    if len(normalized_words) >= 50:
        unique_ratio = len(set(normalized_words)) / len(normalized_words)

        if unique_ratio < 0.25:
            return {
                "abstract_quality": "review",
                "abstract_warning": (
                    "Abstract contains unusually repetitive text; "
                    "verify the original source."
                ),
            }

    section_patterns = [
        r"\b(?:introduction|related work|literature review|methodology|"
        r"materials and methods|results and discussion|conclusion|"
        r"acknowledg(?:e)?ments|references)\s*:",
        r"\bkeywords\s*:",
        r"\b(?:figure|table)\s+\d+\s*[:.]",
    ]

    section_hits = sum(
        len(re.findall(pattern, lower_text, flags=re.IGNORECASE))
        for pattern in section_patterns
    )

    if section_hits >= 3:
        return {
            "abstract_quality": "review",
            "abstract_warning": (
                "Abstract appears to contain multiple paper-section labels; "
                "verify the original source."
            ),
        }

    normalized_text = normalize_text(text)
    tokens = normalized_text.split()

    if len(tokens) >= 80:
        fivegrams = [
            " ".join(tokens[index:index + 5])
            for index in range(len(tokens) - 4)
        ]

        repeated_ratio = 1 - (
            len(set(fivegrams)) / max(1, len(fivegrams))
        )

        if repeated_ratio > 0.20:
            return {
                "abstract_quality": "review",
                "abstract_warning": (
                    "Abstract contains repeated text spans; "
                    "verify the original source."
                ),
            }

    if isinstance(title, str) and title.strip():
        title_terms = extract_terms(title)
        abstract_terms = extract_terms(text)

        if len(title_terms) >= 3:
            overlap = len(title_terms & abstract_terms) / len(title_terms)

            if overlap < 0.10:
                return {
                    "abstract_quality": "review",
                    "abstract_warning": (
                        "Abstract has little vocabulary overlap with the title; "
                        "check for a mismatched or contaminated abstract."
                    ),
                }

    return {
        "abstract_quality": "unchecked",
        "abstract_warning": "",
    }


# ---------------------------------------------------------
# Outcome relevance and claim screening
# ---------------------------------------------------------

def extract_outcome_terms(question: str) -> set[str]:
    """
    Extract terms describing the specific outcome or constraint.

    This is domain-independent and supplements, rather than replaces,
    semantic relevance scoring.
    """
    if not isinstance(question, str):
        return set()

    normalized = normalize_text(question)
    padded = f" {normalized} "

    markers = (
        " while ", " while also ", " while reducing ", " while minimizing ",
        " reduce ", " reducing ", " minimize ", " minimizing ",
        " lower ", " lowering ", " without ", " improve ", " improving ",
        " increase ", " increasing ", " predict ", " predicting ",
        " detect ", " detecting ", " identify ", " identifying ",
        " compare ", " compared with ", " versus ", " and assess ",
    )

    focus_parts = []

    for marker in markers:
        position = padded.find(marker)

        if position >= 0:
            part = padded[position + len(marker):].strip()

            if part:
                focus_parts.append(part)

    focus_text = " ".join(focus_parts)

    return extract_terms(focus_text) if focus_text else set()


def outcome_relevance_score(
    question: str,
    title: str,
    abstract: str,
) -> float:
    """
    Estimate whether a paper addresses the question's specific outcome.

    A lexical score does not prove that the paper measured or achieved
    the outcome.
    """
    focus_terms = extract_outcome_terms(question)

    if not focus_terms:
        return 0.0

    title_terms = extract_terms(title)
    abstract_terms = extract_terms(abstract)

    if not (title_terms or abstract_terms):
        return 0.0

    title_overlap = len(focus_terms & title_terms) / len(focus_terms)
    full_overlap = (
        len(focus_terms & (title_terms | abstract_terms))
        / len(focus_terms)
    )

    return min(1.0, 0.6 * full_overlap + 0.4 * title_overlap)


def check_claim_verification_needs(abstract: Any) -> dict:
    """
    Flag quantitative or emphatic claims for manual verification.

    A warning is not evidence that a claim is false.
    """
    if not isinstance(abstract, str) or not abstract.strip():
        return {
            "claim_verification_warning": (
                "No abstract available; claims cannot be screened."
            )
        }

    patterns = (
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b(?:accuracy|precision|recall|f1[- ]?score|auc|sensitivity|"
        r"specificity|false[- ]positive rate)\b",
        r"\b(?:reduced|increased|improved|outperformed|achieved|reached)\b"
        r".{0,100}\b\d+(?:\.\d+)?",
        r"\b\d{3,}\s+(?:samples|instances|patients|records|websites|"
        r"urls|cases)\b",
    )

    matched = [
        pattern
        for pattern in patterns
        if re.search(pattern, abstract, re.IGNORECASE)
    ]

    if matched:
        return {
            "claim_verification_warning": (
                "Abstract contains quantitative results or performance claims. "
                "Verify the dataset, baseline, metric definition, and results "
                "in the original paper."
            )
        }

    return {
        "claim_verification_warning": (
            "No obvious quantitative claim pattern detected; "
            "this does not verify the abstract."
        )
    }


def get_plan_terms(plan: dict) -> set[str]:
    """Collect additional research context from the planner."""
    if not isinstance(plan, dict):
        return set()

    fields = (
        "research_topic",
        "research_question",
        "keywords",
        "subtopics",
        "search_queries",
        "research_objectives",
        "research_scope",
    )

    collected = []

    for field in fields:
        value = plan.get(field, [])

        if isinstance(value, str):
            collected.append(value)

        elif isinstance(value, list):
            collected.extend(
                item for item in value
                if isinstance(item, str)
            )

    return extract_terms(" ".join(collected))


def build_plan_context(question: str, plan: dict) -> str:
    """Build a bounded context string from the research plan."""
    terms = sorted(get_plan_terms(plan))
    return question + " " + " ".join(terms[:40])


# ---------------------------------------------------------
# Semantic model
# ---------------------------------------------------------

def get_embedding_model():
    """
    Load the embedding model once.

    If loading fails, use lexical scoring instead of crashing.
    """
    global _embedding_model, _model_load_attempted

    if _embedding_model is not None:
        return _embedding_model

    if _model_load_attempted:
        return None

    _model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Semantic relevance model loaded successfully.")
        return _embedding_model

    except Exception as exc:
        print(
            "Semantic model could not be loaded. "
            f"Using lexical relevance fallback: {exc}"
        )
        return None


def lexical_relevance_score(
    question: str,
    title: str,
    abstract: str,
) -> float:
    """Calculate a fallback relevance score using term overlap."""
    question_terms = extract_terms(question)

    if not question_terms:
        return 0.0

    title_terms = extract_terms(title)
    abstract_terms = extract_terms(abstract)
    document_terms = title_terms | abstract_terms

    overall_overlap = (
        len(question_terms & document_terms) / len(question_terms)
    )

    title_overlap = (
        len(question_terms & title_terms) / len(question_terms)
    )

    return 0.65 * overall_overlap + 0.35 * title_overlap


def score_papers(
    question: str,
    papers: list[dict],
    plan: dict | None = None,
) -> list[dict]:
    """
    Rank papers using semantic relevance and a supplementary
    outcome-focused relevance score.
    """
    if not papers:
        return []

    model = get_embedding_model()
    context = build_plan_context(question, plan or {})

    paper_texts = []
    paper_titles = []

    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not isinstance(title, str):
            title = ""

        if not isinstance(abstract, str):
            abstract = ""

        paper_titles.append(title)
        paper_texts.append(f"Title: {title}. Abstract: {abstract}")

    if model is None:
        scored_papers = []

        for paper, title in zip(papers, paper_titles):
            abstract = paper.get("abstract", "")

            if not isinstance(abstract, str):
                abstract = ""

            base_score = lexical_relevance_score(
                question, title, abstract
            )
            focus_score = outcome_relevance_score(
                question, title, abstract
            )

            scored = dict(paper)
            scored["base_relevance_score"] = round(base_score, 4)
            scored["outcome_relevance_score"] = round(focus_score, 4)
            scored["relevance_score"] = round(
                0.85 * base_score + 0.15 * focus_score, 4
            )
            scored["title_relevance_score"] = round(
                lexical_relevance_score(question, title, ""), 4
            )
            scored["relevance_method"] = "lexical_fallback"

            scored_papers.append(scored)

        return sorted(
            scored_papers,
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

    try:
        texts = (
            [question, context]
            + paper_texts
            + paper_titles
        )

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        question_vector = embeddings[0]
        context_vector = embeddings[1]

        count = len(papers)
        paper_vectors = embeddings[2:2 + count]
        title_vectors = embeddings[2 + count:2 + (2 * count)]

        full_scores = paper_vectors @ question_vector
        title_scores = title_vectors @ question_vector
        context_scores = paper_vectors @ context_vector

        scored_papers = []

        for index, paper in enumerate(papers):
            direct_score = (
                0.75 * float(full_scores[index])
                + 0.25 * float(title_scores[index])
            )

            semantic_score = (
                0.90 * direct_score
                + 0.10 * float(context_scores[index])
            )

            abstract = paper.get("abstract", "")

            if not isinstance(abstract, str):
                abstract = ""

            focus_score = outcome_relevance_score(
                question, paper_titles[index], abstract
            )

            final_score = 0.85 * semantic_score + 0.15 * focus_score

            scored = dict(paper)
            scored["base_relevance_score"] = round(semantic_score, 4)
            scored["outcome_relevance_score"] = round(focus_score, 4)
            scored["relevance_score"] = round(final_score, 4)
            scored["title_relevance_score"] = round(
                float(title_scores[index]), 4
            )
            scored["relevance_method"] = EMBEDDING_MODEL_NAME

            scored_papers.append(scored)

        return sorted(
            scored_papers,
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

    except Exception as exc:
        print(
            "Semantic scoring failed. "
            f"Using lexical fallback: {exc}"
        )

        scored_papers = []

        for paper, title in zip(papers, paper_titles):
            abstract = paper.get("abstract", "")

            if not isinstance(abstract, str):
                abstract = ""

            base_score = lexical_relevance_score(
                question, title, abstract
            )
            focus_score = outcome_relevance_score(
                question, title, abstract
            )

            scored = dict(paper)
            scored["base_relevance_score"] = round(base_score, 4)
            scored["outcome_relevance_score"] = round(focus_score, 4)
            scored["relevance_score"] = round(
                0.85 * base_score + 0.15 * focus_score, 4
            )
            scored["title_relevance_score"] = round(
                lexical_relevance_score(question, title, ""), 4
            )
            scored["relevance_method"] = "lexical_fallback"

            scored_papers.append(scored)

        return sorted(
            scored_papers,
            key=lambda item: item["relevance_score"],
            reverse=True,
        )


# ---------------------------------------------------------
# Relevance checking
# ---------------------------------------------------------

def is_relevant(
    paper: dict,
    question: str,
    plan: dict | None = None,
) -> bool:
    """Check relevance while retaining compatibility with existing tests."""
    if not isinstance(paper, dict):
        return False

    title = paper.get("title")

    if not isinstance(title, str) or not title.strip():
        return False

    if not isinstance(question, str) or not question.strip():
        return False

    ranked = score_papers(
        question=question,
        papers=[paper],
        plan=plan or {},
    )

    if not ranked:
        return False

    scored = ranked[0]
    score = scored["relevance_score"]

    if scored["relevance_method"] == "lexical_fallback":
        question_terms = extract_terms(question)
        title_terms = extract_terms(title)

        abstract = paper.get("abstract", "")

        if not isinstance(abstract, str):
            abstract = ""

        document_terms = title_terms | extract_terms(abstract)

        if not question_terms:
            return False

        return (
            score >= 0.25
            and len(question_terms & document_terms)
            >= min(2, len(question_terms))
            and len(question_terms & title_terms) >= 1
        )

    return (
        score >= MIN_RELEVANCE_SCORE
        and scored.get("title_relevance_score", 0.0)
        >= MIN_TITLE_RELEVANCE_SCORE
    )


# ---------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------

def get_paper_identifier(paper: dict) -> str:
    """Create a stable identifier for duplicate detection."""
    doi = paper.get("doi")

    if isinstance(doi, str) and doi.strip():
        doi = doi.strip().lower()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
        return "doi:" + doi

    paper_id = paper.get("paperId") or paper.get("arxiv_id")

    if paper_id:
        return "id:" + str(paper_id).strip().lower()

    url = paper.get("url") or paper.get("paper_url")

    if isinstance(url, str) and url.strip():
        url = url.strip().lower().rstrip("/")
        url = re.sub(r"[?#].*$", "", url)
        return "url:" + url

    title = normalize_text(paper.get("title") or "")

    return "title:" + title if title else ""


def merge_paper_metadata(
    existing: dict,
    incoming: dict,
) -> dict:
    """Merge duplicate records without discarding useful metadata."""
    merged = dict(existing)

    for key, value in incoming.items():
        current = merged.get(key)

        if current is None or current == "" or current == []:
            if value is not None and value != "" and value != []:
                merged[key] = value

    sources = []

    for source in (
        existing.get("source"),
        incoming.get("source"),
    ):
        if isinstance(source, str):
            for item in source.split("; "):
                if item and item not in sources:
                    sources.append(item)

    if sources:
        merged["source"] = "; ".join(sources)

    return merged


# ---------------------------------------------------------
# Main retrieval agent
# ---------------------------------------------------------

def retrieval_agent(
    question: str,
    plan: dict,
    limit: int = 10,
) -> list[dict]:
    """
    Retrieve, rank, filter, and deduplicate papers across domains.

    Quantitative claims are flagged for verification, not automatically
    accepted as true or rejected as false.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The research question cannot be empty.")

    question = question.strip()

    if limit < 1:
        return []

    if not isinstance(plan, dict):
        plan = {}

    # Support workflow state containing a nested plan.
    if isinstance(plan.get("plan"), dict):
        plan = plan["plan"]

    # Search the user's actual question first.
    queries = [question]
    planned_queries = plan.get("search_queries", [])

    if isinstance(planned_queries, list):
        for query in planned_queries:
            if not isinstance(query, str) or not query.strip():
                continue

            query = query.strip()
            normalized_query = normalize_text(query)

            if not normalized_query:
                continue

            if all(
                normalize_text(existing) != normalized_query
                for existing in queries
            ):
                queries.append(query)

            if len(queries) >= MAX_SEARCH_QUERIES:
                break

    # Gather candidates before ranking.
    candidates = {}
    identifier_to_key = {}

    sources = [
        ("Semantic Scholar", search_papers),
        ("arXiv", search_arxiv),
    ]

    for query in queries[:MAX_SEARCH_QUERIES]:
        for source_name, search_function in sources:
            try:
                results = search_function(
                    query,
                    limit=RESULTS_PER_QUERY,
                )

                if not isinstance(results, list):
                    print(
                        f"{source_name} returned an unexpected "
                        f"result for query: {query}"
                    )
                    continue

                for original_paper in results:
                    if not isinstance(original_paper, dict):
                        continue

                    title = original_paper.get("title")

                    if not isinstance(title, str) or not title.strip():
                        continue

                    paper = dict(original_paper)
                    paper["source"] = source_name

                    identifier = get_paper_identifier(paper)
                    normalized_title = normalize_text(title)

                    if not normalized_title:
                        continue

                    # Title-based deduplication supplements provider IDs.
                    key = "title:" + normalized_title
                    existing_key = None

                    if identifier:
                        existing_key = identifier_to_key.get(identifier)

                    if existing_key is None and key in candidates:
                        existing_key = key

                    if existing_key is not None:
                        candidates[existing_key] = merge_paper_metadata(
                            candidates[existing_key],
                            paper,
                        )
                    else:
                        existing_key = key
                        candidates[existing_key] = paper

                    if identifier:
                        identifier_to_key[identifier] = existing_key

                time.sleep(0.5)

            except Exception as exc:
                print(
                    f"{source_name} search failed for "
                    f"'{query}': {exc}"
                )
                continue

    if not candidates:
        return []

    # Check abstract quality and flag claims that need manual verification.
    for paper in candidates.values():
        raw_abstract = paper.get("abstract", "")

        quality = check_abstract_quality(
            raw_abstract,
            paper.get("title", ""),
        )

        paper.update(quality)
        paper.update(check_claim_verification_needs(raw_abstract))

        if quality.get("abstract_quality") == "corrupted":
            # Preserve original text for inspection, but don't pass it
            # downstream as usable research evidence.
            paper["abstract_original"] = raw_abstract
            paper["abstract"] = ""

    # Rank all unique candidates.
    unique_papers = list(candidates.values())

    ranked_papers = score_papers(
        question=question,
        papers=unique_papers,
        plan=plan,
    )

    relevant_papers = []

    for paper in ranked_papers:
        score = paper.get("relevance_score", 0.0)
        method = paper.get("relevance_method", "")
        title_score = paper.get("title_relevance_score", 0.0)

        if method == "lexical_fallback":
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if not isinstance(title, str):
                title = ""

            if not isinstance(abstract, str):
                abstract = ""

            question_terms = extract_terms(question)
            title_terms = extract_terms(title)
            document_terms = title_terms | extract_terms(abstract)

            document_overlap = len(question_terms & document_terms)
            title_overlap = len(question_terms & title_terms)
            minimum_overlap = min(2, len(question_terms))

            if (
                score < 0.25
                or document_overlap < minimum_overlap
                or title_overlap < 1
            ):
                continue

        else:
            if (
                score < MIN_RELEVANCE_SCORE
                or title_score < MIN_TITLE_RELEVANCE_SCORE
            ):
                continue

        # Retain relevant papers with missing/short abstracts, but preserve
        # their quality warnings so downstream agents can handle them safely.
        relevant_papers.append(paper)

        if len(relevant_papers) >= limit:
            break

    return relevant_papers
