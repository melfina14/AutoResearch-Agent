
import json
import re
from urllib.parse import quote

import requests
from app.llm import generate

CROSSREF_API = "https://api.crossref.org/works/"
REQUEST_TIMEOUT = 8
NOT_PROVIDED = "Not provided"


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(x for x in (_text(v) for v in value) if x)
    if isinstance(value, dict):
        if value.get("name"):
            return _text(value["name"])
        return " ".join(
            x for x in (
                _text(value.get("given")),
                _text(value.get("family"))
            ) if x
        )
    return ""


def _title(paper):
    return _text(paper.get("title") or paper.get("paper_title"))


def _year(paper):
    for key in ("year", "publication_year", "published_year"):
        match = re.search(
            r"\b(19|20)\d{2}\b",
            str(paper.get(key, ""))
        )
        if match:
            return match.group(0)
    return ""


def _clean_doi(value):
    """Normalize a DOI or DOI URL into a bare DOI string."""
    value = _text(value)
    if not value:
        return ""

    value = value.strip().strip("<>\"' ")
    value = re.sub(r"^doi\s*:\s*", "", value, flags=re.I)
    value = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value,
        flags=re.I
    )

    match = re.search(r"(10\.\d{4,9}/[^\s<>\"']+)", value, re.I)
    if not match:
        return ""

    return match.group(1).rstrip(".,;)]}\"")


def _doi(paper):
    """Find DOI in common metadata formats."""
    for key in ("doi", "DOI", "external_doi", "paper_doi"):
        doi = _clean_doi(paper.get(key))
        if doi:
            return doi

    # DOI may be nested in Semantic Scholar externalIds or other metadata.
    for container_key in (
        "externalIds",
        "external_ids",
        "identifiers",
        "ids",
        "metadata",
        "citation",
    ):
        container = paper.get(container_key)

        if isinstance(container, dict):
            for key in ("DOI", "doi", "external_doi", "paper_doi"):
                doi = _clean_doi(container.get(key))
                if doi:
                    return doi

    # Some providers supply identifiers as a list.
    for key in ("identifiers", "externalIds", "external_ids"):
        values = paper.get(key)

        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    label = _text(
                        item.get("type")
                        or item.get("name")
                        or item.get("scheme")
                    ).lower()

                    if "doi" in label:
                        doi = _clean_doi(
                            item.get("value")
                            or item.get("id")
                            or item.get("url")
                        )
                        if doi:
                            return doi

                elif isinstance(item, str):
                    doi = _clean_doi(item)
                    if doi:
                        return doi

    # DOI may be embedded in a URL field.
    for key in (
        "url",
        "paper_url",
        "link",
        "doi_url",
        "publicationUrl",
        "publication_url",
    ):
        doi = _clean_doi(paper.get(key))
        if doi:
            return doi

    return ""


def _norm_title(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _doi_from_report(report_text, title):
    """
    Recover a DOI from the matching References entry if the paper
    record omitted the DOI.
    """
    if not report_text or not title:
        return ""

    lines = report_text.splitlines()
    target = _norm_title(title)

    for index, line in enumerate(lines):
        if target and target in _norm_title(line):
            # Search only the matching reference block.
            for candidate in lines[index + 1:index + 18]:
                if re.match(r"\s*#{1,4}\s+\d+\.", candidate):
                    break

                if re.search(r"\bDOI\b", candidate, re.I) and ":" in candidate:
                    doi = _clean_doi(candidate)
                    if doi:
                        return doi

            # Also allow a DOI on the title line itself.
            doi = _clean_doi(line)
            if doi:
                return doi

    return ""


def _url(paper):
    return _text(
        paper.get("url")
        or paper.get("paper_url")
        or paper.get("link")
    )


def _crossref_check(paper):
    """
    Look up DOI metadata in Crossref.

    A failed request means the citation could not be checked,
    not that the citation is necessarily false.
    """
    doi = _doi(paper)

    if not doi:
        return {
            "external_check_status": "Not checked",
            "external_source": "",
            "external_metadata_match": "Not checked",
            "external_notes": (
                "No DOI supplied; Crossref lookup was not attempted."
            ),
        }

    try:
        response = requests.get(
            CROSSREF_API + quote(doi, safe="/"),
            headers={
                "User-Agent": (
                    "AutoResearchAgent/1.0 "
                    "(citation verification)"
                )
            },
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 404:
            return {
                "external_check_status": "Checked - DOI not found in Crossref",
                "external_source": "Crossref",
                "external_metadata_match": "Mismatch or not indexed",
                "external_notes": (
                    "Crossref returned 404. This does not prove the work "
                    "does not exist; the DOI may be incorrect or registered "
                    "elsewhere."
                ),
            }

        response.raise_for_status()
        item = response.json().get("message", {})

        titles = item.get("title") or []
        external_title = _text(titles[0]) if titles else ""

        external_year = ""
        for key in (
            "published-print",
            "published-online",
            "issued",
            "created",
        ):
            parts = item.get(key, {}).get("date-parts", [])
            if parts and parts[0]:
                external_year = str(parts[0][0])
                break

        supplied_title = _title(paper)
        supplied_year = _year(paper)

        title_match = (
            _norm_title(supplied_title) == _norm_title(external_title)
            if supplied_title and external_title
            else None
        )

        year_match = (
            supplied_year == external_year
            if supplied_year and external_year
            else None
        )

        mismatches = []

        if title_match is False:
            mismatches.append(
                "Supplied title differs from Crossref title "
                f"({external_title or 'not available'})."
            )

        if year_match is False:
            mismatches.append(
                f"Supplied year {supplied_year} differs from "
                f"Crossref year {external_year}."
            )

        if mismatches:
            match_status = "Mismatch"
        elif title_match is True and year_match is True:
            match_status = "Title and year matched"
        elif title_match is True:
            match_status = "Title matched; year not checked"
        else:
            match_status = "Partial comparison only"

        return {
            "external_check_status": "Checked",
            "external_source": "Crossref",
            "external_metadata_match": match_status,
            "external_title": external_title,
            "external_year": external_year,
            "external_notes": (
                "; ".join(mismatches)
                if mismatches
                else (
                    "Crossref returned a DOI record. This does not verify "
                    "the abstract or report claims."
                )
            ),
        }

    except requests.RequestException as exc:
        return {
            "external_check_status": "Unable to check",
            "external_source": "Crossref",
            "external_metadata_match": "Not checked",
            "external_notes": (
                f"Crossref request failed ({type(exc).__name__}); "
                "citation remains externally unverified."
            ),
        }

    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {
            "external_check_status": "Unable to check",
            "external_source": "Crossref",
            "external_metadata_match": "Not checked",
            "external_notes": (
                "Crossref response could not be interpreted "
                f"({type(exc).__name__})."
            ),
        }


def _abstract_warning(paper):
    quality = _text(paper.get("abstract_quality")).lower()
    warning = _text(paper.get("abstract_warning"))
    abstract = _text(paper.get("abstract"))

    if quality == "corrupted":
        return (
            warning
            or "Abstract was flagged as corrupted by retrieval quality checks."
        )

    if quality == "missing" or not abstract:
        return (
            warning
            or "No abstract is available; abstract-based evidence cannot be assessed."
        )

    if quality in {"review", "short", "poor"}:
        return (
            warning
            or f"Abstract quality is marked '{quality}'; manual review is required."
        )

    return ""


def _basic_metadata_status(paper):
    title = _title(paper)
    year = _year(paper)
    doi = _doi(paper)
    url = _url(paper)

    if not title or not year or not (doi or url):
        return "Incomplete"

    if doi and not re.match(r"^10\.\d{4,9}/\S+$", doi, re.I):
        return "Needs review"

    return "Complete in supplied record"


def citation_verification_agent(
    papers: list[dict],
    report_text: str = ""
) -> dict:
    """
    Check supplied paper records, query Crossref for DOI metadata
    when possible, and assess report claims against supplied abstracts.

    Crossref metadata matches do not independently verify abstracts,
    scientific findings, or every report statement.
    """
    if not isinstance(papers, list) or not papers:
        return {
            "agent": "Citation Verification Agent",
            "papers_checked": 0,
            "verification": {
                "status": "Unable to verify",
                "verification_scope": "No papers were provided.",
                "checked_citations": [],
                "issues": [],
                "unsupported_claims": [],
                "summary": "No papers were provided.",
            },
        }

    valid_papers = [
        paper for paper in papers
        if isinstance(paper, dict)
    ]

    if not valid_papers:
        return {
            "agent": "Citation Verification Agent",
            "papers_checked": 0,
            "verification": {
                "status": "Unable to verify",
                "verification_scope": (
                    "No valid paper records were provided."
                ),
                "checked_citations": [],
                "issues": [],
                "unsupported_claims": [],
                "summary": "No valid paper records were provided.",
            },
        }

    report_text = (
        report_text.strip()
        if isinstance(report_text, str)
        else ""
    )

    checked = []
    issues = []
    evidence_records = []

    for paper in valid_papers:
        title = _title(paper) or "Untitled record"
        year = _year(paper) or NOT_PROVIDED
        doi = _doi(paper)
        url = _url(paper)

        # The report generator may display a DOI from metadata even when
        # the workflow passes a flattened paper record. Try recovering it
        # from this paper's own References entry as a fallback.
        doi_from_report = False

        if not doi:
            doi = _doi_from_report(report_text, title)
            doi_from_report = bool(doi)

        verification_record = dict(paper)

        if doi and not _doi(verification_record):
            verification_record["doi"] = doi

        identifier = doi or url or NOT_PROVIDED

        metadata_status = _basic_metadata_status(verification_record)
        external = _crossref_check(verification_record)
        abstract_warning = _abstract_warning(paper)

        if metadata_status == "Incomplete":
            issues.append({
                "issue": (
                    "One or more key citation fields "
                    "(title, year, DOI/URL) are missing."
                ),
                "paper_title": title,
                "severity": "Medium",
                "recommended_action": (
                    "Check the original publication page and complete "
                    "missing metadata."
                ),
            })

        elif metadata_status == "Needs review":
            issues.append({
                "issue": (
                    "The supplied DOI does not match expected DOI syntax."
                ),
                "paper_title": title,
                "severity": "High",
                "recommended_action": (
                    "Check the DOI against the publisher or DOI registry."
                ),
            })

        if external.get("external_metadata_match") == "Mismatch":
            metadata_status = "Mismatch with Crossref"

            issues.append({
                "issue": external.get(
                    "external_notes",
                    "Citation metadata differs from Crossref."
                ),
                "paper_title": title,
                "severity": "High",
                "recommended_action": (
                    "Inspect the authoritative publication record "
                    "before correcting metadata."
                ),
            })

        elif external.get("external_check_status") == "Checked":
            metadata_status = external.get(
                "external_metadata_match",
                metadata_status
            )

        elif doi:
            metadata_status = "Not externally verified"

        quality = _text(paper.get("abstract_quality")).lower()

        if abstract_warning:
            issues.append({
                "issue": abstract_warning,
                "paper_title": title,
                "severity": (
                    "High" if quality == "corrupted" else "Medium"
                ),
                "recommended_action": (
                    "Retrieve a clean abstract or inspect the original "
                    "publication before using this paper as evidence."
                ),
            })

        recovery_note = (
            " DOI recovered from the report reference entry."
            if doi_from_report
            else ""
        )

        notes = (
            abstract_warning
            or (
                "Metadata checks do not establish that all research "
                "claims are accurate."
            )
        )
        notes += recovery_note

        checked.append({
            "paper_title": title,
            "year": year,
            "identifier": identifier,
            "metadata_status": metadata_status,
            "evidence_status": (
                "Insufficient evidence"
                if abstract_warning
                else "Pending report-claim assessment"
            ),
            "external_check_status": external.get(
                "external_check_status",
                "Not checked"
            ),
            "external_source": external.get("external_source", ""),
            "external_metadata_match": external.get(
                "external_metadata_match",
                "Not checked"
            ),
            "external_notes": external.get("external_notes", ""),
            "notes": notes,
        })

        evidence_records.append({
            "title": title,
            "year": year,
            "doi": doi,
            "url": url,
            "abstract": _text(paper.get("abstract")),
            "abstract_quality": (
                quality
                or (
                    "missing"
                    if not _text(paper.get("abstract"))
                    else "unspecified"
                )
            ),
            "abstract_warning": abstract_warning,
        })

    unsupported_claims = []
    llm_summary = ""
    claim_scope = "Report-claim assessment was not completed."

    if report_text:
        prompt = f"""
Assess whether statements in this report are supported by the supplied paper records.

Rules:
- Use only the supplied records and report text.
- Do not claim a paper exists or a DOI resolves based only on supplied records.
- Treat abstracts marked corrupted, missing, review, short, or with warnings
  as unreliable or insufficient evidence.
- A claim is supported only when the supplied abstract or metadata directly
  supports it.
- Do not invent findings, datasets, metrics, authors, or publication details.
- List only concrete unsupported claims; do not list every sentence.

Return valid JSON only:
{{
  "unsupported_claims": [
    {{
      "claim": "Report claim",
      "reason": "Why evidence is insufficient",
      "paper_title": "Relevant title or Not specified"
    }}
  ],
  "summary": "Concise assessment"
}}

SUPPLIED RECORDS:
{json.dumps(evidence_records, ensure_ascii=False, indent=2)}

REPORT:
{report_text}
"""

        try:
            response = generate(prompt)
            cleaned = response.strip() if isinstance(response, str) else ""

            if cleaned.startswith("```"):
                lines = cleaned.splitlines()

                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                cleaned = "\n".join(lines).strip()

            result = json.loads(cleaned)

            if not isinstance(result, dict):
                raise ValueError("Response JSON must be an object.")

            raw_claims = result.get("unsupported_claims", [])

            if isinstance(raw_claims, list):
                for claim in raw_claims:
                    if isinstance(claim, dict) and _text(
                        claim.get("claim")
                    ):
                        unsupported_claims.append({
                            "claim": _text(claim.get("claim")),
                            "reason": (
                                _text(claim.get("reason"))
                                or "Supplied evidence is insufficient."
                            ),
                            "paper_title": (
                                _text(claim.get("paper_title"))
                                or NOT_PROVIDED
                            ),
                        })

            llm_summary = _text(result.get("summary"))

            claim_scope = (
                "Report claims assessed by an LLM against supplied records "
                "only; this is not independent scientific validation."
            )

            for citation in checked:
                if citation["evidence_status"] == (
                    "Pending report-claim assessment"
                ):
                    citation["evidence_status"] = (
                        "Supplied evidence assessed; not independently validated"
                    )

        except Exception as exc:
            claim_scope = (
                "Report-claim assessment could not complete "
                f"({type(exc).__name__}); deterministic checks are still reported."
            )

            issues.append({
                "issue": "Report-claim assessment did not complete.",
                "paper_title": NOT_PROVIDED,
                "severity": "Medium",
                "recommended_action": (
                    "Retry and manually compare key claims with original "
                    "publications."
                ),
            })

    else:
        for citation in checked:
            if citation["evidence_status"] != "Insufficient evidence":
                citation["evidence_status"] = (
                    "Not assessed (no report text supplied)"
                )

    crossref_checked = sum(
        citation.get("external_check_status") == "Checked"
        for citation in checked
    )

    mismatch_count = sum(
        citation.get("external_metadata_match") == "Mismatch"
        for citation in checked
    )

    corrupted_count = sum(
        paper["abstract_quality"] == "corrupted"
        for paper in evidence_records
    )

    if mismatch_count:
        status = "Needs review"
    elif crossref_checked:
        status = "Partially externally checked"
    else:
        status = "Supplied records checked; external verification incomplete"

    summary = (
        f"Processed {len(valid_papers)} paper record(s). "
        f"Crossref returned metadata for {crossref_checked} record(s). "
        "Crossref metadata matches do not verify abstracts or report claims."
    )

    if corrupted_count:
        summary += (
            f" {corrupted_count} record(s) have corrupted abstracts "
            "and should not be used as evidence."
        )

    if unsupported_claims:
        summary += (
            f" The report-claim assessment flagged {len(unsupported_claims)} "
            "unsupported or insufficiently supported claim(s)."
        )

    if llm_summary:
        summary += " " + llm_summary

    verification = {
        "status": status,
        "verification_scope": (
            "Checked supplied title/year/DOI/URL fields; looked up DOI "
            "metadata through Crossref where possible; "
            + claim_scope
        ),
        "checked_citations": checked,
        "issues": issues,
        "unsupported_claims": unsupported_claims,
        "summary": summary,
    }

    return {
        "agent": "Citation Verification Agent",
        "papers_checked": len(valid_papers),
        "verification": verification,
    }
