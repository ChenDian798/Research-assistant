from __future__ import annotations

import re
from urllib.parse import urlparse

from .doi import extract_arxiv_id, extract_doi, extract_pmid


GENERIC_TITLES = {
    "paper",
    "article",
    "research paper",
    "untitled",
    "unknown",
    "no title",
    "pubmed",
    "arxiv",
}


def screen_references(papers: list[dict]) -> dict:
    qualified = []
    needs_review = []
    rejected = []
    seen = set()
    for paper in papers:
        item = dict(paper) if isinstance(paper, dict) else {}
        screened = screen_reference(item)
        key = screened.get("dedupe_key", "")
        if key and key in seen:
            screened["screening_status"] = "rejected"
            screened.setdefault("screening_reasons", []).append("duplicate")
            screened.setdefault("screening_risks", []).append("duplicate_record")
        elif key:
            seen.add(key)

        status = screened.get("screening_status")
        if status == "qualified":
            qualified.append(screened)
        elif status == "needs_review":
            needs_review.append(screened)
        else:
            rejected.append(screened)
    return {
        "qualified": qualified,
        "needs_review": needs_review,
        "rejected": rejected,
    }


def screen_reference(reference: dict) -> dict:
    item = dict(reference or {})
    reasons: list[str] = []
    risks: list[str] = []

    if item.get("source_error"):
        return with_screening(item, "rejected", ["source_error"], [str(item.get("source_error"))], "")

    title = clean_text(item.get("title"))
    doi = extract_doi(item) or clean_text(item.get("doi"))
    pmid = extract_pmid(item) or clean_text(item.get("pmid"))
    arxiv_id = extract_arxiv_id(item) or clean_text(item.get("arxiv_id"))
    source = clean_text(item.get("source") or item.get("url") or item.get("abs_url"))
    dedupe_key = dedupe_key_for(item, doi=doi, pmid=pmid, arxiv_id=arxiv_id, source=source)

    if not title:
        return with_screening(item, "rejected", ["missing_title"], ["no_title"], dedupe_key)
    if is_generic_title(title):
        return with_screening(item, "rejected", ["generic_title"], ["title_too_generic"], dedupe_key)

    stable_url = is_stable_source_url(source)
    if doi:
        reasons.append("has_doi")
    if pmid:
        reasons.append("has_pmid")
    if arxiv_id:
        reasons.append("has_arxiv_id")
    if stable_url:
        reasons.append("has_stable_url")
    if item.get("abstract"):
        reasons.append("has_abstract")
    if item.get("authors"):
        reasons.append("has_authors")
    if item.get("year"):
        reasons.append("has_year")

    if doi or pmid or arxiv_id:
        item["doi"] = doi or clean_text(item.get("doi"))
        item["pmid"] = pmid or clean_text(item.get("pmid"))
        item["arxiv_id"] = arxiv_id or clean_text(item.get("arxiv_id"))
        return with_screening(item, "qualified", reasons, risks, dedupe_key)

    support_count = sum(bool(item.get(key)) for key in ("authors", "year", "abstract"))
    if stable_url and support_count >= 2:
        return with_screening(item, "qualified", reasons, risks, dedupe_key)

    if stable_url and support_count >= 1:
        risks.append("stable_url_with_sparse_metadata")
        return with_screening(item, "needs_review", reasons or ["has_stable_url"], risks, dedupe_key)

    risks.append("missing_stable_identifier")
    return with_screening(item, "rejected", ["missing_stable_source_url_or_id"], risks, dedupe_key)


def with_screening(item: dict, status: str, reasons: list[str], risks: list[str], dedupe_key: str) -> dict:
    item["screening_status"] = status
    item["screening_reasons"] = list(dict.fromkeys(reason for reason in reasons if reason))
    item["screening_risks"] = list(dict.fromkeys(risk for risk in risks if risk))
    item["dedupe_key"] = dedupe_key
    return item


def dedupe_key_for(reference: dict, *, doi: str, pmid: str, arxiv_id: str, source: str) -> str:
    if doi:
        return f"doi:{doi.casefold()}"
    if pmid:
        return f"pmid:{pmid}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.casefold()}"
    if source:
        return f"url:{source.rstrip('/').casefold()}"
    title = re.sub(r"\W+", " ", clean_text(reference.get("title")).casefold()).strip()
    year = clean_text(reference.get("year"))
    return f"title:{title}:{year}" if title else ""


def is_stable_source_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    if any(host.endswith(domain) for domain in [
        "doi.org",
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "semanticscholar.org",
        "openalex.org",
        "crossref.org",
        "cnki.net",
        "ncbi.nlm.nih.gov",
    ]):
        return True
    return bool(parsed.path.strip("/") and "." in host)


def is_generic_title(title: str) -> bool:
    normalized = re.sub(r"\W+", " ", str(title or "").casefold()).strip()
    if normalized in GENERIC_TITLES:
        return True
    words = normalized.split()
    return len(words) <= 2 and any(word in GENERIC_TITLES for word in words)


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
