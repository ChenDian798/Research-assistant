from __future__ import annotations

from .doi import (
    extract_arxiv_id,
    extract_doi,
    extract_pmid,
    fetch_arxiv_metadata,
    fetch_crossref_metadata,
    fetch_pubmed_metadata,
)


def verify_references(references: list[dict]) -> list[dict]:
    return [verify_reference(reference) for reference in references if isinstance(reference, dict)]


def verify_reference(reference: dict) -> dict:
    item = dict(reference or {})
    risks: list[str] = list(item.get("verification_risks") or [])
    sources = ["paper-search-mcp"]
    doi = extract_doi(item) or str(item.get("doi") or "").strip()
    pmid = extract_pmid(item) or str(item.get("pmid") or "").strip()
    arxiv_id = extract_arxiv_id(item) or str(item.get("arxiv_id") or "").strip()

    metadata = {}
    verified_by = ""
    if doi:
        metadata = fetch_crossref_metadata(doi)
        verified_by = "Crossref"
    elif pmid:
        metadata = fetch_pubmed_metadata(pmid)
        verified_by = "PubMed"
    elif arxiv_id:
        metadata = fetch_arxiv_metadata(arxiv_id)
        verified_by = "arXiv"

    if metadata:
        sources.append(verified_by)
        conflicts = metadata_conflicts(item, metadata)
        if conflicts:
            status = "needs_review"
            risks.extend(conflicts)
        else:
            status = "verified"
        item = merge_verified_metadata(item, metadata)
        item["verification_status"] = status
        item["verification_sources"] = sources
        item["verification_risks"] = list(dict.fromkeys(risks))
        item["provenance"] = build_provenance(item, verified_by=verified_by, evidence_level=evidence_level(item))
        return item

    if doi or pmid or arxiv_id:
        status = "unverified"
        risks.append("stable_id_lookup_failed")
    elif item.get("source"):
        status = "partial"
        risks.append("no_stable_id_for_secondary_lookup")
    else:
        status = "unverified"
        risks.append("missing_verifiable_source")

    item["verification_status"] = status
    item["verification_sources"] = sources
    item["verification_risks"] = list(dict.fromkeys(risks))
    item["provenance"] = build_provenance(item, verified_by="", evidence_level=evidence_level(item))
    return item


def merge_verified_metadata(reference: dict, metadata: dict) -> dict:
    merged = dict(reference)
    for key in ("doi", "pmid", "arxiv_id", "title", "source", "authors", "year", "journal", "abstract"):
        if metadata.get(key):
            merged[key] = metadata[key]
    return merged


def metadata_conflicts(reference: dict, metadata: dict) -> list[str]:
    risks = []
    ref_title = normalize_title(reference.get("title"))
    meta_title = normalize_title(metadata.get("title"))
    if ref_title and meta_title and not title_compatible(ref_title, meta_title):
        risks.append("title_conflict")
    ref_year = str(reference.get("year") or "").strip()
    meta_year = str(metadata.get("year") or "").strip()
    if ref_year and meta_year and ref_year != meta_year:
        risks.append("year_conflict")
    return risks


def title_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= max(1, min(len(left_tokens), len(right_tokens)) // 2)


def normalize_title(value) -> str:
    return " ".join(str(value or "").casefold().replace(":", " ").split())


def evidence_level(reference: dict) -> str:
    if reference.get("pdf_text_available") or reference.get("evidence_source_text"):
        return "full_text"
    if reference.get("abstract"):
        return "metadata+abstract"
    return "metadata"


def build_provenance(reference: dict, *, verified_by: str, evidence_level: str) -> dict:
    provenance = dict(reference.get("provenance") or {})
    provenance.setdefault("retrieved_from", reference.get("retrieved_from") or reference.get("source_label") or "")
    if verified_by:
        provenance["verified_by"] = verified_by
    provenance["evidence_level"] = evidence_level
    return provenance
