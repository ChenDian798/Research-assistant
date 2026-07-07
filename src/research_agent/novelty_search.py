from __future__ import annotations

import re
from collections import defaultdict

from .paper_search import (
    PaperSearchError,
    filter_papers_by_year,
    normalize_search_payload,
    normalize_source_name,
    normalize_sources,
    paper_identity_key,
    run_paper_search_backend_by_source,
)
from .reference_screening import screen_reference


def run_novelty_search_plan(
    plan: dict,
    sources: str | list[str],
    *,
    year: str = "",
    max_results_per_source: int = 8,
    timeout_seconds: int = 45,
) -> dict:
    requested_sources = normalize_sources(sources)
    queries = [item for item in plan.get("queries") or [] if isinstance(item, dict) and item.get("query")]
    diagnostics_queries: list[dict] = []
    raw_candidates: list[dict] = []
    source_summary: dict[str, dict] = {
        source: {"returned": 0, "kept": 0, "filtered": 0, "error": ""} for source in requested_sources
    }

    for search_query in queries:
        query_sources = [normalize_source_name(source) for source in search_query.get("sources") or []]
        active_sources = [source for source in query_sources if source in requested_sources]
        if not active_sources:
            continue
        query_limit = bounded_int(
            search_query.get("max_results_per_source"),
            default=max_results_per_source,
            minimum=1,
            maximum=max_results_per_source,
        )
        for source in active_sources:
            query_diag = {
                "query_id": search_query.get("query_id", ""),
                "purpose": search_query.get("purpose", ""),
                "claim_id": search_query.get("claim_id", ""),
                "claim_query_type": search_query.get("claim_query_type", ""),
                "source": source,
                "query": search_query.get("query", ""),
                "strictness": search_query.get("strictness", ""),
                "returned": 0,
                "kept": 0,
                "filtered": 0,
                "error": "",
            }
            try:
                payload = run_paper_search_backend_by_source(
                    {source: search_query["query"]},
                    max_results_per_source=query_limit,
                    year=str(year or "").strip(),
                    timeout_seconds=max(1, int(timeout_seconds or 45)),
                )
                papers, source_results, errors = normalize_search_payload(payload, [source])
                papers = filter_papers_by_year(papers, str(year or "").strip())
                returned = int(source_results.get(source, len(papers)) or 0)
                query_diag["returned"] = returned
                source_summary[source]["returned"] += returned
                if errors.get(source):
                    query_diag["error"] = errors[source]
                    source_summary[source]["error"] = merge_error(source_summary[source].get("error"), errors[source])
                for paper in papers:
                    candidate = build_candidate(dict(paper), search_query, plan)
                    if candidate["candidate_status"] == "source_noise":
                        query_diag["filtered"] += 1
                        source_summary[source]["filtered"] += 1
                    else:
                        query_diag["kept"] += 1
                        source_summary[source]["kept"] += 1
                    raw_candidates.append(candidate)
            except Exception as error:
                message = str(error)
                query_diag["error"] = message
                source_summary[source]["error"] = merge_error(source_summary[source].get("error"), message)
            diagnostics_queries.append(query_diag)

    deduped_candidates = merge_duplicate_candidates(raw_candidates)
    strong_candidates = [item for item in deduped_candidates if item.get("candidate_status") == "strong_candidate"]
    weak_candidates = [item for item in deduped_candidates if item.get("candidate_status") == "weak_candidate"]
    noise_candidates = [item for item in deduped_candidates if item.get("candidate_status") == "source_noise"]
    assessment_candidates = [*strong_candidates, *weak_candidates[: max(0, 24 - len(strong_candidates))]]
    diagnostics = build_diagnostics(
        queries=diagnostics_queries,
        source_summary=source_summary,
        raw_count=len(raw_candidates),
        deduped_count=len(deduped_candidates),
        strong_count=len(strong_candidates),
        weak_count=len(weak_candidates),
        noise_count=len(noise_candidates),
        assessment_count=len(assessment_candidates),
    )
    return {
        "status": "done",
        "plan": plan,
        "queries": queries,
        "candidates": assessment_candidates,
        "strong_candidates": strong_candidates,
        "weak_candidates": weak_candidates,
        "source_noise": noise_candidates,
        "raw_candidates": raw_candidates,
        "diagnostics": diagnostics,
        "source_results": {source: summary["returned"] for source, summary in source_summary.items()},
        "errors": {source: summary["error"] for source, summary in source_summary.items() if summary.get("error")},
        "raw_count": len(raw_candidates),
        "deduped_count": len(deduped_candidates),
    }


def build_candidate(paper: dict, query: dict, plan: dict) -> dict:
    screened = screen_reference(paper)
    text = candidate_text(screened)
    required_hits = matched_terms(text, concept_terms(plan.get("required_concepts")))
    method_hits = matched_terms(text, concept_terms(plan.get("method_concepts")))
    context_hits = matched_terms(text, concept_terms(plan.get("context_concepts")))
    baseline_hits = matched_terms(text, [str(item) for item in plan.get("baseline_concepts") or []])
    notes: list[str] = []
    if required_hits:
        notes.append(f"matched core topic: {', '.join(required_hits[:4])}")
    if method_hits:
        notes.append(f"matched method: {', '.join(method_hits[:4])}")
    if context_hits:
        notes.append(f"matched context: {', '.join(context_hits[:4])}")
    if baseline_hits:
        notes.append(f"matched baseline: {', '.join(baseline_hits[:4])}")
    if screened.get("screening_status") == "rejected":
        notes.extend(screened.get("screening_reasons") or [])

    status = classify_candidate(
        screened,
        required_hits=required_hits,
        method_hits=method_hits,
        context_hits=context_hits,
        baseline_hits=baseline_hits,
        required_total=len(concept_terms(plan.get("required_concepts"))),
    )
    screened.update(
        {
            "retrieved_from": normalize_source_name(screened.get("retrieved_from") or paper.get("retrieved_from") or paper.get("source_label")),
            "matched_query_ids": [query.get("query_id", "")],
            "retrieval_purpose": [query.get("purpose", "")],
            "matched_claim_ids": [query.get("claim_id", "")] if query.get("claim_id") else [],
            "claim_query_types": [query.get("claim_query_type", "")] if query.get("claim_query_type") else [],
            "candidate_status": status,
            "screening_notes": list(dict.fromkeys(note for note in notes if note)),
            "matched_concepts": {
                "required": required_hits,
                "method": method_hits,
                "context": context_hits,
                "baseline": baseline_hits,
            },
        }
    )
    return screened


def classify_candidate(
    candidate: dict,
    *,
    required_hits: list[str],
    method_hits: list[str],
    context_hits: list[str],
    baseline_hits: list[str],
    required_total: int,
) -> str:
    if candidate.get("screening_status") == "rejected" and not (required_hits or method_hits or context_hits or baseline_hits):
        return "source_noise"
    if not candidate.get("title") and not candidate.get("abstract"):
        return "source_noise"
    core_hits = len(required_hits)
    if required_total <= 0:
        required_total = 2
    if core_hits >= min(3, required_total) or (core_hits >= 2 and (method_hits or context_hits or baseline_hits)):
        return "strong_candidate"
    if core_hits >= 1 or method_hits or context_hits or baseline_hits:
        return "weak_candidate"
    return "source_noise"


def merge_duplicate_candidates(candidates: list[dict]) -> list[dict]:
    merged: list[dict] = []
    positions: dict[str, int] = {}
    for candidate in candidates:
        key = paper_identity_key(candidate)
        if not key:
            key = title_key(candidate)
        if key and key in positions:
            existing = merged[positions[key]]
            existing["matched_query_ids"] = merge_list(existing.get("matched_query_ids"), candidate.get("matched_query_ids"))
            existing["retrieval_purpose"] = merge_list(existing.get("retrieval_purpose"), candidate.get("retrieval_purpose"))
            existing["matched_claim_ids"] = merge_list(existing.get("matched_claim_ids"), candidate.get("matched_claim_ids"))
            existing["claim_query_types"] = merge_list(existing.get("claim_query_types"), candidate.get("claim_query_types"))
            existing["screening_notes"] = merge_list(existing.get("screening_notes"), candidate.get("screening_notes"))
            existing["candidate_status"] = stronger_status(existing.get("candidate_status"), candidate.get("candidate_status"))
            existing["matched_concepts"] = merge_matched_concepts(existing.get("matched_concepts"), candidate.get("matched_concepts"))
            continue
        if key:
            positions[key] = len(merged)
        merged.append(candidate)
    return merged


def build_diagnostics(
    *,
    queries: list[dict],
    source_summary: dict[str, dict],
    raw_count: int,
    deduped_count: int,
    strong_count: int,
    weak_count: int,
    noise_count: int,
    assessment_count: int,
) -> dict:
    warnings = []
    for source, summary in source_summary.items():
        if summary.get("error"):
            if source == "semantic":
                warnings.append("Semantic Scholar failed or was rate-limited; this run relies on the remaining sources.")
            else:
                warnings.append(f"{source} search failed or returned an error: {summary.get('error')}")
    for item in queries:
        if item.get("source") == "pubmed" and item.get("strictness") == "narrow" and not item.get("returned"):
            warnings.append(f"PubMed narrow query returned 0 for {item.get('query_id')}.")
    if raw_count and assessment_count == 0:
        warnings.append("Search returned records, but all were filtered as weak metadata or source noise.")
    if raw_count == 0:
        warnings.append("No source returned candidate records; broaden the claim or verify source availability.")
    return {
        "queries": queries,
        "source_summary": source_summary,
        "candidate_pool": {
            "raw": raw_count,
            "deduped": deduped_count,
            "strong": strong_count,
            "weak": weak_count,
            "noise": noise_count,
            "sent_to_overlap_assessment": assessment_count,
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def matched_terms(text: str, terms: list[str]) -> list[str]:
    lower = normalize_for_match(text)
    hits = []
    for term in terms:
        normalized = normalize_for_match(term)
        if not normalized:
            continue
        synonyms = [normalized, *[normalize_for_match(item) for item in TERM_MATCH_SYNONYMS.get(normalized, [])]]
        if any(matches_term(lower, synonym) for synonym in synonyms):
            hits.append(term)
    return list(dict.fromkeys(hits))


def matches_term(text: str, term: str) -> bool:
    if not term:
        return False
    if " " in term:
        return term in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def concept_terms(items) -> list[str]:
    terms = []
    for item in items or []:
        term = item.get("term") if isinstance(item, dict) else item
        term = str(term or "").strip()
        if term and term.casefold() not in {existing.casefold() for existing in terms}:
            terms.append(term)
    return terms


def candidate_text(candidate: dict) -> str:
    return " ".join(str(candidate.get(key) or "") for key in ("title", "abstract", "journal", "source_label", "relevance"))


def normalize_for_match(value: str) -> str:
    text = re.sub(r"[\W_]+", " ", str(value or "").casefold())
    return re.sub(r"\s+", " ", text).strip()


def merge_list(left, right) -> list:
    return list(dict.fromkeys([*(left or []), *(right or [])]))


def merge_matched_concepts(left, right) -> dict:
    merged: dict[str, list] = defaultdict(list)
    for source in (left, right):
        if not isinstance(source, dict):
            continue
        for key, values in source.items():
            merged[key] = merge_list(merged[key], values if isinstance(values, list) else [])
    return dict(merged)


def stronger_status(left: str, right: str) -> str:
    order = {"source_noise": 0, "weak_candidate": 1, "strong_candidate": 2}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def title_key(candidate: dict) -> str:
    title = normalize_for_match(candidate.get("title"))
    year = str(candidate.get("year") or "").strip()
    return f"title:{title}:{year}" if title else ""


def merge_error(left: str | None, right: str | None) -> str:
    parts = [part for part in [str(left or "").strip(), str(right or "").strip()] if part]
    return "; ".join(dict.fromkeys(parts))[:1000]


def bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


TERM_MATCH_SYNONYMS = {
    "ischemic stroke": ["ischaemic stroke", "cerebral infarction", "stroke", "brain infarct"],
    "non contrast ct": ["ncct", "computed tomography", "ct"],
    "non contrast": ["ncct", "computed tomography", "ct"],
    "non contrast ct": ["ncct", "computed tomography", "ct"],
    "lesion segmentation": ["segmentation", "delineation", "lesion delineation"],
    "retrieval augmented generation": ["rag"],
    "question answering": ["qa"],
    "large language model": ["llm"],
    "domain generalization": ["domain adaptation", "cross domain"],
    "low resource": ["few shot", "small sample"],
}
