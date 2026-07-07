from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter

from .llm import LLMClient, LLMServiceError
from .novelty_planner import sanitize_innovation_text


NOVELTY_CHECK_SYSTEM_PROMPT = """
You are a scholarly novelty-search reviewer.
Your job is to compare a user's claimed paper innovation against candidate literature.
Return only valid JSON, no markdown.
Schema:
{
  "innovation_claims": ["short claim"],
  "innovation_profile": {
    "domain": "biomedical|computer|engineering|society|general",
    "innovation_types": [
      {
        "type": "technical_route|problem_framing|data_or_sample|application_context|evaluation_design|theory_framework|system_engineering|clinical_pathway|material_or_process|combination",
        "risk": "high|moderate|low|unknown",
        "assessment": "short assessment of this innovation type"
      }
    ]
  },
  "overall": {
    "risk_level": "high|moderate|low|unknown",
    "assessment": "concise assessment",
    "confidence": "high|medium|low"
  },
  "comparisons": [
    {
      "reference_index": 0,
      "overlap_level": "high_overlap|partial_overlap|adjacent|no_clear_overlap",
      "overlap_score": 0.0,
      "overlap_points": ["what appears similar"],
      "difference_points": ["what appears different or not evidenced"],
      "dimension_overlap": {
        "target_problem": "same|similar|different|unknown",
        "data_or_population": "same|similar|different|unknown",
        "method": "same|partial|different|unknown",
        "application_context": "same|similar|different|unknown",
        "evaluation": "same|partial|different|unknown"
      },
      "evidence": "short evidence from title/abstract/metadata",
      "recommendation": "what the researcher should verify or adjust"
    }
  ],
  "next_steps": ["action"]
}
Rules:
- Judge overlap with the user's claimed innovation, not merely broad topic relevance.
- High overlap means the candidate appears to contain substantially the same core idea, method, target, and context.
- Partial overlap means one or more core elements match, but important method/context/claim details differ or are not evidenced.
- Adjacent means same neighborhood but no clear reuse of the claimed innovation.
- No clear overlap means the candidate metadata does not show meaningful overlap.
- Do not say the idea is proven novel, completely original, or has no overlap anywhere.
- Be conservative when only title/abstract metadata is available. Do not claim novelty is proven.
- Use Chinese for narrative fields when the user writes mainly Chinese; otherwise use English.
- Keep every list compact.
""".strip()


class NoveltyCheckWorkflow:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    async def run(
        self,
        *,
        innovation_text: str,
        references: list[dict],
        search_payload: dict | None = None,
    ) -> dict:
        clean_text = normalize_space(sanitize_innovation_text(innovation_text) or innovation_text)
        clean_references = [normalize_reference(reference, index) for index, reference in enumerate(references)]
        clean_references = [reference for reference in clean_references if reference.get("title") or reference.get("abstract")]
        if not clean_text:
            raise ValueError("Innovation description cannot be empty.")

        result = None
        llm_attempted = False
        if clean_references and self._can_use_llm():
            llm_attempted = True
            try:
                result = await self._run_llm_check(clean_text, clean_references)
            except (LLMServiceError, TimeoutError, asyncio.TimeoutError, ValueError, json.JSONDecodeError, RuntimeError):
                result = None
        if not result:
            result = self._run_rules_check(clean_text, clean_references)
            if clean_references:
                warning = llm_degradation_warning(clean_text, status="fallback")
                result["overall"]["confidence"] = "low"
                result["overall"]["assessment"] = prefix_assessment_warning(
                    result.get("overall", {}).get("assessment", ""),
                    warning,
                )
                result.setdefault("next_steps", []).append(warning)
                result["llm_assessment"] = {
                    "status": "fallback",
                    "selected_reference_count": 0 if not llm_attempted else min(
                        len(clean_references),
                        bounded_int_env("NOVELTY_CHECK_MAX_LLM_REFERENCES", 15, minimum=1, maximum=40),
                    ),
                    "batch_count": 0,
                    "succeeded_batch_count": 0,
                    "failed_batch_count": 0,
                    "failure_types": ["LLMUnavailable"] if not llm_attempted else ["LLMAssessmentFailed"],
                    "warnings": [warning],
                }

        result = normalize_novelty_result(result, clean_text, clean_references, search_payload=search_payload or {})
        result["references"] = public_references(clean_references)
        result["search"] = compact_search_payload(search_payload or {})
        result["closest_prior_work"] = build_closest_prior_work(
            result.get("comparisons", []),
            plan=(search_payload or {}).get("plan", {}),
        )
        result["novelty_dimensions"] = build_novelty_dimensions(result.get("comparisons", []))
        result["innovation_profile"] = build_innovation_profile(
            clean_text,
            result.get("comparisons", []),
            search_payload=search_payload or {},
            llm_profile=result.get("innovation_profile", {}),
        )
        return result

    def _can_use_llm(self) -> bool:
        if self.llm is not None:
            return True
        try:
            self.llm = LLMClient()
            return True
        except RuntimeError:
            return False

    async def _run_llm_check(self, innovation_text: str, references: list[dict]) -> dict:
        assert self.llm is not None
        baseline = self._run_rules_check(innovation_text, references)
        selected_references = select_references_for_llm(innovation_text, references)
        if not selected_references:
            return baseline

        batch_size = bounded_int_env("NOVELTY_CHECK_LLM_BATCH_SIZE", 5, minimum=1, maximum=12)
        max_parallel = bounded_int_env("NOVELTY_CHECK_LLM_PARALLEL_BATCHES", 3, minimum=1, maximum=6)
        batches = chunked(selected_references, batch_size)
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_batch(batch: list[dict], batch_index: int):
            async with semaphore:
                return await self._run_llm_check_batch(innovation_text, batch, batch_index=batch_index)

        raw_results = await asyncio.gather(
            *(run_batch(batch, index) for index, batch in enumerate(batches)),
            return_exceptions=True,
        )

        llm_results = [item for item in raw_results if isinstance(item, dict)]
        failures = [item for item in raw_results if isinstance(item, Exception)]
        if not llm_results:
            baseline["overall"]["confidence"] = "low"
            warning = llm_degradation_warning(innovation_text, status="fallback")
            baseline["overall"]["assessment"] = prefix_assessment_warning(
                baseline.get("overall", {}).get("assessment", ""),
                warning,
            )
            baseline.setdefault("next_steps", []).append(warning)
            baseline["llm_assessment"] = {
                "status": "fallback",
                "selected_reference_count": len(selected_references),
                "batch_count": len(batches),
                "succeeded_batch_count": 0,
                "failed_batch_count": len(failures),
                "failure_types": exception_type_names(failures),
                "warnings": [warning],
            }
            return baseline

        return merge_llm_batch_results(
            baseline,
            llm_results,
            selected_count=len(selected_references),
            batch_count=len(batches),
            failed_count=len(failures),
            failure_types=exception_type_names(failures),
            innovation_text=innovation_text,
        )

    async def _run_llm_check_batch(self, innovation_text: str, references: list[dict], *, batch_index: int) -> dict:
        assert self.llm is not None
        prompt = (
            f"User innovation description:\n{innovation_text[:4000]}\n\n"
            f"Candidate batch: {batch_index + 1}\n"
            "Candidate literature metadata:\n"
            f"{json.dumps(references_for_prompt(references), ensure_ascii=False, indent=2)}"
        )
        content = await asyncio.wait_for(
            self.llm.complete(
                system_prompt=NOVELTY_CHECK_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=bounded_int_env("NOVELTY_CHECK_LLM_MAX_TOKENS", 2500, minimum=800, maximum=8000),
            ),
            timeout=bounded_int_env("NOVELTY_CHECK_LLM_TIMEOUT_SECONDS", 120, minimum=15, maximum=300),
        )
        return parse_json_object(content)

    @staticmethod
    def _run_rules_check(innovation_text: str, references: list[dict]) -> dict:
        claims = extract_claims(innovation_text)
        comparisons = []
        for reference in references:
            score, hits = lexical_overlap_score(innovation_text, reference_text(reference))
            if score >= 0.58:
                level = "high_overlap"
            elif score >= 0.38:
                level = "partial_overlap"
            elif score >= 0.2:
                level = "adjacent"
            else:
                level = "no_clear_overlap"
            comparisons.append(
                {
                    "reference_index": reference["reference_index"],
                    "overlap_level": level,
                    "overlap_score": round(score, 3),
                    "overlap_points": hits[:6] or ["No strong shared terms were found in available metadata."],
                    "difference_points": ["Only title/abstract metadata was checked; verify the full text before making a final claim."],
                    "dimension_overlap": dimension_overlap_from_hits(innovation_text, reference, hits),
                    "evidence": evidence_excerpt(reference, hits),
                    "recommendation": recommendation_for_level(level),
                }
            )
        risk = overall_risk_level(comparisons)
        return {
            "innovation_claims": claims,
            "overall": {
                "risk_level": risk,
                "assessment": assessment_for_risk(risk, len(comparisons)),
                "confidence": "low" if not comparisons else "medium-low",
            },
            "comparisons": comparisons,
            "next_steps": [
                "Review the highest-overlap papers manually, especially full-text method and experiment sections.",
                "Tighten the innovation statement around the method, scenario, dataset, baseline, and claimed improvement.",
            ],
        }


def normalize_reference(reference: dict, index: int) -> dict:
    item = dict(reference or {})
    return {
        "reference_index": index,
        "title": normalize_space(item.get("title")),
        "source": normalize_space(item.get("source") or item.get("url")),
        "authors": normalize_space(item.get("authors")),
        "year": normalize_space(item.get("year")),
        "journal": normalize_space(item.get("journal") or item.get("venue")),
        "doi": normalize_space(item.get("doi")),
        "pmid": normalize_space(item.get("pmid")),
        "arxiv_id": normalize_space(item.get("arxiv_id")),
        "source_label": normalize_space(item.get("source_label")),
        "abstract": normalize_space(item.get("abstract") or item.get("relevance")),
        "screening_status": normalize_space(item.get("screening_status")),
        "verification_status": normalize_space(item.get("verification_status")),
        "verification_risks": item.get("verification_risks") if isinstance(item.get("verification_risks"), list) else [],
        "verification_sources": item.get("verification_sources") if isinstance(item.get("verification_sources"), list) else [],
        "provenance": item.get("provenance") if isinstance(item.get("provenance"), dict) else {},
        "retrieved_from": normalize_space(item.get("retrieved_from")),
        "matched_query_ids": item.get("matched_query_ids") if isinstance(item.get("matched_query_ids"), list) else [],
        "retrieval_purpose": item.get("retrieval_purpose") if isinstance(item.get("retrieval_purpose"), list) else [],
        "matched_claim_ids": item.get("matched_claim_ids") if isinstance(item.get("matched_claim_ids"), list) else [],
        "claim_query_types": item.get("claim_query_types") if isinstance(item.get("claim_query_types"), list) else [],
        "candidate_status": normalize_space(item.get("candidate_status")),
        "screening_notes": item.get("screening_notes") if isinstance(item.get("screening_notes"), list) else [],
        "matched_concepts": item.get("matched_concepts") if isinstance(item.get("matched_concepts"), dict) else {},
    }


def references_for_prompt(references: list[dict]) -> list[dict]:
    return [
        {
            "reference_index": reference.get("reference_index"),
            "title": reference.get("title", ""),
            "source": reference.get("source", ""),
            "authors": reference.get("authors", ""),
            "year": reference.get("year", ""),
            "journal": reference.get("journal", ""),
            "abstract": reference.get("abstract", "")[:1200],
            "doi": reference.get("doi", ""),
            "pmid": reference.get("pmid", ""),
            "arxiv_id": reference.get("arxiv_id", ""),
            "verification_status": reference.get("verification_status", ""),
            "verification_risks": reference.get("verification_risks", []),
            "candidate_status": reference.get("candidate_status", ""),
            "matched_concepts": reference.get("matched_concepts", {}),
            "matched_claim_ids": reference.get("matched_claim_ids", []),
        }
        for reference in references
    ]


def select_references_for_llm(innovation_text: str, references: list[dict]) -> list[dict]:
    max_count = bounded_int_env("NOVELTY_CHECK_MAX_LLM_REFERENCES", 15, minimum=1, maximum=40)
    ranked = sorted(
        references,
        key=lambda reference: llm_reference_rank(innovation_text, reference),
        reverse=True,
    )
    return ranked[:max_count]


def llm_reference_rank(innovation_text: str, reference: dict) -> tuple[float, float, int]:
    lexical_score, _hits = lexical_overlap_score(innovation_text, reference_text(reference))
    status_weight = {
        "strong_candidate": 3.0,
        "weak_candidate": 2.0,
        "source_noise": 0.0,
    }.get(str(reference.get("candidate_status") or "").strip(), 1.0)
    screening_weight = {
        "qualified": 0.3,
        "needs_review": 0.1,
        "rejected": -0.5,
    }.get(str(reference.get("screening_status") or "").strip(), 0.0)
    verification_weight = {
        "verified": 0.25,
        "partial": 0.05,
        "needs_review": -0.05,
        "unverified": -0.1,
    }.get(str(reference.get("verification_status") or "").strip(), 0.0)
    concepts = reference.get("matched_concepts") if isinstance(reference.get("matched_concepts"), dict) else {}
    concept_hits = sum(len(values) for values in concepts.values() if isinstance(values, list))
    return status_weight + screening_weight + verification_weight + min(1.5, concept_hits * 0.2), lexical_score, -int(reference.get("reference_index") or 0)


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def merge_llm_batch_results(
    baseline: dict,
    llm_results: list[dict],
    *,
    selected_count: int,
    batch_count: int,
    failed_count: int,
    failure_types: list[str],
    innovation_text: str,
) -> dict:
    merged = dict(baseline)
    comparisons_by_index = {
        int(item.get("reference_index")): dict(item)
        for item in baseline.get("comparisons", [])
        if isinstance(item, dict) and str(item.get("reference_index", "")).strip()
    }
    innovation_claims = list(baseline.get("innovation_claims") or [])
    next_steps = list(baseline.get("next_steps") or [])
    overall_assessments = []
    innovation_profile = baseline.get("innovation_profile") if isinstance(baseline.get("innovation_profile"), dict) else {}
    for result in llm_results:
        for claim in result.get("innovation_claims") if isinstance(result.get("innovation_claims"), list) else []:
            text = normalize_space(claim)
            if text and text not in innovation_claims:
                innovation_claims.append(text)
        for item in result.get("comparisons") if isinstance(result.get("comparisons"), list) else []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("reference_index"))
            except (TypeError, ValueError):
                continue
            comparisons_by_index[index] = item
        for step in result.get("next_steps") if isinstance(result.get("next_steps"), list) else []:
            text = normalize_space(step)
            if text and text not in next_steps:
                next_steps.append(text)
        overall = result.get("overall") if isinstance(result.get("overall"), dict) else {}
        assessment = safe_assessment(normalize_space(overall.get("assessment")))
        if assessment:
            overall_assessments.append(assessment)
        if not innovation_profile and isinstance(result.get("innovation_profile"), dict):
            innovation_profile = result["innovation_profile"]

    merged["innovation_claims"] = innovation_claims[:8]
    merged["comparisons"] = list(comparisons_by_index.values())
    merged["next_steps"] = next_steps[:8]
    if innovation_profile:
        merged["innovation_profile"] = innovation_profile
    if overall_assessments:
        merged["overall"] = {
            **(merged.get("overall") if isinstance(merged.get("overall"), dict) else {}),
            "assessment": overall_assessments[0][:1000],
            "confidence": "medium" if failed_count == 0 else "medium-low",
        }
    warnings = []
    if failed_count:
        warning = llm_degradation_warning(innovation_text, status="partial")
        warnings.append(warning)
        merged["overall"] = {
            **(merged.get("overall") if isinstance(merged.get("overall"), dict) else {}),
            "assessment": prefix_assessment_warning(
                (merged.get("overall") if isinstance(merged.get("overall"), dict) else {}).get("assessment", ""),
                warning,
            ),
            "confidence": "medium-low",
        }
        if warning not in next_steps:
            merged["next_steps"] = [warning, *next_steps][:8]
    merged["llm_assessment"] = {
        "status": "partial" if failed_count else "done",
        "selected_reference_count": selected_count,
        "batch_count": batch_count,
        "succeeded_batch_count": len(llm_results),
        "failed_batch_count": failed_count,
        "failure_types": failure_types,
        "warnings": warnings,
    }
    return merged


def exception_type_names(exceptions: list[Exception]) -> list[str]:
    return list(dict.fromkeys(type(error).__name__ for error in exceptions))


def llm_degradation_warning(innovation_text: str, *, status: str) -> str:
    is_zh = bool(re.search(r"[\u4e00-\u9fff]", str(innovation_text or "")))
    if status == "fallback":
        return (
            "LLM 重合度评估超时或失败，本次结果已降级为本地规则/词面重合评估；请将结论视为初筛结果，并优先人工核验高重合候选。"
            if is_zh
            else "The LLM overlap assessment timed out or failed, so this result falls back to local lexical/rule-based assessment. Treat it as a preliminary screen and manually verify high-overlap candidates."
        )
    return (
        "部分 LLM 批次超时或失败，本次结果混合了模型判断与本地规则兜底；请重点人工核验高重合和部分重合候选。"
        if is_zh
        else "Some LLM overlap-assessment batches timed out or failed, so this result mixes model judgments with local rule-based fallback. Manually verify high- and partial-overlap candidates."
    )


def prefix_assessment_warning(assessment: str, warning: str) -> str:
    clean_warning = normalize_space(warning)
    clean_assessment = normalize_space(assessment)
    if not clean_warning:
        return clean_assessment
    if clean_assessment.startswith(clean_warning):
        return clean_assessment
    if not clean_assessment:
        return clean_warning
    return f"{clean_warning} {clean_assessment}"


def normalize_novelty_result(
    result: dict,
    innovation_text: str,
    references: list[dict],
    *,
    search_payload: dict | None = None,
) -> dict:
    data = result if isinstance(result, dict) else {}
    comparisons_by_index = {}
    for item in data.get("comparisons") if isinstance(data.get("comparisons"), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("reference_index"))
        except (TypeError, ValueError):
            continue
        comparisons_by_index[index] = item

    normalized_comparisons = []
    for reference in references:
        index = int(reference.get("reference_index", len(normalized_comparisons)))
        raw = comparisons_by_index.get(index, {})
        level = normalize_overlap_level(raw.get("overlap_level"))
        score = bounded_float(raw.get("overlap_score"), default=score_from_level(level))
        normalized_comparisons.append(
            {
                "reference_index": index,
                "title": reference.get("title", ""),
                "source": reference.get("source", ""),
                "authors": reference.get("authors", ""),
                "year": reference.get("year", ""),
                "source_label": reference.get("source_label", ""),
                "doi": reference.get("doi", ""),
                "pmid": reference.get("pmid", ""),
                "arxiv_id": reference.get("arxiv_id", ""),
                "verification_status": normalized_verification_status(reference.get("verification_status")),
                "verification_risks": reference.get("verification_risks", []),
                "verification_note": verification_note(reference),
                "overlap_level": level,
                "overlap_score": round(score, 3),
                "overlap_points": clean_string_list(raw.get("overlap_points"), limit=6),
                "difference_points": clean_string_list(raw.get("difference_points"), limit=6),
                "dimension_overlap": normalize_dimension_overlap(raw.get("dimension_overlap"), reference),
                "evidence": evidence_with_verification_note(normalize_space(raw.get("evidence"))[:800], reference),
                "recommendation": recommendation_with_verification_note(normalize_space(raw.get("recommendation"))[:500], reference),
                "retrieved_from": reference.get("retrieved_from", ""),
                "matched_query_ids": reference.get("matched_query_ids", []),
                "retrieval_purpose": reference.get("retrieval_purpose", []),
                "matched_claim_ids": reference.get("matched_claim_ids", []),
                "claim_query_types": reference.get("claim_query_types", []),
                "candidate_status": reference.get("candidate_status", ""),
                "screening_notes": reference.get("screening_notes", []),
            }
        )

    normalized_comparisons.sort(key=lambda item: overlap_sort_key(item), reverse=True)
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    risk = aggregate_risk_level(normalized_comparisons) or normalize_risk_level(overall.get("risk_level") or overall_risk_level(normalized_comparisons))
    normalized = {
        "status": "done",
        "innovation_text": innovation_text,
        "innovation_claims": clean_string_list(data.get("innovation_claims"), limit=8) or extract_claims(innovation_text),
        "overall": {
            "risk_level": risk,
            "assessment": safe_assessment(normalize_space(overall.get("assessment"))[:1000]) or assessment_for_risk(risk, len(normalized_comparisons)),
            "confidence": normalize_space(overall.get("confidence"))[:120] or "medium-low",
        },
        "comparisons": normalized_comparisons,
        "next_steps": clean_string_list(data.get("next_steps"), limit=8) or [
            "Review high-overlap and partial-overlap papers manually.",
            "Refine the innovation statement to emphasize verifiable differences.",
        ],
        "counts": novelty_counts(normalized_comparisons),
        "innovation_profile": normalize_innovation_profile(data.get("innovation_profile"), innovation_text, search_payload or {}),
    }
    if isinstance(data.get("llm_assessment"), dict):
        normalized["llm_assessment"] = normalize_llm_assessment(data["llm_assessment"])
    else:
        normalized["llm_assessment"] = {
            "status": "done",
            "selected_reference_count": 0,
            "batch_count": 0,
            "succeeded_batch_count": 0,
            "failed_batch_count": 0,
            "failure_types": [],
            "warnings": [],
        }
    return normalized


def normalize_llm_assessment(value: dict) -> dict:
    status = normalize_space(value.get("status"))
    if status not in {"done", "partial", "fallback"}:
        status = "done"
    batch_count = safe_int(value.get("batch_count"))
    failed = safe_int(value.get("failed_batch_count"))
    succeeded = safe_int(value.get("succeeded_batch_count"))
    if not succeeded and batch_count:
        succeeded = max(0, batch_count - failed)
    return {
        "status": status,
        "selected_reference_count": safe_int(value.get("selected_reference_count")),
        "batch_count": batch_count,
        "succeeded_batch_count": succeeded,
        "failed_batch_count": failed,
        "failure_types": clean_string_list(value.get("failure_types"), limit=12),
        "warnings": clean_string_list(value.get("warnings"), limit=12),
    }


def safe_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def normalize_innovation_profile(value, innovation_text: str, search_payload: dict) -> dict:
    if not isinstance(value, dict):
        return {"domain": infer_innovation_domain(innovation_text, search_payload), "innovation_types": [], "domain_focus": []}
    domain = normalize_domain(value.get("domain")) or infer_innovation_domain(innovation_text, search_payload)
    return {
        "domain": domain,
        "innovation_types": normalize_profile_rows(value.get("innovation_types"), type_key="type"),
        "domain_focus": normalize_profile_rows(value.get("domain_focus"), type_key="key"),
    }


def normalize_profile_rows(value, *, type_key: str) -> list[dict]:
    rows = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        key = normalize_profile_key(item.get(type_key))
        if not key:
            continue
        rows.append(
            {
                type_key: key,
                "risk": normalize_risk_level(item.get("risk")),
                "assessment": normalize_space(item.get("assessment"))[:500],
                "supporting_references": [
                    safe_int(ref)
                    for ref in item.get("supporting_references", [])
                    if isinstance(item.get("supporting_references"), list)
                ][:6],
            }
        )
    return rows[:10]


def build_innovation_profile(
    innovation_text: str,
    comparisons: list[dict],
    *,
    search_payload: dict | None = None,
    llm_profile: dict | None = None,
) -> dict:
    llm_profile = llm_profile if isinstance(llm_profile, dict) else {}
    domain = normalize_domain(llm_profile.get("domain")) or infer_innovation_domain(innovation_text, search_payload or {})
    active_types = active_innovation_types(innovation_text, domain)
    llm_types = {
        row.get("type"): row
        for row in normalize_profile_rows(llm_profile.get("innovation_types"), type_key="type")
        if row.get("type")
    }
    type_rows = []
    for innovation_type in active_types:
        derived = summarize_innovation_type(innovation_type, comparisons)
        llm_row = llm_types.get(innovation_type, {})
        type_rows.append(
            {
                "type": innovation_type,
                "risk": normalize_risk_level(llm_row.get("risk")) if llm_row else derived["risk"],
                "assessment": normalize_space(llm_row.get("assessment"))[:500] if llm_row.get("assessment") else derived["assessment"],
                "supporting_references": llm_row.get("supporting_references") if llm_row.get("supporting_references") else derived["supporting_references"],
            }
        )
    return {
        "domain": domain,
        "innovation_types": type_rows,
        "domain_focus": build_domain_focus(domain, comparisons),
    }


def infer_innovation_domain(text: str, search_payload: dict) -> str:
    mode = normalize_domain(
        (search_payload or {}).get("search_mode")
        or (search_payload or {}).get("requested_search_mode")
        or (search_payload or {}).get("mode")
    )
    if mode and mode != "general":
        return mode
    lower = normalize_space(text).casefold()
    domain_terms = {
        "biomedical": ["clinical", "patient", "disease", "diagnosis", "treatment", "ct", "mri", "医学", "临床", "患者", "疾病", "诊断", "治疗"],
        "computer": ["algorithm", "model", "neural", "transformer", "llm", "rag", "benchmark", "计算机", "人工智能", "模型", "算法", "神经网络", "大模型"],
        "engineering": ["material", "process", "manufacturing", "sensor", "structure", "工艺", "材料", "结构", "传感器", "制造", "工程"],
        "society": ["policy", "law", "survey", "governance", "education", "economics", "法律", "政策", "社会", "问卷", "治理", "教育"],
    }
    scores = {
        domain: sum(1 for term in terms if term.casefold() in lower)
        for domain, terms in domain_terms.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score else "general"


def normalize_domain(value) -> str:
    clean = normalize_space(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "medical": "biomedical",
        "medicine": "biomedical",
        "bio": "biomedical",
        "ai": "computer",
        "computer_science": "computer",
        "cs": "computer",
        "engineering_science": "engineering",
        "social": "society",
        "social_science": "society",
    }
    clean = aliases.get(clean, clean)
    return clean if clean in {"biomedical", "computer", "engineering", "society", "general"} else ""


def active_innovation_types(text: str, domain: str) -> list[str]:
    lower = normalize_space(text).casefold()
    keyword_map = {
        "technical_route": ["method", "algorithm", "model", "architecture", "training", "pipeline", "方法", "算法", "模型", "架构", "技术路线", "训练"],
        "problem_framing": ["problem", "task", "definition", "objective", "问题", "任务", "定义", "目标"],
        "data_or_sample": ["data", "dataset", "sample", "population", "cohort", "patient", "数据", "样本", "人群", "队列", "患者"],
        "application_context": ["application", "scenario", "domain", "context", "场景", "应用", "领域", "情境"],
        "evaluation_design": ["evaluation", "metric", "benchmark", "baseline", "ablation", "指标", "评价", "评估", "基线", "消融", "对比实验"],
        "theory_framework": ["theory", "framework", "hypothesis", "mechanism", "理论", "框架", "假设", "机制"],
        "system_engineering": ["system", "platform", "deployment", "workflow", "系统", "平台", "部署", "流程", "工程实现"],
        "clinical_pathway": ["clinical", "intervention", "diagnosis", "treatment", "outcome", "临床", "干预", "诊断", "治疗", "结局"],
        "material_or_process": ["material", "process", "structure", "manufacturing", "材料", "工艺", "结构", "制备", "制造"],
        "combination": ["combination", "hybrid", "integrated", "fusion", "联合", "组合", "融合", "集成"],
    }
    active = [
        key
        for key, terms in keyword_map.items()
        if any(term.casefold() in lower for term in terms)
    ]
    domain_defaults = {
        "biomedical": ["technical_route", "data_or_sample", "clinical_pathway", "evaluation_design"],
        "computer": ["technical_route", "problem_framing", "data_or_sample", "evaluation_design"],
        "engineering": ["technical_route", "material_or_process", "system_engineering", "evaluation_design"],
        "society": ["theory_framework", "problem_framing", "data_or_sample", "evaluation_design"],
        "general": ["technical_route", "application_context", "data_or_sample", "evaluation_design"],
    }
    for item in domain_defaults.get(domain, domain_defaults["general"]):
        if item not in active:
            active.append(item)
    if "combination" not in active and len(active) >= 3:
        active.append("combination")
    return active[:8]


def normalize_profile_key(value) -> str:
    clean = normalize_space(value).casefold().replace("-", "_").replace(" ", "_")
    allowed = {
        "technical_route",
        "problem_framing",
        "data_or_sample",
        "application_context",
        "evaluation_design",
        "theory_framework",
        "system_engineering",
        "clinical_pathway",
        "material_or_process",
        "combination",
        "biomedical_population",
        "biomedical_intervention",
        "biomedical_outcome",
        "computer_model",
        "computer_benchmark",
        "computer_training",
        "engineering_structure",
        "engineering_process",
        "engineering_performance",
        "society_theory",
        "society_identification",
        "society_context",
    }
    return clean if clean in allowed else ""


def summarize_innovation_type(innovation_type: str, comparisons: list[dict]) -> dict:
    dimension_map = {
        "technical_route": ["method"],
        "problem_framing": ["target_problem"],
        "data_or_sample": ["data_or_population"],
        "application_context": ["application_context"],
        "evaluation_design": ["evaluation"],
        "theory_framework": ["target_problem", "method"],
        "system_engineering": ["method", "application_context", "combination"],
        "clinical_pathway": ["data_or_population", "application_context", "evaluation"],
        "material_or_process": ["method", "evaluation"],
        "combination": ["combination"],
    }
    return summarize_profile_dimensions(comparisons, dimension_map.get(innovation_type, []), label=innovation_type)


def build_domain_focus(domain: str, comparisons: list[dict]) -> list[dict]:
    focus_map = {
        "biomedical": [
            ("biomedical_population", ["data_or_population"]),
            ("biomedical_intervention", ["method", "application_context"]),
            ("biomedical_outcome", ["evaluation"]),
        ],
        "computer": [
            ("computer_model", ["method"]),
            ("computer_benchmark", ["data_or_population", "evaluation"]),
            ("computer_training", ["method", "combination"]),
        ],
        "engineering": [
            ("engineering_structure", ["method"]),
            ("engineering_process", ["method", "application_context"]),
            ("engineering_performance", ["evaluation"]),
        ],
        "society": [
            ("society_theory", ["target_problem", "method"]),
            ("society_identification", ["evaluation"]),
            ("society_context", ["application_context", "data_or_population"]),
        ],
        "general": [
            ("technical_route", ["method"]),
            ("application_context", ["application_context"]),
            ("evaluation_design", ["evaluation"]),
        ],
    }
    return [
        {"key": key, **summarize_profile_dimensions(comparisons, dimensions, label=key)}
        for key, dimensions in focus_map.get(domain, focus_map["general"])
    ]


def summarize_profile_dimensions(comparisons: list[dict], dimension_keys: list[str], *, label: str) -> dict:
    supporting: list[int] = []
    strongest = "unknown"
    saw_known = False
    for item in comparisons:
        dims = item.get("dimension_overlap") if isinstance(item.get("dimension_overlap"), dict) else {}
        values = [
            combination_dimension_value(dims) if key == "combination" else dims.get(key)
            for key in dimension_keys
        ]
        for value in values:
            if value in {"same", "similar", "partial", "different"}:
                saw_known = True
            if value in {"same", "similar", "partial"}:
                supporting.append(int(item.get("reference_index") or 0))
                strongest = stronger_dimension_value(strongest, value)
    risk = profile_risk_from_value(strongest, saw_known=saw_known, has_comparisons=bool(comparisons))
    return {
        "risk": risk,
        "assessment": profile_assessment(label, risk, supporting),
        "supporting_references": list(dict.fromkeys(supporting))[:6],
    }


def profile_risk_from_value(value: str, *, saw_known: bool, has_comparisons: bool) -> str:
    if value == "same":
        return "high"
    if value in {"similar", "partial"}:
        return "moderate"
    if saw_known and has_comparisons:
        return "low"
    return "unknown"


def profile_assessment(label: str, risk: str, supporting: list[int]) -> str:
    if risk == "high":
        return "This innovation angle appears strongly represented in the current candidate set; verify full texts before claiming it as the core novelty."
    if risk == "moderate":
        return "This innovation angle has partial overlap in the current candidate set; sharpen the exact difference and supporting evidence."
    if risk == "low":
        return "No strong overlap for this innovation angle is visible in the screened metadata, but this is not proof of novelty."
    if supporting:
        return "Related candidates exist, but the available metadata is not enough to judge this innovation angle confidently."
    return "Evidence is insufficient for this innovation angle in the current search results."


def normalized_verification_status(value) -> str:
    status = normalize_space(value).casefold()
    return status if status in {"verified", "needs_review", "unverified", "partial"} else "partial"


def verification_note(reference: dict) -> str:
    status = normalized_verification_status(reference.get("verification_status"))
    if status == "verified":
        return "Stable identifier metadata was verified where available."
    if status == "needs_review":
        return "metadata 存在冲突，需要人工确认。"
    if status == "unverified":
        return "该条文献未通过稳定标识校验，仅作为线索。"
    return "该条文献仅有部分身份线索，只能作为辅助线索。"


def evidence_with_verification_note(evidence: str, reference: dict) -> str:
    status = normalized_verification_status(reference.get("verification_status"))
    if status == "verified":
        return evidence
    return normalize_space(f"{evidence} {verification_note(reference)}")[:900]


def recommendation_with_verification_note(recommendation: str, reference: dict) -> str:
    status = normalized_verification_status(reference.get("verification_status"))
    note = verification_note(reference)
    if status == "verified" or note in recommendation:
        return recommendation
    return normalize_space(f"{recommendation} {note}")[:700]


def build_closest_prior_work(comparisons: list[dict], *, plan: dict | None = None, limit: int = 10) -> list[dict]:
    claim_ids = [
        claim.get("claim_id")
        for claim in (plan or {}).get("claims", [])
        if isinstance(claim, dict) and claim.get("claim_id")
    ]
    rows = []
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        dims = item.get("dimension_overlap") if isinstance(item.get("dimension_overlap"), dict) else {}
        overlap_dimensions = [name for name, value in dims.items() if value in {"same", "similar", "partial"}]
        matched_claim_ids = item.get("matched_claim_ids") if isinstance(item.get("matched_claim_ids"), list) else []
        if not matched_claim_ids and claim_ids and item.get("overlap_level") in {"high_overlap", "partial_overlap"}:
            matched_claim_ids = claim_ids[:2]
        rows.append(
            {
                "reference_index": item.get("reference_index", 0),
                "title": item.get("title", ""),
                "year": item.get("year", ""),
                "source": item.get("source", ""),
                "verification_status": normalized_verification_status(item.get("verification_status")),
                "overlap_level": normalize_overlap_level(item.get("overlap_level")),
                "overlap_score": round(bounded_float(item.get("overlap_score"), default=0.0), 3),
                "matched_claim_ids": matched_claim_ids,
                "overlap_dimensions": overlap_dimensions,
                "key_overlap": first_text(item.get("overlap_points")) or normalize_space(item.get("evidence"))[:360],
                "key_delta": first_text(item.get("difference_points")) or normalize_space(item.get("recommendation"))[:360],
                "risk": risk_for_comparison(item),
            }
        )
    rows.sort(key=closest_prior_sort_key, reverse=True)
    return rows[: max(1, min(10, limit))]


def closest_prior_sort_key(item: dict) -> tuple[float, int, int]:
    level_order = {"high_overlap": 4, "partial_overlap": 3, "adjacent": 2, "no_clear_overlap": 1}
    verification_order = {"verified": 4, "needs_review": 3, "partial": 2, "unverified": 1}
    return (
        float(item.get("overlap_score") or 0),
        level_order.get(item.get("overlap_level"), 0),
        verification_order.get(item.get("verification_status"), 0),
    )


def risk_for_comparison(item: dict) -> str:
    status = normalized_verification_status(item.get("verification_status"))
    level = normalize_overlap_level(item.get("overlap_level"))
    if status in {"unverified", "partial"} and level in {"high_overlap", "partial_overlap"}:
        return "unknown"
    return {
        "high_overlap": "high",
        "partial_overlap": "moderate",
        "adjacent": "low",
        "no_clear_overlap": "low",
    }.get(level, "unknown")


def first_text(value) -> str:
    if isinstance(value, list):
        for item in value:
            text = normalize_space(item)
            if text:
                return text[:360]
    return normalize_space(value)[:360]


def build_novelty_dimensions(comparisons: list[dict]) -> dict:
    dimensions = {
        "method_novelty": ("method", "方法新颖性"),
        "application_novelty": ("application_context", "应用场景新颖性"),
        "dataset_or_scenario_novelty": ("data_or_population", "数据或场景新颖性"),
        "evaluation_novelty": ("evaluation", "评价设计新颖性"),
        "combination_novelty": ("combination", "组合新颖性"),
    }
    return {
        output_key: summarize_dimension_risk(comparisons, dimension_key, label)
        for output_key, (dimension_key, label) in dimensions.items()
    }


def summarize_dimension_risk(comparisons: list[dict], dimension_key: str, label: str) -> dict:
    supporting: list[int] = []
    strongest = "unknown"
    for item in comparisons:
        dims = item.get("dimension_overlap") if isinstance(item.get("dimension_overlap"), dict) else {}
        value = combination_dimension_value(dims) if dimension_key == "combination" else dims.get(dimension_key)
        if value in {"same", "similar", "partial"}:
            supporting.append(int(item.get("reference_index") or 0))
            strongest = stronger_dimension_value(strongest, value)
    risk = risk_from_dimension_value(strongest)
    return {
        "risk": risk,
        "assessment": dimension_assessment(label, risk, supporting),
        "supporting_references": supporting[:6],
    }


def combination_dimension_value(dims: dict) -> str:
    method = dims.get("method")
    context = dims.get("application_context")
    data = dims.get("data_or_population")
    if method == "same" and (context in {"same", "similar"} or data in {"same", "similar"}):
        return "same"
    if method in {"same", "partial"} and (context in {"same", "similar"} or data in {"same", "similar"}):
        return "partial"
    return "unknown"


def stronger_dimension_value(left: str, right: str) -> str:
    order = {"unknown": 0, "different": 0, "partial": 1, "similar": 2, "same": 3}
    return right if order.get(right, 0) > order.get(left, 0) else left


def risk_from_dimension_value(value: str) -> str:
    if value == "same":
        return "high"
    if value in {"similar", "partial"}:
        return "moderate"
    return "unknown"


def dimension_assessment(label: str, risk: str, supporting: list[int]) -> str:
    if risk == "high":
        return f"{label}存在较高重合风险；已在本轮候选中发现相同或高度相似维度，需要人工核验全文。"
    if risk == "moderate":
        return f"{label}存在中等重合风险；候选文献显示部分相似，但差异是否实质仍需核验。"
    if not supporting:
        return f"{label}证据不足；仅可表述为本轮检索候选中未发现该维度的高重合证据。"
    return f"{label}证据不足；相关候选只能作为线索，不能据此断言原创性。"


def public_references(references: list[dict]) -> list[dict]:
    return [{key: value for key, value in reference.items() if key != "abstract" or value} for reference in references]


def compact_search_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    return {
        "query": payload.get("query", ""),
        "search_mode": payload.get("search_mode", ""),
        "requested_search_mode": payload.get("requested_search_mode", ""),
        "query_rewrite_status": payload.get("query_rewrite_status", ""),
        "queries_by_source": payload.get("queries_by_source", {}),
        "sources_used": payload.get("sources_used", []),
        "source_results": payload.get("source_results", {}),
        "errors": payload.get("errors", {}),
        "raw_count": payload.get("raw_count", 0),
        "diagnostics": payload.get("diagnostics", {}),
        "plan": payload.get("plan", {}),
    }


def parse_json_object(content: str) -> dict:
    cleaned = str(content or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned.startswith("{"):
        embedded = extract_first_json_object(cleaned)
        if embedded:
            cleaned = embedded
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object.")
    return data


def extract_first_json_object(text: str) -> str:
    start = str(text or "").find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return ""


def lexical_overlap_score(query: str, document: str) -> tuple[float, list[str]]:
    query_tokens = content_tokens(query)
    document_tokens = content_tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0, []
    query_counts = Counter(query_tokens)
    document_counts = Counter(document_tokens)
    overlap = set(query_counts) & set(document_counts)
    weighted_overlap = sum(min(query_counts[token], document_counts[token]) for token in overlap)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    doc_norm = math.sqrt(sum(value * value for value in document_counts.values()))
    cosine = weighted_overlap / max(1e-9, query_norm * doc_norm)
    coverage = len(overlap) / max(1, len(set(query_tokens)))
    score = min(1.0, cosine * 0.65 + coverage * 0.35)
    hits = sorted(overlap, key=lambda token: (-query_counts[token], token))
    return score, hits


def content_tokens(text: str) -> list[str]:
    normalized = normalize_space(text).casefold()
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9][a-z0-9+.-]{2,}", normalized)
    return [token for token in tokens if token not in STOPWORDS]


STOPWORDS = {
    "and", "or", "the", "for", "with", "using", "based", "study", "paper", "novel",
    "method", "methods", "model", "models", "analysis", "research", "result", "results",
    "proposed", "approach", "system", "data", "dataset", "application", "一种", "方法",
    "研究", "提出", "基于", "创新", "论文", "模型", "系统", "数据", "结果",
}


def extract_claims(text: str) -> list[str]:
    parts = re.split(r"[\n;；。.!?？]+", normalize_space(text))
    claims = [part.strip(" -0123456789.") for part in parts if len(part.strip()) >= 8]
    return list(dict.fromkeys(claims))[:6] or [normalize_space(text)[:240]]


def extract_claims(text: str) -> list[str]:
    text = sanitize_innovation_text(text) or text
    noise_pattern = "|".join(
        [
            "\u9a8c\u6536\u91cd\u70b9",
            "\u6d4b\u8bd5\u8981\u6c42",
            "\u62a5\u544a\u5e94\u63d0\u793a",
            "\u5efa\u8bae\u6765\u6e90",
            "\u68c0\u7d22\u6a21\u5f0f",
        ]
    )
    parts = re.split(r"[\n;；。?!?]+", normalize_space(text))
    parts = [part for part in parts if not re.search(noise_pattern, part, flags=re.IGNORECASE)]
    claims = [part.strip(" -0123456789.") for part in parts if len(part.strip()) >= 8]
    return list(dict.fromkeys(claims))[:6] or [normalize_space(text)[:240]]


def reference_text(reference: dict) -> str:
    return " ".join(
        str(reference.get(key) or "")
        for key in ("title", "abstract", "journal", "source_label")
    )


def evidence_excerpt(reference: dict, hits: list[str]) -> str:
    text = normalize_space(reference_text(reference))
    if not text:
        return ""
    lower = text.casefold()
    positions = [lower.find(hit.casefold()) for hit in hits if hit and lower.find(hit.casefold()) >= 0]
    start = max(0, min(positions) - 120) if positions else 0
    excerpt = text[start : start + 360].strip()
    return excerpt


def dimension_overlap_from_hits(innovation_text: str, reference: dict, hits: list[str]) -> dict:
    matched = set(hits or [])
    reference_lower = reference_text(reference).casefold()
    query_lower = normalize_space(innovation_text).casefold()
    problem_terms = {"stroke", "ischemic", "cerebral", "legal", "question", "segmentation", "classification", "detection"}
    data_terms = {"ct", "ncct", "computed", "tomography", "patient", "dataset", "hospital", "law", "case"}
    method_terms = {"lightweight", "encoder", "domain", "consistency", "rag", "retrieval", "llm", "deep", "learning", "network"}
    context_terms = {"low-resource", "few-shot", "small", "cross-domain", "clinical", "legal"}
    evaluation_terms = {"baseline", "nnunet", "unet", "benchmark", "evaluation", "dice", "auc", "f1"}
    return {
        "target_problem": dimension_value(matched, reference_lower, query_lower, problem_terms, same_word="same", partial_word="similar"),
        "data_or_population": dimension_value(matched, reference_lower, query_lower, data_terms, same_word="same", partial_word="similar"),
        "method": dimension_value(matched, reference_lower, query_lower, method_terms, same_word="same", partial_word="partial"),
        "application_context": dimension_value(matched, reference_lower, query_lower, context_terms, same_word="same", partial_word="similar"),
        "evaluation": dimension_value(matched, reference_lower, query_lower, evaluation_terms, same_word="same", partial_word="partial"),
    }


def dimension_value(
    matched: set[str],
    reference_lower: str,
    query_lower: str,
    terms: set[str],
    *,
    same_word: str,
    partial_word: str,
) -> str:
    query_terms = {term for term in terms if term in query_lower}
    if not query_terms:
        return "unknown"
    hit_terms = {term for term in query_terms if term in reference_lower or term in matched}
    if len(hit_terms) >= max(1, len(query_terms)):
        return same_word
    if hit_terms:
        return partial_word
    return "different"


def normalize_dimension_overlap(value, reference: dict) -> dict:
    if not isinstance(value, dict):
        concepts = reference.get("matched_concepts") if isinstance(reference.get("matched_concepts"), dict) else {}
        required = concepts.get("required") if isinstance(concepts.get("required"), list) else []
        method = concepts.get("method") if isinstance(concepts.get("method"), list) else []
        context = concepts.get("context") if isinstance(concepts.get("context"), list) else []
        baseline = concepts.get("baseline") if isinstance(concepts.get("baseline"), list) else []
        return {
            "target_problem": "same" if len(required) >= 2 else ("similar" if required else "unknown"),
            "data_or_population": "similar" if required else "unknown",
            "method": "partial" if method else "unknown",
            "application_context": "similar" if context else "unknown",
            "evaluation": "partial" if baseline else "unknown",
        }
    return {
        "target_problem": normalize_dimension_value(value.get("target_problem"), similar_values={"similar"}),
        "data_or_population": normalize_dimension_value(value.get("data_or_population"), similar_values={"similar"}),
        "method": normalize_dimension_value(value.get("method"), similar_values={"partial"}),
        "application_context": normalize_dimension_value(value.get("application_context"), similar_values={"similar"}),
        "evaluation": normalize_dimension_value(value.get("evaluation"), similar_values={"partial"}),
    }


def normalize_dimension_value(value, *, similar_values: set[str]) -> str:
    clean = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    allowed = {"same", "different", "unknown", *similar_values}
    return clean if clean in allowed else "unknown"


def aggregate_risk_level(comparisons: list[dict]) -> str:
    if not comparisons:
        return "unknown"
    has_dimension_evidence = any(
        any(value != "unknown" for value in (item.get("dimension_overlap") or {}).values())
        for item in comparisons
        if isinstance(item.get("dimension_overlap"), dict)
    )
    if not has_dimension_evidence:
        return ""
    for item in comparisons:
        dims = item.get("dimension_overlap") if isinstance(item.get("dimension_overlap"), dict) else {}
        if dims.get("target_problem") == "same" and dims.get("data_or_population") in {"same", "similar"} and dims.get("method") == "same":
            return "high"
    for item in comparisons:
        dims = item.get("dimension_overlap") if isinstance(item.get("dimension_overlap"), dict) else {}
        if dims.get("target_problem") in {"same", "similar"} and (
            dims.get("method") in {"same", "partial"} or dims.get("application_context") in {"same", "similar"}
        ):
            return "moderate"
    if comparisons:
        return "low"
    return "unknown"


def safe_assessment(value: str) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    forbidden = [
        "proof of novelty",
        "proven novel",
        "completely original",
        "no overlap anywhere",
        "\u8bc1\u660e\u521b\u65b0",
        "\u5b8c\u5168\u539f\u521b",
        "\u6ca1\u6709\u4efb\u4f55\u91cd\u5408",
    ]
    if any(item.casefold() in text.casefold() for item in forbidden):
        return ""
    return text


def recommendation_for_level(level: str) -> str:
    return {
        "high_overlap": "Treat this as a priority manual check; compare full-text method, experiments, and claimed contribution.",
        "partial_overlap": "Clarify what is different from this work and verify whether the difference is substantive.",
        "adjacent": "Use this as related work context and check whether it cites closer papers.",
        "no_clear_overlap": "No clear overlap appears in metadata, but full-text and database coverage still matter.",
    }.get(level, "Verify manually before making a final novelty claim.")


def assessment_for_risk(risk: str, count: int) -> str:
    if count <= 0:
        return "No candidate literature was available, so novelty cannot be assessed from the current search."
    return {
        "high": "The current search found literature with strong apparent overlap. The innovation claim needs careful narrowing or full-text verification.",
        "moderate": "The current search found partial or adjacent overlap. The idea may still be defensible, but the claimed novelty should be sharpened.",
        "low": "The current search did not show obvious overlap in available metadata. This is not proof of novelty; expand sources and verify full texts.",
        "unknown": "The current evidence is insufficient for a confident novelty-risk judgment.",
    }.get(risk, "The current evidence is insufficient for a confident novelty-risk judgment.")


def overall_risk_level(comparisons: list[dict]) -> str:
    levels = [normalize_overlap_level(item.get("overlap_level")) for item in comparisons if isinstance(item, dict)]
    if "high_overlap" in levels:
        return "high"
    if "partial_overlap" in levels:
        return "moderate"
    if "adjacent" in levels:
        return "low"
    if levels:
        return "low"
    return "unknown"


def novelty_counts(comparisons: list[dict]) -> dict:
    counts = {"high_overlap": 0, "partial_overlap": 0, "adjacent": 0, "no_clear_overlap": 0}
    for item in comparisons:
        level = normalize_overlap_level(item.get("overlap_level"))
        counts[level] = counts.get(level, 0) + 1
    counts["total"] = len(comparisons)
    return counts


def overlap_sort_key(item: dict) -> tuple[float, int]:
    order = {
        "high_overlap": 4,
        "partial_overlap": 3,
        "adjacent": 2,
        "no_clear_overlap": 1,
    }
    return float(item.get("overlap_score") or 0), order.get(item.get("overlap_level"), 0)


def normalize_overlap_level(value) -> str:
    level = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "high": "high_overlap",
        "overlap": "partial_overlap",
        "partial": "partial_overlap",
        "related": "adjacent",
        "none": "no_clear_overlap",
        "low": "no_clear_overlap",
    }
    level = aliases.get(level, level)
    return level if level in {"high_overlap", "partial_overlap", "adjacent", "no_clear_overlap"} else "adjacent"


def normalize_risk_level(value) -> str:
    risk = str(value or "").strip().casefold()
    return risk if risk in {"high", "moderate", "low", "unknown"} else "unknown"


def score_from_level(level: str) -> float:
    return {
        "high_overlap": 0.82,
        "partial_overlap": 0.55,
        "adjacent": 0.3,
        "no_clear_overlap": 0.08,
    }.get(level, 0.3)


def bounded_float(value, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def clean_string_list(value, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = normalize_space(item)
        if text and text not in cleaned:
            cleaned.append(text[:500])
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_space(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
