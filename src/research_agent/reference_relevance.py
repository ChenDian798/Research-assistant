from __future__ import annotations

import asyncio
import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache

from .llm import LLMClient


# Broad medical/search concepts. Chinese terms are written as unicode escapes so
# this file remains stable across Windows terminals with different encodings.
CONCEPTS = [
    {
        "id": "modality_ct",
        "category": "modality",
        "terms": [
            "ct",
            "ncct",
            "non contrast ct",
            "non-contrast ct",
            "computed tomography",
            "\u5934\u9885ct",
            "\u8111ct",
            "\u80f8\u90e8ct",
            "\u8179\u90e8ct",
        ],
        "expansions": ["computed tomography", "CT"],
    },
    {
        "id": "modality_mri",
        "category": "modality",
        "terms": ["mri", "magnetic resonance", "\u6838\u78c1", "\u78c1\u5171\u632f"],
        "expansions": ["MRI", "magnetic resonance imaging"],
    },
    {
        "id": "modality_ultrasound",
        "category": "modality",
        "terms": ["ultrasound", "sonography", "\u8d85\u58f0"],
        "expansions": ["ultrasound", "sonography"],
    },
    {
        "id": "anatomy_brain",
        "category": "anatomy",
        "terms": [
            "brain",
            "cerebral",
            "cranial",
            "intracranial",
            "\u5934\u9885",
            "\u9885\u8111",
            "\u8111\u90e8",
            "\u8111",
        ],
        "expansions": ["brain", "cerebral"],
    },
    {
        "id": "anatomy_lung",
        "category": "anatomy",
        "terms": ["lung", "pulmonary", "chest", "thoracic", "\u80ba", "\u80f8\u90e8"],
        "expansions": ["lung", "pulmonary"],
    },
    {
        "id": "anatomy_heart",
        "category": "anatomy",
        "terms": ["heart", "cardiac", "coronary", "\u5fc3\u810f", "\u5fc3\u8840\u7ba1", "\u51a0\u8109"],
        "expansions": ["cardiac", "heart"],
    },
    {
        "id": "anatomy_liver",
        "category": "anatomy",
        "terms": ["liver", "hepatic", "\u809d", "\u809d\u810f"],
        "expansions": ["liver", "hepatic"],
    },
    {
        "id": "condition_stroke",
        "category": "condition",
        "terms": [
            "stroke",
            "ischemic stroke",
            "ischaemic stroke",
            "cerebral infarction",
            "brain infarct",
            "infarction",
            "\u8111\u6897",
            "\u8111\u6897\u6b7b",
            "\u6897\u6b7b",
            "\u5352\u4e2d",
            "\u7f3a\u8840\u6027\u5352\u4e2d",
        ],
        "expansions": ["ischemic stroke", "cerebral infarction", "stroke"],
    },
    {
        "id": "condition_hemorrhage",
        "category": "condition",
        "terms": ["hemorrhage", "haemorrhage", "bleeding", "\u51fa\u8840"],
        "expansions": ["hemorrhage", "bleeding"],
    },
    {
        "id": "condition_tumor",
        "category": "condition",
        "terms": ["tumor", "tumour", "cancer", "carcinoma", "neoplasm", "\u80bf\u7624", "\u764c"],
        "expansions": ["tumor", "cancer", "neoplasm"],
    },
    {
        "id": "condition_nodule",
        "category": "condition",
        "terms": ["nodule", "nodules", "\u7ed3\u8282"],
        "expansions": ["nodule"],
    },
    {
        "id": "condition_lesion",
        "category": "target",
        "terms": ["lesion", "lesions", "abnormality", "focus", "\u75c5\u7076"],
        "expansions": ["lesion"],
    },
    {
        "id": "task_segmentation",
        "category": "task",
        "terms": [
            "segmentation",
            "segment",
            "segmented",
            "delineation",
            "mask",
            "\u5206\u5272",
            "\u52fe\u753b",
        ],
        "expansions": ["segmentation", "delineation"],
    },
    {
        "id": "task_detection",
        "category": "task",
        "terms": ["detection", "detect", "localization", "localisation", "\u68c0\u6d4b", "\u5b9a\u4f4d"],
        "expansions": ["detection", "localization"],
    },
    {
        "id": "task_classification",
        "category": "task",
        "terms": ["classification", "classify", "diagnosis", "\u5206\u7c7b", "\u8bca\u65ad"],
        "expansions": ["classification", "diagnosis"],
    },
    {
        "id": "method_algorithm",
        "category": "method",
        "terms": [
            "algorithm",
            "model",
            "deep learning",
            "machine learning",
            "cnn",
            "u-net",
            "unet",
            "transformer",
            "foundation model",
            "\u7b97\u6cd5",
            "\u6a21\u578b",
            "\u6df1\u5ea6\u5b66\u4e60",
            "\u673a\u5668\u5b66\u4e60",
        ],
        "expansions": ["algorithm", "deep learning", "machine learning"],
    },
]

MEDICAL_BACKGROUND_TERMS = [
    "medical",
    "clinical",
    "patient",
    "patients",
    "radiology",
    "imaging",
    "\u533b\u5b66",
    "\u4e34\u5e8a",
    "\u5f71\u50cf",
]

GENERIC_NEGATIVE_TERMS = [
    "veterinary",
    "vet anatomy",
    "vet-anatomy",
    "canine",
    "dog",
    "\u72d7",
    "\u72ac",
]


LLM_RELEVANCE_SYSTEM_PROMPT = """
You are a strict but conservative biomedical literature relevance reviewer.
Your job is to decide whether each candidate is about the user's exact research topic.
Return only valid JSON, no markdown.
Schema:
{
  "decisions": [
    {
      "candidate_index": 0,
      "topic_status": "relevant|borderline|off_topic",
      "confidence": 0.0,
      "matched_concepts": ["short concept"],
      "missing_concepts": ["short concept"],
      "reason": "brief reason"
    }
  ]
}
Rules:
- Preserve the user's exact biomedical intent, including disease/condition, anatomy, modality, intervention, population, outcome, and task when specified.
- Mark relevant only when the title/abstract/metadata clearly matches the topic.
- Mark borderline when the candidate is plausibly useful but incomplete, broad, adjacent, or missing important details.
- Mark off_topic only when it is clearly unrelated or clearly about a different condition/task/modality/population.
- Be conservative: uncertain cases should be borderline, not off_topic.
- Do not require exact words if standard synonyms or translations match the same biomedical concept.
- If query_plan.query_intent is title, author+title, citation, or citation_with_title_author, judge primarily whether the candidate is the same bibliographic work, not ordinary topical relevance. Use extracted.title, extracted.authors, extracted.year, and retrieval_channel before topic concepts.
- For title/author+title/citation identity searches, minor title typos, capitalization, punctuation, or word-form differences should remain relevant or borderline; do not reject a near-title match only because topic concepts are missing from the abstract.
""".strip()

SOCIETY_LLM_RELEVANCE_SYSTEM_PROMPT = """
You are a strict but conservative social-science literature relevance reviewer.
Your job is to decide whether each candidate is about the user's exact research topic.
Return only valid JSON, no markdown.
Schema:
{
  "decisions": [
    {
      "candidate_index": 0,
      "topic_status": "relevant|borderline|off_topic",
      "confidence": 0.0,
      "matched_concepts": ["short concept"],
      "missing_concepts": ["short concept"],
      "reason": "brief reason"
    }
  ]
}
Rules:
- Preserve the user's exact social-science intent, including population, institution, policy, behavior, theory, method, geography, period, and outcome when specified.
- Mark relevant only when the title/abstract/metadata clearly matches the topic.
- Mark borderline when the candidate is plausibly useful but incomplete, broad, adjacent, or missing important details.
- Mark off_topic only when it is clearly unrelated or clearly about a different population, institution, policy, behavior, method, geography, period, or outcome.
- Be conservative: uncertain cases should be borderline, not off_topic.
- Do not require exact words if standard synonyms, translations, or closely equivalent social-science constructs match the same concept.
- If query_plan.query_intent is title, author+title, citation, or citation_with_title_author, judge primarily whether the candidate is the same bibliographic work, not ordinary topical relevance. Use extracted.title, extracted.authors, extracted.year, and retrieval_channel before topic concepts.
- For title/author+title/citation identity searches, minor title typos, capitalization, punctuation, or word-form differences should remain relevant or borderline; do not reject a near-title match only because topic concepts are missing from the abstract.
""".strip()
COMPUTER_LLM_RELEVANCE_SYSTEM_PROMPT = """
You are a strict but conservative computer-science and AI literature relevance reviewer.
Your job is to decide whether each candidate is about the user's exact research topic.
Return only valid JSON, no markdown.
Schema:
{
  "decisions": [
    {
      "candidate_index": 0,
      "topic_status": "relevant|borderline|off_topic",
      "confidence": 0.0,
      "matched_concepts": ["short concept"],
      "missing_concepts": ["short concept"],
      "reason": "brief reason"
    }
  ]
}
Rules:
- Preserve the user's exact computer-science or AI intent, including task, method, model family, dataset, benchmark, metric, system, application domain, and constraints when specified.
- Mark relevant only when the title/abstract/metadata clearly matches the topic.
- Mark borderline when the candidate is plausibly useful but incomplete, broad, adjacent, or missing important technical details.
- Mark off_topic only when it is clearly unrelated or clearly about a different task, method, model, system, dataset, benchmark, metric, or application domain.
- Be conservative: uncertain cases should be borderline, not off_topic.
- Do not require exact words if standard technical synonyms, abbreviations, translations, or closely equivalent CS/AI constructs match the same concept.
- If query_plan.query_intent is title, author+title, citation, or citation_with_title_author, judge primarily whether the candidate is the same bibliographic work, not ordinary topical relevance. Use extracted.title, extracted.authors, extracted.year, and retrieval_channel before topic concepts.
- For title/author+title/citation identity searches, minor title typos, capitalization, punctuation, or word-form differences should remain relevant or borderline; do not reject a near-title match only because topic concepts are missing from the abstract.
""".strip()
ENGINEERING_LLM_RELEVANCE_SYSTEM_PROMPT = """
You are a strict but conservative engineering literature relevance reviewer.
Your job is to decide whether each candidate is about the user's exact research topic.
Return only valid JSON, no markdown.
Schema:
{
  "decisions": [
    {
      "candidate_index": 0,
      "topic_status": "relevant|borderline|off_topic",
      "confidence": 0.0,
      "matched_concepts": ["short concept"],
      "missing_concepts": ["short concept"],
      "reason": "brief reason"
    }
  ]
}
Rules:
- Preserve the user's exact engineering intent, including system, material, process, design method, control strategy, optimization target, performance metric, application sector, operating condition, and constraints when specified.
- Mark relevant only when the title/abstract/metadata clearly matches the topic.
- Mark borderline when the candidate is plausibly useful but incomplete, broad, adjacent, or missing important engineering details.
- Mark off_topic only when it is clearly unrelated or clearly about a different system, material, process, method, design, control target, performance metric, sector, or condition.
- Be conservative: uncertain cases should be borderline, not off_topic.
- Do not require exact words if standard engineering synonyms, abbreviations, translations, or closely equivalent applied-technology terms match the same concept.
- If query_plan.query_intent is title, author+title, citation, or citation_with_title_author, judge primarily whether the candidate is the same bibliographic work, not ordinary topical relevance. Use extracted.title, extracted.authors, extracted.year, and retrieval_channel before topic concepts.
- For title/author+title/citation identity searches, minor title typos, capitalization, punctuation, or word-form differences should remain relevant or borderline; do not reject a near-title match only because topic concepts are missing from the abstract.
""".strip()


def apply_relevance_gate(query: str, screened: dict, *, query_plan: dict | None = None) -> dict:
    mode = relevance_gate_mode()
    if mode != "rules":
        llm_gated = apply_llm_relevance_gate(query, screened, query_plan=query_plan)
        if llm_gated is not None:
            return llm_gated
        if mode == "llm":
            return apply_rules_relevance_gate(query, screened, query_plan=query_plan)
    return apply_rules_relevance_gate(query, screened, query_plan=query_plan)


def apply_rules_relevance_gate(query: str, screened: dict, *, query_plan: dict | None = None) -> dict:
    active_mode = search_mode_from_query_plan(query_plan)
    if active_mode == "society":
        return apply_society_rules_relevance_gate(query, screened, query_plan=query_plan)
    if active_mode in {"computer", "engineering"}:
        return apply_technical_rules_relevance_gate(query, screened, query_plan=query_plan, domain=active_mode)
    profile = relevance_query_profile(query, query_plan=query_plan)
    if not profile["concepts"] and not profile["keywords"]:
        return screened
    gated = {"qualified": [], "needs_review": [], "rejected": list(screened.get("rejected", []))}
    for bucket in ("qualified", "needs_review"):
        for reference in screened.get(bucket, []):
            title_match_status = title_intent_relevance_status(query, reference, query_plan=query_plan)
            if title_match_status == "strong":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(mark_title_intent_relevant(reference))
                continue
            if title_match_status == "borderline":
                gated["needs_review"].append(mark_title_intent_borderline(reference))
                continue
            assessed = assess_reference_relevance(query, reference, profile=profile)
            status = assessed.get("topic_relevance_status")
            if status == "relevant":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(assessed)
            elif status == "borderline":
                assessed["screening_status"] = "needs_review"
                assessed.setdefault("screening_risks", []).append("topic_relevance_borderline")
                gated["needs_review"].append(assessed)
            else:
                assessed["screening_status"] = "rejected"
                assessed.setdefault("screening_reasons", []).append("topic_relevance_failed")
                assessed.setdefault("screening_risks", []).append("off_topic_reference")
                gated["rejected"].append(assessed)
    return gated


def apply_society_rules_relevance_gate(query: str, screened: dict, *, query_plan: dict | None = None) -> dict:
    profile = society_query_profile(query, query_plan=query_plan)
    if not profile["concepts"] and not profile["keywords"]:
        return screened
    gated = {"qualified": [], "needs_review": [], "rejected": list(screened.get("rejected", []))}
    for bucket in ("qualified", "needs_review"):
        for reference in screened.get(bucket, []):
            title_match_status = title_intent_relevance_status(query, reference, query_plan=query_plan)
            if title_match_status == "strong":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(mark_title_intent_relevant(reference))
                continue
            if title_match_status == "borderline":
                gated["needs_review"].append(mark_title_intent_borderline(reference))
                continue
            assessed = assess_society_reference_relevance(reference, profile=profile)
            status = assessed.get("topic_relevance_status")
            if status == "relevant":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(assessed)
            elif status == "borderline":
                assessed["screening_status"] = "needs_review"
                assessed.setdefault("screening_risks", []).append("topic_relevance_borderline")
                gated["needs_review"].append(assessed)
            else:
                assessed["screening_status"] = "rejected"
                assessed.setdefault("screening_reasons", []).append("topic_relevance_failed")
                assessed.setdefault("screening_risks", []).append("off_topic_reference")
                gated["rejected"].append(assessed)
    return gated


def assess_society_reference_relevance(reference: dict, *, profile: dict) -> dict:
    item = dict(reference or {})
    text = reference_text(item)
    matched_concepts = [
        concept
        for concept in profile["concepts"]
        if any(normalized_contains(text, term) for term in concept["terms"])
    ]
    matched_ids = {concept["id"] for concept in matched_concepts}
    missing_required = [
        concept
        for concept in profile["concepts"]
        if concept["id"] not in matched_ids and concept.get("required")
    ]
    keyword_hits = [keyword for keyword in profile["keywords"] if normalized_contains(text, keyword)]
    concept_score = len(matched_concepts) / max(1, len(profile["concepts"])) if profile["concepts"] else 0.0
    keyword_score = len(keyword_hits) / max(1, len(profile["keywords"])) if profile["keywords"] else 0.0
    lexical_score = max(concept_score, (concept_score * 0.8) + (keyword_score * 0.2)) if profile["concepts"] else keyword_score

    reasons = [f"society_concept_match:{concept['label']}" for concept in matched_concepts]
    reasons.extend(f"keyword_match:{keyword}" for keyword in keyword_hits[:8])
    risks: list[str] = []
    status = "relevant"
    if missing_required:
        status = (
            "borderline"
            if missing_required_can_review(
                domain="society",
                missing_required=missing_required,
                matched_concepts=matched_concepts,
                lexical_score=lexical_score,
            )
            else "off_topic"
        )
        risks.append("missing_required_social_concepts:" + ",".join(concept["label"] for concept in missing_required[:4]))
    elif lexical_score < society_relevance_min_lexical_score(profile):
        status = "borderline" if lexical_score >= 0.4 else "off_topic"
        risks.append(f"low_social_topic_overlap:{lexical_score:.3f}")

    item["topic_relevance_status"] = status
    item["topic_relevance_score"] = round(float(lexical_score), 4)
    item["topic_relevance_reasons"] = reasons
    item["topic_relevance_risks"] = risks
    if risks:
        existing = item.get("screening_risks") if isinstance(item.get("screening_risks"), list) else []
        item["screening_risks"] = list(dict.fromkeys([*existing, *risks]))
    return item


def apply_technical_rules_relevance_gate(
    query: str,
    screened: dict,
    *,
    query_plan: dict | None = None,
    domain: str,
) -> dict:
    profile = technical_query_profile(query, query_plan=query_plan, domain=domain)
    if not profile["concepts"] and not profile["keywords"]:
        return screened
    gated = {"qualified": [], "needs_review": [], "rejected": list(screened.get("rejected", []))}
    for bucket in ("qualified", "needs_review"):
        for reference in screened.get(bucket, []):
            title_match_status = title_intent_relevance_status(query, reference, query_plan=query_plan)
            if title_match_status == "strong":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(mark_title_intent_relevant(reference))
                continue
            if title_match_status == "borderline":
                gated["needs_review"].append(mark_title_intent_borderline(reference))
                continue
            assessed = assess_technical_reference_relevance(reference, profile=profile, domain=domain)
            status = assessed.get("topic_relevance_status")
            if status == "relevant":
                gated["qualified" if bucket == "qualified" else "needs_review"].append(assessed)
            elif status == "borderline":
                assessed["screening_status"] = "needs_review"
                assessed.setdefault("screening_risks", []).append("topic_relevance_borderline")
                gated["needs_review"].append(assessed)
            else:
                assessed["screening_status"] = "rejected"
                assessed.setdefault("screening_reasons", []).append("topic_relevance_failed")
                assessed.setdefault("screening_risks", []).append("off_topic_reference")
                gated["rejected"].append(assessed)
    return gated


def assess_technical_reference_relevance(reference: dict, *, profile: dict, domain: str) -> dict:
    item = dict(reference or {})
    text = reference_text(item)
    matched_concepts = [
        concept
        for concept in profile["concepts"]
        if any(normalized_contains(text, term) for term in concept["terms"])
    ]
    matched_ids = {concept["id"] for concept in matched_concepts}
    missing_required = [
        concept
        for concept in profile["concepts"]
        if concept["id"] not in matched_ids and concept.get("required")
    ]
    keyword_hits = [keyword for keyword in profile["keywords"] if normalized_contains(text, keyword)]
    concept_score = len(matched_concepts) / max(1, len(profile["concepts"])) if profile["concepts"] else 0.0
    keyword_score = len(keyword_hits) / max(1, len(profile["keywords"])) if profile["keywords"] else 0.0
    lexical_score = max(concept_score, (concept_score * 0.85) + (keyword_score * 0.15)) if profile["concepts"] else keyword_score

    reasons = [f"{domain}_concept_match:{concept['label']}" for concept in matched_concepts]
    reasons.extend(f"keyword_match:{keyword}" for keyword in keyword_hits[:8])
    risks: list[str] = []
    status = "relevant"
    if missing_required:
        status = (
            "borderline"
            if missing_required_can_review(
                domain=domain,
                missing_required=missing_required,
                matched_concepts=matched_concepts,
                lexical_score=lexical_score,
            )
            else "off_topic"
        )
        risks.append(f"missing_required_{domain}_concepts:" + ",".join(concept["label"] for concept in missing_required[:4]))
    elif lexical_score < technical_relevance_min_lexical_score(profile):
        status = "borderline" if lexical_score >= 0.4 else "off_topic"
        risks.append(f"low_{domain}_topic_overlap:{lexical_score:.3f}")

    item["topic_relevance_status"] = status
    item["topic_relevance_score"] = round(float(lexical_score), 4)
    item["topic_relevance_reasons"] = reasons
    item["topic_relevance_risks"] = risks
    if risks:
        existing = item.get("screening_risks") if isinstance(item.get("screening_risks"), list) else []
        item["screening_risks"] = list(dict.fromkeys([*existing, *risks]))
    return item


def title_intent_relevance_match(query: str, reference: dict, *, query_plan: dict | None = None) -> bool:
    return title_intent_relevance_status(query, reference, query_plan=query_plan) == "strong"


def title_intent_relevance_status(query: str, reference: dict, *, query_plan: dict | None = None) -> str:
    if not title_intent_query(query_plan):
        return ""
    candidate_title = str((reference or {}).get("title") or "")
    if not candidate_title:
        return ""
    title_probe = title_intent_query_text(query, query_plan=query_plan)
    similarity = title_similarity(title_probe, candidate_title)
    author_required = bibliographic_authors_required(query_plan)
    author_score = bibliographic_author_similarity(query_plan, reference)
    if author_required and similarity >= 0.9 and author_score >= 0.45:
        return "strong"
    if author_required and similarity >= 0.82 and author_score >= 0.3:
        return "borderline"
    if author_required and similarity >= 0.75:
        return "borderline"
    retrieval_channel = str((reference or {}).get("retrieval_channel") or "")
    if similarity >= 0.9 or (retrieval_channel == "exact_title" and similarity >= 0.75):
        return "strong"
    if similarity >= 0.68 or (retrieval_channel in {"exact_title", "fuzzy_title"} and similarity >= 0.6):
        return "borderline"
    return ""


def title_intent_query(query_plan: dict | None) -> bool:
    if not isinstance(query_plan, dict):
        return False
    intent = query_plan.get("intent") if isinstance(query_plan.get("intent"), dict) else {}
    scores = intent.get("scores") if isinstance(intent.get("scores"), dict) else {}
    query_intent = str(query_plan.get("query_intent") or "").strip().casefold()
    return (
        query_intent in {"title", "author+title", "citation", "citation_with_title_author"}
        or str(intent.get("top_intent") or "").strip().casefold() == "title"
        or str(intent.get("template") or "").strip().casefold() in {"title", "author+title", "citation_with_title_author"}
        or float(scores.get("title") or 0.0) >= 0.65
    )


def title_intent_query_text(query: str, *, query_plan: dict | None = None) -> str:
    if isinstance(query_plan, dict):
        identity = query_plan.get("bibliographic_identity") if isinstance(query_plan.get("bibliographic_identity"), dict) else {}
        if identity.get("title"):
            return str(identity.get("title") or "")
        extracted = query_plan.get("extracted") if isinstance(query_plan.get("extracted"), dict) else {}
        if extracted.get("title"):
            return str(extracted.get("title") or "")
        intent = query_plan.get("intent") if isinstance(query_plan.get("intent"), dict) else {}
        intent_extracted = intent.get("extracted") if isinstance(intent.get("extracted"), dict) else {}
        if intent_extracted.get("title"):
            return str(intent_extracted.get("title") or "")
    return str(query or "")


def bibliographic_authors_required(query_plan: dict | None) -> bool:
    if not isinstance(query_plan, dict):
        return False
    query_intent = str(query_plan.get("query_intent") or "").strip().casefold()
    if query_intent not in {"author+title", "citation", "citation_with_title_author"}:
        return False
    return bool(bibliographic_query_authors(query_plan))


def bibliographic_query_authors(query_plan: dict | None) -> list[str]:
    if not isinstance(query_plan, dict):
        return []
    identity = query_plan.get("bibliographic_identity") if isinstance(query_plan.get("bibliographic_identity"), dict) else {}
    authors = identity.get("authors")
    if isinstance(authors, list) and authors:
        return clean_string_list(authors, limit=8)
    extracted = query_plan.get("extracted") if isinstance(query_plan.get("extracted"), dict) else {}
    authors = extracted.get("authors")
    if isinstance(authors, list) and authors:
        return clean_string_list(authors, limit=8)
    intent = query_plan.get("intent") if isinstance(query_plan.get("intent"), dict) else {}
    intent_extracted = intent.get("extracted") if isinstance(intent.get("extracted"), dict) else {}
    return clean_string_list(intent_extracted.get("authors"), limit=8)


def bibliographic_author_similarity(query_plan: dict | None, reference: dict) -> float:
    authors = bibliographic_query_authors(query_plan)
    if not authors:
        return 0.0
    reference_authors = normalize_text(str((reference or {}).get("authors") or ""))
    if not reference_authors:
        return 0.0
    best = 0.0
    for author in authors:
        author_norm = normalize_text(author)
        if author_norm and normalized_contains(reference_authors, author_norm):
            best = max(best, 1.0)
            continue
        tokens = {token for token in re.findall(r"[a-z][a-z'.-]+", author_norm) if len(token) > 1}
        ref_tokens = {token for token in re.findall(r"[a-z][a-z'.-]+", reference_authors) if len(token) > 1}
        if tokens and ref_tokens:
            best = max(best, len(tokens & ref_tokens) / max(1, len(tokens)))
    return min(1.0, best)


def mark_title_intent_relevant(reference: dict) -> dict:
    item = dict(reference or {})
    item["topic_relevance_status"] = "relevant"
    item["topic_relevance_score"] = 1.0
    append_unique(item, "topic_relevance_reasons", "exact_title_match")
    return item


def mark_title_intent_borderline(reference: dict) -> dict:
    item = dict(reference or {})
    item["screening_status"] = "needs_review"
    item["topic_relevance_status"] = "borderline"
    item["topic_relevance_score"] = max(float(item.get("topic_relevance_score") or 0.0), 0.6)
    append_unique(item, "screening_risks", "title_intent_needs_review")
    append_unique(item, "topic_relevance_reasons", "near_title_match")
    return item


def title_similarity(query_title: str, candidate_title: str) -> float:
    query_norm = normalize_title(query_title)
    candidate_norm = normalize_title(candidate_title)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        return 0.92
    query_tokens = title_tokens(query_norm)
    candidate_tokens = title_tokens(candidate_norm)
    token_score = 0.0
    if query_tokens and candidate_tokens:
        token_score = len(query_tokens & candidate_tokens) / max(1, max(len(query_tokens), len(candidate_tokens)))
    char_score = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    return max(token_score, char_score)


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", str(value or "").casefold()).strip()


def title_tokens(value: str) -> set[str]:
    stopwords = {"a", "an", "and", "the", "of", "for", "to"}
    return {token for token in re.findall(r"[a-z0-9][a-z0-9+-]*|[\u4e00-\u9fff]+", value.casefold()) if token not in stopwords}


REVIEWABLE_MISSING_REQUIRED_CONFIG = {
    "biomedical": {
        "reviewable_terms": {
            "assessment",
            "classification",
            "control",
            "diagnosis",
            "detection",
            "evaluation",
            "intervention",
            "management",
            "method",
            "model",
            "outcome",
            "prediction",
            "prevention",
            "prognosis",
            "risk",
            "screening",
            "segmentation",
            "therapy",
            "treatment",
        },
        "hard_categories": {"anatomy", "condition", "modality", "population", "target"},
        "min_matched_concepts": 1,
        "min_lexical_score": 0.5,
    },
    "computer": {
        "reviewable_terms": {
            "accuracy",
            "algorithm",
            "assessment",
            "baseline",
            "benchmark",
            "dataset",
            "evaluation",
            "experiment",
            "metric",
            "method",
            "model",
            "performance",
            "ranking",
            "retrieval",
            "test",
        },
        "hard_categories": {"application", "domain", "object", "system", "task"},
        "min_matched_concepts": 2,
        "min_lexical_score": 0.5,
    },
    "engineering": {
        "reviewable_terms": {
            "algorithm",
            "assessment",
            "control",
            "detection",
            "diagnosis",
            "estimation",
            "forecast",
            "forecasting",
            "method",
            "model",
            "modeling",
            "modelling",
            "optimization",
            "optimisation",
            "prediction",
            "simulation",
        },
        "hard_categories": {"application", "condition", "material", "process", "sector", "system"},
        "min_matched_concepts": 2,
        "min_lexical_score": 0.65,
        "allow_single_match_if_score": 0.65,
    },
    "society": {
        "reviewable_terms": {
            "assessment",
            "effect",
            "evaluation",
            "governance",
            "impact",
            "management",
            "mechanism",
            "method",
            "model",
            "outcome",
            "policy",
            "regulation",
            "theory",
        },
        "hard_categories": {"geography", "institution", "phenomenon", "population"},
        "min_matched_concepts": 2,
        "min_lexical_score": 0.5,
    },
}


def missing_required_can_review(
    *,
    domain: str,
    missing_required: list[dict],
    matched_concepts: list[dict],
    lexical_score: float,
) -> bool:
    config = REVIEWABLE_MISSING_REQUIRED_CONFIG.get(domain)
    if not config or not missing_required:
        return False
    min_matched = int(config.get("min_matched_concepts", 2))
    min_score = float(config.get("min_lexical_score", 0.5))
    allow_single_score = config.get("allow_single_match_if_score")
    if len(matched_concepts) < min_matched:
        if allow_single_score is None or len(matched_concepts) < 1 or lexical_score < float(allow_single_score):
            return False
    if lexical_score < min_score:
        return False
    hard_categories = set(config.get("hard_categories") or [])
    reviewable_terms = set(config.get("reviewable_terms") or [])
    for concept in missing_required:
        if str(concept.get("category") or "").casefold() in hard_categories:
            return False
        label = str(concept.get("label") or "").casefold()
        if not any(term in label for term in reviewable_terms):
            return False
    return True


def apply_llm_relevance_gate(query: str, screened: dict, *, query_plan: dict | None = None) -> dict | None:
    if not should_use_llm_relevance_gate():
        return None
    candidates = relevance_gate_candidates(screened)
    if not candidates:
        return screened
    try:
        decisions = llm_relevance_decisions(query, candidates, query_plan=query_plan)
    except Exception:
        return None
    if not decisions:
        return None
    return apply_llm_relevance_decisions(screened, candidates, decisions, query=query, query_plan=query_plan)


def relevance_gate_candidates(screened: dict) -> list[dict]:
    candidates = []
    candidate_index = 0
    for bucket in ("qualified", "needs_review"):
        for reference in screened.get(bucket, []) or []:
            if not isinstance(reference, dict):
                continue
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "source_bucket": bucket,
                    "reference": reference,
                }
            )
            candidate_index += 1
    return candidates


def llm_relevance_decisions(query: str, candidates: list[dict], *, query_plan: dict | None = None) -> dict[int, dict]:
    timeout = bounded_float_env("PAPER_RELEVANCE_LLM_TIMEOUT_SECONDS", 25.0, minimum=3.0, maximum=90.0)
    system_prompt = {
        "society": SOCIETY_LLM_RELEVANCE_SYSTEM_PROMPT,
        "computer": COMPUTER_LLM_RELEVANCE_SYSTEM_PROMPT,
        "engineering": ENGINEERING_LLM_RELEVANCE_SYSTEM_PROMPT,
    }.get(search_mode_from_query_plan(query_plan), LLM_RELEVANCE_SYSTEM_PROMPT)
    content = asyncio.run(
        asyncio.wait_for(
            LLMClient().complete(
                system_prompt=system_prompt,
                user_prompt=llm_relevance_user_prompt(query, candidates, query_plan=query_plan),
                model=os.getenv("PAPER_RELEVANCE_LLM_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.0,
                max_tokens=bounded_int_env("PAPER_RELEVANCE_LLM_MAX_TOKENS", 3000, minimum=500, maximum=12000),
            ),
            timeout=timeout,
        )
    )
    data = parse_json_object(content)
    raw_decisions = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(raw_decisions, list):
        return {}
    decisions: dict[int, dict] = {}
    valid_statuses = {"relevant", "borderline", "off_topic"}
    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        try:
            candidate_index = int(item.get("candidate_index"))
        except (TypeError, ValueError):
            continue
        status = str(item.get("topic_status") or "").strip().casefold()
        if status not in valid_statuses:
            continue
        decisions[candidate_index] = {
            "topic_status": status,
            "confidence": bounded_confidence(item.get("confidence")),
            "matched_concepts": clean_string_list(item.get("matched_concepts"), limit=8),
            "missing_concepts": clean_string_list(item.get("missing_concepts"), limit=8),
            "reason": re.sub(r"\s+", " ", str(item.get("reason") or "")).strip()[:500],
        }
    return decisions


def llm_relevance_user_prompt(query: str, candidates: list[dict], *, query_plan: dict | None = None) -> str:
    compact_candidates = []
    for candidate in candidates:
        reference = candidate["reference"]
        candidate_payload = {
            "candidate_index": candidate["candidate_index"],
            "title": str(reference.get("title") or "")[:400],
            "authors": str(reference.get("authors") or "")[:300],
            "abstract": str(reference.get("abstract") or "")[:900],
            "journal": str(reference.get("journal") or reference.get("source_label") or "")[:160],
            "year": str(reference.get("year") or "")[:20],
            "doi": str(reference.get("doi") or "")[:160],
            "source": str(reference.get("source") or "")[:240],
            "retrieval_channel": str(reference.get("retrieval_channel") or "")[:80],
        }
        if title_intent_query(query_plan):
            candidate_payload["title_similarity_to_query"] = round(
                title_similarity(title_intent_query_text(query, query_plan=query_plan), candidate_payload["title"]),
                4,
            )
            candidate_payload["author_similarity_to_query"] = round(bibliographic_author_similarity(query_plan, reference), 4)
        compact_candidates.append(candidate_payload)
    payload = {
        "original_query": str(query or ""),
        "query_plan": compact_query_plan(query_plan),
        "candidates": compact_candidates,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def compact_query_plan(query_plan: dict | None) -> dict:
    if not isinstance(query_plan, dict):
        return {}
    compact = {}
    for key in (
        "backend_query",
        "llm_search_query",
        "llm_pubmed_query",
        "rules_fallback_query",
        "search_mode",
        "requested_search_mode",
        "query_intent",
        "intent_confidence",
        "extracted",
        "core_concepts",
        "synonyms",
        "forbidden_broadenings",
        "rationale",
    ):
        value = query_plan.get(key)
        if value:
            compact[key] = value
    return compact


def search_mode_from_query_plan(query_plan: dict | None) -> str:
    if not isinstance(query_plan, dict):
        return "biomedical"
    mode = str(query_plan.get("search_mode") or "").strip().casefold()
    aliases = {
        "society": "society",
        "social": "society",
        "social_science": "society",
        "social-science": "society",
        "computer": "computer",
        "computer_science": "computer",
        "computer-science": "computer",
        "cs": "computer",
        "ai": "computer",
        "artificial_intelligence": "computer",
        "artificial-intelligence": "computer",
        "engineering": "engineering",
        "eng": "engineering",
    }
    return aliases.get(mode, "biomedical")


def apply_llm_relevance_decisions(
    screened: dict,
    candidates: list[dict],
    decisions: dict[int, dict],
    *,
    query: str = "",
    query_plan: dict | None = None,
) -> dict:
    gated = {"qualified": [], "needs_review": [], "rejected": list(screened.get("rejected", []))}
    min_confidence = bounded_float_env("PAPER_RELEVANCE_LLM_MIN_CONFIDENCE", 0.65, minimum=0.0, maximum=1.0)
    reject_confidence = bounded_float_env("PAPER_RELEVANCE_LLM_REJECT_CONFIDENCE", 0.75, minimum=0.0, maximum=1.0)
    for candidate in candidates:
        reference = dict(candidate["reference"])
        if title_intent_relevance_match(query, reference, query_plan=query_plan):
            target_bucket = "qualified" if candidate["source_bucket"] == "qualified" else "needs_review"
            gated[target_bucket].append(mark_title_intent_relevant(reference))
            continue
        decision = decisions.get(candidate["candidate_index"])
        if not decision:
            reference["screening_status"] = "needs_review"
            append_unique(reference, "screening_risks", "llm_relevance_missing_decision")
            append_unique(reference, "topic_relevance_risks", "llm_relevance_missing_decision")
            gated["needs_review"].append(reference)
            continue

        status = decision["topic_status"]
        confidence = float(decision["confidence"])
        reference["llm_relevance_status"] = status
        reference["llm_relevance_confidence"] = round(confidence, 4)
        reference["llm_relevance_reason"] = decision.get("reason", "")
        reference["llm_relevance_matched_concepts"] = decision.get("matched_concepts", [])
        reference["llm_relevance_missing_concepts"] = decision.get("missing_concepts", [])
        reference["topic_relevance_score"] = round(confidence, 4)
        reference["topic_relevance_reasons"] = [
            f"llm_match:{concept}" for concept in decision.get("matched_concepts", [])[:8]
        ]
        if decision.get("reason"):
            append_unique(reference, "topic_relevance_reasons", f"llm_reason:{decision['reason']}")
        for concept in decision.get("missing_concepts", [])[:8]:
            append_unique(reference, "topic_relevance_risks", f"llm_missing:{concept}")

        if status == "relevant" and confidence >= min_confidence:
            reference["topic_relevance_status"] = "relevant"
            target_bucket = "qualified" if candidate["source_bucket"] == "qualified" else "needs_review"
            gated[target_bucket].append(reference)
        elif status == "off_topic" and confidence >= reject_confidence:
            reference["screening_status"] = "rejected"
            reference["topic_relevance_status"] = "off_topic"
            append_unique(reference, "screening_reasons", "topic_relevance_failed")
            append_unique(reference, "screening_risks", "off_topic_reference")
            gated["rejected"].append(reference)
        else:
            reference["screening_status"] = "needs_review"
            reference["topic_relevance_status"] = "borderline"
            append_unique(reference, "screening_risks", "topic_relevance_borderline")
            if confidence < min_confidence:
                append_unique(reference, "screening_risks", f"llm_relevance_low_confidence:{confidence:.3f}")
                append_unique(reference, "topic_relevance_risks", f"llm_relevance_low_confidence:{confidence:.3f}")
            gated["needs_review"].append(reference)
    return gated


def relevance_query_profile(query: str, *, query_plan: dict | None = None) -> dict:
    text = relevance_query_text(query, query_plan=query_plan)
    profile = query_profile(text)
    plan_concepts = biomedical_query_plan_concepts(query_plan)
    if not plan_concepts:
        return profile

    concepts = list(profile["concepts"])
    seen = {
        normalize_text(str(term))
        for concept in concepts
        for term in concept.get("terms", [])
        if str(term or "").strip()
    }
    for concept in plan_concepts:
        label_key = normalize_text(concept["label"])
        if label_key in seen:
            continue
        concepts.append(concept)
        seen.update(normalize_text(str(term)) for term in concept.get("terms", []) if str(term or "").strip())

    required_categories = set(profile["required_categories"])
    required_categories.update(
        concept["category"]
        for concept in plan_concepts
        if concept.get("required") and concept.get("category")
    )
    return {
        **profile,
        "concepts": concepts[:12],
        "required_categories": required_categories,
        "is_medical": True,
    }


def relevance_query_text(query: str, *, query_plan: dict | None = None) -> str:
    parts = []
    if isinstance(query_plan, dict):
        for key in ("backend_query", "llm_search_query", "llm_pubmed_query", "rules_fallback_query"):
            value = str(query_plan.get(key) or "").strip()
            if value:
                parts.append(value)
        value = query_plan.get("core_concepts")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = str(item.get("concept") or item.get("term") or item.get("name") or "").strip()
                else:
                    text = str(item or "").strip()
                if text:
                    parts.append(text)
    if not parts:
        parts.append(str(query or "").strip())
    return " ".join(dict.fromkeys(part for part in parts if part))


def biomedical_query_plan_concepts(query_plan: dict | None) -> list[dict]:
    if not isinstance(query_plan, dict):
        return []
    concepts = []
    seen = set()
    for index, value in enumerate(query_plan.get("core_concepts") or []):
        if isinstance(value, dict):
            label = str(value.get("concept") or value.get("term") or value.get("name") or "").strip()
            category = biomedical_concept_category(label, explicit_type=value.get("type"))
            required = bool(value.get("must_keep", True))
        else:
            label = str(value or "").strip()
            category = biomedical_concept_category(label)
            required = True
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        concepts.append(
            {
                "id": f"plan_biomedical_{index}",
                "label": label,
                "category": category,
                "terms": biomedical_concept_terms(label),
                "required": required,
            }
        )
    return concepts


def biomedical_concept_terms(label: str) -> list[str]:
    terms = [label]
    lower = label.casefold()
    synonym_groups = [
        ["dental caries", "tooth decay", "caries", "cavity"],
        ["prevention", "preventive", "preventative", "prophylaxis"],
        ["diagnosis", "diagnostic", "classification"],
        ["prediction", "predictive", "prognosis", "prognostic"],
        ["treatment", "therapy", "therapeutic"],
    ]
    for group in synonym_groups:
        if any(term in lower for term in group):
            terms.extend(group)
    return list(dict.fromkeys(term for term in terms if term))


def biomedical_concept_category(label: str, *, explicit_type=None) -> str:
    explicit = str(explicit_type or "").strip().casefold()
    if explicit in {"condition", "disease", "disorder", "diagnosis"}:
        return "condition"
    if explicit in {"population", "patient", "people", "cohort"}:
        return "population"
    if explicit in {"modality", "imaging", "test"}:
        return "modality"
    if explicit in {"target", "anatomy", "organ", "site"}:
        return "target"
    if explicit in {"method", "model", "algorithm"}:
        return "method"
    if explicit in {"task", "intervention"}:
        return "task"
    if explicit in {"outcome", "endpoint", "metric"}:
        return "outcome"

    lower = label.casefold()
    if any(term in lower for term in ("patient", "population", "children", "child", "adult", "elderly", "women")):
        return "population"
    if any(term in lower for term in ("ct", "mri", "ultrasound", "x-ray", "xray", "imaging", "radiograph")):
        return "modality"
    if any(term in lower for term in ("lesion", "tumor", "tumour", "nodule", "organ", "tissue", "tooth", "teeth")):
        return "target"
    if any(
        term in lower
        for term in (
            "cancer",
            "caries",
            "disease",
            "disorder",
            "infection",
            "stroke",
            "syndrome",
            "diabetes",
            "arthritis",
        )
    ):
        return "condition"
    if any(term in lower for term in ("model", "method", "algorithm", "machine learning", "deep learning")):
        return "method"
    if any(
        term in lower
        for term in (
            "classification",
            "control",
            "detection",
            "diagnosis",
            "prediction",
            "prevention",
            "screening",
            "segmentation",
            "treatment",
        )
    ):
        return "task"
    if any(term in lower for term in ("mortality", "survival", "outcome", "risk", "endpoint")):
        return "outcome"
    return "topic"


def society_query_profile(query: str, *, query_plan: dict | None = None) -> dict:
    concept_inputs: list[tuple[str, bool]] = []
    if isinstance(query_plan, dict):
        for value in query_plan.get("core_concepts") or []:
            if isinstance(value, dict):
                text = str(value.get("concept") or value.get("term") or value.get("name") or "").strip()
                required = bool(value.get("must_keep", True))
            else:
                text = str(value or "").strip()
                required = True
            if text:
                concept_inputs.append((text, required))
        for value in query_plan.get("synonyms") or []:
            text = str(value or "").strip()
            if text:
                concept_inputs.append((text, False))

    if not concept_inputs:
        text = relevance_query_text(query, query_plan=query_plan)
        for keyword in query_keywords(normalize_text(text)):
            concept_inputs.append((keyword, True))

    concepts = []
    seen = set()
    for index, (label, required) in enumerate(concept_inputs[:16]):
        clean = re.sub(r"\s+", " ", label).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms = society_concept_terms(clean)
        concepts.append({"id": f"society_{index}", "label": clean, "terms": terms, "required": required})

    keywords = query_keywords(normalize_text(relevance_query_text(query, query_plan=query_plan)))
    return {"concepts": concepts[:12], "keywords": keywords[:10]}


def society_concept_terms(label: str) -> list[str]:
    terms = [label]
    lower = label.casefold()
    synonym_groups = [
        ["labor", "labour", "employment", "work"],
        ["inequality", "disparity", "stratification"],
        ["education", "schooling", "educational"],
        ["migration", "migrant", "mobility"],
        ["policy", "governance", "regulation"],
        ["family", "household"],
        ["fertility", "birth intention", "fertility intention"],
        ["social media", "platform", "digital platform"],
    ]
    for group in synonym_groups:
        if any(term in lower for term in group):
            terms.extend(group)
    return list(dict.fromkeys(term for term in terms if term))


def society_relevance_min_lexical_score(profile: dict) -> float:
    concept_count = len(profile.get("concepts") or [])
    if concept_count >= 5:
        return 0.55
    if concept_count >= 3:
        return 0.5
    return 0.45


def technical_query_profile(query: str, *, query_plan: dict | None = None, domain: str) -> dict:
    concept_inputs: list[tuple[str, bool]] = []
    if isinstance(query_plan, dict):
        for value in query_plan.get("core_concepts") or []:
            if isinstance(value, dict):
                text = str(value.get("concept") or value.get("term") or value.get("name") or "").strip()
                required = bool(value.get("must_keep", True))
            else:
                text = str(value or "").strip()
                required = True
            if text:
                concept_inputs.append((text, required))
        for value in query_plan.get("synonyms") or []:
            text = str(value or "").strip()
            if text:
                concept_inputs.append((text, False))

    if not concept_inputs:
        text = relevance_query_text(query, query_plan=query_plan)
        for keyword in query_keywords(normalize_text(text)):
            concept_inputs.append((keyword, True))

    concepts = []
    seen = set()
    for index, (label, required) in enumerate(concept_inputs[:16]):
        clean = re.sub(r"\s+", " ", label).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms = technical_concept_terms(clean, domain=domain)
        concepts.append({"id": f"{domain}_{index}", "label": clean, "terms": terms, "required": required})

    keywords = query_keywords(normalize_text(relevance_query_text(query, query_plan=query_plan)))
    return {"concepts": concepts[:12], "keywords": keywords[:10]}


def technical_concept_terms(label: str, *, domain: str) -> list[str]:
    terms = [label]
    lower = label.casefold()
    common_groups = [
        ["artificial intelligence", "ai"],
        ["machine learning", "ml"],
        ["deep learning", "neural network", "neural networks"],
        ["large language model", "llm", "language model"],
        ["retrieval augmented generation", "rag"],
        ["natural language processing", "nlp"],
        ["computer vision", "image analysis"],
        ["optimization", "optimisation"],
        ["fault diagnosis", "fault detection"],
    ]
    computer_groups = [
        ["privacy", "differential privacy"],
        ["security", "cybersecurity"],
        ["software engineering", "software development"],
        ["database", "data management"],
        ["distributed system", "distributed systems"],
    ]
    engineering_groups = [
        ["battery", "lithium-ion battery", "li-ion battery"],
        ["control", "controller", "control system"],
        ["materials", "material"],
        ["manufacturing", "machining"],
        ["renewable energy", "energy"],
        ["structural engineering", "structure", "structural"],
        ["concrete", "cementitious"],
        ["thermal", "heat transfer"],
    ]
    groups = common_groups + (engineering_groups if domain == "engineering" else computer_groups)
    for group in groups:
        if any(term in lower for term in group):
            terms.extend(group)
    return list(dict.fromkeys(term for term in terms if term))


def technical_relevance_min_lexical_score(profile: dict) -> float:
    concept_count = len(profile.get("concepts") or [])
    if concept_count >= 5:
        return 0.55
    if concept_count >= 3:
        return 0.5
    return 0.45


def relevance_gate_mode() -> str:
    mode = str(os.getenv("PAPER_RELEVANCE_GATE", "hybrid") or "").strip().casefold()
    if mode in {"0", "false", "off", "disabled", "rule", "rules"}:
        return "rules"
    if mode in {"1", "true", "on", "enabled", "llm"}:
        return "llm"
    return "hybrid"


def should_use_llm_relevance_gate() -> bool:
    if relevance_gate_mode() == "rules":
        return False
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))


def append_unique(item: dict, key: str, value: str) -> None:
    if not value:
        return
    existing = item.get(key) if isinstance(item.get(key), list) else []
    item[key] = list(dict.fromkeys([*existing, value]))


def bounded_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def bounded_float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def clean_string_list(value, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text[:120])
        if len(cleaned) >= limit:
            break
    return cleaned


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
        raise ValueError("Expected a JSON object.")
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


def assess_reference_relevance(query: str, reference: dict, *, profile: dict | None = None) -> dict:
    item = dict(reference or {})
    profile = profile or query_profile(query)
    text = reference_text(item)

    matched_concepts = [
        concept
        for concept in profile["concepts"]
        if group_matches(text, concept["terms"])
    ]
    matched_ids = {concept["id"] for concept in matched_concepts}
    missing = [concept for concept in profile["concepts"] if concept["id"] not in matched_ids]

    keyword_hits = [keyword for keyword in profile["keywords"] if normalized_contains(text, keyword)]
    concept_score = len(matched_concepts) / max(1, len(profile["concepts"])) if profile["concepts"] else 0.0
    keyword_score = len(keyword_hits) / max(1, len(profile["keywords"])) if profile["keywords"] else 0.0
    if profile["concepts"] and profile["keywords"]:
        lexical_score = max(concept_score, (concept_score * 0.85) + (keyword_score * 0.15))
    elif profile["concepts"]:
        lexical_score = concept_score
    else:
        lexical_score = keyword_score

    reasons = [f"concept_match:{concept['id']}" for concept in matched_concepts]
    reasons.extend(f"keyword_match:{keyword}" for keyword in keyword_hits[:8])
    risks: list[str] = []
    status = "relevant"

    negative_hits = negative_topic_hits(profile, text)
    if negative_hits:
        status = "off_topic"
        risks.append("negative_topic_hint:" + ",".join(negative_hits[:3]))
    else:
        missing_required = [
            concept
            for concept in missing
            if concept["category"] in profile["required_categories"]
        ]
        if missing_required:
            status = (
                "borderline"
                if missing_required_can_review(
                    domain="biomedical",
                    missing_required=missing_required,
                    matched_concepts=matched_concepts,
                    lexical_score=lexical_score,
                )
                else "off_topic"
            )
            risks.append("missing_required_topic_concepts:" + ",".join(concept["id"] for concept in missing_required))
        elif lexical_score < relevance_min_lexical_score(profile):
            status = "borderline" if lexical_score >= 0.45 else "off_topic"
            risks.append(f"low_topic_overlap:{lexical_score:.3f}")

    medcpt_score = medcpt_relevance_score(query, item)
    if medcpt_score is not None:
        item["medcpt_relevance_score"] = round(float(medcpt_score), 4)
        min_score = relevance_min_score()
        if medcpt_score < min_score and status == "relevant":
            status = "borderline"
            risks.append(f"medcpt_below_threshold:{medcpt_score:.3f}<{min_score:.3f}")

    item["topic_relevance_status"] = status
    item["topic_relevance_score"] = round(float(lexical_score), 4)
    item["topic_relevance_reasons"] = reasons
    item["topic_relevance_risks"] = risks
    if risks:
        existing = item.get("screening_risks") if isinstance(item.get("screening_risks"), list) else []
        item["screening_risks"] = list(dict.fromkeys([*existing, *risks]))
    return item


def query_profile(query: str) -> dict:
    text = normalize_text(query)
    concepts = [concept for concept in CONCEPTS if group_matches(text, concept["terms"])]
    keywords = query_keywords(text)
    categories = {concept["category"] for concept in concepts}

    required_categories = set()
    for category in ("task", "condition", "target", "modality"):
        if category in categories:
            required_categories.add(category)
    # For method-only or broad queries, do not make method mandatory; otherwise
    # surveys and clinical benchmark papers can be useful candidates.
    if categories == {"method"}:
        required_categories.add("method")

    return {
        "concepts": concepts,
        "keywords": keywords,
        "required_categories": required_categories,
        "is_medical": is_medical_query(text, concepts),
    }


def expand_query_terms(query: str) -> str:
    profile = query_profile(query)
    terms = []
    categories = {concept["category"] for concept in profile["concepts"]}
    for concept in profile["concepts"]:
        if concept["category"] == "method" and categories - {"method", "task"}:
            continue
        terms.extend(concept.get("expansions") or [])
    if not re.search(r"[\u4e00-\u9fff]", str(query or "")):
        terms.extend(profile["keywords"])
    return " ".join(dict.fromkeys(term for term in terms if term)) or str(query or "").strip()


def reference_text(reference: dict) -> str:
    parts = [
        reference.get("title"),
        reference.get("abstract"),
        reference.get("relevance"),
        reference.get("journal"),
        reference.get("source_label"),
        reference.get("source"),
    ]
    return normalize_text(" ".join(str(part or "") for part in parts))


def query_keywords(text: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", text):
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9][a-z0-9+-]{2,}", text)
    else:
        chunks = re.findall(r"[a-z0-9][a-z0-9+-]{2,}", text)
    stopwords = {
        "and",
        "or",
        "the",
        "for",
        "with",
        "using",
        "based",
        "study",
        "method",
        "paper",
        "medical",
        "clinical",
        "image",
        "imaging",
        "\u533b\u5b66",
        "\u5f71\u50cf",
        "\u7814\u7a76",
        "\u65b9\u6cd5",
    }
    keywords = []
    for chunk in chunks:
        if chunk in stopwords:
            continue
        if any(normalized_contains(" ".join(concept["terms"]), chunk) for concept in CONCEPTS):
            continue
        keywords.append(chunk)
    return list(dict.fromkeys(keywords))[:10]


def group_matches(text: str, terms: list[str]) -> bool:
    return any(normalized_contains(text, term) for term in terms)


def normalized_contains(text: str, term: str) -> bool:
    term_text = normalize_text(term)
    if not term_text:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 +.-]*", term_text):
        pattern = r"(?<![a-z0-9])" + re.escape(term_text).replace(r"\ ", r"[\s-]+") + r"(?![a-z0-9])"
        return bool(re.search(pattern, text))
    return term_text in text


def normalize_text(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[\u2010-\u2015_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_medical_query(text: str, concepts: list[dict]) -> bool:
    if any(concept["category"] in {"modality", "anatomy", "condition", "target"} for concept in concepts):
        return True
    return group_matches(text, MEDICAL_BACKGROUND_TERMS)


def negative_topic_hits(profile: dict, text: str) -> list[str]:
    if not profile.get("is_medical"):
        return []
    return [term for term in GENERIC_NEGATIVE_TERMS if normalized_contains(text, term)]


def relevance_min_lexical_score(profile: dict) -> float:
    concept_count = len(profile.get("concepts") or [])
    if concept_count >= 4:
        return 0.62
    if concept_count >= 2:
        return 0.58
    return 0.5


def relevance_min_score() -> float:
    try:
        return float(os.getenv("PAPER_RELEVANCE_MEDCPT_MIN_SCORE", "0.35"))
    except ValueError:
        return 0.35


def medcpt_relevance_score(query: str, reference: dict) -> float | None:
    mode = os.getenv("PAPER_RELEVANCE_RERANKER", "rules").strip().casefold()
    if mode not in {"medcpt", "auto"}:
        return None
    if mode == "auto" and not os.getenv("PAPER_RELEVANCE_ENABLE_MODEL", "").strip():
        return None
    try:
        tokenizer, model, torch = load_medcpt_cross_encoder()
    except Exception:
        return None
    text = " ".join(
        part
        for part in [
            str(reference.get("title") or "").strip(),
            str(reference.get("abstract") or "").strip(),
        ]
        if part
    )[:2500]
    if not text:
        return None
    with torch.no_grad():
        inputs = tokenizer(
            query,
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        if logits.shape[-1] == 1:
            return float(torch.sigmoid(logits[0][0]).item())
        return float(torch.softmax(logits[0], dim=-1)[-1].item())


@lru_cache(maxsize=1)
def load_medcpt_cross_encoder():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = os.getenv("PAPER_RELEVANCE_MEDCPT_MODEL", "ncbi/MedCPT-Cross-Encoder").strip()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch
