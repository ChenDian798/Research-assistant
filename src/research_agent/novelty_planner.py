from __future__ import annotations

import asyncio
import json
import os
import re

from .llm import LLMClient, LLMServiceError
from .paper_search import normalize_search_mode


NOISE_LABELS = [
    "\u9a8c\u6536\u91cd\u70b9",
    "\u9a8c\u6536\u8981\u6c42",
    "\u6d4b\u8bd5\u8981\u6c42",
    "\u6d4b\u8bd5\u8bf4\u660e",
    "\u62a5\u544a\u5e94\u63d0\u793a",
    "\u5efa\u8bae\u6765\u6e90",
    "\u68c0\u7d22\u6a21\u5f0f",
    "\u5bfc\u5e08\u8981\u6c42",
    "\u5907\u6ce8",
]

NOVELTY_PLANNER_SYSTEM_PROMPT = """
You are a scholarly novelty-search planner.
Return only valid JSON, no markdown.
Schema:
{
  "language": "zh|en",
  "clean_innovation_text": "claim text only, with acceptance notes removed",
  "ignored_context": ["removed headings or notes"],
  "domain": "biomedical|computer|engineering|society",
  "core_claim": "one concise English claim",
  "claims": [
    {
      "claim_id": "C1",
      "claim": "one concise claim in English or Chinese",
      "claim_type": "method|application|dataset_or_scenario|evaluation|combination|finding",
      "required_concepts": [{"term":"English concept","type":"condition|modality|task|domain|other","must_keep":true}],
      "method_concepts": [{"term":"English method","type":"method"}],
      "context_concepts": [{"term":"English context","type":"data_setting|population|scenario|other"}],
      "baseline_concepts": ["known baseline/model/dataset"]
    }
  ],
  "required_concepts": [{"term":"English concept","type":"condition|modality|task|domain|other","must_keep":true}],
  "method_concepts": [{"term":"English method","type":"method"}],
  "context_concepts": [{"term":"English context","type":"data_setting|population|scenario|other"}],
  "baseline_concepts": ["known baseline/model/dataset"],
  "forbidden_noise": ["noise that must not guide retrieval"]
}
Rules:
- Do not create one over-constrained Boolean query.
- Separate required topic concepts from methods, constraints, and baselines.
- Split the innovation into 3-5 concise claim-level units when the text supports it.
- Translate Chinese research terms into practical English scholarly-search terms.
- Remove acceptance criteria, testing instructions, report-format notes, and source-selection notes from the innovation claim.
""".strip()


def build_novelty_plan(innovation_text: str, search_mode: str = "auto") -> dict:
    clean_text, ignored = sanitize_innovation_text_with_ignored(innovation_text)
    if not clean_text:
        clean_text = normalize_space(innovation_text)
    plan = None
    if os.getenv("NOVELTY_PLANNER_LLM", os.getenv("PAPER_SEARCH_QUERY_REWRITE", "true")).lower() not in {"0", "false", "no"}:
        try:
            plan = build_novelty_plan_with_llm(clean_text, search_mode)
        except (RuntimeError, LLMServiceError, ValueError, json.JSONDecodeError, TimeoutError):
            plan = None
    if not plan:
        plan = build_novelty_plan_rules_fallback(clean_text, search_mode)
    plan["clean_innovation_text"] = plan.get("clean_innovation_text") or clean_text
    plan["ignored_context"] = list(dict.fromkeys([*(plan.get("ignored_context") or []), *ignored]))
    plan["forbidden_noise"] = list(dict.fromkeys([*(plan.get("forbidden_noise") or []), *NOISE_LABELS]))
    plan["claims"] = normalize_claims(plan.get("claims"), plan, clean_text)
    plan["queries"] = build_layered_queries(plan)
    return normalize_plan(plan, clean_text, search_mode)


def build_novelty_plan_with_llm(innovation_text: str, search_mode: str = "auto") -> dict:
    llm = LLMClient()

    async def run() -> dict:
        content = await asyncio.wait_for(
            llm.complete(
                system_prompt=NOVELTY_PLANNER_SYSTEM_PROMPT,
                user_prompt=f"Requested domain/search mode: {search_mode}\n\nInnovation text:\n{innovation_text[:4000]}",
                temperature=0.1,
                max_tokens=2400,
            ),
            timeout=90,
        )
        data = parse_json_object(content)
        if not isinstance(data, dict):
            raise ValueError("Novelty planner must return an object.")
        return data

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run())
    raise RuntimeError("Cannot run synchronous novelty planner inside an active event loop.")


def build_novelty_plan_rules_fallback(innovation_text: str, search_mode: str = "auto") -> dict:
    text = normalize_space(innovation_text)
    language = "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
    domain = normalize_search_mode(search_mode, text)
    terms = detect_terms(text)
    required = [concept(term, ctype, True) for term, ctype in terms["required"]]
    methods = [concept(term, ctype, False) for term, ctype in terms["method"]]
    contexts = [concept(term, ctype, False) for term, ctype in terms["context"]]
    baselines = terms["baseline"]

    if not required:
        required = fallback_required_concepts(text, domain)
    if not methods and domain in {"biomedical", "computer"}:
        methods = [concept("deep learning", "method", False)]

    core_claim_terms = [item["term"] for item in required[:4]]
    if not core_claim_terms:
        core_claim_terms = english_tokens(text)[:8]
    core_claim = " ".join(core_claim_terms) or text[:180]
    return {
        "language": language,
        "clean_innovation_text": text,
        "ignored_context": [],
        "domain": domain,
        "core_claim": core_claim,
        "required_concepts": required[:6],
        "method_concepts": methods[:8],
        "context_concepts": contexts[:6],
        "baseline_concepts": baselines[:8],
        "claims": build_fallback_claims(
            text,
            required_concepts=required[:6],
            method_concepts=methods[:8],
            context_concepts=contexts[:6],
            baseline_concepts=baselines[:8],
        ),
        "forbidden_noise": list(NOISE_LABELS),
    }


def sanitize_innovation_text(innovation_text: str) -> str:
    return sanitize_innovation_text_with_ignored(innovation_text)[0]


def sanitize_innovation_text_with_ignored(innovation_text: str) -> tuple[str, list[str]]:
    text = str(innovation_text or "").replace("\r\n", "\n").replace("\r", "\n")
    ignored: list[str] = []
    kept: list[str] = []
    label_pattern = "|".join(re.escape(label) for label in NOISE_LABELS)
    heading_re = re.compile(rf"^\s*(?:[#*\-\d.、\s]*)?(?P<label>{label_pattern})\s*[:：]?", re.IGNORECASE)
    inline_re = re.compile(rf"(?:^|[;\n。.!?])\s*(?P<label>{label_pattern})\s*[:：].*$", re.IGNORECASE | re.DOTALL)
    for line in text.split("\n"):
        if not line.strip():
            continue
        match = heading_re.search(line)
        if match:
            ignored.append(match.group("label"))
            continue
        cleaned = inline_re.sub(lambda m: ignored.append(m.group("label")) or "", line).strip()
        if cleaned:
            kept.append(cleaned)
    clean = normalize_space(" ".join(kept))
    return clean, list(dict.fromkeys(ignored))


def build_layered_queries(plan: dict) -> list[dict]:
    domain = str(plan.get("domain") or "computer")
    required = concept_terms(plan.get("required_concepts"))
    methods = concept_terms(plan.get("method_concepts"))
    contexts = concept_terms(plan.get("context_concepts"))
    baselines = [str(item).strip() for item in plan.get("baseline_concepts") or [] if str(item).strip()]
    core = required[:3] or split_claim_terms(plan.get("core_claim"))
    topic = join_query(core)
    method_topic = join_query([*(required[:2] or core[:2]), *(methods[:3] or ["deep learning"])])
    context_topic = join_query([*(required[:2] or core[:2]), *contexts[:3]]) if contexts else join_query([*core[:2], "low-resource"])

    queries: list[dict] = []
    if domain == "biomedical":
        queries.extend(
            [
                query("pubmed_broad_topic", "broad_topic", ["pubmed"], pubmed_query(core), "broad", 10),
                query("pubmed_method_overlap", "method_overlap", ["pubmed"], pubmed_query([*(required[:2] or core[:2]), *(methods[:2] or ["deep learning"])]), "medium", 10),
                query("pubmed_context_overlap", "context_overlap", ["pubmed"], pubmed_query([contexts[0] if contexts else "medical image segmentation", *(contexts[1:3] or ["domain generalization", "few-shot"])]), "medium", 8),
            ]
        )
    else:
        queries.append(query("academic_broad_topic", "broad_topic", ["semantic", "openalex", "crossref"], topic, "broad", 10))

    queries.extend(
        [
            query("general_broad_topic", "broad_topic", ["semantic", "openalex", "crossref"], topic, "broad", 10),
            query("general_core_topic", "core_topic", ["semantic", "openalex", "crossref"], join_query(core[:4] or required[:4]), "medium", 10),
            query("semantic_method_overlap", "method_overlap", ["semantic", "openalex", "crossref", "arxiv"], method_topic, "medium", 10),
            query("semantic_context_overlap", "context_overlap", ["semantic", "openalex", "crossref", "arxiv"], context_topic, "medium", 8),
        ]
    )
    for index, baseline in enumerate(baselines[:4]):
        queries.append(
            query(
                f"baseline_overlap_{index + 1}",
                "baseline_overlap",
                ["semantic", "openalex", "crossref", "arxiv", "pubmed"] if domain == "biomedical" else ["semantic", "openalex", "crossref", "arxiv"],
                join_query([baseline, *(core[:3] or required[:3])]),
                "narrow",
                8,
            )
        )
    if not baselines:
        queries.append(query("baseline_overlap_generic", "baseline_overlap", ["semantic", "openalex", "crossref", "arxiv"], join_query([*(methods[:2] or core[:2]), "baseline comparison"]), "medium", 8))
    queries.extend(build_claim_queries(plan, domain))
    return dedupe_queries(queries)


def build_claim_queries(plan: dict, domain: str) -> list[dict]:
    queries: list[dict] = []
    for claim in plan.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = normalize_space(claim.get("claim_id"))
        if not claim_id:
            continue
        claim_text = normalize_space(claim.get("claim"))
        required = concept_terms(claim.get("required_concepts")) or concept_terms(plan.get("required_concepts"))
        methods = concept_terms(claim.get("method_concepts")) or concept_terms(plan.get("method_concepts"))
        contexts = concept_terms(claim.get("context_concepts")) or concept_terms(plan.get("context_concepts"))
        baselines = [str(item).strip() for item in claim.get("baseline_concepts") or [] if str(item).strip()]
        core = required[:3] or split_claim_terms(claim_text) or split_claim_terms(plan.get("core_claim"))
        exact_terms = [*(core[:3]), *(methods[:2]), *(contexts[:2])]
        method_terms = [*(methods[:3] or core[:2]), *(baselines[:1])]
        task_terms = [*(core[:3]), *(contexts[:2])]
        if not exact_terms:
            exact_terms = split_claim_terms(claim_text)[:5]
        source_set = ["semantic", "openalex", "crossref", "arxiv"]
        if domain == "biomedical":
            exact_text = pubmed_query(core[:2] + methods[:1] + contexts[:1])
            method_text = pubmed_query((methods[:2] or core[:2]) + baselines[:1])
            task_text = pubmed_query(core[:2] + contexts[:1])
            queries.extend(
                [
                    query(
                        f"{claim_id}_exact_claim",
                        "claim_exact",
                        ["pubmed", "semantic", "openalex", "crossref"],
                        exact_text or join_query(exact_terms),
                        "medium",
                        8,
                        claim_id=claim_id,
                        claim_query_type="exact_claim_query",
                    ),
                    query(
                        f"{claim_id}_method_generalized",
                        "claim_method_generalized",
                        ["semantic", "openalex", "crossref", "arxiv", "pubmed"],
                        method_text or join_query(method_terms),
                        "broad",
                        8,
                        claim_id=claim_id,
                        claim_query_type="method_generalized_query",
                    ),
                    query(
                        f"{claim_id}_task_generalized",
                        "claim_task_generalized",
                        ["pubmed", "semantic", "openalex", "crossref"],
                        task_text or join_query(task_terms),
                        "broad",
                        8,
                        claim_id=claim_id,
                        claim_query_type="task_generalized_query",
                    ),
                ]
            )
        else:
            queries.extend(
                [
                    query(
                        f"{claim_id}_exact_claim",
                        "claim_exact",
                        source_set,
                        join_query(exact_terms),
                        "medium",
                        8,
                        claim_id=claim_id,
                        claim_query_type="exact_claim_query",
                    ),
                    query(
                        f"{claim_id}_method_generalized",
                        "claim_method_generalized",
                        ["semantic", "openalex", "crossref", "arxiv"],
                        join_query(method_terms or exact_terms),
                        "broad",
                        8,
                        claim_id=claim_id,
                        claim_query_type="method_generalized_query",
                    ),
                    query(
                        f"{claim_id}_task_generalized",
                        "claim_task_generalized",
                        ["semantic", "openalex", "crossref"],
                        join_query(task_terms or exact_terms),
                        "broad",
                        8,
                        claim_id=claim_id,
                        claim_query_type="task_generalized_query",
                    ),
                ]
            )
    return queries


def pubmed_query(terms: list[str]) -> str:
    groups = [pubmed_group(term) for term in terms if str(term).strip()]
    return " AND ".join(group for group in groups if group)


def pubmed_group(term: str) -> str:
    clean = normalize_space(term)
    synonyms = TERM_SYNONYMS.get(clean.casefold(), [clean])
    parts = []
    for synonym in synonyms[:4]:
        safe = re.sub(r"[\[\]\"]+", "", synonym).strip()
        if safe:
            parts.append(f"{safe}[Title/Abstract]")
    if not parts:
        return ""
    return f"({' OR '.join(dict.fromkeys(parts))})"


def query(
    query_id: str,
    purpose: str,
    sources: list[str],
    text: str,
    strictness: str,
    limit: int,
    *,
    claim_id: str = "",
    claim_query_type: str = "",
) -> dict:
    item = {
        "query_id": query_id,
        "purpose": purpose,
        "sources": sources,
        "query": normalize_space(text),
        "strictness": strictness,
        "max_results_per_source": limit,
    }
    if claim_id:
        item["claim_id"] = claim_id
    if claim_query_type:
        item["claim_query_type"] = claim_query_type
    return item


def detect_terms(text: str) -> dict[str, list]:
    lower = normalize_space(text).casefold()
    required: list[tuple[str, str]] = []
    method: list[tuple[str, str]] = []
    context: list[tuple[str, str]] = []
    baseline: list[str] = []
    for needles, term, ctype, bucket in TERM_RULES:
        if any(needle.casefold() in lower for needle in needles):
            if bucket == "required":
                required.append((term, ctype))
            elif bucket == "method":
                method.append((term, ctype))
            elif bucket == "context":
                context.append((term, ctype))
    for pattern in BASELINE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_space(match.group(0))
            if value and value.casefold() not in {item.casefold() for item in baseline}:
                baseline.append(value)
    return {
        "required": list(dict.fromkeys(required)),
        "method": list(dict.fromkeys(method)),
        "context": list(dict.fromkeys(context)),
        "baseline": baseline,
    }


def fallback_required_concepts(text: str, domain: str) -> list[dict]:
    tokens = english_tokens(text)
    if domain == "biomedical":
        defaults = ["medical image", "segmentation"]
    elif domain == "society":
        defaults = ["social science", "policy"]
    elif domain == "engineering":
        defaults = ["engineering system", "optimization"]
    else:
        defaults = ["artificial intelligence", "task"]
    terms = tokens[:4] or defaults
    return [concept(term, "domain" if index == 0 else "task", True) for index, term in enumerate(terms)]


def concept(term: str, ctype: str, must_keep: bool) -> dict:
    return {"term": term, "type": ctype, "must_keep": bool(must_keep)}


def concept_terms(items) -> list[str]:
    terms = []
    for item in items or []:
        term = item.get("term") if isinstance(item, dict) else item
        term = normalize_space(term)
        if term and term.casefold() not in {existing.casefold() for existing in terms}:
            terms.append(term)
    return terms


def build_fallback_claims(
    text: str,
    *,
    required_concepts: list[dict],
    method_concepts: list[dict],
    context_concepts: list[dict],
    baseline_concepts: list[str],
) -> list[dict]:
    parts = split_claim_sentences(text)
    if not parts:
        parts = [text]
    claims = []
    for index, part in enumerate(parts[:5], start=1):
        claims.append(
            {
                "claim_id": f"C{index}",
                "claim": normalize_space(part)[:300],
                "claim_type": infer_claim_type(part, method_concepts, context_concepts, baseline_concepts),
                "required_concepts": required_concepts[:6],
                "method_concepts": method_concepts[:8],
                "context_concepts": context_concepts[:6],
                "baseline_concepts": baseline_concepts[:8],
            }
        )
    return claims or [
        {
            "claim_id": "C1",
            "claim": normalize_space(text)[:300],
            "claim_type": "combination",
            "required_concepts": required_concepts[:6],
            "method_concepts": method_concepts[:8],
            "context_concepts": context_concepts[:6],
            "baseline_concepts": baseline_concepts[:8],
        }
    ]


def split_claim_sentences(text: str) -> list[str]:
    parts = re.split(r"[\n。！？!?；;]+|(?:\s+\b(?:and|plus)\b\s+)", normalize_space(text))
    cleaned = [part.strip(" -0123456789.") for part in parts if len(part.strip()) >= 8]
    return list(dict.fromkeys(cleaned))[:5]


def infer_claim_type(text: str, methods: list[dict], contexts: list[dict], baselines: list[str]) -> str:
    lower = normalize_space(text).casefold()
    if any(word in lower for word in ["evaluation", "benchmark", "compare", "comparison", "对比", "评估", "验证"]):
        return "evaluation"
    if any(word in lower for word in ["dataset", "scenario", "low-resource", "few-shot", "小样本", "低资源", "场景"]):
        return "dataset_or_scenario"
    if any(word in lower for word in ["apply", "application", "应用", "面向"]):
        return "application"
    if len(methods or []) >= 2 or ((methods or []) and (contexts or baselines)):
        return "combination"
    if methods:
        return "method"
    return "finding"


def normalize_claims(value, plan: dict, clean_text: str) -> list[dict]:
    base_required = normalize_concept_list(plan.get("required_concepts"), must_keep=True)
    base_methods = normalize_concept_list(plan.get("method_concepts"), must_keep=False)
    base_contexts = normalize_concept_list(plan.get("context_concepts"), must_keep=False)
    base_baselines = clean_string_list(plan.get("baseline_concepts"), limit=12)
    raw_claims = value if isinstance(value, list) else []
    normalized: list[dict] = []
    for index, raw in enumerate(raw_claims[:5], start=1):
        if not isinstance(raw, dict):
            claim_text = normalize_space(raw)
            raw = {}
        else:
            claim_text = normalize_space(raw.get("claim"))
        if not claim_text:
            continue
        claim_type = normalize_space(raw.get("claim_type")).replace("-", "_")
        if claim_type not in {"method", "application", "dataset_or_scenario", "evaluation", "combination", "finding"}:
            claim_type = infer_claim_type(claim_text, base_methods, base_contexts, base_baselines)
        normalized.append(
            {
                "claim_id": normalize_space(raw.get("claim_id")) or f"C{index}",
                "claim": claim_text[:300],
                "claim_type": claim_type,
                "required_concepts": normalize_concept_list(raw.get("required_concepts"), must_keep=True) or base_required[:6],
                "method_concepts": normalize_concept_list(raw.get("method_concepts"), must_keep=False) or base_methods[:8],
                "context_concepts": normalize_concept_list(raw.get("context_concepts"), must_keep=False) or base_contexts[:6],
                "baseline_concepts": clean_string_list(raw.get("baseline_concepts"), limit=8) or base_baselines[:8],
            }
        )
    if not normalized:
        normalized = build_fallback_claims(
            clean_text,
            required_concepts=base_required[:6],
            method_concepts=base_methods[:8],
            context_concepts=base_contexts[:6],
            baseline_concepts=base_baselines[:8],
        )
    for index, claim in enumerate(normalized, start=1):
        claim["claim_id"] = claim.get("claim_id") or f"C{index}"
    return normalized[:5]


def split_claim_terms(value) -> list[str]:
    text = normalize_space(value)
    if not text:
        return []
    return [part for part in re.split(r"\s+(?:and|with|for|on|in)\s+|[,;]", text, flags=re.IGNORECASE) if normalize_space(part)][:5]


def join_query(terms: list[str]) -> str:
    return normalize_space(" ".join(dict.fromkeys(str(term).strip() for term in terms if str(term).strip())))


def dedupe_queries(queries: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in queries:
        key = (item.get("query_id"), item.get("query"))
        if not item.get("query") or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def normalize_plan(plan: dict, clean_text: str, search_mode: str) -> dict:
    domain = normalize_search_mode(plan.get("domain") or search_mode, clean_text)
    normalized = {
        "language": plan.get("language") if plan.get("language") in {"zh", "en"} else ("zh" if re.search(r"[\u4e00-\u9fff]", clean_text) else "en"),
        "clean_innovation_text": normalize_space(plan.get("clean_innovation_text") or clean_text),
        "ignored_context": clean_string_list(plan.get("ignored_context"), limit=12),
        "domain": domain,
        "core_claim": normalize_space(plan.get("core_claim") or clean_text[:240]),
        "required_concepts": normalize_concept_list(plan.get("required_concepts"), must_keep=True),
        "method_concepts": normalize_concept_list(plan.get("method_concepts"), must_keep=False),
        "context_concepts": normalize_concept_list(plan.get("context_concepts"), must_keep=False),
        "baseline_concepts": clean_string_list(plan.get("baseline_concepts"), limit=12),
        "claims": normalize_claims(plan.get("claims"), plan, clean_text),
        "forbidden_noise": clean_string_list(plan.get("forbidden_noise"), limit=20),
        "queries": [item for item in plan.get("queries") or [] if isinstance(item, dict) and item.get("query")],
    }
    return normalized


def normalize_concept_list(value, *, must_keep: bool) -> list[dict]:
    items = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            term = normalize_space(item.get("term") or item.get("concept"))
            ctype = normalize_space(item.get("type")) or "other"
            keep = bool(item.get("must_keep", must_keep))
        else:
            term = normalize_space(item)
            ctype = "other"
            keep = must_keep
        if term and term.casefold() not in {existing["term"].casefold() for existing in items}:
            items.append({"term": term, "type": ctype, "must_keep": keep})
    return items[:12]


def clean_string_list(value, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = normalize_space(item)
        if text and text.casefold() not in {existing.casefold() for existing in cleaned}:
            cleaned.append(text[:300])
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
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def english_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}(?:\s+[a-zA-Z][a-zA-Z0-9+-]{2,})?", text)


def normalize_space(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


TERM_RULES = [
    (["ischemic stroke", "ischaemic stroke", "cerebral infarction", "stroke", "\u8111\u6897", "\u5352\u4e2d", "\u7f3a\u8840\u6027\u5352\u4e2d"], "ischemic stroke", "condition", "required"),
    (["non-contrast ct", "ncct", "computed tomography", "\u975e\u589e\u5f3a ct", "\u975e\u589e\u5f3act", "\u5934\u9885ct", "\u8111ct", "ct"], "non-contrast CT", "modality", "required"),
    (["lesion segmentation", "segmentation", "\u75c5\u7076\u5206\u5272", "\u5206\u5272"], "lesion segmentation", "task", "required"),
    (["legal", "law", "\u6cd5\u5f8b", "\u53f8\u6cd5"], "legal question answering", "task", "required"),
    (["retrieval augmented generation", "rag", "\u68c0\u7d22\u589e\u5f3a"], "retrieval augmented generation", "method", "method"),
    (["question answering", "qa", "\u95ee\u7b54"], "question answering", "task", "required"),
    (["hallucination", "\u5e7b\u89c9"], "hallucination mitigation", "method", "method"),
    (["lightweight", "efficient", "\u8f7b\u91cf", "\u9ad8\u6548"], "lightweight encoder", "method", "method"),
    (["cross-domain", "domain generalization", "\u8de8\u57df", "\u6cdb\u5316"], "domain generalization", "method", "method"),
    (["consistency", "\u4e00\u81f4\u6027"], "consistency constraint", "method", "method"),
    (["few-shot", "few shot", "small sample", "low-resource", "\u5c0f\u6837\u672c", "\u4f4e\u8d44\u6e90"], "low-resource", "data_setting", "context"),
    (["small hospital", "\u5c0f\u533b\u9662"], "small hospital dataset", "data_setting", "context"),
    (["medical image", "\u533b\u5b66\u5f71\u50cf"], "medical image segmentation", "task", "required"),
    (["deep learning", "\u6df1\u5ea6\u5b66\u4e60"], "deep learning", "method", "method"),
    (["large language model", "llm", "\u5927\u8bed\u8a00\u6a21\u578b"], "large language model", "method", "method"),
]

TERM_SYNONYMS = {
    "ischemic stroke": ["ischemic stroke", "ischaemic stroke", "cerebral infarction", "stroke"],
    "non-contrast ct": ["non-contrast CT", "NCCT", "computed tomography", "CT"],
    "lesion segmentation": ["lesion segmentation", "segmentation", "delineation"],
    "medical image segmentation": ["medical image segmentation", "image segmentation"],
    "deep learning": ["deep learning", "neural network", "machine learning"],
    "domain generalization": ["domain generalization", "domain adaptation", "cross-domain"],
    "low-resource": ["low-resource", "few-shot", "small sample"],
}

BASELINE_PATTERNS = [
    r"\bnnU-?Net\b",
    r"\bUNet\+\+\b",
    r"\bU-?Net\b",
    r"\bSwinUNETR\b",
    r"\bTransUNet\b",
    r"\bRAG\b",
    r"\bBM25\b",
]
