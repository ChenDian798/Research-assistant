from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .llm import LLMClient
from .reference_relevance import expand_query_terms, group_matches, normalize_text, query_profile


SOURCE_LABELS = {
    "arxiv": "arXiv",
    "pubmed": "PubMed",
    "semantic": "Semantic Scholar",
    "semanticscholar": "Semantic Scholar",
    "crossref": "Crossref",
    "openalex": "OpenAlex",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "google_scholar": "Google Scholar",
    "cnki": "知网 CNKI",
}
USER_AGENT = "ResearchAgent-LiteratureSearch/0.1"
LLM_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a biomedical literature search query planner.
Your job is to rewrite a user's research topic into precise English search queries for academic and biomedical literature databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as Crossref, OpenAlex, Semantic Scholar, arXiv, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty if not biomedical",
  "core_concepts": [
    {
      "concept": "English concept",
      "type": "condition|anatomy|modality|task|method|population|outcome|procedure|target|other",
      "must_keep": true
    }
  ],
  "synonyms": ["standard biomedical synonym or spelling variant"],
  "forbidden_broadenings": ["overbroad concept that must not replace the user intent"],
  "recommended_sources": ["pubmed", "crossref", "openalex", "semantic", "arxiv", "cnki"],
  "avoid_sources": ["arxiv"],
  "rationale": "brief explanation of the rewrite"
}
Rules:
- Preserve the user's exact biomedical intent.
- Return syntactically valid JSON that can be parsed by Python json.loads. Do not return comments, markdown, trailing commas, or unescaped control characters.
- Never put raw double quotes inside a JSON string value. If a database query truly needs phrase quotes, escape them as \"; otherwise prefer PubMed-safe terms without phrase quotes, for example (cerebral infarction[Title/Abstract] OR ischemic stroke[Title/Abstract]).
- If the user specifies a disease, anatomy, imaging modality, clinical procedure, population, outcome, target, or task, it must remain represented in both search_query and pubmed_query when applicable.
- For Chinese biomedical topics, translate the full phrase into precise English medical terms. Do not translate word-by-word if a standard medical expression exists.
- Do not broaden a specific topic into a generic one. For example:
  - Do not broaden "ischemic stroke lesion segmentation on non-contrast CT" into "stroke diagnosis".
  - Do not broaden "gastric cancer postoperative anastomotic leakage" into "cancer surgery".
  - Do not broaden "lung nodule detection on CT" into "medical imaging AI".
- Use only standard biomedical synonyms, abbreviations, and spelling variants. Do not invent diseases, procedures, datasets, metrics, or MeSH terms.
- If a term is ambiguous, prefer the narrower interpretation supported by the full user topic.
- PubMed queries should use biomedical terms and may use Title/Abstract style Boolean expressions. Keep them practical rather than overly complex.
- search_query should not use PubMed-only field tags such as [Title/Abstract]; reserve those for pubmed_query.
- Prefer PubMed/Crossref/OpenAlex for clinical medicine. Recommend arXiv only for computational, mathematical, AI-method, or preprint-heavy topics.
- search_query must be English, concise, and under 350 characters.
- pubmed_query must be English and under 600 characters.
- Do not include Chinese characters in search_query or pubmed_query.
- Before returning, mentally validate that the entire response is exactly one JSON object and that every string value is properly escaped.
- If you cannot produce a safe biomedical rewrite, return empty strings for the queries and explain briefly in rationale.
""".strip()
SOCIETY_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a social-science scholarly search query planner.
Your job is to rewrite a user's research topic into precise English search queries for social-science and interdisciplinary academic databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as Crossref, OpenAlex, Semantic Scholar, arXiv, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty unless the topic has a clear health, public-health, psychology, or medical-social component",
  "core_concepts": [
    {
      "concept": "English concept",
      "type": "population|phenomenon|theory|method|policy|institution|outcome|context|region|period|other",
      "must_keep": true
    }
  ],
  "synonyms": ["standard synonym, spelling variant, or related social-science term"],
  "forbidden_broadenings": ["overbroad concept that must not replace the user intent"],
  "recommended_sources": ["crossref", "openalex", "semantic", "arxiv", "cnki", "pubmed"],
  "avoid_sources": ["pubmed"],
  "rationale": "brief explanation of the rewrite"
}
Rules:
- Preserve the user's exact social-science intent, including population, institution, policy, behavior, theory, method, geography, period, and outcome when specified.
- For Chinese social-science topics, translate the full phrase into precise English scholarly terms. Keep CNKI useful by allowing the caller to use the original Chinese query for CNKI.
- Do not broaden a specific topic into a generic one. For example:
  - Do not broaden "rural left-behind children education inequality" into "education".
  - Do not broaden "platform labor algorithmic management in China" into "labor market".
  - Do not broaden "housing affordability and fertility intentions" into "urban policy".
- Use standard social-science synonyms and related constructs, but do not invent theories, datasets, measures, or countries.
- Prefer Crossref, OpenAlex, and Semantic Scholar for most social-science topics. Recommend arXiv only for computational social science, quantitative methods, networks, NLP, economics preprints, or model-heavy topics.
- PubMed should be empty or avoided unless the topic is clearly public health, mental health, psychology, epidemiology, health policy, or medical sociology.
- search_query must be English, concise, and under 350 characters.
- pubmed_query must be English and under 600 characters.
- Do not include Chinese characters in search_query or pubmed_query.
- Before returning, mentally validate that the entire response is exactly one JSON object and that every string value is properly escaped.
- If you cannot produce a safe social-science rewrite, return empty strings for the queries and explain briefly in rationale.
""".strip()
QUERY_EXPANSIONS = {
    "ct": ["non-contrast CT", "NCCT", "computed tomography"],
    "stroke_infarct": ["acute ischemic stroke", "cerebral infarction", "stroke"],
    "lesion": ["lesion"],
    "segmentation": ["segmentation", "delineation"],
    "algorithm": ["deep learning", "algorithm"],
}

SOCIETY_QUERY_EXPANSIONS = [
    {
        "terms": ["education", "educational", "\u6559\u80b2"],
        "expansions": ["education", "educational inequality"],
    },
    {
        "terms": ["inequality", "disparity", "\u4e0d\u5e73\u7b49", "\u5dee\u5f02"],
        "expansions": ["inequality", "disparity"],
    },
    {
        "terms": ["policy", "governance", "\u653f\u7b56", "\u6cbb\u7406"],
        "expansions": ["policy", "governance"],
    },
    {
        "terms": ["labor", "labour", "employment", "\u52b3\u52a8", "\u5c31\u4e1a"],
        "expansions": ["labor", "employment"],
    },
    {
        "terms": ["migration", "migrant", "\u8fc1\u79fb", "\u6d41\u52a8"],
        "expansions": ["migration", "migrants"],
    },
    {
        "terms": ["family", "fertility", "\u5bb6\u5ead", "\u751f\u80b2"],
        "expansions": ["family", "fertility"],
    },
    {
        "terms": ["survey", "interview", "ethnography", "\u95ee\u5377", "\u8bbf\u8c08", "\u6c11\u65cf\u5fd7"],
        "expansions": ["survey", "interview", "ethnography"],
    },
    {
        "terms": ["social media", "platform", "\u793e\u4ea4\u5a92\u4f53", "\u5e73\u53f0"],
        "expansions": ["social media", "platform"],
    },
]

SOCIETY_TOPIC_TERMS = [
    "sociology",
    "social science",
    "education",
    "policy",
    "governance",
    "labor",
    "labour",
    "employment",
    "inequality",
    "migration",
    "urban",
    "rural",
    "family",
    "fertility",
    "gender",
    "class",
    "poverty",
    "welfare",
    "public opinion",
    "media",
    "\u793e\u4f1a",
    "\u793e\u4f1a\u5b66",
    "\u6559\u80b2",
    "\u653f\u7b56",
    "\u6cbb\u7406",
    "\u52b3\u52a8",
    "\u5c31\u4e1a",
    "\u4e0d\u5e73\u7b49",
    "\u8fc1\u79fb",
    "\u57ce\u5e02",
    "\u4e61\u6751",
    "\u5bb6\u5ead",
    "\u751f\u80b2",
    "\u6027\u522b",
    "\u8d2b\u56f0",
    "\u798f\u5229",
    "\u8206\u8bba",
    "\u5a92\u4f53",
]


class PaperSearchError(RuntimeError):
    pass


class LLMQueryRewriteParseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def expand_academic_query(query: str) -> str:
    text = str(query or "").strip()
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    lower = text.casefold()
    expansions: list[str] = []
    if any(term in lower for term in ["头颅ct", "脑ct", "ct", "头颅", "颅脑"]):
        expansions.extend(QUERY_EXPANSIONS["ct"])
    if any(term in lower for term in ["脑梗", "梗死", "卒中", "中风"]):
        expansions.extend(QUERY_EXPANSIONS["stroke_infarct"])
    if "病灶" in lower:
        expansions.extend(QUERY_EXPANSIONS["lesion"])
    if any(term in lower for term in ["分割", "勾画"]):
        expansions.extend(QUERY_EXPANSIONS["segmentation"])
    if any(term in lower for term in ["算法", "模型", "深度学习", "机器学习"]):
        expansions.extend(QUERY_EXPANSIONS["algorithm"])
    if not expansions:
        return text
    return " ".join(dict.fromkeys(expansions))


def expand_academic_query(query: str, *, search_mode: str = "auto") -> str:
    if normalize_search_mode(search_mode, query) == "society":
        return expand_society_query_terms(query)
    return expand_query_terms(query)


def expand_society_query_terms(query: str) -> str:
    text = str(query or "").strip()
    terms: list[str] = []
    lower = text.casefold()
    for concept in SOCIETY_QUERY_EXPANSIONS:
        if any(term.casefold() in lower for term in concept["terms"]):
            terms.extend(concept["expansions"])
    if not re.search(r"[\u4e00-\u9fff]", text):
        terms.extend(query_keyword_terms(text))
    return " ".join(dict.fromkeys(term for term in terms if term)) or text


def normalize_search_mode(search_mode: str | None, query: str = "") -> str:
    mode = str(search_mode or "auto").strip().casefold()
    aliases = {
        "social": "society",
        "social_science": "society",
        "social-science": "society",
        "soc": "society",
        "medical": "biomedical",
        "medicine": "biomedical",
        "bio": "biomedical",
    }
    mode = aliases.get(mode, mode)
    if mode in {"biomedical", "society"}:
        return mode
    if mode not in {"", "auto"}:
        return "auto"
    if looks_like_clinical_topic(query):
        return "biomedical"
    if looks_like_society_topic(query):
        return "society"
    return "biomedical"


def build_academic_search_plan(query: str, requested_sources: list[str], *, search_mode: str = "auto") -> dict:
    active_mode = normalize_search_mode(search_mode, query)
    fallback_query = expand_academic_query(query, search_mode=active_mode)
    plan = {
        "search_mode": active_mode,
        "requested_search_mode": str(search_mode or "auto").strip().casefold() or "auto",
        "backend_query": fallback_query,
        "rules_fallback_query": fallback_query,
        "sources": requested_sources,
        "rewrite_status": "rules",
        "llm_search_query": "",
        "llm_pubmed_query": "",
        "llm_raw_response": "",
        "llm_error": "",
        "core_concepts": [],
        "synonyms": [],
        "forbidden_broadenings": [],
        "avoid_sources": [],
        "rationale": "",
    }
    plan["queries_by_source"] = build_queries_by_source(query, plan, requested_sources)
    if not should_use_llm_query_rewrite(query, search_mode=active_mode):
        return plan
    try:
        try:
            llm_plan = asyncio.run(build_academic_search_plan_with_llm(query, requested_sources, search_mode=active_mode))
        except TypeError as error:
            if "search_mode" not in str(error):
                raise
            llm_plan = asyncio.run(build_academic_search_plan_with_llm(query, requested_sources))
    except LLMQueryRewriteParseError as error:
        plan["rewrite_status"] = f"rules_fallback:{type(error).__name__}"
        plan["llm_error"] = str(error)[:500]
        plan["llm_raw_response"] = clean_text(error.raw_response)[:2000]
        plan["queries_by_source"] = build_queries_by_source(query, plan, requested_sources)
        return plan
    except Exception as error:
        plan["rewrite_status"] = f"rules_fallback:{type(error).__name__}"
        plan["llm_error"] = str(error)[:500]
        plan["queries_by_source"] = build_queries_by_source(query, plan, requested_sources)
        return plan

    plan["llm_search_query"] = clean_text(llm_plan.get("search_query"))[:700] if isinstance(llm_plan, dict) else ""
    plan["llm_pubmed_query"] = clean_text(llm_plan.get("pubmed_query"))[:900] if isinstance(llm_plan, dict) else ""
    plan["llm_raw_response"] = clean_text(llm_plan.get("_raw_response"))[:2000] if isinstance(llm_plan, dict) else ""
    validation_error = llm_plan_guardrail_error(llm_plan, query, search_mode=active_mode)
    if validation_error:
        plan["rewrite_status"] = f"rules_fallback:llm_guardrail:{validation_error}"
        plan["llm_rationale"] = clean_text(llm_plan.get("rationale"))[:500] if isinstance(llm_plan, dict) else ""
        return plan

    backend_query = best_llm_backend_query(llm_plan)
    if backend_query:
        plan["backend_query"] = backend_query
        plan["rewrite_status"] = "llm"
    avoid_sources = normalize_sources_list(llm_plan.get("avoid_sources"))
    plan["sources"] = requested_sources
    plan["core_concepts"] = clean_string_list(llm_plan.get("core_concepts"), limit=8)
    plan["synonyms"] = clean_string_list(llm_plan.get("synonyms"), limit=12)
    plan["forbidden_broadenings"] = clean_string_list(llm_plan.get("forbidden_broadenings"), limit=8)
    plan["avoid_sources"] = avoid_sources
    plan["rationale"] = clean_text(llm_plan.get("rationale"))[:500]
    plan["queries_by_source"] = build_queries_by_source(query, plan, plan["sources"])
    return plan


async def build_academic_search_plan_with_llm(query: str, requested_sources: list[str], *, search_mode: str = "biomedical") -> dict:
    timeout = bounded_float_env("PAPER_SEARCH_QUERY_REWRITE_TIMEOUT_SECONDS", 20.0, minimum=3.0, maximum=60.0)
    system_prompt = (
        SOCIETY_QUERY_REWRITE_SYSTEM_PROMPT
        if normalize_search_mode(search_mode, query) == "society"
        else LLM_QUERY_REWRITE_SYSTEM_PROMPT
    )
    planner_label = "social-science" if normalize_search_mode(search_mode, query) == "society" else "biomedical"
    user_prompt = (
        f"User topic:\n{query}\n\n"
        f"Infer the exact {planner_label} concepts yourself from the full user topic. "
        "Do not rely on a fixed local vocabulary being complete.\n\n"
        f"Currently selected sources:\n{json.dumps(requested_sources, ensure_ascii=False)}"
    )
    content = await asyncio.wait_for(
        LLMClient().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=os.getenv("SEARCH_QUERY_REWRITE_MODEL") or os.getenv("RESEARCH_MODEL"),
            temperature=0.0,
            max_tokens=900,
        ),
        timeout=timeout,
    )
    try:
        data = parse_json_object(content)
    except (json.JSONDecodeError, ValueError) as error:
        raise LLMQueryRewriteParseError(f"{type(error).__name__}: {error}", content) from error
    if isinstance(data, dict):
        data["_raw_response"] = content
    return data if isinstance(data, dict) else {}


def query_profile_for_prompt(profile: dict) -> dict:
    required_categories = set(profile.get("required_categories") or [])
    concepts = []
    for concept in profile.get("concepts") or []:
        english_terms = [term for term in concept.get("terms", []) if not re.search(r"[\u4e00-\u9fff]", term)]
        concepts.append(
            {
                "id": concept.get("id", ""),
                "category": concept.get("category", ""),
                "required": concept.get("category") in required_categories,
                "english_terms": english_terms[:8],
                "expansions": concept.get("expansions", [])[:8],
            }
        )
    return {
        "is_medical": bool(profile.get("is_medical")),
        "required_categories": sorted(required_categories),
        "concepts": concepts,
    }


def llm_plan_guardrail_error(plan: dict, query: str, *, search_mode: str = "biomedical") -> str:
    if not isinstance(plan, dict):
        return "invalid_plan"

    search_query = clean_search_query(plan.get("search_query"), max_length=350)
    pubmed_query = clean_search_query(plan.get("pubmed_query"), max_length=600)
    if clean_text(plan.get("search_query")) and not search_query:
        return "invalid_search_query"
    if clean_text(plan.get("pubmed_query")) and not pubmed_query:
        return "invalid_pubmed_query"
    if not best_llm_backend_query(plan):
        return "empty_query"

    if normalize_search_mode(search_mode, query) == "biomedical" and should_use_local_query_guardrail():
        missing_category = first_missing_required_category(plan, query_profile(query))
        if missing_category:
            return f"missing_required_{missing_category}"

    recommended_sources = normalize_sources_list(plan.get("recommended_sources"))
    if normalize_search_mode(search_mode, query) == "biomedical" and looks_like_clinical_topic(query) and recommended_sources == ["arxiv"]:
        return "medical_sources_too_narrow"

    return ""


def should_use_local_query_guardrail() -> bool:
    mode = str(os.getenv("PAPER_SEARCH_LOCAL_QUERY_GUARDRAIL", "false") or "").strip().casefold()
    return mode in {"1", "true", "on", "enabled", "yes"}


def first_missing_required_category(plan: dict, profile: dict) -> str:
    required_categories = set(profile.get("required_categories") or [])
    if not required_categories:
        return ""
    candidate_text = normalize_text(
        " ".join(
            [
                clean_text(plan.get("search_query")),
                clean_text(plan.get("pubmed_query")),
                " ".join(clean_string_list(plan.get("core_concepts"), limit=12)),
                " ".join(clean_string_list(plan.get("synonyms"), limit=20)),
            ]
        )
    )
    for category in sorted(required_categories):
        concepts = [
            concept
            for concept in profile.get("concepts") or []
            if concept.get("category") == category
        ]
        if concepts and not any(llm_text_matches_concept(candidate_text, concept) for concept in concepts):
            return category
    return ""


def llm_text_matches_concept(text: str, concept: dict) -> bool:
    terms = list(concept.get("terms") or []) + list(concept.get("expansions") or [])
    return group_matches(text, terms)


def should_use_llm_query_rewrite(query: str, *, search_mode: str = "auto") -> bool:
    mode = str(os.getenv("PAPER_SEARCH_QUERY_REWRITE", "auto") or "").strip().casefold()
    if mode in {"0", "false", "off", "disabled", "rules"}:
        return False
    if mode in {"1", "true", "on", "enabled", "llm"}:
        return True
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")):
        return False
    active_mode = normalize_search_mode(search_mode, query)
    if active_mode == "society":
        return True
    text = str(query or "")
    return bool(re.search(r"[\u4e00-\u9fff]", text) or looks_like_clinical_topic(text))


def looks_like_clinical_topic(query: str) -> bool:
    text = str(query or "")
    return bool(query_profile(text).get("is_medical")) or bool(
        re.search(
            r"cancer|carcinoma|tumou?r|patient|patients|clinical|surgery|surgical|postoperative|"
            r"gastrectomy|anastomotic|leak|fistula|术|术后|癌|瘘|吻合|患者|临床|治疗|诊断",
            text,
            flags=re.IGNORECASE,
        )
    )


def looks_like_society_topic(query: str) -> bool:
    text = str(query or "")
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in SOCIETY_TOPIC_TERMS)


def best_llm_backend_query(plan: dict) -> str:
    value = clean_search_query(plan.get("search_query"), max_length=350)
    if value:
        return value
    value = clean_search_query(plan.get("pubmed_query"), max_length=600)
    if value:
        return value
    concepts = clean_string_list(plan.get("core_concepts"), limit=8)
    return " AND ".join(concepts[:4]) if concepts else ""


def build_queries_by_source(query: str, plan: dict, sources: list[str]) -> dict[str, str]:
    general_query = clean_search_query(plan.get("llm_search_query"), max_length=350)
    if not general_query:
        general_query = clean_search_query(plan.get("backend_query"), max_length=700)
    if not general_query:
        general_query = clean_search_query(plan.get("rules_fallback_query"), max_length=700)
    if not general_query:
        general_query = expand_academic_query(query)

    pubmed_query = clean_search_query(plan.get("llm_pubmed_query"), max_length=600)
    if not pubmed_query and normalize_search_mode(plan.get("search_mode"), query) == "biomedical":
        pubmed_query = build_pubmed_fallback_query(query, general_query)
    if not pubmed_query:
        pubmed_query = general_query

    queries = {}
    for source in sources:
        if source == "pubmed":
            queries[source] = pubmed_query
        elif source == "cnki" and re.search(r"[\u4e00-\u9fff]", str(query or "")):
            queries[source] = clean_text(query)
        else:
            queries[source] = general_query
    return queries


def build_pubmed_fallback_query(query: str, fallback_query: str) -> str:
    profile = query_profile(query)
    groups = []
    preferred_categories = {"condition", "modality", "target", "task", "anatomy"}
    for concept in profile.get("concepts") or []:
        if concept.get("category") not in preferred_categories:
            continue
        terms = []
        for term in list(concept.get("expansions") or []) + list(concept.get("terms") or []):
            cleaned = clean_pubmed_term(term)
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
            if len(terms) >= 4:
                break
        if terms:
            groups.append("(" + " OR ".join(f"{term}[Title/Abstract]" for term in terms) + ")")
        if len(groups) >= 5:
            break
    return " AND ".join(groups) if groups else fallback_query


def clean_pubmed_term(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"[\[\]\"']", "", text)
    text = re.sub(r"[^A-Za-z0-9 +.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return ""
    return text


def query_keyword_terms(text: str) -> list[str]:
    chunks = re.findall(r"[a-z0-9][a-z0-9+-]{2,}", str(text or "").casefold())
    stopwords = {
        "and",
        "or",
        "the",
        "for",
        "with",
        "using",
        "based",
        "study",
        "research",
        "paper",
        "analysis",
        "effect",
        "effects",
        "impact",
    }
    return list(dict.fromkeys(chunk for chunk in chunks if chunk not in stopwords))[:10]


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


def clean_search_query(value, *, max_length: int = 700) -> str:
    if not isinstance(value, str):
        return ""
    text = clean_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) > max_length:
        return ""
    if re.search(r"[\u4e00-\u9fff]", text):
        return ""
    return text


def clean_string_list(value, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if isinstance(item, dict):
            text = clean_text(item.get("concept") or item.get("term") or item.get("name"))
        else:
            text = clean_text(item)
        if text and text not in cleaned:
            cleaned.append(text[:120])
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_sources_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = {"arxiv", "pubmed", "semantic", "crossref", "openalex", "biorxiv", "medrxiv", "google_scholar", "cnki"}
    normalized = []
    for item in value:
        source = normalize_source_name(item)
        if source in allowed and source not in normalized:
            normalized.append(source)
    return normalized


def bounded_float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def search_papers(
    query: str,
    *,
    sources: str,
    max_results_per_source: int,
    year: str = "",
    timeout_seconds: int = 45,
    search_mode: str = "auto",
) -> dict:
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")

    source_names = normalize_sources(sources)
    max_results = max(1, min(int(max_results_per_source or 5), 50))
    search_plan = build_academic_search_plan(clean_query, source_names, search_mode=search_mode)
    backend_query = str(search_plan.get("backend_query") or expand_academic_query(clean_query, search_mode=search_plan.get("search_mode", "auto"))).strip()
    source_names = list(search_plan.get("sources") or source_names)
    queries_by_source = build_queries_by_source(clean_query, search_plan, source_names)
    search_plan["queries_by_source"] = queries_by_source
    payload = run_paper_search_backend_by_source(
        queries_by_source,
        max_results_per_source=max_results,
        year=str(year or "").strip(),
        timeout_seconds=max(1, int(timeout_seconds or 45)),
    )
    papers, source_results, errors = normalize_search_payload(payload, source_names)
    papers = filter_papers_by_year(papers, str(year or "").strip())
    deduped = dedupe_papers(papers)
    return {
        "query": clean_query,
        "search_mode": search_plan.get("search_mode", normalize_search_mode(search_mode, clean_query)),
        "requested_search_mode": search_plan.get("requested_search_mode", search_mode),
        "backend_query": backend_query,
        "rules_fallback_query": search_plan.get("rules_fallback_query", ""),
        "llm_search_query": search_plan.get("llm_search_query", ""),
        "llm_pubmed_query": search_plan.get("llm_pubmed_query", ""),
        "llm_raw_response": search_plan.get("llm_raw_response", ""),
        "llm_error": search_plan.get("llm_error", ""),
        "query_rewrite_status": search_plan.get("rewrite_status", "rules"),
        "query_plan": search_plan,
        "queries_by_source": queries_by_source,
        "sources_used": source_names,
        "source_results": source_results,
        "errors": errors,
        "raw_count": len(papers),
        "papers": deduped,
    }


def run_paper_search_backend_by_source(
    queries_by_source: dict[str, str],
    *,
    max_results_per_source: int,
    year: str,
    timeout_seconds: int,
) -> dict:
    grouped: dict[str, list[str]] = {}
    for source, query in queries_by_source.items():
        clean_query = str(query or "").strip()
        if not clean_query:
            continue
        grouped.setdefault(clean_query, []).append(source)

    merged_results: dict[str, list[dict]] = {}
    merged_errors: dict[str, str] = {}
    for query, sources in grouped.items():
        payload = run_paper_search_backend(
            query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            year=year,
            timeout_seconds=timeout_seconds,
        )
        papers, source_results, errors = normalize_search_payload(payload, sources)
        for source in sources:
            merged_results.setdefault(source, [])
        for paper in papers:
            source = normalize_source_name(paper.get("retrieved_from") or paper.get("source_label") or paper.get("source"))
            if source not in sources:
                source = normalize_source_name(paper.get("raw_source_record", {}).get("source")) if isinstance(paper.get("raw_source_record"), dict) else ""
            if source not in sources:
                source = sources[0]
            merged_results.setdefault(source, []).append(paper.get("raw_source_record") or paper)
        for source in sources:
            if source_results.get(source, 0) == 0:
                merged_results.setdefault(source, merged_results.get(source, []))
        merged_errors.update(errors)
    return {"results": merged_results, "errors": merged_errors}


def run_paper_search_cli(
    query: str,
    *,
    sources: list[str],
    max_results_per_source: int,
    year: str,
    timeout_seconds: int,
) -> dict:
    command = os.getenv("PAPER_SEARCH_COMMAND", "paper-search").strip() or "paper-search"
    args = [
        command,
        "search",
        "--query",
        query,
        "--sources",
        ",".join(sources),
        "--max-results",
        str(max_results_per_source),
        "--format",
        "json",
    ]
    if year:
        args.extend(["--year", year])
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise PaperSearchError("paper-search command was not found.") from error
    except subprocess.TimeoutExpired as error:
        raise PaperSearchError(f"paper-search timed out after {timeout_seconds} seconds.") from error
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "paper-search failed").strip()
        raise PaperSearchError(message[:1000])
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PaperSearchError("paper-search returned invalid JSON.") from error
    if not isinstance(data, dict):
        raise PaperSearchError("paper-search JSON response must be an object.")
    return data


def run_paper_search_backend(
    query: str,
    *,
    sources: list[str],
    max_results_per_source: int,
    year: str,
    timeout_seconds: int,
) -> dict:
    command = os.getenv("PAPER_SEARCH_COMMAND", "paper-search").strip() or "paper-search"
    if shutil.which(command):
        return run_paper_search_cli(
            query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            year=year,
            timeout_seconds=timeout_seconds,
        )
    return run_paper_search_mcp_python(
        query,
        sources=sources,
        max_results_per_source=max_results_per_source,
    )


def run_paper_search_mcp_python(
    query: str,
    *,
    sources: list[str],
    max_results_per_source: int,
) -> dict:
    try:
        from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher
        from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher
        from paper_search_mcp.academic_platforms.biorxiv import BioRxivSearcher
        from paper_search_mcp.academic_platforms.medrxiv import MedRxivSearcher
        from paper_search_mcp.academic_platforms.google_scholar import GoogleScholarSearcher
    except ImportError as error:
        raise PaperSearchError(
            "paper-search command was not found, and paper-search-mcp Python package is not installed."
        ) from error

    searchers = {
        "arxiv": ArxivSearcher,
        "pubmed": PubMedSearcher,
        "biorxiv": BioRxivSearcher,
        "medrxiv": MedRxivSearcher,
        "google_scholar": GoogleScholarSearcher,
        "googlescholar": GoogleScholarSearcher,
    }
    api_searchers = {
        "semantic": search_semantic_scholar_api,
        "crossref": search_crossref_api,
        "openalex": search_openalex_api,
        "cnki": search_cnki_api,
    }
    results: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for source in sources:
        if source in api_searchers:
            try:
                results[source] = api_searchers[source](query, max_results_per_source)
            except Exception as error:
                results[source] = []
                errors[source] = str(error)
            continue
        searcher_class = searchers.get(source)
        if searcher_class is None:
            results[source] = []
            errors[source] = "This source is not supported by the installed paper-search-mcp Python package."
            continue
        try:
            papers = searcher_class().search(query, max_results_per_source)
            results[source] = [paper.to_dict() for paper in papers]
        except Exception as error:
            results[source] = []
            errors[source] = str(error)
    return {"results": results, "errors": errors}


def search_semantic_scholar_api(query: str, max_results: int) -> list[dict]:
    fields = "title,authors,abstract,year,url,venue,externalIds"
    headers = {}
    api_key = os.getenv("PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    payload = fetch_json_url(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": query, "limit": max_results, "fields": fields},
        headers=headers,
    )
    papers = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        papers.append(
            {
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "authors": item.get("authors", []),
                "abstract": item.get("abstract", ""),
                "year": item.get("year", ""),
                "url": item.get("url", ""),
                "venue": item.get("venue", ""),
                "doi": external.get("DOI", ""),
                "pmid": external.get("PubMed", ""),
                "arxiv_id": external.get("ArXiv", ""),
                "source": "semantic",
            }
        )
    return papers


def search_cnki_api(query: str, max_results: int) -> list[dict]:
    del query, max_results
    raise PaperSearchError(
        "CNKI search is listed as a selectable source, but no CNKI adapter is configured yet. "
        "CNKI usually requires institution/login access or a licensed API; configure an adapter before enabling live CNKI retrieval."
    )


def search_crossref_api(query: str, max_results: int) -> list[dict]:
    payload = fetch_json_url(
        "https://api.crossref.org/works",
        {"query": query, "rows": max_results, "sort": "relevance"},
    )
    items = ((payload.get("message") or {}).get("items") or []) if isinstance(payload, dict) else []
    papers = []
    for item in items:
        title = first_list_text(item.get("title"))
        authors = []
        for author in item.get("author", [])[:8] if isinstance(item.get("author"), list) else []:
            if isinstance(author, dict):
                authors.append(" ".join(part for part in [author.get("given"), author.get("family")] if part))
        doi = item.get("DOI", "")
        papers.append(
            {
                "paper_id": doi,
                "title": title,
                "authors": authors,
                "abstract": item.get("abstract", ""),
                "year": crossref_year(item),
                "url": item.get("URL", ""),
                "venue": first_list_text(item.get("container-title")),
                "doi": doi,
                "source": "crossref",
            }
        )
    return papers


def search_openalex_api(query: str, max_results: int) -> list[dict]:
    payload = fetch_json_url(
        "https://api.openalex.org/works",
        {"search": query, "per-page": max_results},
    )
    papers = []
    for item in payload.get("results", []) if isinstance(payload, dict) else []:
        authorships = item.get("authorships", []) if isinstance(item.get("authorships"), list) else []
        authors = [
            ((authorship.get("author") or {}).get("display_name") or "")
            for authorship in authorships[:8]
            if isinstance(authorship, dict)
        ]
        primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
        papers.append(
            {
                "paper_id": item.get("id", ""),
                "title": item.get("display_name", ""),
                "authors": authors,
                "abstract": openalex_abstract(item.get("abstract_inverted_index")),
                "year": item.get("publication_year", ""),
                "url": item.get("doi") or item.get("id", ""),
                "venue": source.get("display_name", ""),
                "doi": item.get("doi", ""),
                "source": "openalex",
            }
        )
    return papers


def fetch_json_url(url: str, params: dict, *, headers: dict | None = None) -> dict:
    request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = Request(
        f"{url}?{urlencode(params)}",
        headers=request_headers,
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as error:
        host = urlparse(url).netloc
        if error.code == 429 and "semanticscholar" in host:
            raise PaperSearchError(
                "Semantic Scholar rate limit reached (HTTP 429). "
                "Uncheck Semantic Scholar or set PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY."
            ) from error
        raise PaperSearchError(f"{host} search failed: HTTP Error {error.code}") from error
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise PaperSearchError(f"{urlparse(url).netloc} search failed: {error}") from error


def first_list_text(value) -> str:
    if isinstance(value, list) and value:
        return clean_text(value[0])
    return clean_text(value)


def crossref_year(item: dict) -> str:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return str(parts[0][0])
    return ""


def openalex_abstract(index) -> str:
    if not isinstance(index, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, str(word)))
    return " ".join(word for _, word in sorted(words))


def normalize_sources(sources: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(sources, (list, tuple)):
        items = [str(item).strip().lower() for item in sources]
    else:
        items = [item.strip().lower() for item in str(sources or "").split(",")]
    normalized = []
    for item in items:
        if not item:
            continue
        if item == "semantic-scholar":
            item = "semantic"
        if item not in normalized:
            normalized.append(item)
    return normalized or ["arxiv", "pubmed", "semantic"]


def normalize_search_payload(payload: dict, requested_sources: list[str]) -> tuple[list[dict], dict, dict]:
    errors: dict[str, str] = {}
    papers: list[dict] = []
    source_results: dict[str, int] = {}

    raw_errors = payload.get("errors", {})
    if isinstance(raw_errors, dict):
        errors.update({str(key): str(value) for key, value in raw_errors.items() if value})

    if isinstance(payload.get("papers"), list):
        for item in payload["papers"]:
            paper = normalize_paper(item, default_source=item.get("source") if isinstance(item, dict) else "")
            if paper:
                papers.append(paper)
                source = paper.get("retrieved_from") or "unknown"
                source_results[source] = source_results.get(source, 0) + 1

    raw_results = payload.get("results") or payload.get("source_results")
    if isinstance(raw_results, dict):
        for source, result in raw_results.items():
            source_key = normalize_source_name(source)
            if isinstance(result, list):
                for item in result:
                    paper = normalize_paper(item, default_source=source_key)
                    if paper:
                        papers.append(paper)
                source_results[source_key] = source_results.get(source_key, 0) + len(result)
            elif isinstance(result, dict):
                if result.get("error"):
                    errors[source_key] = str(result.get("error"))
                items = result.get("papers") or result.get("results") or result.get("items")
                if isinstance(items, list):
                    for item in items:
                        paper = normalize_paper(item, default_source=source_key)
                        if paper:
                            papers.append(paper)
                    source_results[source_key] = source_results.get(source_key, 0) + len(items)
                elif isinstance(result.get("count"), int):
                    source_results[source_key] = int(result.get("count") or 0)
            elif isinstance(result, int):
                source_results[source_key] = result

    for source in requested_sources:
        source_results.setdefault(source, 0)
    return papers, source_results, errors


def normalize_paper(item, *, default_source: str = "") -> dict:
    if not isinstance(item, dict):
        return {}
    if item.get("error") and not item.get("title"):
        return {
            "title": "",
            "retrieved_from": normalize_source_name(default_source or item.get("source")),
            "source_error": str(item.get("error") or ""),
        }

    source = normalize_source_name(item.get("retrieved_from") or item.get("source") or default_source)
    doi = first_value(item, "doi", "DOI")
    pmid = first_value(item, "pmid", "PMID", "pubmed_id")
    arxiv_id = first_value(item, "arxiv_id", "arxivId", "arxiv")
    url = first_value(item, "url", "source", "abs_url", "paper_url", "external_url")
    paper_id = first_value(item, "paper_id", "paperId", "id", "paperID")
    if not arxiv_id:
        arxiv_id = infer_arxiv_id(url or paper_id)
    if not pmid:
        pmid = infer_pmid(url)
    authors = normalize_authors(item.get("authors") or item.get("author"))
    abstract = first_value(item, "abstract", "summary", "description")
    year = first_year(first_value(item, "year", "published", "published_date", "publicationDate", "publication_date"))
    title = clean_text(first_value(item, "title", "name"))
    source_url = canonical_source_url(doi=doi, pmid=pmid, arxiv_id=arxiv_id, url=url)
    source_label = SOURCE_LABELS.get(source, source.title() if source else "")

    relevance_parts = []
    if authors:
        relevance_parts.append(f"Authors: {authors}")
    if year:
        relevance_parts.append(f"Year: {year}")
    if item.get("journal") or item.get("venue"):
        relevance_parts.append(f"Source: {clean_text(first_value(item, 'journal', 'venue'))}")
    if abstract:
        relevance_parts.append(f"Abstract: {clean_text(abstract)[:900]}")

    return {
        "id": paper_id or doi or pmid or arxiv_id or source_url,
        "title": title,
        "source": source_url,
        "source_origin": "paper_search_mcp",
        "source_label": source_label,
        "retrieved_from": source,
        "authors": authors,
        "year": year,
        "journal": clean_text(first_value(item, "journal", "venue", "container_title")),
        "doi": normalize_doi(doi),
        "pmid": str(pmid or "").strip(),
        "arxiv_id": str(arxiv_id or "").strip(),
        "abstract": clean_text(abstract),
        "relevance": "；".join(relevance_parts),
        "raw_source_record": item,
    }


def filter_papers_by_year(papers: list[dict], year: str) -> list[dict]:
    if not year:
        return papers
    bounds = parse_year_filter(year)
    if not bounds:
        return papers
    start_year, end_year = bounds
    filtered = []
    for paper in papers:
        paper_year = first_year(paper.get("year"))
        if not paper_year:
            filtered.append(paper)
            continue
        year_value = int(paper_year)
        if start_year <= year_value <= end_year:
            filtered.append(paper)
    return filtered


def parse_year_filter(value: str) -> tuple[int, int] | None:
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:19|20)\d{2}\b", str(value or ""))]
    if not years:
        return None
    if len(years) == 1:
        return years[0], years[0]
    return min(years[0], years[1]), max(years[0], years[1])


def dedupe_papers(papers: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for paper in papers:
        key = paper_identity_key(paper)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(paper)
    return deduped


def paper_identity_key(paper: dict) -> str:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi.casefold()}"
    pmid = str(paper.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"arxiv:{arxiv_id.casefold()}"
    source = str(paper.get("source") or "").strip().rstrip("/")
    if source:
        return f"url:{source.casefold()}"
    title = re.sub(r"\W+", " ", str(paper.get("title") or "").casefold()).strip()
    year = str(paper.get("year") or "").strip()
    return f"title:{title}:{year}" if title else ""


def first_value(values: dict, *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, list):
            if value:
                return clean_text(value[0])
            continue
        if value not in (None, ""):
            return clean_text(value)
    return ""


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_authors(value) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if not isinstance(value, list):
        return ""
    names = []
    for author in value[:8]:
        if isinstance(author, str):
            name = author
        elif isinstance(author, dict):
            name = author.get("name") or " ".join(
                part for part in [author.get("given"), author.get("family")] if part
            )
        else:
            name = ""
        if name:
            names.append(clean_text(name))
    if len(value) > 8:
        names.append("et al.")
    return ", ".join(names)


def normalize_doi(value) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().rstrip(".)]}")


def normalize_source_name(value) -> str:
    source = str(value or "").strip().lower().replace(" ", "")
    if source in {"semanticscholar", "semantic-scholar"}:
        return "semantic"
    if source in {"知网", "cnki", "cnki.net", "中国知网"}:
        return "cnki"
    return source


def first_year(value: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def infer_arxiv_id(value: str) -> str:
    match = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", str(value or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def infer_pmid(value: str) -> str:
    parsed = urlparse(str(value or ""))
    if parsed.netloc.lower().endswith("pubmed.ncbi.nlm.nih.gov"):
        match = re.search(r"/(\d{6,9})(?:/|$)", parsed.path)
        if match:
            return match.group(1)
    return ""


def canonical_source_url(*, doi: str, pmid: str, arxiv_id: str, url: str) -> str:
    doi = normalize_doi(doi)
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return str(url or "").strip()
