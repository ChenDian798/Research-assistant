from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .doi import (
    doi_resolution_status,
    extract_arxiv_id as extract_stable_arxiv_id,
    extract_doi as extract_stable_doi,
    extract_pmid as extract_stable_pmid,
    extract_url as extract_stable_url,
    fetch_arxiv_metadata as fetch_stable_arxiv_metadata,
    fetch_crossref_metadata as fetch_stable_crossref_metadata,
    fetch_pubmed_metadata as fetch_stable_pubmed_metadata,
    fetch_webpage_metadata as fetch_stable_webpage_metadata,
)
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
ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_REQUEST_LOCK = threading.Lock()
_ARXIV_LAST_REQUEST_AT = 0.0
AUTHOR_FIELD_QUERY_RE = re.compile(r'^(?:au|author):(?:"([^"]+)"|(.+))$', re.IGNORECASE)
INTENT_LABELS = ("title", "author", "citation", "topic", "method_task", "abstract")
BIBLIOGRAPHIC_INTENTS = {"title", "author+title", "citation", "citation_with_title_author"}
INTENT_SEARCH_CONTRACTS = {
    "title": {
        "allowed_channels": ("exact_title", "fuzzy_title", "topic"),
        "forbid_channels": ("author", "citation", "method_task", "abstract_claim"),
        "primary_channel": "exact_title",
        "query_fields": ("channel_queries.exact_title", "extracted.title", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.title"),
        "fallback_channel": "topic",
    },
    "author": {
        "allowed_channels": ("author", "topic"),
        "forbid_channels": ("exact_title", "fuzzy_title", "citation", "method_task", "abstract_claim"),
        "primary_channel": "author",
        "query_fields": ("channel_queries.author", "extracted.authors", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.authors"),
        "fallback_channel": "topic",
    },
    "author+title": {
        "allowed_channels": ("exact_title", "author", "fuzzy_title", "topic"),
        "forbid_channels": ("citation", "method_task", "abstract_claim"),
        "primary_channel": "exact_title",
        "query_fields": ("channel_queries.exact_title", "channel_queries.author", "extracted.title", "extracted.authors", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.title", "extracted.authors"),
        "fallback_channel": "topic",
    },
    "citation": {
        "allowed_channels": ("citation", "exact_title", "author"),
        "forbid_channels": ("fuzzy_title", "topic", "method_task", "abstract_claim"),
        "primary_channel": "citation",
        "query_fields": ("channel_queries.citation", "extracted.authors", "extracted.year", "extracted.title", "extracted.venue", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.year", "extracted.title", "extracted.authors", "extracted.venue"),
        "fallback_channel": "citation",
    },
    "citation_with_title_author": {
        "allowed_channels": ("citation", "exact_title", "author", "fuzzy_title"),
        "forbid_channels": ("topic", "method_task", "abstract_claim"),
        "primary_channel": "citation",
        "query_fields": ("channel_queries.citation", "channel_queries.exact_title", "channel_queries.author", "extracted.title", "extracted.authors", "extracted.year", "extracted.venue", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.title", "extracted.authors", "extracted.year", "extracted.venue"),
        "fallback_channel": "exact_title",
    },
    "topic": {
        "allowed_channels": ("topic",),
        "forbid_channels": ("exact_title", "fuzzy_title", "author", "citation", "method_task", "abstract_claim"),
        "primary_channel": "topic",
        "query_fields": ("channel_queries.topic", "search_query", "core_concepts"),
        "validation_fields": ("query_intent", "intent_confidence", "core_concepts", "must_match_concepts"),
        "fallback_channel": "topic",
    },
    "method_task": {
        "allowed_channels": ("method_task", "topic"),
        "forbid_channels": ("exact_title", "fuzzy_title", "author", "citation", "abstract_claim"),
        "primary_channel": "method_task",
        "query_fields": ("channel_queries.method_task", "extracted.method_terms", "extracted.task_terms", "extracted.domain_terms", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "extracted.method_terms", "extracted.task_terms", "extracted.domain_terms"),
        "fallback_channel": "topic",
    },
    "abstract": {
        "allowed_channels": ("abstract_claim", "topic"),
        "forbid_channels": ("exact_title", "fuzzy_title", "author", "citation", "method_task"),
        "primary_channel": "abstract_claim",
        "query_fields": ("channel_queries.abstract_claim", "channel_queries.topic", "core_concepts", "search_query"),
        "validation_fields": ("query_intent", "intent_confidence", "core_concepts", "extracted.method_terms", "extracted.task_terms", "extracted.domain_terms"),
        "fallback_channel": "topic",
    },
}
RANK_SIGNAL_LABELS = ("title", "author", "year", "venue", "topic", "method", "task", "abstract", "source")
TEMPLATE_WEIGHTS = {
    "title": {"title": 50, "author": 15, "year": 10, "venue": 5, "topic": 5, "method": 5, "task": 0, "abstract": 0, "source": 10},
    "author": {"title": 10, "author": 55, "year": 10, "venue": 5, "topic": 5, "method": 5, "task": 0, "abstract": 0, "source": 10},
    "author+title": {"title": 35, "author": 35, "year": 10, "venue": 5, "topic": 5, "method": 5, "task": 0, "abstract": 0, "source": 10},
    "citation": {"title": 35, "author": 25, "year": 20, "venue": 10, "topic": 0, "method": 0, "task": 0, "abstract": 0, "source": 10},
    "topic": {"title": 5, "author": 0, "year": 0, "venue": 0, "topic": 30, "method": 20, "task": 25, "abstract": 10, "source": 10},
    "method_task": {"title": 10, "author": 0, "year": 0, "venue": 0, "topic": 20, "method": 35, "task": 25, "abstract": 0, "source": 10},
    "abstract": {"title": 5, "author": 0, "year": 0, "venue": 0, "topic": 25, "method": 25, "task": 25, "abstract": 15, "source": 5},
}
BASE_DYNAMIC_WEIGHTS = {
    "title": 15,
    "author": 10,
    "year": 5,
    "venue": 5,
    "topic": 20,
    "method": 15,
    "task": 15,
    "abstract": 10,
    "source": 5,
}
CHANNEL_QUERY_KEYS = ("exact_title", "author", "citation", "topic", "method_task", "abstract_claim")
LLM_CHANNEL_QUERY_SCHEMA_PROMPT = """
Additional schema fields for the same JSON object:
{
  "extracted": {
    "identifiers": {
      "doi": "DOI if present, else empty",
      "pmid": "PMID if present, else empty",
      "arxiv_id": "arXiv ID if present, else empty"
    },
    "domain_terms": ["domain/entity terms explicitly present"]
  },
  "channel_queries": {
    "exact_title": "clean exact paper title only, or empty",
    "author": "author-focused query only, or empty",
    "citation": "citation-focused query preserving author/year/title/venue, or empty",
    "topic": "topic/domain query, or empty",
    "method_task": "method+task query, or empty",
    "abstract_claim": "concise claim/abstract-derived query, or empty"
  },
  "must_match_concepts": ["concepts that should remain mandatory for topical recall"],
  "do_not_mix": ["materials that must not be blended into the wrong channel"]
}
Channel-query rules:
- You may suggest channel_queries, but the backend intent contract decides whether any channel is enabled.
- Do not blend title, author, citation, topic, method/task, and abstract material into one search_query.
- search_query is only a general fallback query. It must not override a more specific channel query.
- For title or author+title intent, channel_queries.exact_title should contain only the clean title, without author/year/venue.
- For author intent, channel_queries.author should preserve the author name.
- For citation intent, channel_queries.citation should preserve author, year, title, and venue details when present.
- For topic intent, keep bibliographic fields out of channel_queries.topic.
- For method_task intent, focus channel_queries.method_task on method/model/task/domain terms.
- For abstract intent, use channel_queries.abstract_claim for concise claim-derived recall, not the full paragraph.
- Put known DOI, PMID, and arXiv identifiers in extracted.identifiers rather than mixing them into search_query.
""".strip()
METHOD_TERMS = {
    "transformer", "cnn", "rnn", "bert", "gpt", "llm", "rag", "nnunet", "nn-u-net",
    "u-net", "unet", "swin", "graph neural", "gnn", "federated", "diffusion",
    "segmentation", "classification", "detection", "retrieval", "ranking",
    "深度学习", "机器学习", "模型", "算法", "分割", "检测", "分类", "检索", "排序",
}
TASK_TERMS = {
    "segmentation", "classification", "detection", "retrieval", "generation", "forecasting",
    "prediction", "diagnosis", "question answering", "summarization", "translation",
    "分割", "检测", "分类", "预测", "诊断", "生成", "问答", "摘要", "翻译",
}
DOMAIN_TERMS = {
    "stroke", "ct", "mri", "cancer", "patient", "clinical", "brain", "legal",
    "education", "policy", "battery", "concrete", "remote sensing", "medical",
    "脑梗", "卒中", "头颅", "颅脑", "肿瘤", "患者", "医学", "法律", "教育", "政策", "电池", "混凝土",
}
SOURCE_VERIFICATION_SCORE = {
    "arxiv": 0.95,
    "pubmed": 0.95,
    "crossref": 0.9,
    "semantic": 0.85,
    "biorxiv": 0.8,
    "medrxiv": 0.8,
    "openalex": 0.65,
    "google_scholar": 0.55,
    "cnki": 0.55,
}
LLM_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a biomedical literature search query planner.
Your job is to rewrite a user's research topic into precise English search queries for academic and biomedical literature databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as Crossref, OpenAlex, Semantic Scholar, arXiv, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty if not biomedical",
  "query_intent": "title|author|author+title|citation|topic|method_task|abstract",
  "intent_confidence": 0.0,
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  },
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
- Classify query_intent before writing queries. Title, author+title, citation, and abstract inputs need different search behavior than broad topics.
- If the input contains both a paper title and author names, use query_intent author+title even when it looks citation-like.
- Return syntactically valid JSON that can be parsed by Python json.loads. Do not return comments, markdown, trailing commas, or unescaped control characters.
- Never put raw double quotes inside a JSON string value. If a database query truly needs phrase quotes, escape them as \"; otherwise prefer PubMed-safe terms without phrase quotes, for example (cerebral infarction[Title/Abstract] OR ischemic stroke[Title/Abstract]).
- If the user specifies a disease, anatomy, imaging modality, clinical procedure, population, outcome, target, or task, it must remain represented in both search_query and pubmed_query when applicable.
- For title intent, keep the exact title words in search_query and do not replace the title with only a topic.
- For author+title or citation inputs, put the clean paper title in extracted.title and author names in extracted.authors; do not merge them into an exact-title query.
- For author intent, keep the author name in search_query. For citation intent, preserve author, year, title, and venue details when present.
- For abstract intent, extract the core condition/domain, method, task, population, and outcome before creating concise queries.
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
""".strip() + "\n\n" + LLM_CHANNEL_QUERY_SCHEMA_PROMPT
SOCIETY_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a social-science scholarly search query planner.
Your job is to rewrite a user's research topic into precise English search queries for social-science and interdisciplinary academic databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as Crossref, OpenAlex, Semantic Scholar, arXiv, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty unless the topic has a clear health, public-health, psychology, or medical-social component",
  "query_intent": "title|author|author+title|citation|topic|method_task|abstract",
  "intent_confidence": 0.0,
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  },
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
- Classify query_intent before writing queries. Title, author+title, citation, and abstract inputs need different search behavior than broad topics.
- If the input contains both a paper title and author names, use query_intent author+title even when it looks citation-like.
- For title intent, keep the exact title words in search_query and do not replace the title with only a topic.
- For author+title or citation inputs, put the clean paper title in extracted.title and author names in extracted.authors; do not merge them into an exact-title query.
- For author intent, keep the author name in search_query. For citation intent, preserve author, year, title, and venue details when present.
- For abstract intent, extract core population, phenomenon, method, context, and outcome before creating concise queries.
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
""".strip() + "\n\n" + LLM_CHANNEL_QUERY_SCHEMA_PROMPT
COMPUTER_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a computer-science and AI scholarly search query planner.
Your job is to rewrite a user's research topic into precise English search queries for computer-science, artificial-intelligence, and interdisciplinary technical literature databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as arXiv, Semantic Scholar, Crossref, OpenAlex, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty unless the topic has a clear biomedical, clinical, public-health, or health-AI component",
  "query_intent": "title|author|author+title|citation|topic|method_task|abstract",
  "intent_confidence": 0.0,
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  },
  "core_concepts": [
    {
      "concept": "English concept",
      "type": "task|method|model|dataset|benchmark|metric|system|application|domain|security|software|hardware|other",
      "must_keep": true
    }
  ],
  "synonyms": ["standard computer-science synonym, abbreviation, spelling variant, or closely equivalent technical term"],
  "forbidden_broadenings": ["overbroad concept that must not replace the user intent"],
  "recommended_sources": ["arxiv", "semantic", "openalex", "crossref", "cnki", "pubmed"],
  "avoid_sources": ["pubmed"],
  "rationale": "brief explanation of the rewrite"
}
Rules:
- Preserve the user's exact computer-science or AI intent, including task, model family, algorithm, dataset, benchmark, metric, system context, application domain, and constraints when specified.
- Classify query_intent before writing queries. Title, author+title, citation, and abstract inputs need different search behavior than broad topics.
- If the input contains both a paper title and author names, use query_intent author+title even when it looks citation-like.
- For title intent, keep the exact title words in search_query and do not replace the title with only a topic.
- For author+title or citation inputs, put the clean paper title in extracted.title and author names in extracted.authors; do not merge them into an exact-title query.
- For author intent, keep the author name in search_query. For citation intent, preserve author, year, title, and venue details when present.
- For abstract intent, extract core method/model, task, dataset/benchmark, domain, and metric before creating concise queries.
- Treat AI, machine learning, deep learning, NLP, computer vision, robotics, data mining, software engineering, security, systems, databases, HCI, and hardware architecture as computer-science topics.
- For Chinese computer-science topics, translate the full phrase into precise English technical terms. Keep CNKI useful by allowing the caller to use the original Chinese query for CNKI.
- Do not broaden a specific topic into a generic one. For example:
  - Do not broaden "retrieval augmented generation for legal question answering" into "large language models".
  - Do not broaden "federated learning with differential privacy" into "machine learning".
  - Do not broaden "graph neural networks for traffic forecasting" into "neural networks".
- Use standard technical synonyms and abbreviations, but do not invent datasets, benchmarks, metrics, systems, or algorithms.
- Prefer arXiv and Semantic Scholar for AI/CS topics; Crossref and OpenAlex are useful for published proceedings and journals. Recommend PubMed only for biomedical or health-related AI.
- search_query must be English, concise, and under 350 characters.
- pubmed_query must be English and under 600 characters.
- Do not include Chinese characters in search_query or pubmed_query.
- Before returning, mentally validate that the entire response is exactly one JSON object and that every string value is properly escaped.
- If you cannot produce a safe computer-science rewrite, return empty strings for the queries and explain briefly in rationale.
""".strip() + "\n\n" + LLM_CHANNEL_QUERY_SCHEMA_PROMPT
ENGINEERING_QUERY_REWRITE_SYSTEM_PROMPT = """
You are an engineering scholarly search query planner.
Your job is to rewrite a user's research topic into precise English search queries for engineering and applied-technology academic databases. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_query": "single concise query for general academic search APIs such as Crossref, OpenAlex, Semantic Scholar, arXiv, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty unless the topic has a clear biomedical, clinical, environmental-health, or safety-health component",
  "query_intent": "title|author|author+title|citation|topic|method_task|abstract",
  "intent_confidence": 0.0,
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  },
  "core_concepts": [
    {
      "concept": "English concept",
      "type": "system|material|process|method|design|control|optimization|performance|application|sector|condition|other",
      "must_keep": true
    }
  ],
  "synonyms": ["standard engineering synonym, abbreviation, spelling variant, or related applied-technology term"],
  "forbidden_broadenings": ["overbroad concept that must not replace the user intent"],
  "recommended_sources": ["crossref", "openalex", "semantic", "arxiv", "cnki", "pubmed"],
  "avoid_sources": ["pubmed"],
  "rationale": "brief explanation of the rewrite"
}
Rules:
- Preserve the user's exact engineering intent, including system, material, process, design method, control strategy, optimization target, performance metric, application sector, operating condition, and constraints when specified.
- Classify query_intent before writing queries. Title, author+title, citation, and abstract inputs need different search behavior than broad topics.
- If the input contains both a paper title and author names, use query_intent author+title even when it looks citation-like.
- For title intent, keep the exact title words in search_query and do not replace the title with only a topic.
- For author+title or citation inputs, put the clean paper title in extracted.title and author names in extracted.authors; do not merge them into an exact-title query.
- For author intent, keep the author name in search_query. For citation intent, preserve author, year, title, and venue details when present.
- For abstract intent, extract core system/material, method, performance target, application sector, and operating condition before creating concise queries.
- Treat mechanical, electrical, civil, chemical, materials, energy, manufacturing, transportation, aerospace, robotics hardware, and industrial engineering as engineering topics.
- For Chinese engineering topics, translate the full phrase into precise English engineering terms. Keep CNKI useful by allowing the caller to use the original Chinese query for CNKI.
- Do not broaden a specific topic into a generic one. For example:
  - Do not broaden "lithium-ion battery thermal runaway prediction" into "battery management".
  - Do not broaden "wind turbine blade fault diagnosis using vibration signals" into "renewable energy".
  - Do not broaden "self-healing concrete crack repair" into "construction materials".
- Use standard engineering synonyms and abbreviations, but do not invent materials, standards, datasets, devices, or test conditions.
- Prefer Crossref, OpenAlex, and Semantic Scholar for engineering topics. Recommend arXiv for control, robotics, optimization, signal processing, computational engineering, and model-heavy topics.
- PubMed should be empty or avoided unless the topic is clearly biomedical engineering, clinical devices, occupational health, environmental health, or safety-health.
- search_query must be English, concise, and under 350 characters.
- pubmed_query must be English and under 600 characters.
- Do not include Chinese characters in search_query or pubmed_query.
- Before returning, mentally validate that the entire response is exactly one JSON object and that every string value is properly escaped.
- If you cannot produce a safe engineering rewrite, return empty strings for the queries and explain briefly in rationale.
""".strip() + "\n\n" + LLM_CHANNEL_QUERY_SCHEMA_PROMPT
SEARCH_MODE_CLASSIFIER_SYSTEM_PROMPT = """
You classify a user's literature-search topic into exactly one search domain.
Return only valid JSON, no markdown.
Schema:
{
  "search_mode": "biomedical|computer|engineering|society",
  "confidence": 0.0,
  "rationale": "brief explanation"
}
Definitions:
- biomedical: clinical medicine, biology, disease, anatomy, patients, medical imaging, drugs, public health, biomedical AI.
- computer: computer science, AI, machine learning, deep learning, NLP, computer vision, remote sensing image analysis, software, systems, databases, security, HCI.
- engineering: mechanical/electrical/civil/chemical/materials/energy/manufacturing/transportation/aerospace/control/applied industrial systems.
- society: social science, education, labor, policy, governance, economics, sociology, communication, psychology when not clinical.
Rules:
- Use the full topic, not isolated words.
- If a topic is AI applied to non-medical imagery, remote sensing, documents, software, or general data, classify as computer.
- If a topic is AI applied to clinical images, patients, disease, or biomedical data, classify as biomedical.
- If uncertain, choose the domain that best determines the databases and query-writing style.
""".strip()
UNIFIED_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a scholarly literature search planner.
Your job is to infer the search domain and rewrite the user's input into precise database queries in one step. You are not answering the research question, summarizing papers, or judging final paper inclusion.
Return only valid JSON, no markdown.
Schema:
{
  "search_mode": "biomedical|computer|engineering|society",
  "search_mode_confidence": 0.0,
  "query_intent": "title|author|author+title|citation|topic|method_task|abstract",
  "intent_confidence": 0.0,
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  },
  "search_query": "single concise query for general academic search APIs such as arXiv, Crossref, OpenAlex, Semantic Scholar, and CNKI",
  "pubmed_query": "PubMed-ready English Boolean query, or empty unless biomedical/health-related",
  "core_concepts": [
    {
      "concept": "English concept",
      "type": "condition|anatomy|modality|task|method|model|population|phenomenon|system|material|policy|other",
      "must_keep": true
    }
  ],
  "synonyms": ["standard synonym, abbreviation, spelling variant, or closely equivalent scholarly term"],
  "forbidden_broadenings": ["overbroad concept that must not replace the user intent"],
  "recommended_sources": ["pubmed", "crossref", "openalex", "semantic", "arxiv", "cnki"],
  "avoid_sources": ["pubmed"],
  "rationale": "brief explanation of the domain and rewrite"
}
Rules:
- Infer search_mode from the full input and use it to choose the query style.
- Classify query_intent before writing queries. Title, author+title, citation, and abstract inputs need different search behavior than broad topics.
- Preserve exact paper titles, author/title fragments, citation strings, and stable bibliographic details. Do not turn an exact title into only a broad topic.
- If the input is a paper title such as "Attention Is All You Need", keep the title words in search_query.
- If the input contains both a paper title and author names, use query_intent author+title even when it looks citation-like; keep author names in extracted.authors, not in the exact-title query.
- For author intent, keep the author name in search_query. For citation intent, preserve author, year, title, and venue details when present.
- For abstract intent, extract the core concepts and create concise topic/method/task queries instead of searching the whole paragraph.
- For biomedical topics, produce a PubMed query when useful; for non-biomedical topics, pubmed_query should usually be empty.
- For Chinese topics, translate the full phrase into precise English scholarly terms while preserving the exact intent.
- Do not broaden a specific topic into a generic one.
- search_query must be English, concise, and under 350 characters.
- pubmed_query must be English and under 600 characters.
- Return every required key.
""".strip() + "\n\n" + LLM_CHANNEL_QUERY_SCHEMA_PROMPT
QUERY_INTENT_CLASSIFIER_SYSTEM_PROMPT = """
You classify a user's literature-search input into multiple simultaneous intents.
Return only valid JSON, no markdown.
Schema:
{
  "title": 0.0,
  "author": 0.0,
  "citation": 0.0,
  "topic": 0.0,
  "method_task": 0.0,
  "abstract": 0.0,
  "rationale": "brief explanation",
  "extracted": {
    "title": "paper title if present, else empty",
    "authors": ["author names if present"],
    "year": "year if present, else empty",
    "venue": "venue if present, else empty",
    "method_terms": ["methods/models explicitly present"],
    "task_terms": ["tasks/applications explicitly present"]
  }
}
Definitions:
- title: the user appears to be looking for a specific paper by title or near-title.
- author: the input is an author name or has an author-name constraint.
- citation: the input resembles a bibliographic citation, author-year-title string, or reference line.
- topic: the input asks for a set of papers about a research area.
- method_task: the input is primarily methods/models/tasks such as nnU-Net stroke segmentation.
- abstract: the input is long text, an abstract, claim list, or innovation description.
Rules:
- Scores are independent probabilities from 0.0 to 1.0, not a single-label distribution.
- Do not convert a famous exact paper title into a topic just because you know its research area.
- For exact or near-exact paper titles such as "Attention Is All You Need", title must be high.
- For author-only inputs, author must be high and title should be low unless title words are present.
- For long abstracts or innovation paragraphs, abstract must be high.
- Return every required key.
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

COMPUTER_QUERY_EXPANSIONS = [
    {"terms": ["ai", "artificial intelligence", "\u4eba\u5de5\u667a\u80fd"], "expansions": ["artificial intelligence", "AI"]},
    {"terms": ["machine learning", "ml", "\u673a\u5668\u5b66\u4e60"], "expansions": ["machine learning", "ML"]},
    {"terms": ["deep learning", "neural network", "\u6df1\u5ea6\u5b66\u4e60", "\u795e\u7ecf\u7f51\u7edc"], "expansions": ["deep learning", "neural networks"]},
    {"terms": ["large language model", "llm", "\u5927\u8bed\u8a00\u6a21\u578b"], "expansions": ["large language model", "LLM"]},
    {"terms": ["retrieval augmented generation", "rag", "\u68c0\u7d22\u589e\u5f3a"], "expansions": ["retrieval augmented generation", "RAG"]},
    {"terms": ["computer vision", "image", "\u8ba1\u7b97\u673a\u89c6\u89c9", "\u56fe\u50cf"], "expansions": ["computer vision", "image analysis"]},
    {"terms": ["remote sensing", "\u9065\u611f"], "expansions": ["remote sensing"]},
    {"terms": ["change detection", "\u53d8\u5316\u68c0\u6d4b"], "expansions": ["change detection"]},
    {"terms": ["siamese network", "siamese neural network", "\u5b6a\u751f\u7f51\u7edc"], "expansions": ["siamese network", "siamese neural network"]},
    {"terms": ["nlp", "natural language processing", "\u81ea\u7136\u8bed\u8a00\u5904\u7406"], "expansions": ["natural language processing", "NLP"]},
    {"terms": ["security", "privacy", "\u5b89\u5168", "\u9690\u79c1"], "expansions": ["security", "privacy"]},
]

ENGINEERING_QUERY_EXPANSIONS = [
    {"terms": ["battery", "lithium", "\u7535\u6c60", "\u9502\u7535"], "expansions": ["battery", "lithium-ion battery"]},
    {"terms": ["control", "controller", "\u63a7\u5236"], "expansions": ["control", "controller"]},
    {"terms": ["optimization", "optimisation", "\u4f18\u5316"], "expansions": ["optimization", "optimisation"]},
    {"terms": ["fault diagnosis", "fault detection", "\u6545\u969c\u8bca\u65ad", "\u6545\u969c\u68c0\u6d4b"], "expansions": ["fault diagnosis", "fault detection"]},
    {"terms": ["material", "materials", "\u6750\u6599"], "expansions": ["materials", "material properties"]},
    {"terms": ["manufacturing", "machining", "\u5236\u9020", "\u52a0\u5de5"], "expansions": ["manufacturing", "machining"]},
    {"terms": ["energy", "renewable", "\u80fd\u6e90", "\u53ef\u518d\u751f\u80fd\u6e90"], "expansions": ["energy", "renewable energy"]},
    {"terms": ["structural", "concrete", "\u7ed3\u6784", "\u6df7\u51dd\u571f"], "expansions": ["structural engineering", "concrete"]},
]

COMPUTER_TOPIC_TERMS = [
    "computer science", "artificial intelligence", "ai", "machine learning", "deep learning",
    "neural network", "large language model", "llm", "retrieval augmented generation", "rag",
    "natural language processing", "nlp", "computer vision", "image analysis", "data mining", "algorithm",
    "remote sensing", "change detection", "siamese network", "siamese neural network",
    "software engineering", "cybersecurity", "database", "distributed system", "operating system",
    "human-computer interaction", "\u8ba1\u7b97\u673a", "\u4eba\u5de5\u667a\u80fd", "\u673a\u5668\u5b66\u4e60",
    "\u6df1\u5ea6\u5b66\u4e60", "\u795e\u7ecf\u7f51\u7edc", "\u5927\u8bed\u8a00\u6a21\u578b",
    "\u68c0\u7d22\u589e\u5f3a", "\u81ea\u7136\u8bed\u8a00\u5904\u7406", "\u8ba1\u7b97\u673a\u89c6\u89c9",
    "\u56fe\u50cf", "\u56fe\u50cf\u5206\u6790", "\u9065\u611f", "\u53d8\u5316\u68c0\u6d4b",
    "\u5b6a\u751f\u7f51\u7edc", "\u76ee\u6807\u68c0\u6d4b", "\u8bed\u4e49\u5206\u5272",
    "\u7b97\u6cd5", "\u8f6f\u4ef6\u5de5\u7a0b", "\u7f51\u7edc\u5b89\u5168", "\u6570\u636e\u5e93",
]

ENGINEERING_TOPIC_TERMS = [
    "engineering", "mechanical", "electrical", "civil engineering", "chemical engineering",
    "materials", "manufacturing", "robotics", "control system", "battery", "power system",
    "renewable energy", "fault diagnosis", "structural", "concrete", "aerospace",
    "transportation", "thermal", "\u5de5\u7a0b", "\u673a\u68b0", "\u7535\u6c14", "\u571f\u6728",
    "\u5316\u5de5", "\u6750\u6599", "\u5236\u9020", "\u673a\u5668\u4eba", "\u63a7\u5236\u7cfb\u7edf",
    "\u7535\u6c60", "\u7535\u529b\u7cfb\u7edf", "\u80fd\u6e90", "\u6545\u969c\u8bca\u65ad",
    "\u7ed3\u6784", "\u6df7\u51dd\u571f", "\u822a\u7a7a\u822a\u5929", "\u4ea4\u901a", "\u70ed",
]


class PaperSearchError(RuntimeError):
    pass


class LLMQueryRewriteParseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def expand_academic_query(query: str, *, search_mode: str = "auto") -> str:
    active_mode = normalize_search_mode(search_mode, query)
    if active_mode == "society":
        return expand_society_query_terms(query)
    if active_mode == "computer":
        return expand_domain_query_terms(query, COMPUTER_QUERY_EXPANSIONS)
    if active_mode == "engineering":
        return expand_domain_query_terms(query, ENGINEERING_QUERY_EXPANSIONS)
    return expand_query_terms(query)


def expand_society_query_terms(query: str) -> str:
    return expand_domain_query_terms(query, SOCIETY_QUERY_EXPANSIONS)


def expand_domain_query_terms(query: str, expansions: list[dict]) -> str:
    text = str(query or "").strip()
    terms: list[str] = []
    lower = text.casefold()
    for concept in expansions:
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
        "computer_science": "computer",
        "computer-science": "computer",
        "cs": "computer",
        "ai": "computer",
        "artificial_intelligence": "computer",
        "artificial-intelligence": "computer",
        "eng": "engineering",
    }
    mode = aliases.get(mode, mode)
    if mode in {"biomedical", "society", "computer", "engineering"}:
        return mode
    if mode not in {"", "auto"}:
        return "auto"
    if looks_like_clinical_topic(query):
        return "biomedical"
    if looks_like_computer_topic(query):
        return "computer"
    if looks_like_engineering_topic(query):
        return "engineering"
    if looks_like_society_topic(query):
        return "society"
    return "biomedical"


def infer_search_mode(query: str, search_mode: str | None = "auto", *, allow_llm: bool = True) -> dict:
    requested = str(search_mode or "auto").strip().casefold() or "auto"
    explicit_mode = normalize_explicit_search_mode(requested)
    if explicit_mode in {"biomedical", "society", "computer", "engineering"}:
        return {
            "search_mode": explicit_mode,
            "mode_inference_status": "manual",
            "mode_inference_error": "",
            "mode_inference_rationale": "",
            "mode_inference_confidence": "",
        }
    if requested not in {"", "auto"}:
        return {
            "search_mode": normalize_search_mode("auto", query),
            "mode_inference_status": f"rules_fallback:invalid_requested_mode:{requested[:40]}",
            "mode_inference_error": "",
            "mode_inference_rationale": "",
            "mode_inference_confidence": "",
        }

    if allow_llm and should_use_llm_search_mode_inference(query):
        try:
            decision = asyncio.run(classify_search_mode_with_llm(query))
            llm_mode = normalize_llm_search_mode(decision.get("search_mode"))
            if llm_mode:
                return {
                    "search_mode": llm_mode,
                    "mode_inference_status": "llm",
                    "mode_inference_error": "",
                    "mode_inference_rationale": clean_text(decision.get("rationale"))[:500],
                    "mode_inference_confidence": str(bounded_confidence(decision.get("confidence"))),
                }
            error = f"invalid_mode:{clean_text(decision.get('search_mode'))[:80]}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        fallback_mode = normalize_search_mode("auto", query)
        return {
            "search_mode": fallback_mode,
            "mode_inference_status": "rules_fallback:llm",
            "mode_inference_error": error,
            "mode_inference_rationale": "",
            "mode_inference_confidence": "",
        }

    return {
        "search_mode": normalize_search_mode("auto", query),
        "mode_inference_status": "rules",
        "mode_inference_error": "",
        "mode_inference_rationale": "",
        "mode_inference_confidence": "",
    }


def normalize_explicit_search_mode(search_mode: str | None) -> str:
    mode = str(search_mode or "").strip().casefold()
    aliases = {
        "social": "society",
        "social_science": "society",
        "social-science": "society",
        "soc": "society",
        "medical": "biomedical",
        "medicine": "biomedical",
        "bio": "biomedical",
        "computer_science": "computer",
        "computer-science": "computer",
        "cs": "computer",
        "ai": "computer",
        "artificial_intelligence": "computer",
        "artificial-intelligence": "computer",
        "eng": "engineering",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"biomedical", "society", "computer", "engineering"} else ""


def build_academic_search_plan(query: str, requested_sources: list[str], *, search_mode: str = "auto") -> dict:
    requested_mode = str(search_mode or "auto").strip().casefold() or "auto"
    explicit_mode = normalize_explicit_search_mode(requested_mode)
    use_unified_llm_plan = (
        explicit_mode == ""
        and requested_mode in {"", "auto"}
        and should_use_llm_query_rewrite(query, search_mode="auto")
    )
    mode_decision = infer_search_mode(query, search_mode, allow_llm=not use_unified_llm_plan)
    active_mode = mode_decision["search_mode"]
    fallback_query = expand_academic_query(query, search_mode=active_mode)
    plan = {
        "search_mode": active_mode,
        "requested_search_mode": str(search_mode or "auto").strip().casefold() or "auto",
        "mode_inference_status": mode_decision.get("mode_inference_status", ""),
        "mode_inference_error": mode_decision.get("mode_inference_error", ""),
        "mode_inference_rationale": mode_decision.get("mode_inference_rationale", ""),
        "mode_inference_confidence": mode_decision.get("mode_inference_confidence", ""),
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
        "query_intent": "",
        "intent_confidence": "",
        "intent_scores": {},
        "intent": {},
        "extracted": {},
        "channel_queries": {},
        "must_match_concepts": [],
        "do_not_mix": [],
    }
    plan["queries_by_source"] = build_queries_by_source(query, plan, requested_sources)
    if not should_use_llm_query_rewrite(query, search_mode=active_mode):
        return plan
    try:
        try:
            llm_plan = asyncio.run(
                build_academic_search_plan_with_llm(
                    query,
                    requested_sources,
                    search_mode="auto" if use_unified_llm_plan else active_mode,
                )
            )
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
    plan["must_match_concepts"] = clean_string_list(llm_plan.get("must_match_concepts"), limit=12) if isinstance(llm_plan, dict) else []
    plan["do_not_mix"] = clean_string_list(llm_plan.get("do_not_mix"), limit=12) if isinstance(llm_plan, dict) else []
    if isinstance(llm_plan, dict):
        plan["extracted"] = normalize_extracted_fields(llm_plan.get("extracted"))
        normalization_probe = dict(llm_plan)
        normalization_probe["extracted"] = plan["extracted"]
        normalization_probe["backend_query"] = plan.get("backend_query", "")
        normalization_probe["must_match_concepts"] = plan["must_match_concepts"]
        plan["channel_queries"] = normalize_channel_queries(normalization_probe, query)
    llm_mode = normalize_llm_search_mode(llm_plan.get("search_mode")) if isinstance(llm_plan, dict) else ""
    if use_unified_llm_plan and llm_mode:
        active_mode = llm_mode
        plan["search_mode"] = llm_mode
        plan["mode_inference_status"] = "llm"
        plan["mode_inference_error"] = ""
        plan["mode_inference_rationale"] = clean_text(llm_plan.get("rationale"))[:500]
        confidence = llm_plan.get("search_mode_confidence", llm_plan.get("confidence"))
        plan["mode_inference_confidence"] = str(bounded_confidence(confidence))
    elif use_unified_llm_plan and not llm_mode:
        plan["mode_inference_status"] = "rules_fallback:llm_missing_search_mode"
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
    planner_intent = normalize_planner_intent(llm_plan, query)
    if planner_intent:
        plan["intent"] = planner_intent
        plan["intent_scores"] = planner_intent["scores"]
        plan["query_intent"] = planner_intent["template"] if planner_intent.get("template") in BIBLIOGRAPHIC_INTENTS else planner_intent["top_intent"]
        plan["intent_confidence"] = str(planner_intent["confidence"])
        plan["extracted"] = merge_extracted_fields(plan.get("extracted"), planner_intent.get("extracted"))
        identity = normalize_bibliographic_identity(plan, query)
        if identity.get("query_intent") in BIBLIOGRAPHIC_INTENTS:
            plan["query_intent"] = identity["query_intent"]
        plan["bibliographic_identity"] = identity
    plan["queries_by_source"] = build_queries_by_source(query, plan, plan["sources"])
    return plan


async def build_academic_search_plan_with_llm(query: str, requested_sources: list[str], *, search_mode: str = "biomedical") -> dict:
    timeout = bounded_float_env("PAPER_SEARCH_QUERY_REWRITE_TIMEOUT_SECONDS", 20.0, minimum=3.0, maximum=60.0)
    raw_mode = str(search_mode or "auto").strip().casefold()
    unified_mode = raw_mode in {"", "auto"}
    active_mode = normalize_search_mode(search_mode, query)
    if unified_mode:
        system_prompt = UNIFIED_QUERY_REWRITE_SYSTEM_PROMPT
        planner_label = "scholarly"
    else:
        system_prompt = {
            "society": SOCIETY_QUERY_REWRITE_SYSTEM_PROMPT,
            "computer": COMPUTER_QUERY_REWRITE_SYSTEM_PROMPT,
            "engineering": ENGINEERING_QUERY_REWRITE_SYSTEM_PROMPT,
        }.get(active_mode, LLM_QUERY_REWRITE_SYSTEM_PROMPT)
        planner_label = {
            "society": "social-science",
            "computer": "computer-science or AI",
            "engineering": "engineering",
        }.get(active_mode, "biomedical")
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

    intent_label = planner_intent_label(plan)
    if intent_label == "citation" and planner_extracted_title(plan) and planner_extracted_authors(plan):
        intent_label = "citation_with_title_author"
    extracted = normalize_extracted_fields(plan.get("extracted"))
    explicit_channels = clean_channel_query_map(plan.get("channel_queries"))
    if intent_label in {"title", "author+title", "citation_with_title_author"}:
        title = planner_extracted_title(plan) or clean_text(explicit_channels.get("exact_title"))
        if intent_label == "title" and not title:
            return "title_missing"
        exact_title_query = planner_channel_query(plan, "exact_title") or search_query
        if intent_label in {"author+title", "citation_with_title_author"} and not exact_title_query:
            return "title_missing"
        if exact_title_query and title_match_score(title, exact_title_query) < 0.5:
            return "title_query_broadened"
        if intent_label == "author+title" and exact_title_contains_author(exact_title_query, planner_extracted_authors(plan)):
            return "exact_title_contains_author"
        citation_year = first_year(query)
        citation_probe = planner_channel_query(plan, "citation") or search_query
        if (
            intent_label == "citation_with_title_author"
            and citation_year
            and citation_year not in citation_probe
            and not planner_stable_identifier_query(plan)
        ):
            return "citation_year_dropped"
    if intent_label == "author":
        author_probe = " ".join(planner_extracted_authors(plan)) or author_channel_query(query) or query
        author_query = planner_channel_query(plan, "author") or search_query
        if explicit_channels.get("exact_title") and not planner_extracted_title(plan):
            return "author_exact_title_channel_forbidden"
        if not planner_extracted_authors(plan) and not author_query:
            return "author_missing"
        if author_probe and author_query and author_match_score(author_probe, author_query) < 0.5:
            return "author_query_missing_author"
        if author_query_too_long(author_query):
            return "author_query_too_long"
    if intent_label == "citation":
        stable_identifier_query = planner_stable_identifier_query(plan)
        citation_year = planner_extracted_year(plan) or first_year(query)
        citation_query = planner_channel_query(plan, "citation") or search_query
        if stable_identifier_query and not text_contains_identifier(citation_query, stable_identifier_query):
            return "citation_identifier_dropped"
        if citation_year and citation_year not in citation_query and not stable_identifier_query:
            return "citation_year_dropped"
        citation_title = planner_extracted_title(plan)
        if citation_title and citation_query and title_match_score(citation_title, citation_query) < 0.35 and not stable_identifier_query:
            return "citation_title_dropped"
        if citation_topic_channel_mixes_reference(explicit_channels.get("topic"), query):
            return "citation_topic_channel_mixed"
    if intent_label == "citation_with_title_author":
        stable_identifier_query = planner_stable_identifier_query(plan)
        citation_query = planner_channel_query(plan, "citation") or search_query
        if stable_identifier_query and not text_contains_identifier(citation_query, stable_identifier_query):
            return "citation_identifier_dropped"
        if citation_topic_channel_mixes_reference(explicit_channels.get("topic"), query):
            return "citation_topic_channel_mixed"
    if intent_label == "method_task":
        method_error = method_task_channel_guardrail_error(plan)
        if method_error:
            return method_error
    if planner_intent_label(plan) == "abstract":
        if query_features(query).get("is_long_text"):
            abstract_query = planner_channel_query(plan, "abstract_claim") or search_query
            if len(abstract_query) > 220 or len(query_keyword_terms(abstract_query)) > 24:
                return "abstract_query_too_long"
            core_terms = abstract_core_terms(plan, extracted)
            if not core_terms:
                return "abstract_missing_core_concepts"
            if abstract_query and not text_matches_any_term(abstract_query, core_terms):
                return "abstract_query_missing_core_concepts"

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


def planner_intent_label(plan: dict) -> str:
    if not isinstance(plan, dict):
        return ""
    label = normalize_intent_label(plan.get("query_intent") or plan.get("intent"))
    if label:
        return label
    scores = {intent: bounded_confidence(plan.get(intent)) for intent in INTENT_LABELS}
    if any(scores.values()):
        return max(scores, key=scores.get)
    return ""


def normalize_intent_label(value) -> str:
    label = clean_text(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "author_title": "author+title",
        "title_author": "author+title",
        "title+author": "author+title",
        "paper_title_author": "author+title",
        "author_and_title": "author+title",
        "title_and_author": "author+title",
        "citation_title_author": "citation_with_title_author",
        "citation_with_author_title": "citation_with_title_author",
        "paper_title": "title",
        "exact_title": "title",
        "near_title": "title",
        "author_name": "author",
        "reference": "citation",
        "bibliographic": "citation",
        "bibliographic_citation": "citation",
        "research_topic": "topic",
        "topic_search": "topic",
        "method": "method_task",
        "task": "method_task",
        "method_and_task": "method_task",
        "long_text": "abstract",
        "claim": "abstract",
        "innovation": "abstract",
    }
    label = aliases.get(label, label)
    if label == "author_title":
        label = "author+title"
    return label if label in set(INTENT_LABELS) | {"author+title", "citation_with_title_author"} else ""


def allowed_channels_for_intent(intent) -> tuple[str, ...]:
    label = normalize_intent_label(intent)
    contract = INTENT_SEARCH_CONTRACTS.get(label) or INTENT_SEARCH_CONTRACTS["topic"]
    return tuple(contract["allowed_channels"])


def search_contract_for_intent(intent) -> dict:
    label = normalize_intent_label(intent)
    return INTENT_SEARCH_CONTRACTS.get(label) or INTENT_SEARCH_CONTRACTS["topic"]


def planner_extracted_title(plan: dict) -> str:
    extracted = plan.get("extracted") if isinstance(plan, dict) and isinstance(plan.get("extracted"), dict) else {}
    return clean_text(extracted.get("title"))[:300]


def planner_extracted_authors(plan: dict) -> list[str]:
    extracted = plan.get("extracted") if isinstance(plan, dict) and isinstance(plan.get("extracted"), dict) else {}
    return clean_string_list(extracted.get("authors"), limit=8)


def planner_extracted_year(plan: dict) -> str:
    extracted = plan.get("extracted") if isinstance(plan, dict) and isinstance(plan.get("extracted"), dict) else {}
    return first_year(clean_text(extracted.get("year")))


def planner_stable_identifier_query(plan: dict) -> str:
    if not isinstance(plan, dict):
        return ""
    extracted = normalize_extracted_fields(plan.get("extracted"))
    return stable_identifier_channel_query(extracted.get("identifiers") or {})


def planner_channel_query(plan: dict, channel: str) -> str:
    if not isinstance(plan, dict):
        return ""
    queries = normalize_channel_queries(plan, "")
    return queries.get(channel, "")


def exact_title_contains_author(exact_title_query: str, authors: list[str]) -> bool:
    text = normalize_text_for_match(exact_title_query)
    if not text:
        return False
    for author in authors or []:
        author_text = normalize_text_for_match(author)
        if not author_text:
            continue
        parts = [part for part in author_text.split() if len(part) > 1]
        if author_text in text or (parts and parts[-1] in text and len(parts[-1]) >= 4):
            return True
    return False


def author_query_too_long(author_query: str) -> bool:
    if not author_query:
        return False
    return len(author_query) > 160 or len(query_keyword_terms(author_query)) > 12


def text_contains_identifier(text: str, identifier_query: str) -> bool:
    haystack = normalize_identifier_text(text)
    needle = normalize_identifier_text(identifier_query)
    return bool(needle and needle in haystack)


def normalize_identifier_text(value: str) -> str:
    text = clean_text(value).casefold()
    text = re.sub(r"\b(?:doi|pmid|pubmed|arxiv)\b[:\s]*", "", text)
    text = re.sub(r"https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/", "", text)
    text = re.sub(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", text)
    text = re.sub(r"https?://arxiv\.org/(?:abs|pdf)/", "", text)
    return re.sub(r"[^a-z0-9./-]+", "", text).strip("/")


def citation_topic_channel_mixes_reference(topic_query: str, original_query: str) -> bool:
    topic = clean_text(topic_query)
    if not topic:
        return False
    if normalize_text_for_match(topic) == normalize_text_for_match(original_query):
        return True
    return bool(first_year(topic) and re.search(r"\bet al\.?\b|[,;]", topic, flags=re.IGNORECASE))


def method_task_channel_guardrail_error(plan: dict) -> str:
    extracted = normalize_extracted_fields(plan.get("extracted") if isinstance(plan, dict) else {})
    method_terms = clean_string_list(extracted.get("method_terms"), limit=8)
    task_terms = clean_string_list(extracted.get("task_terms"), limit=8)
    if not (method_terms and task_terms):
        return ""
    method_query = planner_channel_query(plan, "method_task") or clean_search_query(plan.get("search_query"), max_length=350)
    if not method_query:
        return "method_task_query_missing"
    has_method = text_matches_any_term(method_query, method_terms)
    has_task = text_matches_any_term(method_query, task_terms)
    if not has_method:
        return "method_task_query_missing_method"
    if not has_task:
        return "method_task_query_missing_task"
    return ""


def abstract_core_terms(plan: dict, extracted: dict | None = None) -> list[str]:
    extracted = extracted if isinstance(extracted, dict) else normalize_extracted_fields(plan.get("extracted") if isinstance(plan, dict) else {})
    terms = clean_string_list(plan.get("core_concepts"), limit=8) if isinstance(plan, dict) else []
    terms.extend(clean_string_list(plan.get("must_match_concepts"), limit=8) if isinstance(plan, dict) else [])
    terms.extend(clean_string_list(extracted.get("method_terms"), limit=8))
    terms.extend(clean_string_list(extracted.get("task_terms"), limit=8))
    terms.extend(clean_string_list(extracted.get("domain_terms"), limit=8))
    return list(dict.fromkeys(term for term in terms if term))


def text_matches_any_term(text: str, terms: list[str]) -> bool:
    normalized = normalize_text_for_match(text)
    if not normalized:
        return False
    for term in terms or []:
        term_text = normalize_text_for_match(term)
        if term_text and (term_text in normalized or group_matches(normalized, [term_text])):
            return True
    return False


def normalize_bibliographic_identity(search_plan: dict, query: str) -> dict:
    plan = search_plan if isinstance(search_plan, dict) else {}
    intent = plan.get("intent") if isinstance(plan.get("intent"), dict) else {}
    extracted = {}
    if isinstance(intent.get("extracted"), dict):
        extracted.update(intent.get("extracted") or {})
    if isinstance(plan.get("extracted"), dict):
        extracted.update(plan.get("extracted") or {})

    authors = clean_string_list(extracted.get("authors"), limit=8)
    title = clean_text(extracted.get("title"))[:300]
    year = first_year(clean_text(extracted.get("year")))
    venue = clean_text(extracted.get("venue"))[:200]
    label = normalize_intent_label(plan.get("query_intent"))
    if not label:
        label = normalize_intent_label(intent.get("template") or intent.get("top_intent"))
    template = normalize_intent_label(intent.get("template")) or label
    scores = intent.get("scores") if isinstance(intent.get("scores"), dict) else {}

    if label == "citation" and title and authors:
        label = "citation_with_title_author"
        template = "author+title"
    elif title and authors and label in {"title", "citation", ""}:
        label = "author+title"
        template = "author+title"

    if not title and template in {"title", "author+title", "citation_with_title_author"}:
        title = clean_title_probe_from_query(query, authors=authors, year=year)
    if not authors and label in {"author", "author+title", "citation", "citation_with_title_author"}:
        authors = query_author_names(query)
    if not year:
        year = first_year(query)

    title_confidence = bounded_confidence(extracted.get("title_confidence"))
    if not title_confidence:
        title_confidence = max(bounded_confidence(scores.get("title")), 0.85 if title else 0.0)
    author_confidence = bounded_confidence(extracted.get("author_confidence"))
    if not author_confidence:
        author_confidence = max(bounded_confidence(scores.get("author")), 0.85 if authors else 0.0)
    year_confidence = bounded_confidence(extracted.get("year_confidence")) or (0.9 if year else 0.0)
    venue_confidence = bounded_confidence(extracted.get("venue_confidence")) or (0.8 if venue else 0.0)

    return {
        "query_intent": label,
        "template": template,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "title_confidence": round(title_confidence, 4),
        "author_confidence": round(author_confidence, 4),
        "year_confidence": round(year_confidence, 4),
        "venue_confidence": round(venue_confidence, 4),
    }


def bibliographic_identity_intent(search_plan: dict, query: str = "") -> bool:
    identity = normalize_bibliographic_identity(search_plan, query)
    return identity.get("query_intent") in BIBLIOGRAPHIC_INTENTS and bool(identity.get("title") or identity.get("authors"))


def clean_title_probe_from_query(query: str, *, authors: list[str] | None = None, year: str = "") -> str:
    text = clean_text(query)
    if not text:
        return ""
    for author in authors or []:
        if author:
            text = re.sub(r"\b" + re.escape(author) + r"\b", " ", text, flags=re.IGNORECASE)
            parts = author.split()
            if len(parts) >= 2:
                text = re.sub(r"\b" + re.escape(parts[-1]) + r"\b", " ", text, flags=re.IGNORECASE)
    if year:
        text = re.sub(r"\b" + re.escape(year) + r"\b", " ", text)
    text = re.sub(r"\b(et\s+al\.?|and)\b|[,;:()\[\]{}]", " ", text, flags=re.IGNORECASE)
    text = clean_text(text)
    return text[:300] if should_run_exact_title_search(text) else ""


def query_author_names(query: str) -> list[str]:
    text = clean_text(query)
    names = re.findall(r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3}\b", text)
    return list(dict.fromkeys(name for name in names if name))[:8]


def normalize_planner_intent(plan: dict, query: str) -> dict:
    payload = planner_intent_payload(plan)
    if not payload:
        return {}
    rules_intent = predict_query_intent_rules(query)
    try:
        intent = normalize_llm_intent(payload, query, rules_intent, source="llm_planner")
    except Exception:
        return {}
    if float(intent.get("confidence") or 0.0) < planner_intent_min_confidence():
        return {}
    rules_confidence = float(rules_intent.get("confidence") or 0.0)
    if (
        intent.get("top_intent") != rules_intent.get("top_intent")
        and float(intent.get("confidence") or 0.0) < 0.65
        and rules_confidence >= 0.75
    ):
        return {}
    return intent


def planner_intent_min_confidence() -> float:
    return bounded_float_env("PAPER_SEARCH_PLANNER_INTENT_MIN_CONFIDENCE", 0.55, minimum=0.0, maximum=1.0)


def normalized_intent(value) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("scores"), dict)
        and normalize_intent_label(value.get("top_intent"))
        and "confidence" in value
    )


def planner_intent_payload(plan: dict) -> dict:
    if not isinstance(plan, dict):
        return {}
    if any(bounded_confidence(plan.get(label)) > 0 for label in INTENT_LABELS):
        payload = dict(plan)
        payload.setdefault("rationale", plan.get("intent_rationale") or plan.get("rationale") or "")
        return payload
    label = planner_intent_label(plan)
    if not label:
        return {}
    confidence = bounded_confidence(
        plan.get("intent_confidence")
        if plan.get("intent_confidence") is not None
        else plan.get("query_intent_confidence", plan.get("confidence"))
    )
    if confidence <= 0:
        confidence = 0.7
    payload = {intent: 0.0 for intent in INTENT_LABELS}
    if label == "author+title":
        payload["title"] = confidence
        payload["author"] = confidence
    elif label == "citation_with_title_author":
        payload["citation"] = confidence
        payload["title"] = max(payload["title"], confidence)
        payload["author"] = max(payload["author"], confidence)
    elif label in payload:
        payload[label] = confidence
    payload["rationale"] = plan.get("intent_rationale") or plan.get("rationale") or ""
    payload["query_intent"] = label
    payload["extracted"] = plan.get("extracted") if isinstance(plan.get("extracted"), dict) else {}
    payload["_raw_response"] = plan.get("_raw_response", "")
    return payload


def should_use_llm_search_mode_inference(query: str) -> bool:
    mode = str(os.getenv("PAPER_SEARCH_MODE_INFERENCE", "auto") or "").strip().casefold()
    if mode in {"0", "false", "off", "disabled", "rules"}:
        return False
    if not str(query or "").strip():
        return False
    has_model = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
    if mode in {"1", "true", "on", "enabled", "llm"}:
        return has_model
    return has_model


async def classify_search_mode_with_llm(query: str) -> dict:
    timeout = bounded_float_env("PAPER_SEARCH_MODE_INFERENCE_TIMEOUT_SECONDS", 8.0, minimum=2.0, maximum=30.0)
    content = await asyncio.wait_for(
        LLMClient().complete(
            system_prompt=SEARCH_MODE_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=f"User topic:\n{query}",
            model=os.getenv("PAPER_SEARCH_MODE_INFERENCE_MODEL")
            or os.getenv("SEARCH_QUERY_REWRITE_MODEL")
            or os.getenv("RESEARCH_MODEL"),
            temperature=0.0,
            max_tokens=300,
        ),
        timeout=timeout,
    )
    data = parse_json_object(content)
    return data if isinstance(data, dict) else {}


def normalize_llm_search_mode(value) -> str:
    return normalize_explicit_search_mode(value)


def bounded_confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def should_use_llm_query_rewrite(query: str, *, search_mode: str = "auto") -> bool:
    mode = str(os.getenv("PAPER_SEARCH_QUERY_REWRITE", "auto") or "").strip().casefold()
    if mode in {"0", "false", "off", "disabled", "rules"}:
        return False
    if mode in {"1", "true", "on", "enabled", "llm"}:
        return True
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")):
        return False
    active_mode = normalize_search_mode(search_mode, query)
    if active_mode in {"society", "computer", "engineering"}:
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


def looks_like_computer_topic(query: str) -> bool:
    text = str(query or "")
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in COMPUTER_TOPIC_TERMS)


def looks_like_engineering_topic(query: str) -> bool:
    text = str(query or "")
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in ENGINEERING_TOPIC_TERMS)


def best_llm_backend_query(plan: dict) -> str:
    value = clean_search_query(plan.get("search_query"), max_length=350)
    if value:
        return value
    value = clean_search_query(plan.get("pubmed_query"), max_length=600)
    if value:
        return value
    channel_queries = normalize_channel_queries(plan, "")
    for key in ("topic", "method_task", "abstract_claim", "exact_title", "author", "citation"):
        if channel_queries.get(key):
            return channel_queries[key]
    concepts = clean_string_list(plan.get("core_concepts"), limit=8)
    return " AND ".join(concepts[:4]) if concepts else ""


def build_queries_by_source(query: str, plan: dict, sources: list[str]) -> dict[str, str]:
    general_query = planner_contract_channel_query(plan, "topic")
    if not general_query:
        general_query = clean_search_query(plan.get("llm_search_query"), max_length=350)
    if not general_query:
        general_query = clean_search_query(plan.get("backend_query"), max_length=700)
    if not general_query:
        general_query = clean_search_query(plan.get("rules_fallback_query"), max_length=700)
    if not general_query:
        general_query = expand_academic_query(query, search_mode=plan.get("search_mode", "auto"))

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


def queries_by_source_for_channel(channel_name: str, query: str, plan: dict, sources: list[str]) -> dict[str, str]:
    channel = clean_text(channel_name)
    topic_queries = build_queries_by_source(query, plan, sources)
    channel_query = channel_query_for_recall(channel, query, plan)
    if channel == "topic" or not channel_query:
        return topic_queries
    queries = {}
    for source in sources:
        if source == "pubmed":
            queries[source] = pubmed_query_for_channel(channel, query, plan, channel_query, topic_queries.get(source, ""))
        elif source == "cnki" and re.search(r"[\u4e00-\u9fff]", str(query or "")) and channel in {"topic", "method_task", "abstract_claim"}:
            queries[source] = clean_text(query)
        else:
            queries[source] = source_specific_channel_query(source, channel, channel_query)
    return queries


def source_specific_channel_query(source: str, channel: str, channel_query: str) -> str:
    if clean_text(channel) != "author":
        return channel_query
    author = clean_fielded_author_query(channel_query)
    if not author:
        return channel_query
    source_name = normalize_source_name(source)
    escaped = author.replace('"', " ")
    if source_name == "arxiv":
        terms = [term for term in re.findall(r"[A-Za-z][A-Za-z'.-]*", escaped) if len(term) > 1]
        return " AND ".join(f"au:{term}" for term in terms[:4]) or f"au:{escaped}"
    if source_name in {"crossref", "openalex"}:
        return f'author:"{escaped}"'
    return author


def parse_fielded_author_query(query: str) -> str:
    match = AUTHOR_FIELD_QUERY_RE.match(clean_text(query))
    if not match:
        return ""
    return clean_fielded_author_query(match.group(1) or match.group(2) or "")


def is_author_field_query(query: str) -> bool:
    text = clean_text(query)
    if parse_fielded_author_query(text):
        return True
    parts = [part.strip() for part in re.split(r"\s+AND\s+", text, flags=re.IGNORECASE) if part.strip()]
    return bool(parts) and all(re.fullmatch(r"au:[A-Za-z][A-Za-z'.-]*", part, flags=re.IGNORECASE) for part in parts)


def clean_fielded_author_query(query: str) -> str:
    text = clean_search_query(query, max_length=160)
    text = re.sub(r'^(?:au|author):', "", text, flags=re.IGNORECASE).strip()
    text = text.strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text if text and not re.search(r"\b(?:AND|OR|NOT)\b|[()[\]{}?]", text) else ""


def channel_query_for_recall(channel_name: str, query: str, plan: dict) -> str:
    channel = clean_text(channel_name)
    normalized = normalize_channel_queries(plan, query)
    extracted = normalize_extracted_fields(plan.get("extracted") if isinstance(plan, dict) else {})
    if channel == "exact_title":
        return clean_search_query(normalized.get("exact_title"), max_length=300) or clean_search_query(extracted.get("title"), max_length=300)
    if channel == "fuzzy_title":
        return fuzzy_title_query(clean_text(extracted.get("title")) or query)
    if channel == "author":
        author_query = clean_search_query(normalized.get("author"), max_length=300)
        if author_query:
            return author_query
        authors = clean_string_list(extracted.get("authors"), limit=8)
        if not authors and isinstance(plan, dict) and isinstance(plan.get("bibliographic_identity"), dict):
            authors = clean_string_list(plan["bibliographic_identity"].get("authors"), limit=8)
        return clean_search_query(" ".join(authors[:3]), max_length=300)
    if channel == "citation":
        identifier_query = stable_identifier_channel_query(extracted.get("identifiers") or {})
        if identifier_query:
            return identifier_query
        citation_query = clean_search_query(normalized.get("citation"), max_length=450)
        if citation_query:
            return citation_query
        parts = []
        parts.extend(clean_string_list(extracted.get("authors"), limit=2))
        for key in ("year", "title", "venue"):
            value = clean_text(extracted.get(key))
            if value:
                parts.append(value)
        return clean_search_query(" ".join(parts), max_length=450)
    if channel == "method_task":
        method_query = clean_search_query(normalized.get("method_task"), max_length=350)
        method_terms = clean_string_list(extracted.get("method_terms"), limit=8)
        task_terms = clean_string_list(extracted.get("task_terms"), limit=8)
        if method_terms and task_terms and text_matches_any_term(method_query, method_terms) and text_matches_any_term(method_query, task_terms):
            return method_query
        fallback = clean_search_query(" ".join(dict.fromkeys([*method_terms, *task_terms])), max_length=350)
        return fallback or method_query
    if channel == "abstract_claim":
        abstract_query = clean_search_query(normalized.get("abstract_claim"), max_length=220)
        if abstract_query and len(query_keyword_terms(abstract_query)) <= 24:
            return abstract_query
        terms = abstract_core_terms(plan)
        return clean_search_query(" ".join(dict.fromkeys(terms[:8])), max_length=220)
    if channel == "topic":
        return clean_search_query(normalized.get("topic"), max_length=350) or first_nonempty_query(build_queries_by_source(query, plan, ["_"]), ["_"])
    return ""


def pubmed_query_for_channel(channel_name: str, query: str, plan: dict, channel_query: str, topic_pubmed_query: str) -> str:
    if channel_name in {"topic", "method_task", "abstract_claim"}:
        pubmed_query = clean_search_query(plan.get("llm_pubmed_query"), max_length=600)
        if pubmed_query:
            return pubmed_query
        if normalize_search_mode(plan.get("search_mode"), query) == "biomedical":
            return build_pubmed_fallback_query(query, channel_query)
        return topic_pubmed_query or channel_query
    return channel_query


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


def clean_channel_query_map(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    limits = {
        "exact_title": 300,
        "author": 300,
        "citation": 450,
        "topic": 350,
        "method_task": 350,
        "abstract_claim": 350,
    }
    normalized = {}
    for key in CHANNEL_QUERY_KEYS:
        query = clean_search_query(value.get(key), max_length=limits[key])
        if query:
            normalized[key] = query
    return normalized


def normalize_channel_queries(plan: dict, query: str) -> dict[str, str]:
    if not isinstance(plan, dict):
        return {}
    normalized = clean_channel_query_map(plan.get("channel_queries"))
    extracted = normalize_extracted_fields(plan.get("extracted"))
    identifiers = extracted.get("identifiers") if isinstance(extracted.get("identifiers"), dict) else {}

    title = clean_search_query(extracted.get("title"), max_length=300)
    if title and "exact_title" not in normalized:
        normalized["exact_title"] = title

    authors = clean_string_list(extracted.get("authors"), limit=8)
    if authors and "author" not in normalized:
        normalized["author"] = clean_search_query(" ".join(authors[:3]), max_length=300)

    if "citation" not in normalized:
        citation_query = stable_identifier_channel_query(identifiers)
        if not citation_query:
            citation_parts = [*authors[:2]]
            year = clean_text(extracted.get("year"))
            if year:
                citation_parts.append(year)
            if title:
                citation_parts.append(title)
            venue = clean_text(extracted.get("venue"))
            if venue:
                citation_parts.append(venue)
            citation_query = clean_search_query(" ".join(citation_parts), max_length=450)
        if citation_query:
            normalized["citation"] = citation_query

    if "topic" not in normalized:
        topic_query = clean_search_query(plan.get("search_query"), max_length=350)
        if not topic_query:
            topic_query = clean_search_query(plan.get("backend_query"), max_length=350)
        if not topic_query:
            topic_query = clean_search_query(plan.get("llm_search_query"), max_length=350)
        if not topic_query:
            topic_query = clean_search_query(plan.get("rules_fallback_query"), max_length=350)
        if topic_query:
            normalized["topic"] = topic_query

    if "method_task" not in normalized:
        method_terms = clean_string_list(extracted.get("method_terms"), limit=8)
        task_terms = clean_string_list(extracted.get("task_terms"), limit=8)
        method_query = clean_search_query(" ".join(dict.fromkeys([*method_terms, *task_terms])), max_length=350)
        if method_query:
            normalized["method_task"] = method_query

    if "abstract_claim" not in normalized:
        abstract_terms = clean_string_list(plan.get("core_concepts"), limit=8)
        abstract_terms.extend(clean_string_list(plan.get("must_match_concepts"), limit=8))
        abstract_query = clean_search_query(" ".join(dict.fromkeys(abstract_terms)), max_length=350)
        if abstract_query:
            normalized["abstract_claim"] = abstract_query

    return {key: value for key, value in normalized.items() if key in CHANNEL_QUERY_KEYS and value}


def stable_identifier_channel_query(identifiers: dict) -> str:
    if not isinstance(identifiers, dict):
        return ""
    doi = clean_text(identifiers.get("doi"))
    if doi:
        return clean_search_query(doi, max_length=200)
    pmid = clean_text(identifiers.get("pmid"))
    if pmid:
        return clean_search_query(f"PMID {pmid}", max_length=80)
    arxiv_id = clean_text(identifiers.get("arxiv_id"))
    if arxiv_id:
        return clean_search_query(f"arXiv {arxiv_id}", max_length=120)
    return ""


def normalize_extracted_fields(value) -> dict:
    if not isinstance(value, dict):
        value = {}
    identifiers = normalize_extracted_identifiers(value.get("identifiers"), value)
    extracted = {
        "title": clean_text(value.get("title"))[:300],
        "authors": clean_string_list(value.get("authors"), limit=8),
        "year": first_year(clean_text(value.get("year"))),
        "venue": clean_text(value.get("venue"))[:200],
        "method_terms": clean_string_list(value.get("method_terms"), limit=8),
        "task_terms": clean_string_list(value.get("task_terms"), limit=8),
        "domain_terms": clean_string_list(value.get("domain_terms"), limit=8),
        "identifiers": identifiers,
    }
    for key in ("title_confidence", "author_confidence", "year_confidence", "venue_confidence"):
        confidence = bounded_confidence(value.get(key))
        if confidence:
            extracted[key] = confidence
    return extracted


def normalize_extracted_identifiers(value, fallback: dict | None = None) -> dict[str, str]:
    data = value if isinstance(value, dict) else {}
    fallback = fallback if isinstance(fallback, dict) else {}
    doi_probe = {
        "doi": clean_text(data.get("doi") or fallback.get("doi")),
        "source": clean_text(data.get("doi") or fallback.get("doi")),
    }
    pmid_probe = {
        "pmid": clean_text(data.get("pmid") or fallback.get("pmid")),
        "source": clean_text(data.get("pmid") or fallback.get("pmid")),
    }
    arxiv_probe = {
        "arxiv_id": clean_text(data.get("arxiv_id") or fallback.get("arxiv_id")),
        "source": clean_text(data.get("arxiv_id") or fallback.get("arxiv_id")),
    }
    return {
        "doi": extract_stable_doi(doi_probe),
        "pmid": extract_stable_pmid(pmid_probe),
        "arxiv_id": extract_stable_arxiv_id(arxiv_probe),
    }


def merge_extracted_fields(base, override) -> dict:
    merged = normalize_extracted_fields(base) if isinstance(base, dict) else {}
    incoming = normalize_extracted_fields(override) if isinstance(override, dict) else {}
    for key, value in incoming.items():
        if key == "identifiers":
            identifiers = dict(merged.get("identifiers") or {})
            for identifier_key, identifier_value in (value or {}).items():
                if identifier_value:
                    identifiers[identifier_key] = identifier_value
            merged["identifiers"] = identifiers
        elif isinstance(value, list):
            existing = merged.get(key) if isinstance(merged.get(key), list) else []
            merged[key] = list(dict.fromkeys([*existing, *value]))
        elif value not in ("", None, 0.0):
            merged[key] = value
    return merged


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


def detect_stable_identifier(query: str) -> dict:
    text = clean_text(query)
    probe = {"title": text, "source": text, "relevance": text, "bibliographic_identity": text}
    doi = extract_stable_doi(probe)
    pmid = extract_stable_pmid(probe)
    arxiv_id = extract_stable_arxiv_id(probe)
    url = extract_stable_url(probe)
    if doi:
        return {"type": "doi", "value": doi, "raw": text}
    if pmid:
        return {"type": "pmid", "value": pmid, "raw": text}
    if arxiv_id:
        return {"type": "arxiv", "value": arxiv_id, "raw": text}
    if url:
        url_probe = {"source": url, "title": url, "relevance": url, "bibliographic_identity": url}
        if doi := extract_stable_doi(url_probe):
            return {"type": "doi", "value": doi, "raw": text, "url": url}
        if pmid := extract_stable_pmid(url_probe):
            return {"type": "pmid", "value": pmid, "raw": text, "url": url}
        if arxiv_id := extract_stable_arxiv_id(url_probe):
            return {"type": "arxiv", "value": arxiv_id, "raw": text, "url": url}
        return {"type": "url", "value": url, "raw": text}
    return {}


def stable_identifier_search_result(identifier: dict, source_names: list[str], *, search_mode: str, requested_search_mode: str) -> dict:
    id_type = str(identifier.get("type") or "")
    value = str(identifier.get("value") or "").strip()
    metadata: dict = {}
    risks: list[str] = []
    resolution_status = ""
    verified_by = ""
    if id_type == "doi":
        metadata = fetch_stable_crossref_metadata(value)
        resolution_status = doi_resolution_status(value)
        verified_by = "Crossref" if metadata else ""
        if resolution_status == "failed":
            risks.append("doi_resolution_failed")
        elif resolution_status == "unknown":
            risks.append("doi_resolution_unchecked")
    elif id_type == "pmid":
        metadata = fetch_stable_pubmed_metadata(value)
        verified_by = "PubMed" if metadata else ""
    elif id_type == "arxiv":
        metadata = fetch_stable_arxiv_metadata(value)
        verified_by = "arXiv" if metadata else ""
    elif id_type == "url":
        metadata = fetch_stable_webpage_metadata(value)
        verified_by = "webpage_metadata" if metadata else ""

    paper = normalize_stable_identifier_paper(identifier, metadata, risks, verified_by=verified_by)
    source = normalize_source_name(paper.get("retrieved_from"))
    source_results = {name: 0 for name in source_names}
    if source:
        source_results[source] = 1
    plan = {
        "search_mode": search_mode,
        "requested_search_mode": requested_search_mode,
        "stable_identifier": identifier,
        "identifier_short_circuit": True,
        "backend_query": "",
        "rules_fallback_query": "",
        "sources": source_names,
        "queries_by_source": {},
        "rewrite_status": "stable_identifier",
        "ranking_weights": {},
        "intent_scores": {},
        "rationale": "Stable identifier detected; skipped topic recall and weighted ranking.",
    }
    return {
        "query": str(identifier.get("raw") or value),
        "search_mode": search_mode,
        "requested_search_mode": requested_search_mode,
        "mode_inference_status": "stable_identifier",
        "mode_inference_error": "",
        "mode_inference_rationale": "Stable identifier short-circuit.",
        "mode_inference_confidence": "1.0",
        "backend_query": "",
        "rules_fallback_query": "",
        "llm_search_query": "",
        "llm_pubmed_query": "",
        "llm_raw_response": "",
        "llm_error": "",
        "query_rewrite_status": "stable_identifier",
        "query_plan": plan,
        "queries_by_source": {},
        "sources_used": source_names,
        "source_results": source_results,
        "exact_title_source_results": {},
        "channel_results": {},
        "errors": {},
        "raw_count": 1 if paper else 0,
        "papers": [paper] if paper else [],
    }


def normalize_stable_identifier_paper(identifier: dict, metadata: dict, risks: list[str], *, verified_by: str) -> dict:
    id_type = str(identifier.get("type") or "")
    value = str(identifier.get("value") or "").strip()
    item = dict(metadata or {})
    if id_type == "doi":
        item.setdefault("doi", value)
        item.setdefault("source", f"https://doi.org/{normalize_doi(value)}")
        source = "crossref" if metadata else "doi"
    elif id_type == "pmid":
        item.setdefault("pmid", value)
        item.setdefault("source", f"https://pubmed.ncbi.nlm.nih.gov/{value}/")
        source = "pubmed"
    elif id_type == "arxiv":
        item.setdefault("arxiv_id", value)
        item.setdefault("source", f"https://arxiv.org/abs/{value}")
        source = "arxiv"
    else:
        item.setdefault("source", value)
        source = "web"
    item.setdefault("title", item.get("title") or f"{id_type.upper()}: {value}")
    item.setdefault("retrieved_from", source)
    item.setdefault("source_label", SOURCE_LABELS.get(source, source.title()))
    item.setdefault("source_origin", "stable_identifier_short_circuit")
    item["stable_identifier"] = {"type": id_type, "value": value}
    if verified_by:
        item["verification_status"] = "verified"
        item["verification_sources"] = ["stable_identifier", verified_by]
    elif id_type == "doi" and "doi_resolution_failed" in risks:
        item["verification_status"] = "needs_review"
        item["verification_sources"] = ["stable_identifier"]
    else:
        item["verification_status"] = "partial" if id_type == "url" else "unverified"
        item["verification_sources"] = ["stable_identifier"]
    item["verification_risks"] = list(dict.fromkeys(risks))
    item["selection_reasons"] = ["stable_identifier_short_circuit"]
    item["candidate_score"] = 1.0 if verified_by and not risks else 0.0
    return normalize_paper(item, default_source=source) | {
        "verification_status": item["verification_status"],
        "verification_sources": item["verification_sources"],
        "verification_risks": item["verification_risks"],
        "selection_reasons": item["selection_reasons"],
        "candidate_score": item["candidate_score"],
        "stable_identifier": item["stable_identifier"],
    }


def predict_query_intent(query: str, *, search_mode: str = "auto") -> dict:
    rules_intent = predict_query_intent_rules(query)
    if not should_use_llm_intent_prediction(query):
        return rules_intent
    try:
        llm_intent = asyncio.run(classify_query_intent_with_llm(query, search_mode=search_mode))
        return normalize_llm_intent(llm_intent, query, rules_intent)
    except Exception as error:
        fallback = dict(rules_intent)
        fallback["intent_source"] = "rules_fallback:llm"
        fallback["intent_error"] = f"{type(error).__name__}: {str(error)[:300]}"
        return fallback


def predict_query_intent_rules(query: str) -> dict:
    features = query_features(query)
    scores = {label: 0.0 for label in INTENT_LABELS}
    token_count = features["token_count"]
    if features["has_citation_pattern"]:
        scores["citation"] += 0.75
        scores["title"] += 0.25
        scores["author"] += 0.25
    if features["has_person_name"]:
        scores["author"] += 0.55
        if features["has_year"] or token_count >= 3:
            scores["citation"] += 0.2
    if features["title_like"]:
        scores["title"] += 0.65
    if features["has_method_terms"] or features["has_task_terms"]:
        scores["method_task"] += 0.55
        scores["topic"] += 0.25
    if features["has_domain_terms"] or re.search(r"[\u4e00-\u9fff]", query):
        scores["topic"] += 0.55
    if features["is_long_text"]:
        scores["abstract"] += 0.8
        scores["topic"] += 0.15
    if features["is_question_like"]:
        scores["topic"] += 0.35
        scores["title"] -= 0.15
    if token_count <= 2 and features["has_person_name"]:
        scores["author"] += 0.25
    if token_count <= 2 and not features["has_person_name"]:
        scores["topic"] += 0.2
    if features["has_year"] and scores["author"] > 0.3:
        scores["citation"] += 0.2
    scores = {key: round(max(0.0, min(1.0, value)), 4) for key, value in scores.items()}
    top_intent = max(scores, key=scores.get)
    template = top_intent
    if scores["author"] >= 0.45 and scores["title"] >= 0.45:
        template = "author+title"
    return {
        "scores": scores,
        "top_intent": top_intent,
        "template": template,
        "confidence": scores[top_intent],
        "features": features,
        "intent_source": "rules",
        "intent_rationale": "",
        "intent_error": "",
        "extracted": {},
    }


def should_use_llm_intent_prediction(query: str) -> bool:
    mode = str(os.getenv("PAPER_SEARCH_INTENT_PREDICTION", "auto") or "").strip().casefold()
    if mode in {"0", "false", "off", "disabled", "rules"}:
        return False
    if not clean_text(query):
        return False
    has_model = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
    if mode in {"1", "true", "on", "enabled", "llm"}:
        return has_model
    return has_model


async def classify_query_intent_with_llm(query: str, *, search_mode: str = "auto") -> dict:
    timeout = bounded_float_env("PAPER_SEARCH_INTENT_TIMEOUT_SECONDS", 8.0, minimum=2.0, maximum=30.0)
    active_mode = normalize_search_mode(search_mode, query)
    content = await asyncio.wait_for(
        LLMClient().complete(
            system_prompt=QUERY_INTENT_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=(
                f"Search domain: {active_mode}\n"
                f"User input:\n{query}"
            ),
            model=os.getenv("PAPER_SEARCH_INTENT_MODEL")
            or os.getenv("PAPER_SEARCH_MODE_INFERENCE_MODEL")
            or os.getenv("SEARCH_QUERY_REWRITE_MODEL")
            or os.getenv("RESEARCH_MODEL"),
            temperature=0.0,
            max_tokens=500,
        ),
        timeout=timeout,
    )
    data = parse_json_object(content)
    if isinstance(data, dict):
        data["_raw_response"] = content
    return data if isinstance(data, dict) else {}


def normalize_llm_intent(plan: dict, query: str, rules_intent: dict, *, source: str = "llm") -> dict:
    if not isinstance(plan, dict):
        raise ValueError("invalid_intent_plan")
    scores = {}
    for label in INTENT_LABELS:
        scores[label] = round(bounded_confidence(plan.get(label)), 4)
    if not any(scores.values()):
        raise ValueError("empty_intent_scores")
    explicit_label = normalize_intent_label(plan.get("query_intent") or plan.get("intent"))
    top_intent = max(scores, key=scores.get)
    template = explicit_label if explicit_label in {"author+title", "citation_with_title_author"} else top_intent
    if scores["author"] >= 0.45 and scores["title"] >= 0.45:
        template = "author+title"
    extracted = normalize_extracted_fields(plan.get("extracted"))
    extracted_title = clean_text(extracted.get("title"))[:300]
    extracted_authors = clean_string_list(extracted.get("authors"), limit=8)
    if explicit_label == "citation" and extracted_title and extracted_authors:
        template = "citation_with_title_author"
    if explicit_label == "author+title":
        scores["title"] = max(scores.get("title", 0.0), bounded_confidence(plan.get("intent_confidence")), 0.7)
        scores["author"] = max(scores.get("author", 0.0), bounded_confidence(plan.get("intent_confidence")), 0.7)
        top_intent = "title" if scores["title"] >= scores["author"] else "author"
        template = "author+title"
    features = dict(rules_intent.get("features") or query_features(query))
    for key in ("method_terms", "task_terms", "domain_terms"):
        values = extracted.get(key)
        if isinstance(values, list):
            merged = list(features.get(key) or [])
            merged.extend(clean_text(value) for value in values if clean_text(value))
            features[key] = list(dict.fromkeys(merged))
            features[f"has_{key[:-6]}_terms"] = bool(features[key])
    return {
        "scores": scores,
        "top_intent": top_intent,
        "template": template,
        "confidence": scores[top_intent],
        "features": features,
        "intent_source": source,
        "intent_rationale": clean_text(plan.get("rationale"))[:500],
        "intent_error": "",
        "intent_raw_response": clean_text(plan.get("_raw_response"))[:2000],
        "extracted": {
            "title": extracted_title,
            "authors": extracted_authors,
            "year": first_year(clean_text(extracted.get("year"))),
            "venue": clean_text(extracted.get("venue"))[:200],
            "method_terms": clean_string_list(extracted.get("method_terms"), limit=8),
            "task_terms": clean_string_list(extracted.get("task_terms"), limit=8),
            "domain_terms": clean_string_list(extracted.get("domain_terms"), limit=8),
            "identifiers": normalize_extracted_identifiers(extracted.get("identifiers")),
            "title_confidence": bounded_confidence(extracted.get("title_confidence") or plan.get("title_confidence")),
            "author_confidence": bounded_confidence(extracted.get("author_confidence") or plan.get("author_confidence")),
            "year_confidence": bounded_confidence(extracted.get("year_confidence") or plan.get("year_confidence")),
            "venue_confidence": bounded_confidence(extracted.get("venue_confidence") or plan.get("venue_confidence")),
        },
    }


def query_features(query: str) -> dict:
    text = clean_text(query)
    words = re.findall(r"[A-Za-z][A-Za-z0-9+.-]*|[\u4e00-\u9fff]+|\d{4}", text)
    lower = text.casefold()
    person_name = bool(re.fullmatch(r"[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3}", text))
    person_name = person_name or bool(re.search(r"\b[A-Z][A-Za-z'.-]+\s+(?:et al\.?|and)\b", text))
    citation_pattern = bool(re.search(r"\b(?:19|20)\d{2}\b", text) and ("," in text or person_name or "et al" in lower))
    method_terms = sorted(term for term in METHOD_TERMS if term.casefold() in lower)
    task_terms = sorted(term for term in TASK_TERMS if term.casefold() in lower)
    domain_terms = sorted(term for term in DOMAIN_TERMS if term.casefold() in lower)
    token_count = len(words)
    long_text = len(text) >= 360 or token_count >= 55
    title_like = 3 <= token_count <= 14 and not long_text and not text.endswith("?")
    if method_terms or task_terms or domain_terms or re.search(r"[\u4e00-\u9fff]", text):
        title_like = title_like and not (method_terms or task_terms or domain_terms)
    return {
        "token_count": token_count,
        "has_title_case": bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)),
        "title_like": title_like,
        "has_person_name": person_name,
        "has_year": bool(re.search(r"\b(?:19|20)\d{2}\b", text)),
        "has_citation_pattern": citation_pattern,
        "has_method_terms": bool(method_terms),
        "has_task_terms": bool(task_terms),
        "has_domain_terms": bool(domain_terms),
        "method_terms": method_terms,
        "task_terms": task_terms,
        "domain_terms": domain_terms,
        "is_long_text": long_text,
        "is_question_like": bool(text.endswith("?") or re.search(r"^(how|what|which|why|can|does|do)\b", text, flags=re.IGNORECASE)),
    }


def ranking_weights_for_intent(intent: dict) -> dict:
    template = intent.get("template") if isinstance(intent, dict) else ""
    confidence = float(intent.get("confidence") or 0.0) if isinstance(intent, dict) else 0.0
    features = intent.get("features") or {}
    if template in TEMPLATE_WEIGHTS and confidence >= 0.65:
        raw = dict(TEMPLATE_WEIGHTS[template])
        strategy = f"template:{template}"
    else:
        raw = dict(BASE_DYNAMIC_WEIGHTS)
        if features.get("has_title_case") or 3 <= int(features.get("token_count") or 0) <= 12:
            raw["title"] += 20
        if features.get("has_person_name"):
            raw["author"] += 25
        if features.get("has_year"):
            raw["year"] += 15
        if features.get("has_citation_pattern"):
            raw["title"] += 20
            raw["author"] += 20
            raw["year"] += 15
            raw["venue"] += 10
        if features.get("has_method_terms"):
            raw["method"] += 20
        if features.get("has_task_terms"):
            raw["task"] += 20
        if features.get("has_domain_terms"):
            raw["topic"] += 15
        if features.get("is_long_text"):
            raw["abstract"] += 25
            raw["title"] -= 10
            raw["author"] -= 5
        if features.get("is_question_like"):
            raw["topic"] += 20
            raw["title"] -= 10
        strategy = "dynamic"
    total = sum(max(0.0, float(raw.get(label, 0))) for label in RANK_SIGNAL_LABELS) or 1.0
    return {
        "strategy": strategy,
        "raw": {label: raw.get(label, 0) for label in RANK_SIGNAL_LABELS},
        "normalized": {label: round(max(0.0, float(raw.get(label, 0))) / total, 6) for label in RANK_SIGNAL_LABELS},
    }


def recall_multiplier() -> int:
    try:
        value = int(os.getenv("PAPER_SEARCH_RECALL_MULTIPLIER", "2"))
    except ValueError:
        return 2
    return max(1, min(value, 5))


def build_recall_channels(query: str, search_plan: dict, sources: list[str], *, max_internal_results: int) -> list[dict]:
    weights = (search_plan.get("ranking_weights") or {}).get("normalized") or {}
    budgets = recall_channel_budgets(weights, max_internal_results)
    channels: list[dict] = []
    topic_queries = dict(search_plan.get("queries_by_source") or build_queries_by_source(query, search_plan, sources))
    identity = normalize_bibliographic_identity(search_plan, query)
    contract = recall_channel_contract(search_plan, identity=identity)
    contract_intent = contract["intent"]
    allowed_channels = set(contract["allowed_channels"])
    title_query = identity.get("title") or query
    fuzzy_query = fuzzy_title_query(title_query)
    if search_plan.get("rewrite_status") == "llm":
        if "exact_title" in allowed_channels and should_add_exact_title_channel(title_query, search_plan, budgets):
            channels.append(
                {
                    "name": "exact_title",
                    "queries_by_source": queries_by_source_for_channel("exact_title", query, search_plan, sources),
                    "budget": max(1, budgets.get("exact_title", max_internal_results // 2 or 1)),
                    "exact": True,
                }
            )
        if (
            "fuzzy_title" in allowed_channels
            and fuzzy_query
            and identity.get("query_intent") in {"author+title", "citation", "citation_with_title_author"}
            and budgets.get("fuzzy_title", 0) > 0
        ):
            channels.append({"name": "fuzzy_title", "queries_by_source": queries_by_source_for_channel("fuzzy_title", query, search_plan, sources), "budget": budgets["fuzzy_title"]})
        author_query = planner_contract_channel_query(search_plan, "author", identity=identity)
        if "author" in allowed_channels and author_query and budgets.get("author", 0) > 0:
            author_budget = max_internal_results if contract_intent == "author" else budgets["author"]
            channels.append({"name": "author", "queries_by_source": queries_by_source_for_channel("author", query, search_plan, sources), "budget": author_budget})
        citation_query = planner_contract_channel_query(search_plan, "citation", identity=identity)
        if "citation" in allowed_channels and citation_query and budgets.get("citation", 0) > 0:
            channels.append({"name": "citation", "queries_by_source": queries_by_source_for_channel("citation", query, search_plan, sources), "budget": budgets["citation"]})
        if "topic" in allowed_channels and topic_queries:
            channels.append({"name": "topic", "queries_by_source": queries_by_source_for_channel("topic", query, search_plan, sources), "budget": max(1, budgets.get("topic", max_internal_results))})
        method_query = planner_contract_channel_query(search_plan, "method_task", identity=identity)
        if "method_task" in allowed_channels and method_query and budgets.get("method_task", 0) > 0:
            channels.append({"name": "method_task", "queries_by_source": queries_by_source_for_channel("method_task", query, search_plan, sources), "budget": budgets["method_task"]})
        abstract_query = planner_contract_channel_query(search_plan, "abstract_claim", identity=identity)
        if "abstract_claim" in allowed_channels and abstract_query and budgets.get("abstract", 0) > 0:
            channels.append({"name": "abstract_claim", "queries_by_source": queries_by_source_for_channel("abstract_claim", query, search_plan, sources), "budget": budgets["abstract"]})
    else:
        if "exact_title" in allowed_channels and should_run_exact_title_search(title_query) and budgets.get("exact_title", 0) > 0:
            channels.append({"name": "exact_title", "queries_by_source": queries_by_source_for_channel("exact_title", query, search_plan, sources), "budget": budgets["exact_title"], "exact": True})
        if "fuzzy_title" in allowed_channels and fuzzy_query and budgets.get("fuzzy_title", 0) > 0:
            channels.append({"name": "fuzzy_title", "queries_by_source": queries_by_source_for_channel("fuzzy_title", query, search_plan, sources), "budget": budgets["fuzzy_title"]})
        author_query = planner_contract_channel_query(search_plan, "author", identity=identity) or bibliographic_author_query(identity, query)
        if "author" in allowed_channels and author_query and budgets.get("author", 0) > 0:
            author_budget = max_internal_results if contract_intent == "author" else budgets["author"]
            channels.append({"name": "author", "queries_by_source": queries_by_source_for_channel("author", query, search_plan, sources), "budget": author_budget})
        citation_query = planner_contract_channel_query(search_plan, "citation", identity=identity) or bibliographic_citation_query(identity, query)
        if "citation" in allowed_channels and citation_query and budgets.get("citation", 0) > 0:
            channels.append({"name": "citation", "queries_by_source": queries_by_source_for_channel("citation", query, search_plan, sources), "budget": budgets["citation"]})
        if "topic" in allowed_channels and topic_queries and budgets.get("topic", 0) > 0:
            channels.append({"name": "topic", "queries_by_source": queries_by_source_for_channel("topic", query, search_plan, sources), "budget": budgets["topic"]})
        method_query = planner_contract_channel_query(search_plan, "method_task", identity=identity) or method_task_channel_query(query, search_plan)
        if "method_task" in allowed_channels and method_query and budgets.get("method_task", 0) > 0:
            channels.append({"name": "method_task", "queries_by_source": queries_by_source_for_channel("method_task", query, search_plan, sources), "budget": budgets["method_task"]})
        abstract_query = planner_contract_channel_query(search_plan, "abstract_claim", identity=identity) or abstract_claim_channel_query(query)
        if "abstract_claim" in allowed_channels and abstract_query and budgets.get("abstract", 0) > 0:
            channels.append({"name": "abstract_claim", "queries_by_source": queries_by_source_for_channel("abstract_claim", query, search_plan, sources), "budget": budgets["abstract"]})
    if not channels:
        channels.append(fallback_recall_channel(query, topic_queries, search_plan, contract_intent, sources, max_internal_results))
    return filter_recall_channels_by_contract(channels, query, topic_queries, search_plan, contract, sources, max_internal_results)


def recall_channel_contract(search_plan: dict, *, identity: dict | None = None) -> dict:
    intent = search_plan.get("intent") if isinstance(search_plan, dict) and isinstance(search_plan.get("intent"), dict) else {}
    label = routing_intent_for_search_plan(search_plan, identity=identity)
    confidence = float(intent.get("confidence") or 0.0) if normalized_intent(intent) else 0.0
    if not label:
        contract = search_contract_for_intent("topic")
        return {
            "intent": "topic",
            "raw_intent": "",
            "confidence": confidence,
            "allowed_channels": tuple(contract["allowed_channels"]),
            "forbid_channels": tuple(contract["forbid_channels"]),
            "fallback_channel": contract["fallback_channel"],
            "reason": "missing_intent_topic_fallback",
        }
    if confidence < planner_intent_min_confidence():
        contract = search_contract_for_intent("topic")
        return {
            "intent": "topic",
            "raw_intent": label,
            "confidence": confidence,
            "allowed_channels": tuple(contract["allowed_channels"]),
            "forbid_channels": tuple(channel for channel in CHANNEL_QUERY_KEYS if channel != "topic"),
            "fallback_channel": "topic",
            "reason": "low_confidence_topic_fallback",
        }
    contract = search_contract_for_intent(label)
    return {
        "intent": label,
        "raw_intent": label,
        "confidence": confidence,
        "allowed_channels": tuple(contract["allowed_channels"]),
        "forbid_channels": tuple(contract["forbid_channels"]),
        "fallback_channel": contract["fallback_channel"],
        "reason": "intent_contract",
    }


def filter_recall_channels_by_contract(
    channels: list[dict],
    query: str,
    topic_queries: dict,
    search_plan: dict,
    contract: dict,
    sources: list[str],
    max_internal_results: int,
) -> list[dict]:
    allowed = set(contract.get("allowed_channels") or ())
    forbidden = set(contract.get("forbid_channels") or ())
    reasons: list[str] = []
    filtered: list[dict] = []
    for channel in channels:
        name = clean_text(channel.get("name"))
        if not name:
            reasons.append("drop:missing_channel_name")
            continue
        if name in forbidden:
            reasons.append(f"drop:{name}:forbidden_for_{contract.get('intent', 'topic')}")
            continue
        if name not in allowed:
            reasons.append(f"drop:{name}:not_allowed_for_{contract.get('intent', 'topic')}")
            continue
        filtered.append(channel)
    if not filtered:
        fallback = fallback_recall_channel(
            query,
            topic_queries,
            search_plan,
            clean_text(contract.get("intent")) or "topic",
            sources,
            max_internal_results,
        )
        fallback_name = clean_text(fallback.get("name"))
        if fallback_name in forbidden or fallback_name not in allowed:
            fallback = {"name": "topic", "queries_by_source": topic_queries or {source: query for source in sources}, "budget": max_internal_results}
            fallback_name = "topic"
        filtered.append(fallback)
        reasons.append(f"fallback:{fallback_name}:{contract.get('reason', 'intent_contract')}")
    else:
        reasons.append(str(contract.get("reason") or "intent_contract"))

    search_plan["opened_channels"] = [clean_text(channel.get("name")) for channel in filtered if clean_text(channel.get("name"))]
    search_plan["forbidden_channels"] = sorted(forbidden)
    search_plan["channel_filter_reasons"] = reasons
    return filtered


def routing_intent_for_search_plan(search_plan: dict, *, identity: dict | None = None) -> str:
    identity = identity if isinstance(identity, dict) else {}
    label = normalize_intent_label(identity.get("query_intent"))
    if label:
        return label
    label = normalize_intent_label(search_plan.get("query_intent") if isinstance(search_plan, dict) else "")
    if label:
        return label
    intent = search_plan.get("intent") if isinstance(search_plan, dict) and isinstance(search_plan.get("intent"), dict) else {}
    return normalize_intent_label(intent.get("template") or intent.get("top_intent"))


def fallback_recall_channel(query: str, topic_queries: dict, search_plan: dict, intent: str, sources: list[str], budget: int) -> dict:
    contract = search_contract_for_intent(intent)
    channel = contract["fallback_channel"]
    if channel == "exact_title":
        return {"name": "exact_title", "queries_by_source": queries_by_source_for_channel("exact_title", query, search_plan, sources), "budget": budget, "exact": True}
    if channel == "author":
        return {"name": "author", "queries_by_source": queries_by_source_for_channel("author", query, search_plan, sources), "budget": budget}
    if channel == "citation":
        return {"name": "citation", "queries_by_source": queries_by_source_for_channel("citation", query, search_plan, sources), "budget": budget}
    if channel == "method_task":
        return {"name": "method_task", "queries_by_source": queries_by_source_for_channel("method_task", query, search_plan, sources), "budget": budget}
    if channel == "abstract_claim":
        return {"name": "abstract_claim", "queries_by_source": queries_by_source_for_channel("abstract_claim", query, search_plan, sources), "budget": budget}
    return {"name": "topic", "queries_by_source": topic_queries or queries_by_source_for_channel("topic", query, search_plan, sources), "budget": budget}


def planner_contract_channel_query(search_plan: dict, channel: str, *, identity: dict | None = None) -> str:
    query = planner_channel_query(search_plan, channel)
    if not query:
        return ""
    intent = search_plan.get("intent") if isinstance(search_plan.get("intent"), dict) else {}
    if not normalized_intent(intent):
        return ""
    if intent.get("intent_source") != "llm_planner":
        return ""
    confidence = float(intent.get("confidence") or 0.0)
    if confidence < planner_intent_min_confidence():
        return ""
    scores = intent.get("scores") if isinstance(intent.get("scores"), dict) else {}
    top_intent = normalize_intent_label(intent.get("top_intent"))
    template = normalize_intent_label(intent.get("template")) or top_intent
    identity = identity if isinstance(identity, dict) else normalize_bibliographic_identity(search_plan, "")
    bibliographic_intent = normalize_intent_label(identity.get("query_intent"))
    contract_intent = bibliographic_intent or template or top_intent
    if channel not in allowed_channels_for_intent(contract_intent):
        return ""
    if channel == "exact_title":
        return query if bibliographic_intent in BIBLIOGRAPHIC_INTENTS and identity.get("title") and should_run_exact_title_search(query) else ""
    if channel == "author":
        return query if bibliographic_intent in {"author", "author+title", "citation", "citation_with_title_author"} and identity.get("authors") else ""
    if channel == "citation":
        return query if bibliographic_intent in {"citation", "citation_with_title_author", "author+title"} else ""
    if channel == "method_task":
        return query if template == "method_task" or top_intent == "method_task" or float(scores.get("method_task") or 0.0) >= 0.55 else ""
    if channel == "abstract_claim":
        return query if template == "abstract" or top_intent == "abstract" or float(scores.get("abstract") or 0.0) >= 0.55 else ""
    if channel == "topic":
        return query if template == "topic" or top_intent == "topic" or float(scores.get("topic") or 0.0) >= 0.55 else ""
    return ""


def should_add_exact_title_channel(query: str, search_plan: dict, budgets: dict[str, int]) -> bool:
    if not should_run_exact_title_search(query):
        return False
    identity = normalize_bibliographic_identity(search_plan, query)
    if identity.get("query_intent") in BIBLIOGRAPHIC_INTENTS and identity.get("title"):
        return True
    intent = search_plan.get("intent") if isinstance(search_plan.get("intent"), dict) else {}
    scores = intent.get("scores") if isinstance(intent.get("scores"), dict) else {}
    template = str(intent.get("template") or "")
    title_score = float(scores.get("title") or 0.0)
    citation_score = float(scores.get("citation") or 0.0)
    if template in {"title", "citation", "author+title"}:
        return True
    if title_score >= 0.5 or citation_score >= 0.5:
        return True
    return budgets.get("exact_title", 0) > 0 and title_score >= 0.35


def recall_channel_budgets(weights: dict, max_internal_results: int) -> dict[str, int]:
    title_budget = float(weights.get("title", 0)) + 0.5 * float(weights.get("author", 0)) + 0.5 * float(weights.get("year", 0))
    author_budget = float(weights.get("author", 0)) + 0.3 * float(weights.get("title", 0))
    citation_budget = float(weights.get("year", 0)) + float(weights.get("venue", 0)) + 0.3 * float(weights.get("author", 0))
    topic_budget = float(weights.get("topic", 0)) + float(weights.get("method", 0)) + float(weights.get("task", 0))
    abstract_budget = float(weights.get("abstract", 0))
    raw = {
        "exact_title": max(0.0, title_budget * 0.55),
        "fuzzy_title": max(0.0, title_budget * 0.45),
        "author": max(0.0, author_budget),
        "citation": max(0.0, citation_budget),
        "topic": max(0.0, topic_budget),
        "method_task": max(0.0, topic_budget * 0.35),
        "abstract": max(0.0, abstract_budget),
    }
    total = sum(raw.values()) or 1.0
    return {
        name: max(1, min(max_internal_results, round(max_internal_results * value / total)))
        for name, value in raw.items()
        if value > 0.01
    }


def fuzzy_title_query(query: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", str(query or "")):
        return ""
    terms = query_keyword_terms(query)
    return " ".join(terms[:8]) if 3 <= len(terms) <= 12 else ""


def author_channel_query(query: str) -> str:
    text = clean_text(query)
    match = re.search(r"\b([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3})\b", text)
    if not match:
        return ""
    year = first_year(text)
    title_terms = [term for term in query_keyword_terms(text) if term not in {match.group(1).casefold(), year}]
    return clean_text(" ".join([match.group(1), year, " ".join(title_terms[:4])]))


def bibliographic_author_query(identity: dict, query: str) -> str:
    authors = clean_string_list(identity.get("authors"), limit=3) if isinstance(identity, dict) else []
    if not authors:
        return author_channel_query(query)
    title = clean_text(identity.get("title")) if isinstance(identity, dict) else ""
    year = clean_text(identity.get("year")) if isinstance(identity, dict) else ""
    title_terms = query_keyword_terms(title)
    return clean_text(" ".join([authors[0], year, " ".join(title_terms[:4])]))


def citation_channel_query(query: str) -> str:
    text = clean_text(query)
    if not (first_year(text) or re.search(r"\bet al\.?\b|,", text, flags=re.IGNORECASE)):
        return ""
    return re.sub(r"[,:;()\[\]{}]+", " ", text).strip()


def bibliographic_citation_query(identity: dict, query: str) -> str:
    if not isinstance(identity, dict) or identity.get("query_intent") not in {"citation", "citation_with_title_author", "author+title"}:
        return citation_channel_query(query)
    parts = []
    parts.extend(clean_string_list(identity.get("authors"), limit=2))
    for key in ("year", "title", "venue"):
        value = clean_text(identity.get(key))
        if value:
            parts.append(value)
    return clean_text(" ".join(parts)) or citation_channel_query(query)


def method_task_channel_query(query: str, search_plan: dict) -> str:
    features = (search_plan.get("intent") or {}).get("features") or query_features(query)
    terms = list(features.get("method_terms") or []) + list(features.get("task_terms") or []) + list(features.get("domain_terms") or [])
    concepts = clean_string_list(search_plan.get("core_concepts"), limit=6)
    return " ".join(dict.fromkeys([*terms, *concepts])) or ""


def abstract_claim_channel_query(query: str) -> str:
    features = query_features(query)
    if not features.get("is_long_text"):
        return ""
    terms = query_keyword_terms(query)
    return " ".join(terms[:12]) or clean_text(query)[:300]


def prepare_search_execution(
    clean_query: str,
    source_names: list[str],
    *,
    max_internal_results: int,
    search_mode: str,
) -> tuple[dict, list[str], str, dict[str, str], list[dict]]:
    search_plan = build_academic_search_plan(clean_query, source_names, search_mode=search_mode)
    backend_query = str(
        search_plan.get("backend_query")
        or expand_academic_query(clean_query, search_mode=search_plan.get("search_mode", "auto"))
    ).strip()
    source_names = list(search_plan.get("sources") or source_names)
    queries_by_source = build_queries_by_source(clean_query, search_plan, source_names)
    search_plan["queries_by_source"] = queries_by_source

    intent = search_plan.get("intent") if normalized_intent(search_plan.get("intent")) else predict_query_intent(
        clean_query,
        search_mode=search_plan.get("search_mode", search_mode),
    )
    search_plan["intent"] = intent
    search_plan["intent_scores"] = intent["scores"]
    if not isinstance(search_plan.get("extracted"), dict) or not search_plan.get("extracted"):
        search_plan["extracted"] = intent.get("extracted") or {}

    identity = normalize_bibliographic_identity(search_plan, clean_query)
    search_plan["bibliographic_identity"] = identity
    if identity.get("query_intent") in BIBLIOGRAPHIC_INTENTS:
        search_plan["query_intent"] = identity["query_intent"]
    elif not search_plan.get("query_intent"):
        search_plan["query_intent"] = intent.get("top_intent", "")
    if search_plan.get("query_intent") in BIBLIOGRAPHIC_INTENTS and intent.get("template") not in BIBLIOGRAPHIC_INTENTS:
        intent = {**intent, "template": "author+title" if identity.get("authors") and identity.get("title") else search_plan["query_intent"]}
        search_plan["intent"] = intent

    search_plan["ranking_weights"] = ranking_weights_for_intent(intent)
    search_plan["recall_multiplier"] = recall_multiplier()
    search_plan["max_internal_results_per_source"] = max_internal_results
    recall_channels = build_recall_channels(clean_query, search_plan, source_names, max_internal_results=max_internal_results)
    search_plan["recall_channels"] = [
        {"name": channel["name"], "budget": channel["budget"], "queries_by_source": channel.get("queries_by_source", {})}
        for channel in recall_channels
    ]
    return search_plan, source_names, backend_query, queries_by_source, recall_channels


def execute_recall_channels(
    recall_channels: list[dict],
    source_names: list[str],
    *,
    clean_query: str,
    max_internal_results: int,
    year: str,
    timeout_seconds: int,
) -> tuple[list[dict], dict[str, int], dict[str, str], dict[str, int], dict[str, dict[str, int]]]:
    papers: list[dict] = []
    source_results: dict[str, int] = {source: 0 for source in source_names}
    errors: dict[str, str] = {}
    exact_title_source_results: dict[str, int] = {}
    channel_results: dict[str, dict[str, int]] = {}

    for channel in recall_channels:
        channel_name = str(channel.get("name") or "topic")
        channel_budget = max(1, int(channel.get("budget") or max_internal_results))
        channel_queries = channel.get("queries_by_source") or {}
        if channel.get("exact"):
            exact_title_query = first_nonempty_query(channel_queries, source_names) or clean_query
            payload = run_exact_title_search_by_source(exact_title_query, source_names, max_results_per_source=channel_budget)
        else:
            payload = run_paper_search_backend_by_source(
                channel_queries,
                max_results_per_source=channel_budget,
                year=year,
                timeout_seconds=timeout_seconds,
            )

        channel_papers, channel_source_results, channel_errors = normalize_search_payload(payload, source_names)
        retrieval_query = first_nonempty_query(channel_queries, source_names)
        for paper in channel_papers:
            paper["retrieval_channel"] = channel_name
            paper["retrieval_query"] = retrieval_query
        papers.extend(channel_papers)
        channel_results[channel_name] = channel_source_results
        if channel_name == "exact_title":
            exact_title_source_results = channel_source_results
        for source, count in channel_source_results.items():
            source_results[source] = source_results.get(source, 0) + int(count or 0)
        for source, message in channel_errors.items():
            errors.setdefault(source, message)

    return papers, source_results, errors, exact_title_source_results, channel_results


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
    max_internal_results = max_results * recall_multiplier()
    mode_decision = infer_search_mode(clean_query, search_mode)
    stable_identifier = detect_stable_identifier(clean_query)
    if stable_identifier:
        return stable_identifier_search_result(
            stable_identifier,
            source_names,
            search_mode=mode_decision["search_mode"],
            requested_search_mode=str(search_mode or "auto").strip().casefold() or "auto",
        )

    search_plan, source_names, backend_query, queries_by_source, recall_channels = prepare_search_execution(
        clean_query,
        source_names,
        max_internal_results=max_internal_results,
        search_mode=search_mode,
    )
    papers, source_results, errors, exact_title_source_results, channel_results = execute_recall_channels(
        recall_channels,
        source_names,
        clean_query=clean_query,
        max_internal_results=max_internal_results,
        year=str(year or "").strip(),
        timeout_seconds=max(1, int(timeout_seconds or 45)),
    )
    papers = repair_canonical_metadata_candidates(papers, clean_query, search_plan, source_names, max_results_per_source=max_internal_results)
    papers = filter_papers_by_year(papers, str(year or "").strip())
    internal_source_results = dict(source_results)
    deduped = rank_papers(dedupe_papers(papers), clean_query, search_plan)
    source_results = source_counts_for_papers(deduped, source_names)
    return {
        "query": clean_query,
        "search_mode": search_plan.get("search_mode", normalize_search_mode(search_mode, clean_query)),
        "requested_search_mode": search_plan.get("requested_search_mode", search_mode),
        "mode_inference_status": search_plan.get("mode_inference_status", ""),
        "mode_inference_error": search_plan.get("mode_inference_error", ""),
        "mode_inference_rationale": search_plan.get("mode_inference_rationale", ""),
        "mode_inference_confidence": search_plan.get("mode_inference_confidence", ""),
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
        "internal_source_results": internal_source_results,
        "exact_title_source_results": exact_title_source_results,
        "channel_results": channel_results,
        "errors": errors,
        "raw_count": len(papers),
        "papers": deduped,
    }


def rank_papers(papers: list[dict], query: str, search_plan: dict) -> list[dict]:
    weights = ((search_plan.get("ranking_weights") or {}).get("normalized") or BASE_DYNAMIC_WEIGHTS)
    ranked = []
    for index, paper in enumerate(papers):
        item = dict(paper)
        score_detail = candidate_rank_score(item, query, weights, search_plan)
        item["candidate_score"] = round(score_detail["score"], 6)
        item["ranking_signals"] = score_detail["signals"]
        item["selection_reasons"] = score_detail["reasons"]
        item["ranking_penalties"] = score_detail["penalties"]
        ranked.append((item["candidate_score"], -index, item))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in ranked]


def source_counts_for_papers(papers: list[dict], sources: list[str]) -> dict[str, int]:
    counts = {source: 0 for source in sources}
    for paper in papers:
        raw_source = paper.get("raw_source_record", {}).get("source") if isinstance(paper.get("raw_source_record"), dict) else ""
        source = normalize_source_name(paper.get("retrieved_from") or raw_source)
        if source not in counts:
            source = infer_source_bucket_from_url(paper.get("source"), sources)
        if source not in counts and sources:
            source = sources[0]
        if source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def repair_canonical_metadata_candidates(
    papers: list[dict],
    query: str,
    search_plan: dict,
    requested_sources: list[str],
    *,
    max_results_per_source: int,
) -> list[dict]:
    identity = normalize_bibliographic_identity(search_plan, query)
    title = clean_text(identity.get("title"))
    if identity.get("query_intent") not in BIBLIOGRAPHIC_INTENTS or not title:
        return papers
    if not should_run_exact_title_search(title):
        return papers

    strong_existing = [
        paper for paper in papers
        if strong_bibliographic_candidate(paper, identity)
    ]
    if not strong_existing and len(papers) >= max(3, max_results_per_source):
        return papers

    repair_sources = canonical_repair_sources(requested_sources)
    if not repair_sources:
        return papers
    try:
        payload = run_exact_title_search_by_source(title, repair_sources, max_results_per_source=min(5, max(2, max_results_per_source)))
    except Exception:
        return papers
    repair_papers, _, _ = normalize_search_payload(payload, repair_sources)
    canonical_candidates = []
    for paper in repair_papers:
        if not strong_bibliographic_candidate(paper, identity):
            continue
        item = dict(paper)
        item["retrieval_channel"] = "canonical_repair"
        item["retrieval_query"] = title
        item["canonical_repair"] = True
        item["canonical_repair_basis"] = {
            "title": title,
            "authors": identity.get("authors") or [],
            "year": identity.get("year") or "",
        }
        canonical_candidates.append(item)
    if not canonical_candidates:
        return papers
    canonical_candidates.sort(key=canonical_preference_key)
    return [*canonical_candidates, *papers]


def canonical_repair_sources(requested_sources: list[str]) -> list[str]:
    preferred = ["arxiv", "semantic", "openalex", "crossref"]
    available = [source for source in preferred if source in {"arxiv", "semantic", "openalex", "crossref"}]
    requested = [normalize_source_name(source) for source in requested_sources or []]
    return list(dict.fromkeys([*available, *(source for source in requested if source in available)]))


def canonical_preference_key(paper: dict) -> tuple[int, int, str]:
    source = normalize_source_name(paper.get("retrieved_from") or paper.get("source_label"))
    source_rank = {"arxiv": 0, "semantic": 1, "openalex": 2, "crossref": 3}.get(source, 9)
    has_stable_id = 0 if (paper.get("arxiv_id") or paper.get("pmid") or paper.get("doi")) else 1
    return (source_rank, has_stable_id, str(paper.get("title") or ""))


def strong_bibliographic_candidate(paper: dict, identity: dict) -> bool:
    title = clean_text(identity.get("title"))
    if not title:
        return False
    title_score = strict_title_identity_score(title, paper.get("title"))
    if title_score < 0.9:
        return False
    authors = clean_string_list(identity.get("authors"), limit=8)
    if authors and author_match_score(" ".join(authors), paper.get("authors")) < 0.45:
        return False
    year = clean_text(identity.get("year"))
    if year and paper.get("year") and year_match_score(year, paper.get("year")) <= 0:
        return False
    return True


def strict_title_identity_score(query_title, candidate_title) -> float:
    query_norm = normalize_title_for_match(query_title)
    candidate_norm = normalize_title_for_match(candidate_title)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    query_tokens = useful_tokens(query_norm)
    candidate_tokens = useful_tokens(candidate_norm)
    token_score = 0.0
    if query_tokens and candidate_tokens:
        token_score = len(query_tokens & candidate_tokens) / max(1, max(len(query_tokens), len(candidate_tokens)))
    char_score = title_character_similarity(query_norm, candidate_norm)
    if token_score >= 0.9 and char_score >= 0.88:
        return max(token_score, char_score)
    return min(token_score, char_score)


def title_character_similarity(left: str, right: str) -> float:
    # Keep this local to avoid pulling difflib into the hot path unless identity repair is active.
    from difflib import SequenceMatcher

    return SequenceMatcher(None, str(left or ""), str(right or "")).ratio()


def infer_source_bucket_from_url(url, sources: list[str]) -> str:
    parsed = urlparse(str(url or ""))
    host = parsed.netloc.lower()
    candidates = []
    if host.endswith("arxiv.org"):
        candidates.append("arxiv")
    if host.endswith("pubmed.ncbi.nlm.nih.gov") or host.endswith("ncbi.nlm.nih.gov"):
        candidates.append("pubmed")
    if host.endswith("doi.org"):
        candidates.extend(["crossref", "doi"])
    if "semanticscholar" in host:
        candidates.append("semantic")
    if "openalex" in host:
        candidates.append("openalex")
    return next((source for source in candidates if source in sources), "")


def candidate_rank_score(reference: dict, query: str, weights: dict, search_plan: dict) -> dict:
    identity = normalize_bibliographic_identity(search_plan, query)
    title_probe = identity.get("title") or query
    author_probe = " ".join(identity.get("authors") or []) or query
    year_probe = identity.get("year") or query
    venue_probe = identity.get("venue") or query
    signals = {
        "title": title_match_score(title_probe, reference.get("title")),
        "author": author_match_score(author_probe, reference.get("authors")),
        "year": year_match_score(year_probe, reference.get("year")),
        "venue": venue_match_score(venue_probe, reference.get("journal")),
        "topic": lexical_coverage_score(query, reference_text(reference), category="topic"),
        "method": term_coverage_score(query, reference_text(reference), METHOD_TERMS),
        "task": term_coverage_score(query, reference_text(reference), TASK_TERMS),
        "abstract": abstract_match_score(query, reference.get("abstract")),
        "source": source_quality_score(reference),
    }
    if reference.get("retrieval_channel") == "exact_title" and signals["title"] >= 0.6:
        signals["title"] = max(signals["title"], 0.98)
    channel_score = retrieval_channel_score(reference)
    identity_score = candidate_identity_score(signals, reference, identity, channel_score=channel_score)
    signals["title_similarity"] = signals["title"]
    signals["author_match"] = signals["author"]
    signals["year_match"] = signals["year"]
    signals["source_quality"] = signals["source"]
    signals["retrieval_channel"] = channel_score
    signals["identity"] = identity_score
    penalties = candidate_penalties(reference, query, signals, search_plan)
    weighted = sum(float(weights.get(label, 0)) * signals.get(label, 0.0) for label in RANK_SIGNAL_LABELS)
    if identity.get("query_intent") in BIBLIOGRAPHIC_INTENTS and identity.get("title"):
        weighted = max(weighted, identity_score)
    penalty_value = sum(penalties.values()) / 100.0
    score = max(0.0, weighted - penalty_value)
    return {
        "score": score,
        "signals": {key: round(value, 4) for key, value in signals.items()},
        "penalties": penalties,
        "reasons": ranking_reasons(signals, reference),
    }


def reference_text(reference: dict) -> str:
    return " ".join(
        clean_text(reference.get(key))
        for key in ("title", "abstract", "journal", "authors", "relevance")
        if reference.get(key)
    )


def title_match_score(query, title) -> float:
    query_title = normalize_title_for_match(query)
    candidate_title = normalize_title_for_match(title)
    if not query_title or not candidate_title:
        return 0.0
    if query_title == candidate_title:
        return 1.0
    query_tokens = useful_tokens(query_title)
    title_tokens = useful_tokens(candidate_title)
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
    if overlap >= 0.9:
        return 0.9
    if overlap >= 0.5:
        return min(0.85, overlap)
    if query_title in candidate_title or candidate_title in query_title:
        return 0.7
    return min(0.45, overlap)


def author_match_score(query, authors) -> float:
    author_text = normalize_text_for_match(authors)
    query_text = normalize_text_for_match(query)
    if not author_text or not query_text:
        return 0.0
    name_match = re.search(r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3}\b", str(query or ""))
    if name_match and normalize_text_for_match(name_match.group(0)) in author_text:
        return 1.0
    query_tokens = useful_tokens(query_text)
    author_tokens = useful_tokens(author_text)
    overlap = query_tokens & author_tokens
    if not overlap:
        return 0.0
    return min(0.8, len(overlap) / max(1, min(len(query_tokens), len(author_tokens))))


def year_match_score(query, year) -> float:
    query_years = re.findall(r"\b(?:19|20)\d{2}\b", str(query or ""))
    if not query_years:
        return 0.0
    paper_year = first_year(str(year or ""))
    if not paper_year:
        return 0.0
    paper_value = int(paper_year)
    best_delta = min(abs(int(value) - paper_value) for value in query_years)
    if best_delta == 0:
        return 1.0
    if best_delta == 1:
        return 0.7
    return 0.0


def venue_match_score(query, venue) -> float:
    query_tokens = useful_tokens(normalize_text_for_match(query))
    venue_tokens = useful_tokens(normalize_text_for_match(venue))
    if not query_tokens or not venue_tokens:
        return 0.0
    return min(1.0, len(query_tokens & venue_tokens) / max(1, len(venue_tokens)))


def lexical_coverage_score(query: str, text: str, *, category: str) -> float:
    del category
    query_tokens = useful_tokens(normalize_text_for_match(query))
    text_tokens = useful_tokens(normalize_text_for_match(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return min(1.0, len(query_tokens & text_tokens) / max(1, len(query_tokens)))


def term_coverage_score(query: str, text: str, terms: set[str]) -> float:
    combined_query = normalize_text_for_match(query)
    combined_text = normalize_text_for_match(text)
    matched_query_terms = [term for term in terms if term.casefold() in combined_query]
    if not matched_query_terms:
        return 0.0
    hits = [term for term in matched_query_terms if term.casefold() in combined_text]
    return len(hits) / max(1, len(matched_query_terms))


def abstract_match_score(query, abstract) -> float:
    if not abstract:
        return 0.0
    return lexical_coverage_score(query, str(abstract), category="abstract")


def source_quality_score(reference: dict) -> float:
    status = str(reference.get("verification_status") or "").casefold()
    if status == "verified":
        return 1.0
    source = normalize_source_name(reference.get("retrieved_from") or reference.get("source_label"))
    return SOURCE_VERIFICATION_SCORE.get(source, 0.4)


def retrieval_channel_score(reference: dict) -> float:
    channel = str(reference.get("retrieval_channel") or "").strip().casefold()
    if channel == "canonical_repair":
        return 1.0
    if channel == "exact_title":
        return 0.95
    if channel == "fuzzy_title":
        return 0.75
    if channel in {"citation", "author"}:
        return 0.65
    if channel == "topic":
        return 0.35
    return 0.25


def candidate_identity_score(signals: dict, reference: dict, identity: dict, *, channel_score: float) -> float:
    if not isinstance(identity, dict) or identity.get("query_intent") not in BIBLIOGRAPHIC_INTENTS:
        return 0.0
    title_score = float(signals.get("title") or 0.0)
    author_score = float(signals.get("author") or 0.0)
    year_score = float(signals.get("year") or 0.0)
    source_score = float(signals.get("source") or 0.0)
    has_authors = bool(identity.get("authors"))
    has_year = bool(identity.get("year"))
    if has_authors:
        score = 0.58 * title_score + 0.27 * author_score + 0.07 * year_score + 0.05 * source_score + 0.03 * channel_score
    elif has_year:
        score = 0.7 * title_score + 0.12 * year_score + 0.1 * source_score + 0.08 * channel_score
    else:
        score = 0.82 * title_score + 0.1 * source_score + 0.08 * channel_score
    if reference.get("arxiv_id"):
        score += 0.04
    return max(0.0, min(1.0, score))


def candidate_penalties(reference: dict, query: str, signals: dict, search_plan: dict) -> dict[str, float]:
    penalties: dict[str, float] = {}
    status = str(reference.get("verification_status") or "").strip().casefold()
    if status in {"unverified", "partial", ""}:
        penalties["unverified_or_partial"] = 10.0
    risks = " ".join(str(risk) for risk in reference.get("verification_risks") or reference.get("screening_risks") or [])
    if "doi_resolution_failed" in risks:
        penalties["doi_resolution_failed"] = 25.0
    if "year_conflict" in risks:
        penalties["year_conflict"] = 25.0
    if signals.get("title", 0.0) >= 0.85 and signals.get("author", 0.0) == 0.0 and query_features(query).get("has_person_name"):
        penalties["title_author_conflict"] = 20.0
    weights = ((search_plan.get("ranking_weights") or {}).get("normalized") or {})
    if weights.get("topic", 0.0) > 0.25 and signals.get("topic", 0.0) > 0.0 and max(signals.get("title", 0.0), signals.get("author", 0.0), signals.get("abstract", 0.0)) == 0.0:
        penalties["generic_topic_only"] = 15.0
    if normalize_source_name(reference.get("retrieved_from")) == "openalex" and status not in {"verified"}:
        penalties["aggregator_without_secondary_verification"] = 10.0
    return penalties


def ranking_reasons(signals: dict, reference: dict) -> list[str]:
    reasons = []
    if signals.get("title", 0) >= 0.85:
        reasons.append("title_match")
    elif signals.get("title", 0) >= 0.5:
        reasons.append("partial_title_match")
    if signals.get("author", 0) >= 0.5:
        reasons.append("author_match")
    if signals.get("year", 0) >= 0.7:
        reasons.append("year_match")
    if signals.get("identity", 0) >= 0.85:
        reasons.append("identity_match")
    elif signals.get("identity", 0) >= 0.65:
        reasons.append("partial_identity_match")
    if signals.get("topic", 0) >= 0.45:
        reasons.append("topic_match")
    if signals.get("method", 0) >= 0.5:
        reasons.append("method_match")
    if signals.get("task", 0) >= 0.5:
        reasons.append("task_match")
    if str(reference.get("verification_status") or "").casefold() == "verified" or signals.get("source", 0) >= 0.85:
        reasons.append("verified_source")
    if reference.get("retrieval_channel"):
        reasons.append(f"channel:{reference['retrieval_channel']}")
    if reference.get("canonical_repair"):
        reasons.append("canonical_repair")
    return list(dict.fromkeys(reasons)) or ["metadata_candidate"]


def normalize_title_for_match(value) -> str:
    return re.sub(r"\W+", " ", str(value or "").casefold()).strip()


def normalize_text_for_match(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def useful_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "the", "to", "using", "with", "all", "you", "need",
    }
    tokens = re.findall(r"[a-z0-9][a-z0-9+-]{1,}|[\u4e00-\u9fff]{1,}", str(text or "").casefold())
    return {token for token in tokens if token not in stopwords}


def should_run_exact_title_search(query: str) -> bool:
    text = clean_text(query)
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    if not (3 <= len(words) <= 18):
        return False
    if re.search(r"\b(AND|OR|NOT)\b|\[|\]|\(|\)|\{|\}|[?]", text):
        return False
    return True


def run_exact_title_search_by_source(title: str, sources: list[str], *, max_results_per_source: int) -> dict:
    results: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for source in sources:
        try:
            if source == "arxiv":
                results[source] = search_arxiv_title_api(title, max_results_per_source)
            elif source == "crossref":
                results[source] = search_crossref_title_api(title, max_results_per_source)
            elif source == "openalex":
                results[source] = search_openalex_title_api(title, max_results_per_source)
            elif source == "semantic":
                results[source] = search_semantic_scholar_api(title, max_results_per_source)
        except Exception as error:
            results[source] = []
            errors[source] = str(error)
    return {"results": results, "errors": errors}


def first_nonempty_query(queries_by_source: dict, sources: list[str]) -> str:
    if not isinstance(queries_by_source, dict):
        return ""
    for source in sources:
        value = clean_text(queries_by_source.get(source))
        if value:
            return value
    for value in queries_by_source.values():
        text = clean_text(value)
        if text:
            return text
    return ""


def run_paper_search_backend_by_source(
    queries_by_source: dict[str, str],
    *,
    max_results_per_source: int,
    year: str,
    timeout_seconds: int,
) -> dict:
    grouped: dict[str, list[str]] = {}
    merged_results: dict[str, list[dict]] = {}
    merged_errors: dict[str, str] = {}
    for source, query in queries_by_source.items():
        clean_query = str(query or "").strip()
        if not clean_query:
            continue
        handled = run_source_specific_query(
            source,
            clean_query,
            max_results_per_source=max_results_per_source,
            timeout_seconds=timeout_seconds,
        )
        if handled is not None:
            papers, error = handled
            merged_results.setdefault(source, papers)
            if error:
                merged_errors[source] = error
            continue
        grouped.setdefault(clean_query, []).append(source)

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


def run_source_specific_query(
    source: str,
    query: str,
    *,
    max_results_per_source: int,
    timeout_seconds: int,
) -> tuple[list[dict], str] | None:
    source_name = normalize_source_name(source)
    author_query = parse_fielded_author_query(query)
    if not author_query and not is_author_field_query(query):
        return None
    normalized_query = source_specific_channel_query(source_name, "author", author_query) if author_query else query
    try:
        if source_name == "arxiv":
            return search_arxiv_api(normalized_query, max_results_per_source, timeout_seconds=timeout_seconds), ""
        if source_name == "crossref":
            return search_crossref_api(normalized_query, max_results_per_source), ""
        if source_name == "openalex":
            return search_openalex_api(normalized_query, max_results_per_source), ""
    except Exception as error:
        return [], str(error)
    return None


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
        timeout_seconds=timeout_seconds,
    )


def run_paper_search_mcp_python(
    query: str,
    *,
    sources: list[str],
    max_results_per_source: int,
    timeout_seconds: int,
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
        if source == "arxiv":
            try:
                results[source] = search_arxiv_api(
                    query,
                    max_results_per_source,
                    timeout_seconds=max(1, int(timeout_seconds or 45)),
                )
            except Exception as error:
                results[source] = []
                errors[source] = str(error)
            continue
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
    author_query = parse_fielded_author_query(query)
    params = {"rows": max_results, "sort": "relevance"}
    if author_query:
        params["query.author"] = author_query
    else:
        params["query"] = query
    payload = fetch_json_url(
        "https://api.crossref.org/works",
        params,
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


def search_crossref_title_api(title: str, max_results: int) -> list[dict]:
    payload = fetch_json_url(
        "https://api.crossref.org/works",
        {"query.title": title, "rows": max_results, "sort": "relevance"},
    )
    items = ((payload.get("message") or {}).get("items") or []) if isinstance(payload, dict) else []
    return crossref_items_to_papers(items)


def crossref_items_to_papers(items: list[dict]) -> list[dict]:
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
    author_query = parse_fielded_author_query(query)
    if author_query:
        return search_openalex_author_api(author_query, max_results)
    payload = fetch_json_url(
        "https://api.openalex.org/works",
        {"search": query, "per-page": max_results},
    )
    return openalex_items_to_papers(payload.get("results", []) if isinstance(payload, dict) else [])


def search_openalex_author_api(author: str, max_results: int) -> list[dict]:
    author_payload = fetch_json_url(
        "https://api.openalex.org/authors",
        {"search": author, "per-page": 3},
    )
    author_ids = []
    for item in author_payload.get("results", []) if isinstance(author_payload, dict) else []:
        openalex_id = clean_text(item.get("id"))
        if openalex_id:
            author_ids.append(openalex_id.rsplit("/", 1)[-1])
    if not author_ids:
        return search_openalex_api(author, max_results)
    payload = fetch_json_url(
        "https://api.openalex.org/works",
        {
            "filter": f"authorships.author.id:{author_ids[0]}",
            "per-page": max_results,
            "sort": "cited_by_count:desc",
        },
    )
    return openalex_items_to_papers(payload.get("results", []) if isinstance(payload, dict) else [])


def openalex_items_to_papers(items: list[dict]) -> list[dict]:
    papers = []
    for item in items:
        authorships = item.get("authorships", []) if isinstance(item.get("authorships"), list) else []
        authors = [
            ((authorship.get("author") or {}).get("display_name") or "")
            for authorship in authorships[:8]
            if isinstance(authorship, dict)
        ]
        primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
        landing_url = clean_text(primary_location.get("landing_page_url") or primary_location.get("pdf_url"))
        papers.append(
            {
                "paper_id": item.get("id", ""),
                "title": item.get("display_name", ""),
                "authors": authors,
                "abstract": openalex_abstract(item.get("abstract_inverted_index")),
                "year": item.get("publication_year", ""),
                "url": landing_url or item.get("doi") or item.get("id", ""),
                "venue": source.get("display_name", ""),
                "doi": item.get("doi", ""),
                "source": "openalex",
            }
        )
    return papers


def search_openalex_title_api(title: str, max_results: int) -> list[dict]:
    return search_openalex_api(title, max_results)


def search_arxiv_api(query: str, max_results: int, *, timeout_seconds: int = 20) -> list[dict]:
    author_query = is_author_field_query(query)
    xml = fetch_arxiv_atom(
        {
            "search_query": clean_text(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance" if author_query else "submittedDate",
            "sortOrder": "descending",
        },
        timeout_seconds=timeout_seconds,
    )
    return parse_arxiv_atom(xml)


def search_arxiv_title_api(title: str, max_results: int) -> list[dict]:
    search_query = f'ti:"{clean_text(title).replace(chr(34), " ")}"'
    xml = fetch_arxiv_atom(
        {"search_query": search_query, "start": 0, "max_results": max_results, "sortBy": "relevance"},
        timeout_seconds=20,
    )
    return parse_arxiv_atom(xml)


def fetch_arxiv_atom(params: dict, *, timeout_seconds: int) -> str:
    attempts = max(1, min(int(os.getenv("PAPER_SEARCH_ARXIV_RETRIES", "2") or 2), 4))
    last_error: Exception | None = None
    for attempt in range(attempts):
        wait_for_arxiv_turn()
        request = Request(
            f"{ARXIV_API_URL}?{urlencode(params)}",
            headers={"Accept": "application/atom+xml", "User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=max(1, int(timeout_seconds or 20))) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            last_error = error
            if error.code == 429 and attempt + 1 < attempts:
                time.sleep(arxiv_retry_delay_seconds(error, attempt))
                continue
            if error.code == 429:
                raise PaperSearchError(
                    "export.arxiv.org search rate limited (HTTP 429). Wait a moment and retry, "
                    "or reduce repeated arXiv-only searches."
                ) from error
            raise PaperSearchError(f"export.arxiv.org search failed: HTTP Error {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(arxiv_retry_delay_seconds(error, attempt))
                continue
            raise PaperSearchError(f"export.arxiv.org search failed after {attempts} attempt(s): {error}") from error
    raise PaperSearchError(f"export.arxiv.org search failed: {last_error}")


def wait_for_arxiv_turn() -> None:
    global _ARXIV_LAST_REQUEST_AT
    delay = arxiv_request_delay_seconds()
    if delay <= 0:
        return
    with _ARXIV_REQUEST_LOCK:
        elapsed = time.monotonic() - _ARXIV_LAST_REQUEST_AT
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _ARXIV_LAST_REQUEST_AT = time.monotonic()


def arxiv_request_delay_seconds() -> float:
    try:
        return max(0.0, min(float(os.getenv("PAPER_SEARCH_ARXIV_DELAY_SECONDS", "3.2") or 3.2), 30.0))
    except ValueError:
        return 3.2


def arxiv_retry_delay_seconds(error: Exception, attempt: int) -> float:
    if isinstance(error, HTTPError):
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 60.0))
            except ValueError:
                pass
    return min(2.0 * (attempt + 1), 10.0)


def parse_arxiv_atom(xml: str) -> list[dict]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise PaperSearchError(f"export.arxiv.org search failed: {error}") from error
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        url = text_from_node(entry.find("atom:id", ns))
        arxiv_id = infer_arxiv_id(url)
        authors = [text_from_node(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)]
        authors = [author for author in authors if author]
        published = text_from_node(entry.find("atom:published", ns))
        doi = text_from_node(entry.find("arxiv:doi", ns))
        papers.append(
            {
                "paper_id": arxiv_id or url,
                "title": text_from_node(entry.find("atom:title", ns)),
                "authors": authors,
                "abstract": text_from_node(entry.find("atom:summary", ns)),
                "year": first_year(published),
                "url": url,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "source": "arxiv",
            }
        )
    return papers

def text_from_node(node) -> str:
    return clean_text(node.text if node is not None else "")


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
