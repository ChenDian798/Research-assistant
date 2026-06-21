from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.research_agent.literature_workflow import LiteratureAnalysisWorkflow
from src.research_agent.citations import format_references
from src.research_agent.doi import extract_arxiv_id, extract_doi, extract_pmid
from src.research_agent.paper_search import (
    LLMQueryRewriteParseError,
    PaperSearchError,
    build_academic_search_plan,
    normalize_search_payload,
    parse_json_object,
    search_cnki_api,
    search_papers,
)
from src.research_agent.reference_relevance import apply_relevance_gate, assess_reference_relevance
from src.research_agent.reference_screening import screen_reference
from src.research_agent.reference_verification import verify_reference
import web_app
from web_app import ResearchWebHandler


@pytest.fixture(autouse=True)
def default_rules_relevance_gate(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "rules")
    monkeypatch.setattr(web_app, "HISTORY_PATH", tmp_path / "history_records.json")


class CaptureLiteratureLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        del model, temperature, max_tokens
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if "evidence packet optimizer" in system_prompt:
            return (
                '{"references":[{"reference_index":0,"title":"Paper A",'
                '"evidence_brief":"Use the supplied excerpt and candidate evidence.",'
                '"gap_audit":["Check author-stated limitations"],'
                '"evidence_quotes":["Full text excerpt"]}]}'
            )
        if "final integrator" in system_prompt:
            return (
                '{"rows":[{"reference_index":0,"title":"Paper A","source":"paper-a.pdf",'
                '"contribution":"Contribution","methodology":"Method","evidence_strength":"Moderate",'
                '"dataset":"Dataset A","modality":"NCCT","metrics":"Dice","key_results":"Dice 0.5"}],'
                '"summary":{"overall_assessment":"Paper A is covered.","common_strengths":[],'
                '"common_weaknesses":[],"methodological_patterns":["Paper A uses Method"],'
                '"evidence_gaps":[],"research_gaps":[],"recommended_reading_order":["Paper A"],'
                '"next_actions":[],"confidence":"Medium"}}}'
            )
        return (
            '{"analyst":"Contribution Analyst","rows":[{"title":"Paper A","source":"paper-a.pdf",'
            '"contribution":"Contribution","methodology":"Method","evidence_strength":"Moderate",'
            '"dataset":"Dataset A","modality":"NCCT","metrics":"Dice","key_results":"Dice 0.5"}]}'
        )


class FailingIntegratorThenSummaryLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        del model, temperature, max_tokens
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if "final integrator" in system_prompt:
            return "not json"
        if "fallback synthesizer" in system_prompt:
            return (
                '{"summary":{"overall_assessment":"Paper A and Paper B are compared across datasets and metrics.",'
                '"common_strengths":["Both report structured validation"],"common_weaknesses":["External evidence is limited"],'
                '"methodological_patterns":["Paper A uses Method A; Paper B uses Method B"],'
                '"evidence_gaps":["More external validation is needed"],"research_gaps":["Clinical workflow validation remains open"],'
                '"recommended_reading_order":["Paper A","Paper B"],"next_actions":[],"confidence":"Medium"}}'
            )
        return '{"rows":[]}'


class StaticRelevanceLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        del model, temperature, max_tokens
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.content


class FailingRelevanceLLM:
    async def complete(self, **kwargs) -> str:
        del kwargs
        raise RuntimeError("LLM unavailable")


class TranslatingLiteratureLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        del model, temperature, max_tokens
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if "Chinese translation post-processor" in system_prompt:
            return (
                '{"rows":[{"reference_index":0,"title":"Paper A","source":"paper-a.pdf",'
                '"contribution":"提出一种值得核验的分割方法","methodology":"方法逻辑需要结合全文确认",'
                '"dataset":"Dataset A","metrics":"Dice","key_results":"Dice 0.5"}],'
                '"summary":{"overall_assessment":"Paper A 已被纳入比较。","common_strengths":[],'
                '"common_weaknesses":[],"methodological_patterns":["Paper A 使用该方法"],'
                '"evidence_gaps":[],"research_gaps":[],"recommended_reading_order":["Paper A：先读方法和结果"],'
                '"next_actions":["核验全文证据"],"confidence":"中等"}}'
            )
        if "final integrator" in system_prompt:
            return (
                '{"rows":[{"reference_index":0,"title":"Paper A","source":"paper-a.pdf",'
                '"contribution":"Contribution should be translated","methodology":"Method should be translated",'
                '"dataset":"Dataset A","metrics":"Dice","key_results":"Dice 0.5"}],'
                '"summary":{"overall_assessment":"Paper A is covered.","common_strengths":[],"common_weaknesses":[],'
                '"methodological_patterns":["Paper A uses Method"],"evidence_gaps":[],"research_gaps":[],'
                '"recommended_reading_order":["Paper A"],"next_actions":[],"confidence":"Medium"}}'
            )
        return '{"rows":[]}'


def test_paper_search_disabled_returns_clear_error(monkeypatch) -> None:
    sent = {}
    handler = object.__new__(ResearchWebHandler)

    monkeypatch.setenv("PAPER_SEARCH_ENABLED", "false")
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: sent.update(payload=payload, status=status))

    handler._handle_literature_search()

    assert "Academic search is not enabled" in sent["payload"]["error"]
    assert sent["payload"]["search_enabled"] is False


def test_search_result_maps_to_reference_shape(monkeypatch) -> None:
    payload = {
        "results": {
            "semantic": [
                {
                    "title": "Foundation Models for Medical Image Segmentation",
                    "authors": [{"name": "Ada Lovelace"}],
                    "year": "2024",
                    "doi": "10.1000/example",
                    "abstract": "A real abstract.",
                    "venue": "Medical AI",
                }
            ]
        }
    }

    monkeypatch.setattr("src.research_agent.paper_search.run_paper_search_backend", lambda *args, **kwargs: payload)

    result = search_papers("medical image segmentation", sources="semantic", max_results_per_source=1)

    assert result["source_results"]["semantic"] == 1
    assert result["papers"][0]["source_origin"] == "paper_search_mcp"
    assert result["papers"][0]["source_label"] == "Semantic Scholar"
    assert result["papers"][0]["source"] == "https://doi.org/10.1000/example"


def test_search_papers_uses_source_specific_queries(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources):
        del query, requested_sources
        return {
            "search_query": "ischemic stroke CT lesion segmentation",
            "pubmed_query": "ischemic stroke[Title/Abstract] AND CT[Title/Abstract] AND segmentation[Title/Abstract]",
            "core_concepts": [
                {"concept": "ischemic stroke", "type": "condition", "must_keep": True},
                {"concept": "CT", "type": "modality", "must_keep": True},
                {"concept": "lesion segmentation", "type": "task", "must_keep": True},
            ],
            "recommended_sources": ["pubmed", "crossref"],
            "avoid_sources": [],
            "rationale": "Use PubMed syntax only for PubMed.",
        }

    calls = []

    def fake_backend(query, *, sources, max_results_per_source, year, timeout_seconds):
        del max_results_per_source, year, timeout_seconds
        calls.append({"query": query, "sources": sources})
        return {
            "results": {
                source: [
                    {
                        "title": f"{source} paper",
                        "doi": f"10.1000/{source}",
                        "source": source,
                        "abstract": "stroke CT segmentation",
                    }
                ]
                for source in sources
            }
        }

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)
    monkeypatch.setattr("src.research_agent.paper_search.run_paper_search_backend", fake_backend)

    result = search_papers("头颅CT脑梗死病灶分割算法", sources="pubmed,crossref", max_results_per_source=1)

    assert calls == [
        {
            "query": "ischemic stroke[Title/Abstract] AND CT[Title/Abstract] AND segmentation[Title/Abstract]",
            "sources": ["pubmed"],
        },
        {"query": "ischemic stroke CT lesion segmentation", "sources": ["crossref"]},
    ]
    assert result["queries_by_source"]["pubmed"].endswith("segmentation[Title/Abstract]")
    assert result["queries_by_source"]["crossref"] == "ischemic stroke CT lesion segmentation"
    assert result["source_results"] == {"pubmed": 1, "crossref": 1}


def test_cnki_source_preserves_user_selection_and_uses_chinese_query(monkeypatch) -> None:
    calls = []

    def fake_backend(query, *, sources, max_results_per_source, year, timeout_seconds):
        del max_results_per_source, year, timeout_seconds
        calls.append({"query": query, "sources": sources})
        return {"results": {source: [] for source in sources}}

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "false")
    monkeypatch.setattr("src.research_agent.paper_search.run_paper_search_backend", fake_backend)

    result = search_papers("头颅CT脑梗死病灶分割算法", sources="cnki,openalex", max_results_per_source=1)

    assert result["sources_used"] == ["cnki", "openalex"]
    assert result["queries_by_source"]["cnki"] == "头颅CT脑梗死病灶分割算法"
    assert result["queries_by_source"]["openalex"] != result["queries_by_source"]["cnki"]
    assert calls[0] == {"query": "头颅CT脑梗死病灶分割算法", "sources": ["cnki"]}


def test_cnki_source_reports_adapter_not_configured() -> None:
    with pytest.raises(PaperSearchError, match="CNKI search is listed as a selectable source"):
        search_cnki_api("头颅CT脑梗死病灶分割算法", 1)


def test_society_search_mode_uses_social_science_query_plan(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources, *, search_mode="auto"):
        assert search_mode == "society"
        del query, requested_sources
        return {
            "search_query": "platform labor algorithmic management China",
            "pubmed_query": "",
            "core_concepts": [
                {"concept": "platform labor", "type": "institution", "must_keep": True},
                {"concept": "algorithmic management", "type": "phenomenon", "must_keep": True},
                {"concept": "China", "type": "context", "must_keep": True},
            ],
            "synonyms": ["gig work", "digital labor platforms"],
            "recommended_sources": ["crossref", "openalex", "semantic"],
            "avoid_sources": ["pubmed"],
            "rationale": "This is a social-science topic about platform labor governance.",
        }

    calls = []

    def fake_backend(query, *, sources, max_results_per_source, year, timeout_seconds):
        del max_results_per_source, year, timeout_seconds
        calls.append({"query": query, "sources": sources})
        return {"results": {source: [] for source in sources}}

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)
    monkeypatch.setattr("src.research_agent.paper_search.run_paper_search_backend", fake_backend)

    result = search_papers("中国平台劳动算法管理", sources="pubmed,crossref", max_results_per_source=1, search_mode="society")

    assert result["search_mode"] == "society"
    assert result["query_plan"]["search_mode"] == "society"
    assert result["queries_by_source"]["pubmed"] == "platform labor algorithmic management China"
    assert "Title/Abstract" not in result["queries_by_source"]["pubmed"]
    assert calls == [
        {"query": "platform labor algorithmic management China", "sources": ["pubmed", "crossref"]},
    ]


def test_auto_search_mode_routes_obvious_society_topic(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "false")

    plan = build_academic_search_plan("rural education inequality and family background", ["crossref"], search_mode="auto")

    assert plan["search_mode"] == "society"
    assert "education" in plan["backend_query"]


def test_biomedical_llm_query_rewrite_keeps_required_concepts_and_sources(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources):
        del query, requested_sources
        return {
            "search_query": '("acute ischemic stroke" OR "cerebral infarction") AND ("non-contrast CT" OR NCCT) AND lesion AND segmentation',
            "pubmed_query": '("acute ischemic stroke"[Title/Abstract] OR "cerebral infarction"[Title/Abstract]) AND ("non-contrast CT"[Title/Abstract] OR NCCT[Title/Abstract]) AND lesion[Title/Abstract] AND segmentation[Title/Abstract]',
            "core_concepts": [
                {"concept": "acute ischemic stroke", "type": "condition", "must_keep": True},
                {"concept": "non-contrast CT", "type": "modality", "must_keep": True},
                {"concept": "lesion segmentation", "type": "task", "must_keep": True},
            ],
            "synonyms": ["cerebral infarction", "NCCT", "delineation"],
            "forbidden_broadenings": ["stroke diagnosis", "medical imaging AI"],
            "recommended_sources": ["pubmed", "crossref", "openalex"],
            "avoid_sources": ["arxiv"],
            "rationale": "The topic is biomedical imaging retrieval.",
        }

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)

    plan = build_academic_search_plan("头颅CT脑梗死病灶分割算法", ["arxiv"])

    assert plan["rewrite_status"] == "llm"
    assert plan["rules_fallback_query"]
    assert "acute ischemic stroke" in plan["llm_search_query"]
    assert "Title/Abstract" in plan["llm_pubmed_query"]
    assert "acute ischemic stroke" in plan["backend_query"]
    assert "non-contrast CT" in plan["backend_query"]
    assert "segmentation" in plan["backend_query"]
    assert plan["sources"] == ["arxiv"]
    assert plan["queries_by_source"]["arxiv"] == plan["llm_search_query"]
    assert "stroke diagnosis" in plan["forbidden_broadenings"]


def test_biomedical_llm_query_rewrite_is_llm_first_by_default(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources):
        del query, requested_sources
        return {
            "search_query": "stroke diagnosis medical imaging artificial intelligence",
            "pubmed_query": "stroke diagnosis medical imaging",
            "core_concepts": [{"concept": "stroke diagnosis", "type": "condition", "must_keep": True}],
            "synonyms": [],
            "recommended_sources": ["pubmed", "crossref", "openalex"],
            "avoid_sources": ["arxiv"],
            "rationale": "Overly broad rewrite.",
        }

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)

    plan = build_academic_search_plan("头颅CT脑梗死病灶分割算法", ["arxiv"])

    assert plan["rewrite_status"] == "llm"
    assert plan["backend_query"] == "stroke diagnosis medical imaging artificial intelligence"
    assert plan["queries_by_source"]["arxiv"] == "stroke diagnosis medical imaging artificial intelligence"
    assert plan["sources"] == ["arxiv"]


def test_biomedical_llm_query_rewrite_can_enable_local_guardrail(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources):
        del query, requested_sources
        return {
            "search_query": "stroke diagnosis medical imaging artificial intelligence",
            "pubmed_query": "stroke diagnosis medical imaging",
            "core_concepts": [{"concept": "stroke diagnosis", "type": "condition", "must_keep": True}],
            "synonyms": [],
            "recommended_sources": ["pubmed", "crossref", "openalex"],
            "avoid_sources": ["arxiv"],
            "rationale": "Overly broad rewrite.",
        }

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setenv("PAPER_SEARCH_LOCAL_QUERY_GUARDRAIL", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)

    plan = build_academic_search_plan("头颅CT脑梗死病灶分割算法", ["arxiv"])

    assert plan["rewrite_status"].startswith("rules_fallback:llm_guardrail:missing_required_")
    assert plan["rules_fallback_query"] == plan["backend_query"]
    assert plan["llm_search_query"] == "stroke diagnosis medical imaging artificial intelligence"
    assert plan["llm_pubmed_query"] == "stroke diagnosis medical imaging"
    assert "stroke diagnosis" not in plan["backend_query"]
    assert "computed tomography" in plan["backend_query"]
    assert "segmentation" in plan["backend_query"]
    assert plan["sources"] == ["arxiv"]
    assert plan["queries_by_source"]["arxiv"] == plan["backend_query"]


def test_llm_query_rewrite_records_raw_response_when_json_parse_fails(monkeypatch) -> None:
    async def fake_llm_plan(query, requested_sources):
        del query, requested_sources
        raise LLMQueryRewriteParseError(
            "JSONDecodeError: Expecting value",
            "I would search PubMed for acute ischemic stroke CT segmentation.",
        )

    monkeypatch.setenv("PAPER_SEARCH_QUERY_REWRITE", "true")
    monkeypatch.setattr("src.research_agent.paper_search.build_academic_search_plan_with_llm", fake_llm_plan)

    plan = build_academic_search_plan("非增强CT急性缺血性卒中病灶分割算法", ["pubmed"])

    assert plan["rewrite_status"] == "rules_fallback:LLMQueryRewriteParseError"
    assert "JSONDecodeError" in plan["llm_error"]
    assert "acute ischemic stroke" in plan["llm_raw_response"]
    assert plan["llm_search_query"] == ""
    assert plan["backend_query"] == plan["rules_fallback_query"]


def test_parse_json_object_extracts_embedded_llm_json() -> None:
    parsed = parse_json_object(
        'Here is the JSON plan:\\n{"search_query":"stroke segmentation","pubmed_query":"","core_concepts":["stroke"]}\\nDone.'
    )

    assert parsed["search_query"] == "stroke segmentation"
    assert parsed["core_concepts"] == ["stroke"]


def test_reference_screening_rejects_unstable_record() -> None:
    screened = screen_reference({"title": "Paper", "source": "not-a-url"})

    assert screened["screening_status"] == "rejected"
    assert "generic_title" in screened["screening_reasons"]


def test_reference_screening_keeps_doi_pmid_arxiv_records() -> None:
    records = [
        {"title": "A DOI paper", "doi": "10.1000/example"},
        {"title": "A PMID paper", "pmid": "12345678"},
        {"title": "An arXiv paper", "arxiv_id": "2401.12345"},
    ]

    assert [screen_reference(record)["screening_status"] for record in records] == [
        "qualified",
        "qualified",
        "qualified",
    ]


def test_final_search_output_dedupes_exact_title_with_different_doi() -> None:
    duplicate_title = (
        "Fig. 1. An example of cerebral infarction segmentation using artificial intelligence: "
        "T2-weighted images in the axial, frontal and sagittal planes"
    )
    screened = {"rejected": []}

    qualified, needs_review = ResearchWebHandler._dedupe_final_search_candidates(
        [
            {"title": duplicate_title, "doi": "10.17816/clinpract642757-4328213", "screening_status": "qualified"},
            {"title": duplicate_title, "doi": "10.17816/clinpract642757-4226733", "screening_status": "qualified"},
        ],
        [],
        screened,
    )

    assert len(qualified) == 1
    assert needs_review == []
    assert len(screened["rejected"]) == 1
    assert screened["rejected"][0]["screening_status"] == "rejected"
    assert "duplicate" in screened["rejected"][0]["screening_reasons"]
    assert "duplicate_final_output" in screened["rejected"][0]["screening_risks"]


def test_topic_relevance_gate_rejects_real_but_off_topic_doi() -> None:
    query = "头颅CT脑梗死病灶分割算法"
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "狗的头部",
                    "doi": "10.37019/vet-anatomy/382521.cn",
                    "source": "https://doi.org/10.37019/vet-anatomy/382521.cn",
                    "authors": "Antoine Micheau, Denis Hoa",
                    "year": "2016",
                    "journal": "vet-Anatomy",
                }
            )
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate(query, screened)

    assert gated["qualified"] == []
    assert gated["rejected"][0]["screening_status"] == "rejected"
    assert gated["rejected"][0]["topic_relevance_status"] == "off_topic"
    assert "topic_relevance_failed" in gated["rejected"][0]["screening_reasons"]


def test_topic_relevance_gate_keeps_stroke_ncct_segmentation_paper() -> None:
    reference = {
        "title": "Random Expert Sampling for Deep Learning Segmentation of Acute Ischemic Stroke on Non-contrast CT",
        "doi": "10.1136/jnis-2023-020418",
        "source": "https://doi.org/10.1136/jnis-2023-020418",
        "abstract": "We study acute ischemic stroke lesion segmentation on non-contrast CT using deep learning.",
    }

    assessed = assess_reference_relevance("头颅CT脑梗死病灶分割算法", reference)

    assert assessed["topic_relevance_status"] == "relevant"
    assert assessed["topic_relevance_score"] >= 0.8


def test_topic_relevance_gate_uses_search_plan_for_translated_topics() -> None:
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "Ecological Approaches to Dental Caries Prevention",
                    "doi": "10.1159/000484985",
                    "source": "https://doi.org/10.1159/000484985",
                    "abstract": "This paper discusses ecological caries-preventive measures for dental caries.",
                    "authors": "Nebu Philip, Bharat Suneja, Laurence J. Walsh",
                    "year": "2018",
                }
            ),
            screen_reference(
                {
                    "title": "Electric-Field Mapping of Optically Perturbed CdTe Radiation Detectors",
                    "doi": "10.48550/arXiv.2606.13622",
                    "source": "https://arxiv.org/abs/2606.13622",
                    "abstract": "We probe the two-dimensional electric field in a Schottky CdTe detector.",
                    "authors": "Adriano Cola",
                    "year": "2026",
                }
            ),
        ],
        "needs_review": [],
        "rejected": [],
    }
    query_plan = {
        "backend_query": "dental caries prevention",
        "llm_search_query": "dental caries prevention",
        "core_concepts": ["dental caries", "prevention"],
        "synonyms": ["tooth decay prevention", "cavity prevention"],
    }

    gated = apply_relevance_gate("蛀牙预防", screened, query_plan=query_plan)

    assert [item["title"] for item in gated["qualified"]] == [
        "Ecological Approaches to Dental Caries Prevention"
    ]
    assert gated["qualified"][0]["topic_relevance_status"] == "relevant"
    assert gated["qualified"][0]["topic_relevance_score"] >= 0.5
    assert gated["rejected"][0]["title"] == "Electric-Field Mapping of Optically Perturbed CdTe Radiation Detectors"
    assert gated["rejected"][0]["topic_relevance_status"] == "off_topic"


def test_llm_relevance_gate_handles_unlisted_chinese_medical_topics(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm = StaticRelevanceLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "candidate_index": 0,
                        "topic_status": "relevant",
                        "confidence": 0.94,
                        "matched_concepts": ["dental caries", "prevention"],
                        "missing_concepts": [],
                        "reason": "The paper is about preventive approaches for dental caries.",
                    },
                    {
                        "candidate_index": 1,
                        "topic_status": "off_topic",
                        "confidence": 0.91,
                        "matched_concepts": [],
                        "missing_concepts": ["dental caries", "prevention"],
                        "reason": "The paper is about radiation detectors, not dentistry.",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr("src.research_agent.reference_relevance.LLMClient", lambda: llm)
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "Ecological Approaches to Dental Caries Prevention",
                    "doi": "10.1159/000484985",
                    "source": "https://doi.org/10.1159/000484985",
                    "abstract": "Several ecological preventive approaches have been developed for caries control.",
                }
            ),
            screen_reference(
                {
                    "title": "Electric-Field Mapping of Optically Perturbed CdTe Radiation Detectors",
                    "doi": "10.48550/arXiv.2606.13622",
                    "source": "https://arxiv.org/abs/2606.13622",
                    "abstract": "We probe the two-dimensional electric field in a Schottky CdTe detector.",
                }
            ),
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate(
        "蛀牙预防",
        screened,
        query_plan={"backend_query": "dental caries prevention", "core_concepts": ["dental caries", "prevention"]},
    )

    assert [item["title"] for item in gated["qualified"]] == [
        "Ecological Approaches to Dental Caries Prevention"
    ]
    assert gated["qualified"][0]["llm_relevance_confidence"] == 0.94
    assert gated["rejected"][0]["title"] == "Electric-Field Mapping of Optically Perturbed CdTe Radiation Detectors"
    assert "topic_relevance_failed" in gated["rejected"][0]["screening_reasons"]
    assert llm.calls


def test_society_relevance_gate_uses_social_science_prompt(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm = StaticRelevanceLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "candidate_index": 0,
                        "topic_status": "relevant",
                        "confidence": 0.92,
                        "matched_concepts": ["platform labor", "algorithmic management"],
                        "missing_concepts": [],
                        "reason": "The paper studies algorithmic management in platform labor.",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("src.research_agent.reference_relevance.LLMClient", lambda: llm)
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "Algorithmic Management and Platform Labor in China",
                    "doi": "10.1000/platform-labor",
                    "source": "https://doi.org/10.1000/platform-labor",
                    "abstract": "This study analyzes digital labor platforms and algorithmic management.",
                }
            )
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate(
        "中国平台劳动算法管理",
        screened,
        query_plan={
            "search_mode": "society",
            "backend_query": "platform labor algorithmic management China",
            "core_concepts": ["platform labor", "algorithmic management", "China"],
        },
    )

    assert "social-science literature relevance reviewer" in llm.calls[0]["system_prompt"]
    assert gated["qualified"][0]["llm_relevance_status"] == "relevant"
    assert gated["qualified"][0]["topic_relevance_score"] == 0.92


def test_society_rules_relevance_gate_filters_off_topic(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "rules")
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "Algorithmic Management and Platform Labor in China",
                    "doi": "10.1000/platform-labor",
                    "source": "https://doi.org/10.1000/platform-labor",
                    "abstract": "This article studies algorithmic management and platform labor in China.",
                }
            ),
            screen_reference(
                {
                    "title": "Computed Tomography Reconstruction With Neural Networks",
                    "doi": "10.1000/ct-reconstruction",
                    "source": "https://doi.org/10.1000/ct-reconstruction",
                    "abstract": "We propose a neural reconstruction model for CT images.",
                }
            ),
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate(
        "中国平台劳动算法管理",
        screened,
        query_plan={
            "search_mode": "society",
            "backend_query": "platform labor algorithmic management China",
            "core_concepts": ["platform labor", "algorithmic management", "China"],
        },
    )

    assert [item["title"] for item in gated["qualified"]] == [
        "Algorithmic Management and Platform Labor in China"
    ]
    assert gated["rejected"][0]["title"] == "Computed Tomography Reconstruction With Neural Networks"
    assert gated["rejected"][0]["topic_relevance_status"] == "off_topic"


def test_llm_relevance_gate_sends_low_confidence_to_review(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        "src.research_agent.reference_relevance.LLMClient",
        lambda: StaticRelevanceLLM(
            '{"decisions":[{"candidate_index":0,"topic_status":"off_topic","confidence":0.4,'
            '"matched_concepts":[],"missing_concepts":["topic"],"reason":"uncertain"}]}'
        ),
    )
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "A sparse but maybe adjacent paper",
                    "doi": "10.1000/maybe",
                    "source": "https://doi.org/10.1000/maybe",
                }
            )
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate("罕见医学主题", screened, query_plan={"backend_query": "rare medical topic"})

    assert gated["qualified"] == []
    assert gated["rejected"] == []
    assert gated["needs_review"][0]["topic_relevance_status"] == "borderline"
    assert any("llm_relevance_low_confidence" in risk for risk in gated["needs_review"][0]["screening_risks"])


def test_llm_relevance_gate_falls_back_to_rules_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_RELEVANCE_GATE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr("src.research_agent.reference_relevance.LLMClient", lambda: FailingRelevanceLLM())
    screened = {
        "qualified": [
            screen_reference(
                {
                    "title": "Ecological Approaches to Dental Caries Prevention",
                    "doi": "10.1159/000484985",
                    "source": "https://doi.org/10.1159/000484985",
                    "abstract": "This paper discusses ecological caries-preventive measures for dental caries.",
                }
            )
        ],
        "needs_review": [],
        "rejected": [],
    }

    gated = apply_relevance_gate(
        "蛀牙预防",
        screened,
        query_plan={"backend_query": "dental caries prevention", "core_concepts": ["dental caries", "prevention"]},
    )

    assert gated["qualified"][0]["title"] == "Ecological Approaches to Dental Caries Prevention"
    assert "llm_relevance_status" not in gated["qualified"][0]


def test_reference_verification_marks_crossref_verified(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.research_agent.reference_verification.fetch_crossref_metadata",
        lambda doi: {
            "doi": doi,
            "title": "A DOI paper",
            "source": f"https://doi.org/{doi}",
            "authors": "Ada Lovelace",
            "year": "2024",
            "abstract": "Verified abstract.",
        },
    )

    verified = verify_reference({"title": "A DOI paper", "doi": "10.1000/example"})

    assert verified["verification_status"] == "verified"
    assert "Crossref" in verified["verification_sources"]
    assert verified["provenance"]["evidence_level"] == "metadata+abstract"


def test_search_candidates_do_not_start_analysis_job(monkeypatch) -> None:
    sent = {}
    handler = object.__new__(ResearchWebHandler)
    web_app.JOBS.clear()
    monkeypatch.setenv("PAPER_SEARCH_ENABLED", "true")
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {
            "query": "stroke segmentation",
            "sources": "semantic",
            "max_results_per_source": 1,
            "include_needs_review": True,
        },
    )
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: sent.update(payload=payload, status=status))
    monkeypatch.setattr(
        web_app,
        "search_papers",
        lambda *args, **kwargs: {
            "query": "stroke segmentation",
            "sources_used": ["semantic"],
            "source_results": {"semantic": 1},
            "errors": {},
            "raw_count": 1,
                "papers": [
                    {
                        "title": "Deep learning stroke lesion segmentation on non-contrast head CT",
                        "doi": "10.1000/example",
                        "source": "https://doi.org/10.1000/example",
                        "abstract": "An algorithm for acute ischemic stroke lesion segmentation on non-contrast CT.",
                        "source_origin": "paper_search_mcp",
                        "source_label": "Semantic Scholar",
                    }
            ],
        },
    )
    monkeypatch.setattr(
        web_app,
        "verify_references",
        lambda references: [
            {
                **reference,
                "verification_status": "verified",
                "verification_sources": ["paper-search-mcp", "Crossref"],
                "verification_risks": [],
                "provenance": {"retrieved_from": "semantic", "verified_by": "Crossref", "evidence_level": "metadata+abstract"},
            }
            for reference in references
        ],
    )

    handler._handle_literature_search()

    assert sent["payload"]["status"] == "done"
    assert len(sent["payload"]["qualified_references"]) == 1
    assert web_app.JOBS == {}


def test_async_literature_search_returns_history_task(monkeypatch) -> None:
    sent = {}
    handler = object.__new__(ResearchWebHandler)
    handler.server = type("FakeServer", (), {"server_port": 8125})()
    web_app.JOBS.clear()
    monkeypatch.setenv("PAPER_SEARCH_ENABLED", "true")
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {
            "query": "stroke segmentation",
            "sources": "semantic",
            "max_results_per_source": 1,
            "include_needs_review": True,
            "run_async": True,
        },
    )
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: sent.update(payload=payload, status=status))
    monkeypatch.setattr(
        web_app,
        "search_papers",
        lambda *args, **kwargs: {
            "query": "stroke segmentation",
            "sources_used": ["semantic"],
            "source_results": {"semantic": 1},
            "errors": {},
            "raw_count": 1,
            "papers": [
                {
                    "title": "Deep learning stroke lesion segmentation on non-contrast head CT",
                    "doi": "10.1000/example",
                    "source": "https://doi.org/10.1000/example",
                    "abstract": "An algorithm for acute ischemic stroke lesion segmentation on non-contrast CT.",
                }
            ],
        },
    )
    monkeypatch.setattr(
        web_app,
        "verify_references",
        lambda references: [
            {
                **reference,
                "verification_status": "verified",
                "verification_sources": ["Crossref"],
                "verification_risks": [],
                "provenance": {"evidence_level": "metadata+abstract"},
            }
            for reference in references
        ],
    )

    handler._handle_literature_search()

    assert sent["status"] == 202
    assert sent["payload"]["status"] == "queued"
    history_id = sent["payload"]["history_id"]
    entry = ResearchWebHandler._history_entry(history_id)
    assert entry["status"] in {"queued", "running", "done"}
    job = web_app.JOBS[sent["payload"]["job_id"]]
    assert job["kind"] == "literature_search"


def test_literature_search_job_endpoint_returns_job() -> None:
    sent = {}
    job_id = "search-job-1"
    web_app.JOBS.clear()
    web_app.JOBS[job_id] = {
        "status": "running",
        "kind": "literature_search",
        "stage": "Searching literature...",
    }
    handler = object.__new__(ResearchWebHandler)
    handler.path = f"/api/literature-search/{job_id}"
    handler._send_json = lambda payload, status=200: sent.update(payload=payload, status=status)

    handler.do_GET()

    assert sent["status"] == 200
    assert sent["payload"]["kind"] == "literature_search"
    assert sent["payload"]["stage"] == "Searching literature..."


def test_history_entries_persist_to_local_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(web_app, "HISTORY_PATH", tmp_path / "history_records.json")
    history_id = ResearchWebHandler._create_history_entry(
        kind="literature_search",
        source="search",
        title="stroke segmentation",
        status="done",
        request={"query": "stroke segmentation"},
        result={"qualified_references": [{"title": "Paper A"}]},
        counts={"qualified": 1},
    )

    entries = ResearchWebHandler._history_entries()
    entry = ResearchWebHandler._history_entry(history_id)

    assert entries[0]["id"] == history_id
    assert entry["title"] == "stroke segmentation"
    assert entry["counts"]["qualified"] == 1


def test_delete_history_entry_removes_local_record(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(web_app, "HISTORY_PATH", tmp_path / "history_records.json")
    history_id = ResearchWebHandler._create_history_entry(
        kind="literature_search",
        source="search",
        title="stroke segmentation",
        status="done",
        request={"query": "stroke segmentation"},
    )

    assert ResearchWebHandler._delete_history_entry(history_id) is True

    assert ResearchWebHandler._history_entry(history_id) is None
    assert ResearchWebHandler._history_entries() == []


def test_history_delete_endpoint_returns_ok(monkeypatch, tmp_path) -> None:
    sent = {}
    monkeypatch.setattr(web_app, "HISTORY_PATH", tmp_path / "history_records.json")
    history_id = ResearchWebHandler._create_history_entry(
        kind="direct_analysis",
        source="direct",
        title="paper",
        status="done",
    )
    handler = object.__new__(ResearchWebHandler)
    handler.path = f"/api/history/{history_id}"
    handler._send_json = lambda payload, status=200: sent.update(payload=payload, status=status)

    handler.do_DELETE()

    assert sent["status"] == 200
    assert sent["payload"] == {"ok": True, "history_id": history_id}
    assert ResearchWebHandler._history_entry(history_id) is None


def test_direct_analysis_history_title_is_compact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(web_app, "HISTORY_PATH", tmp_path / "history_records.json")
    long_file_topic = (
        "07_clinically_informed_preprocessing_stroke_segmentation_low_resource_arxiv.pdf "
        "08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf "
        "09_ischemic_stroke_lesion_segmentation_adversarial_learning_ctp_arxiv.pdf"
    )

    history_id = ResearchWebHandler._create_history_entry(
        kind="direct_analysis",
        source="direct",
        title=long_file_topic,
        status="queued",
        request={
            "topic": long_file_topic,
            "reference_count": 3,
            "references": [
                {"title": "Clinically-Informed Preprocessing Improves Stroke Segmentation in Low-Resource Settings"},
                {"title": "Random Expert Sampling for Deep Learning Segmentation of Acute Ischemic Stroke"},
            ],
        },
        counts={"references": 3},
    )

    entry = ResearchWebHandler._history_entry(history_id)

    assert entry["title"] == "直接分析 · 3篇 · 卒中分割"
    assert entry["request"]["topic"] == long_file_topic


def test_history_references_omit_extracted_document_text() -> None:
    references = ResearchWebHandler._history_references(
        [
            {
                "title": "Uploaded paper",
                "source": "paper.pdf",
                "content_excerpt": "long extracted text",
                "full_text_for_evidence": "full text",
                "raw_source_record": {"large": True},
                "uploaded_filename": "paper.pdf",
            }
        ]
    )

    assert references == [{"title": "Uploaded paper", "source": "paper.pdf", "uploaded_filename": "paper.pdf"}]


def test_literature_search_writes_audit_log(monkeypatch, tmp_path) -> None:
    sent = {}
    handler = object.__new__(ResearchWebHandler)
    handler.server = type("FakeServer", (), {"server_port": 8123})()
    monkeypatch.setattr(web_app, "LOG_DIR", tmp_path)
    monkeypatch.setattr(web_app, "ANNOTATION_RECORD_PATH", tmp_path / "检索标注记录.md")
    monkeypatch.setenv("PAPER_SEARCH_ENABLED", "true")
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {
            "query": "头颅CT脑梗死病灶分割算法",
            "sources": "arxiv",
            "max_results_per_source": 1,
            "include_needs_review": True,
        },
    )
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: sent.update(payload=payload, status=status))
    monkeypatch.setattr(
        web_app,
        "search_papers",
        lambda *args, **kwargs: {
            "query": "头颅CT脑梗死病灶分割算法",
            "backend_query": "acute ischemic stroke non-contrast CT lesion segmentation",
            "rules_fallback_query": "computed tomography ischemic stroke lesion segmentation",
            "llm_search_query": "acute ischemic stroke non-contrast CT lesion segmentation",
            "llm_pubmed_query": '("acute ischemic stroke"[Title/Abstract]) AND ("non-contrast CT"[Title/Abstract]) AND segmentation[Title/Abstract]',
            "llm_error": "",
            "llm_raw_response": '{"search_query":"acute ischemic stroke non-contrast CT lesion segmentation"}',
            "query_rewrite_status": "llm",
            "query_plan": {
                "backend_query": "acute ischemic stroke non-contrast CT lesion segmentation",
                "rules_fallback_query": "computed tomography ischemic stroke lesion segmentation",
                "llm_search_query": "acute ischemic stroke non-contrast CT lesion segmentation",
                "llm_pubmed_query": '("acute ischemic stroke"[Title/Abstract]) AND ("non-contrast CT"[Title/Abstract]) AND segmentation[Title/Abstract]',
                "rewrite_status": "llm",
                "core_concepts": ["acute ischemic stroke", "non-contrast CT", "lesion segmentation"],
            },
            "sources_used": ["pubmed", "crossref"],
            "source_results": {"pubmed": 1, "crossref": 0},
            "errors": {},
            "raw_count": 1,
            "papers": [
                {
                    "title": "Deep learning stroke lesion segmentation on non-contrast head CT",
                    "doi": "10.5555/audit-test",
                    "source": "https://doi.org/10.5555/audit-test",
                    "abstract": "An algorithm for acute ischemic stroke lesion segmentation on non-contrast CT.",
                    "source_origin": "paper_search_mcp",
                    "source_label": "PubMed",
                    "raw_source_record": {"title": "Raw title", "unexpected_field": "kept for metadata audit"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        web_app,
        "verify_references",
        lambda references: [
            {
                **reference,
                "verification_status": "verified",
                "verification_sources": ["Crossref"],
                "verification_risks": [],
            }
            for reference in references
        ],
    )

    handler._handle_literature_search()

    audit_path = Path(sent["payload"]["search_audit_log"])
    assert audit_path.exists()
    assert "port8123" in audit_path.name
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert sent["payload"]["port"] == 8123
    assert audit["port"] == 8123
    assert audit["query"] == "头颅CT脑梗死病灶分割算法"
    assert audit["query_rewrite_status"] == "llm"
    assert audit["rules_fallback_query"] == "computed tomography ischemic stroke lesion segmentation"
    assert audit["llm_search_query"] == "acute ischemic stroke non-contrast CT lesion segmentation"
    assert "Title/Abstract" in audit["llm_pubmed_query"]
    assert audit["llm_error"] == ""
    assert "search_query" in audit["llm_raw_response"]
    assert audit["query_plan"]["core_concepts"] == [
        "acute ischemic stroke",
        "non-contrast CT",
        "lesion segmentation",
    ]
    assert audit["sources_used"] == ["pubmed", "crossref"]
    assert audit["qualified_references"][0]["audit_summary"]["matched_concepts_or_keywords"]
    assert audit["qualified_references"][0]["raw_source_record"]["unexpected_field"] == "kept for metadata audit"
    annotation_path = Path(sent["payload"]["annotation_record"])
    assert annotation_path.exists()
    annotation = annotation_path.read_text(encoding="utf-8")
    assert "port8123" in annotation
    assert "## " in annotation
    assert "头颅CT脑梗死病灶分割算法" in annotation
    assert "Deep learning stroke lesion segmentation on non-contrast head CT" in annotation
    assert "- 建议人工判断：应收" in annotation
    assert "- 建议依据：" in annotation
    assert "- 人工判断：应收" in annotation
    assert "- 人工判断：" in annotation
    assert "- 错误类型：" in annotation
    assert "- 备注：" in annotation


def test_literature_search_can_skip_annotation_record(monkeypatch, tmp_path) -> None:
    sent = {}
    handler = object.__new__(ResearchWebHandler)
    handler.server = type("FakeServer", (), {"server_port": 8124})()
    annotation_path = tmp_path / "annotation.md"
    monkeypatch.setattr(web_app, "LOG_DIR", tmp_path)
    monkeypatch.setattr(web_app, "ANNOTATION_RECORD_PATH", annotation_path)
    monkeypatch.setenv("PAPER_SEARCH_ENABLED", "true")
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {
            "query": "stroke segmentation",
            "sources": "semantic",
            "max_results_per_source": 1,
            "include_needs_review": True,
            "append_annotation_record": False,
        },
    )
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: sent.update(payload=payload, status=status))
    monkeypatch.setattr(
        web_app,
        "search_papers",
        lambda *args, **kwargs: {
            "query": "stroke segmentation",
            "backend_query": "stroke segmentation",
            "rules_fallback_query": "stroke segmentation",
            "query_rewrite_status": "rules",
            "query_plan": {"rewrite_status": "rules"},
            "sources_used": ["semantic"],
            "source_results": {"semantic": 0},
            "errors": {},
            "raw_count": 0,
            "papers": [],
        },
    )

    handler._handle_literature_search()

    assert sent["payload"]["annotation_record_enabled"] is False
    assert sent["payload"]["annotation_record"] == ""
    assert Path(sent["payload"]["search_audit_log"]).exists()
    assert not annotation_path.exists()


def test_analysis_rows_preserve_verification_metadata() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "A DOI paper", "source": "https://doi.org/10.1000/example"}],
        [
            {
                "index": 0,
                "title": "A DOI paper",
                "source": "https://doi.org/10.1000/example",
                "source_origin": "paper_search_mcp",
                "source_label": "Semantic Scholar",
                "screening_status": "qualified",
                "verification_status": "verified",
                "verification_risks": [],
                "provenance": {"retrieved_from": "semantic", "verified_by": "Crossref", "evidence_level": "metadata+abstract"},
            }
        ],
        output_language="en",
    )

    assert rows[0]["source_origin"] == "paper_search_mcp"
    assert rows[0]["source_label"] == "Semantic Scholar"
    assert rows[0]["verification_status"] == "verified"
    assert rows[0]["provenance"]["evidence_level"] == "metadata+abstract"


def test_pdf_symbol_normalization_removes_superscript_artifacts() -> None:
    assert ResearchWebHandler._normalize_pdf_symbols("10⁻⁶ ≤ x ≥ 3 × y") == "10^-6 <= x >= 3 x y"


def test_extracted_text_normalization_repairs_utf8_latin1_mojibake() -> None:
    utf8_bytes = b"\xe4\xb8\xad\xe6\x96\x87\xe6\x91\x98\xe8\xa6\x81"
    mojibake = utf8_bytes.decode("latin-1")

    assert ResearchWebHandler._normalize_extracted_text(mojibake) == "中文摘要"


def test_extracted_text_normalization_cleans_pdf_artifacts() -> None:
    raw = "of\ufb01ce\r\nB\u00adreak\u200b\n\n\nC"

    assert ResearchWebHandler._normalize_extracted_text(raw) == "office\nBreak\nC"


def test_pdf_extract_page_limit_is_runtime_configurable(monkeypatch) -> None:
    monkeypatch.delenv("PDF_EXTRACT_PAGE_LIMIT", raising=False)
    assert ResearchWebHandler._pdf_extract_page_limit() == 120

    monkeypatch.setenv("PDF_EXTRACT_PAGE_LIMIT", "200")
    assert ResearchWebHandler._pdf_extract_page_limit() == 200

    monkeypatch.setenv("PDF_EXTRACT_PAGE_LIMIT", "all")
    assert ResearchWebHandler._pdf_extract_page_limit() is None


def test_pdf_export_detects_wide_tables_and_normalizes_symbols() -> None:
    markdown = """
| A | B | C | D | E |
| --- | --- | --- | --- | --- |
| p < 10⁻⁶ | Non–Contrast | x | y | z |
""".strip()

    assert ResearchWebHandler._markdown_has_wide_table(markdown)
    assert ResearchWebHandler._normalize_pdf_symbols("p < 10⁻⁶; Non–Contrast") == "p < 10^-6; Non-Contrast"


def test_uploaded_pdf_reference_uses_apa_document_format() -> None:
    references = format_references(
        [
            {
                "title": "08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf",
                "source": "08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf",
                "source_origin": "user_upload",
                "source_label": "Uploaded PDF",
                "document_type": "PDF",
                "authors": "",
                "year": "",
            }
        ],
        "APA",
    )

    assert references == [
        "Random expert sampling deep learning segmentation ais ncct arxiv. (n.d.). [PDF]. User-provided document."
    ]
    assert "Uploaded PDF" not in references[0]


def test_uploaded_pdf_reference_keeps_extracted_authors() -> None:
    references = format_references(
        [
            {
                "title": "Clinically-aligned ischemic stroke segmentation and ASPECTS scoring on NCCT imaging using a slice-gated loss on foundation representations",
                "source_origin": "user_upload",
                "document_type": "PDF",
                "authors": ["Azeem, H.", "Khan, B.", "Syed, T. Q."],
                "year": "2025",
            }
        ],
        "APA",
    )

    assert references == [
        "Azeem, H., Khan, B., & Syed, T. Q. (2025). Clinically-aligned ischemic stroke segmentation and ASPECTS scoring on NCCT imaging using a slice-gated loss on foundation representations. [PDF]. User-provided document."
    ]


def test_uploaded_pdf_reference_with_doi_uses_scholarly_source() -> None:
    references = format_references(
        [
            {
                "title": "Random Expert Sampling for Deep Learning Segmentation of Acute Ischemic Stroke on Non-contrast CT",
                "source_origin": "user_upload",
                "document_type": "PDF",
                "authors": ["Ostmeier, S.", "Axelrod, B., PhD,", "Pulli, B.", "Verhaaren, B. F."],
                "year": "2023",
                "journal": "Journal of NeuroInterventional Surgery",
                "doi": "10.1136/jnis-2023-020418",
            }
        ],
        "APA",
    )

    assert references == [
        "Ostmeier, S., Axelrod, B., Pulli, B., & Verhaaren, B. F. (2023). Random Expert Sampling for Deep Learning Segmentation of Acute Ischemic Stroke on Non-contrast CT. Journal of NeuroInterventional Surgery. https://doi.org/10.1136/jnis-2023-020418"
    ]
    assert "User-provided document" not in references[0]
    assert "PhD" not in references[0]
    assert "[PDF]" not in references[0]


def test_uploaded_pdf_reference_with_arxiv_uses_arxiv_source() -> None:
    references = format_references(
        [
            {
                "title": "Clinically-Informed Preprocessing Improves Stroke Segmentation in Low-Resource Settings",
                "source_origin": "user_upload",
                "document_type": "PDF",
                "authors": "Heras Rivera, J. E.; Oswal, H.; Ren, T.",
                "year": "2025",
                "arxiv_id": "2508.16004",
            }
        ],
        "APA",
    )

    assert references == [
        "Heras Rivera, J. E., Oswal, H., & Ren, T. (2025). Clinically-Informed Preprocessing Improves Stroke Segmentation in Low-Resource Settings. arXiv. https://arxiv.org/abs/2508.16004"
    ]
    assert "User-provided document" not in references[0]


def test_author_cleanup_removes_connector_and_degree_noise() -> None:
    references = format_references(
        [
            {
                "title": "Ischemic Stroke Lesion Segmentation Using Adversarial Learning",
                "authors": "Islam, M., Vaidyanathan, N. R., Jose, V. J. M., & and,",
                "year": "2018",
                "source": "arXiv",
                "arxiv_id": "1808.00000",
            }
        ],
        "APA",
    )

    assert references == [
        "Islam, M., Vaidyanathan, N. R., & Jose, V. J. M. (2018). Ischemic Stroke Lesion Segmentation Using Adversarial Learning. arXiv. https://arxiv.org/abs/1808.00000"
    ]
    assert "& and" not in references[0]


def test_identifier_extraction_reads_bibliographic_identity_only() -> None:
    reference = {
        "title": "Uploaded PDF",
        "source": "paper.pdf",
        "bibliographic_identity": "Published at arXiv:2508.16004 and DOI 10.48550/arXiv.2508.16004.",
        "evidence_source_text": "References include DOI 10.1056/NEJMoa2214403.",
    }

    assert extract_arxiv_id(reference) == "2508.16004"
    assert extract_doi(reference) == "10.48550/arXiv.2508.16004"


def test_identifier_extraction_ignores_cited_references_in_full_text() -> None:
    reference = {
        "title": "Clinically-Informed Preprocessing Improves Stroke Segmentation in Low-Resource Settings",
        "source": "07_clinically_informed_preprocessing_stroke_segmentation_low_resource_arxiv.pdf",
        "evidence_source_text": (
            "The method uses clinical thresholds. References: Bandera et al. "
            "https://doi.org/10.1161/01.STR.0000217418.29609.22"
        ),
    }

    assert extract_doi(reference) == ""


def test_pmid_extraction_requires_explicit_pubmed_marker() -> None:
    uploaded_reference = {
        "title": "Random Forests",
        "source": "random_forests.pdf",
        "source_origin": "user_upload",
        "bibliographic_identity": (
            "Machine Learning 45 5-32 2001. "
            "The experiment reports identifiers such as 994018 and 8352055 in extracted text."
        ),
    }

    assert extract_pmid(uploaded_reference) == ""
    assert extract_pmid({"source": "https://pubmed.ncbi.nlm.nih.gov/994018/"}) == "994018"
    assert extract_pmid({"bibliographic_identity": "PMID: 8352055"}) == "8352055"


def test_uploaded_pdf_ieee_reference_uses_document_format() -> None:
    references = format_references(
        [
            {
                "title": "08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf",
                "source": "08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf",
                "source_origin": "user_upload",
                "document_type": "PDF",
                "authors": "",
                "year": "2023",
            }
        ],
        "IEEE",
    )

    assert references == [
        '[1] "Random expert sampling deep learning segmentation ais ncct arxiv," User-provided document, PDF, 2023.'
    ]
    assert "Unknown author" not in references[0]
    assert ".pdf" not in references[0]


def test_ieee_arxiv_reference_uses_id_not_url() -> None:
    references = format_references(
        [
            {
                "title": "Synchronous Image-Label Diffusion Probability Model with Application to Stroke Lesion Segmentation on Non-contrast CT",
                "authors": ["Wu, J. Z."],
                "source": "arXiv",
                "id": "https://arxiv.org/abs/2307.01740",
                "abs_url": "https://arxiv.org/abs/2307.01740",
                "year": "2023",
            }
        ],
        "IEEE",
    )

    assert references == [
        '[1] J. Z. Wu, "Synchronous Image-Label Diffusion Probability Model with Application to Stroke Lesion Segmentation on Non-contrast CT," arXiv:2307.01740, 2023.'
    ]


def test_literature_analysis_rows_preserve_uploaded_reference_title() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "01_automatic_segmentation_stroke_lesions_ncct_cnn_openreview",
                "source": "Uploaded PDF",
                "contribution": "Uses CNNs for stroke lesion segmentation.",
            },
            {
                "title": "Acute ischemic stroke lesion segmentation in non-contrast CT images using 3D convolutional neural networks A.V. Dobshik1",
                "source": "",
                "contribution": "Segments acute ischemic stroke lesions.",
            },
        ],
        [
            {
                "index": 0,
                "title": "Automatic segmentation of stroke lesions in non-contrast CT using convolutional neural networks",
                "source": "01_automatic_segmentation_stroke_lesions_ncct_cnn_openreview.pdf",
                "source_origin": "user_upload",
            },
            {
                "index": 1,
                "title": "Acute ischemic stroke lesion segmentation in non-contrast CT images using 3D convolutional neural networks",
                "source": "acute_ischemic_stroke_lesion_segmentation.pdf",
                "source_origin": "user_upload",
            },
        ],
        output_language="en",
    )

    assert [row["title"] for row in rows] == [
        "Automatic segmentation of stroke lesions in non-contrast CT using convolutional neural networks",
        "Acute ischemic stroke lesion segmentation in non-contrast CT images using 3D convolutional neural networks",
    ]
    assert rows[0]["source"] == "01_automatic_segmentation_stroke_lesions_ncct_cnn_openreview.pdf"


def test_literature_rows_do_not_backfill_evidence_from_conflicting_reference_index() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Paper B",
                "source": "paper-b.pdf",
                "dataset_or_material": "unclear",
            }
        ],
        [
            {
                "index": 0,
                "title": "Paper A",
                "source": "paper-a.pdf",
                "evidence_source_text": "Paper A reports 12 dogs and coronary venous blood samples.",
            },
            {
                "index": 1,
                "title": "Paper B",
                "source": "paper-b.pdf",
                "evidence_source_text": "Paper B reports 60,000 training and 10,000 testing images.",
            },
        ],
        output_language="en",
    )

    assert len(rows) == 1
    assert rows[0]["title"] == "Paper B"
    assert "60,000 training" in rows[0]["sample_size"]
    assert "10,000 testing" in rows[0]["sample_size"]
    assert "12 dogs" not in rows[0].get("sample_size", "")
    assert "coronary" not in rows[0].get("evidence_candidates", "").casefold()


def test_literature_summary_coverage_flags_missing_paper() -> None:
    rows = [
        {"title": "Clinical preprocessing for stroke segmentation", "methodology": "Preprocessing"},
        {"title": "Random expert sampling for NCCT stroke segmentation", "methodology": "Sampling"},
        {"title": "Adversarial learning for CTP stroke segmentation", "methodology": "Adversarial U-Net"},
        {"title": "Foundation representations for ASPECTS scoring", "methodology": "DINO and TAGL"},
    ]
    summary = {
        "overall_assessment": "Compares preprocessing, random expert sampling, and adversarial learning.",
        "common_strengths": [],
        "common_weaknesses": [],
        "methodological_patterns": [],
        "evidence_gaps": [],
        "research_gaps": [],
        "recommended_reading_order": [],
        "next_actions": [],
        "confidence": "Medium",
    }

    checked = LiteratureAnalysisWorkflow._ensure_summary_coverage(summary, rows, output_language="en")

    assert "Foundation representations for ASPECTS scoring" in " ".join(checked["next_actions"])
    assert "Coverage incomplete" in checked["confidence"]


def test_integration_unavailable_summary_does_not_fake_synthesis() -> None:
    rows = [
        {"title": "Paper A", "methodology": "Method A"},
        {"title": "Paper B", "methodology": "Method B"},
    ]

    summary = LiteratureAnalysisWorkflow._integration_unavailable_summary(rows, output_language="en")

    assert "did not produce a reliable synthesis" in summary["overall_assessment"]
    assert summary["common_strengths"] == []
    assert summary["methodological_patterns"] == []
    assert "Low" in summary["confidence"]


def test_literature_analyst_prompt_keeps_grounding_constraints() -> None:
    llm = CaptureLiteratureLLM()
    workflow = LiteratureAnalysisWorkflow(llm=llm)
    group = {
        "name": "Contribution Analyst",
        "focus": "contribution",
        "references": [
            {
                "index": 0,
                "title": "Paper A",
                "source": "paper-a.pdf",
                "abstract": "Abstract text",
                "content_excerpt": "Full text excerpt",
                "pdf_text_available": True,
            }
        ],
    }

    asyncio.run(workflow._run_analyst("stroke segmentation", group, "", "en"))

    prompt = next(call["system_prompt"] for call in llm.calls if "Return only valid JSON with this schema" in call["system_prompt"])
    user_prompt = next(call["user_prompt"] for call in llm.calls if "Return only valid JSON with this schema" in call["system_prompt"])
    assert "Preserve the original title and source URL when provided." in prompt
    assert "Do not fabricate experiments, datasets, results, or claims" in prompt
    assert "When pdf_text_available is true, ground the analysis in content_excerpt" in prompt
    assert "treat the row as metadata-only" in prompt
    assert "Return only valid JSON" in prompt
    assert "dataset" in prompt
    assert "key_results" in prompt
    assert "study_type" in prompt
    assert "statistical_evidence" in prompt
    assert "Do not specialize the extraction to any medical field" in prompt
    assert "put inferred limitations" in prompt.lower()
    assert "llm_evidence_brief" in user_prompt
    assert "llm_gap_audit" in user_prompt
    assert "Reference isolation rule" in user_prompt
    assert "Do not use evidence from any other reference" in user_prompt
    assert "actual novelty" in prompt
    assert "真实新意" in prompt
    assert "不要用泛泛表扬" in prompt
    assert "generic scholarly-paper strategy" in prompt
    assert "Abstract; Methods / Methodology / Materials and Methods" in prompt
    assert "NCCT, CTA, CTP" not in prompt


def test_literature_integrator_prompt_keeps_synthesis_constraints() -> None:
    llm = CaptureLiteratureLLM()
    workflow = LiteratureAnalysisWorkflow(llm=llm)
    references = [
        {
            "index": 0,
            "title": "Paper A",
            "source": "paper-a.pdf",
            "abstract": "Abstract text",
        }
    ]
    analyst_outputs = [
        {
            "analyst": "Contribution Analyst",
            "rows": [
                {
                    "reference_index": 0,
                    "title": "Paper A",
                    "source": "paper-a.pdf",
                    "contribution": "Contribution",
                    "methodology": "Method",
                }
            ],
        }
    ]

    asyncio.run(workflow._integrate("stroke segmentation", references, analyst_outputs, "", "en"))

    prompt = llm.calls[0]["system_prompt"]
    assert "Keep one row per important reference." in prompt
    assert "Preserve source URLs." in prompt
    assert "Preserve the original reference index, title, and source exactly as supplied." in prompt
    assert "Do not invent unsupported details" in prompt
    assert "The summary must synthesize across references; do not simply restate each row." in prompt
    assert "The summary must cover every original reference at least once." in prompt
    assert "Any cross-paper claim using absolute wording" in prompt
    assert "limitations must contain only author-acknowledged limitations" in prompt
    assert "study_type" in prompt
    assert "statistical_evidence" in prompt
    assert "actual novelty" in prompt
    assert "真实新意" in prompt
    assert "不要用泛泛表扬" in prompt
    assert "Return only valid JSON" in prompt
    assert "Structured analyst rows" in llm.calls[0]["user_prompt"]
    assert "Comparison matrix" in llm.calls[0]["user_prompt"]


def test_literature_fact_consistency_checks_flag_generic_risks() -> None:
    summary = {
        "overall_assessment": "All papers only report one metric and no external validation.",
        "next_actions": [],
        "confidence": "Medium",
    }
    rows = [
        {
            "title": "Paper A",
            "study_type": "experimental study",
            "research_objective": "Objective",
            "dataset_or_material": "Dataset A",
            "sample_size": "unclear",
            "domain_or_modality": "Text",
            "method": "Model A",
            "baseline_or_comparator": "compared with baseline",
            "evaluation_protocol": "external validation and test split",
            "metrics": "Accuracy, F1",
            "key_results": "n=120 participants; F1 0.82",
            "statistical_evidence": "95% confidence interval",
            "availability": "unclear",
            "limitations": "unclear",
            "evidence_locations": "Results",
            "evidence_quotes": "n=120 participants",
        }
    ]

    checked = LiteratureAnalysisWorkflow._apply_fact_consistency_checks(summary, rows, output_language="en")

    risk_text = " ".join(checked["fact_risks"])
    assert "absolute wording" in risk_text
    assert "metric single-ness" in risk_text
    assert "missing validation/comparators" in risk_text
    assert "Possible missed sample size extraction" in risk_text


def test_literature_fact_risk_normalization_cleans_list_strings_and_unclear_wrappers() -> None:
    risks = LiteratureAnalysisWorkflow._dedupe_fact_risks(
        LiteratureAnalysisWorkflow._normalize_risk_list(
            "['Sample size unclear', 'unclear（作者未在提取内容中明确讨论局限性）', 'Availability unclear,']"
        )
        + [
            "unclear",
            "Paper A: Author-acknowledged limitations were not clearly extracted; do not replace them with inferred limitations.",
            "Paper A: Author-acknowledged limitations were not clearly extracted; this inferred limitation belongs in fact_risks.",
        ]
    )

    assert "Sample size unclear" in risks
    assert "作者未在提取内容中明确讨论局限性" in risks
    assert "Availability unclear" in risks
    assert "unclear" not in risks
    assert len([risk for risk in risks if "Author-acknowledged limitations" in risk]) == 1


def test_literature_fact_consistency_checks_clean_existing_summary_risks() -> None:
    checked = LiteratureAnalysisWorkflow._apply_fact_consistency_checks(
        {
            "overall_assessment": "All papers make the same claim.",
            "fact_risks": "['unclear(sample size not stated)', 'Availability unclear,']",
            "next_actions": [],
            "confidence": "Medium",
        },
        [
            {
                "title": "Paper A",
                "study_type": "experimental study",
                "research_objective": "Objective",
                "dataset_or_material": "Dataset A",
                "sample_size": "unclear",
                "domain_or_modality": "Text",
                "method": "Model A",
                "baseline_or_comparator": "Baseline",
                "evaluation_protocol": "Test split",
                "metrics": "Accuracy",
                "key_results": "Accuracy 0.8",
                "statistical_evidence": "unclear",
                "availability": "unclear",
                "limitations": "unclear",
                "evidence_locations": "Abstract",
            }
        ],
        output_language="en",
    )

    assert "sample size not stated" in checked["fact_risks"]
    assert "Availability unclear" in checked["fact_risks"]
    assert checked["overall_assessment"].startswith("Based on the extracted structured evidence")
    assert any("qualified claims" in action for action in checked["next_actions"])


def test_literature_row_normalization_protects_proper_noun_fields() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "High System Metric Dataset",
                "source": "paper.pdf",
                "study_type": "Experimental study",
                "dataset_or_material": "High System Dataset",
                "method": "HighNet",
                "metrics": "HighScore",
            }
        ],
        [{"index": 0, "title": "High System Metric Dataset", "source": "paper.pdf"}],
        output_language="zh",
    )

    row = rows[0]
    assert row["study_type"] == "实验研究"
    assert row["title"] == "High System Metric Dataset"
    assert row["dataset_or_material"] == "High System Dataset"
    assert row["method"] == "HighNet"
    assert row["metrics"] == "HighScore"


def test_literature_row_normalization_preserves_reader_quality_fields() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Paper A",
                "source": "paper-a.pdf",
                "innovation_point": "Frames evaluation around reader-verifiable evidence.",
                "reader_takeaway": "Use this paper as a method reference.",
                "reader_next_step": "Check the ablation table and data availability statement.",
            }
        ],
        [{"index": 0, "title": "Paper A", "source": "paper-a.pdf"}],
        output_language="en",
    )

    row = rows[0]
    assert row["innovation_point"] == "Frames evaluation around reader-verifiable evidence."
    assert row["reader_next_step"] == "Check the ablation table and data availability statement."
    assert row["innovation"] == row["innovation_point"]
    assert row["next_step"] == row["reader_next_step"]


def test_frontend_analysis_export_uses_innovation_label_and_fallbacks() -> None:
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")

    assert '<th>创新点</th>' in index_html
    assert "主要优势" not in index_html
    assert '{ label: "创新点", value: (row) => row.innovation_point || row.innovation || row.strengths || "" }' in app_js
    assert 'fallback: "未明确报告样本量/规模"' in app_js
    assert 'fallback: "未明确报告统计证据"' in app_js
    assert 'fallback: "未明确说明公开性"' in app_js
    assert 'fallback: "未定位到证据位置"' in app_js
    assert '{ label: "主要局限", value: (row) => row.limitations || row.weaknesses || row.limitation || "", fallback: "作者未明确说明" }' in app_js
    assert '{ label: "limitations / 作者承认的局限", value: (row) => row.limitations, fallback: "作者未明确说明" }' in app_js
    assert 'reviewCell(row.limitations || row.weaknesses || row.limitation, "作者未明确说明")' in app_js
    assert "主要优势" not in app_js
    assert "return row.what_is_new || row.innovation_point || row.innovation || row.strengths || \"\";" in app_js


def test_literature_row_normalization_separates_inferred_limitations_from_fact_risks() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Paper A",
                "source": "paper-a.pdf",
                "study_type": "Experimental study",
                "research_objective": "Objective",
                "dataset_or_material": "Dataset A",
                "sample_size": "not clearly reported",
                "domain_or_modality": "Text",
                "method": "Method A",
                "baseline_or_comparator": "Baseline",
                "evaluation_protocol": "Test set",
                "metrics": "Accuracy",
                "key_results": "Accuracy 0.8",
                "statistical_evidence": "not reported",
                "availability": "not stated",
                "limitations": "Authors did not explicitly state limitations; inferred limitation: the dataset may be small.",
                "evidence_locations": "Abstract",
            }
        ],
        [{"index": 0, "title": "Paper A", "source": "paper-a.pdf"}],
        output_language="en",
    )

    row = rows[0]
    assert row["limitations"] == "unclear"
    assert "inferred limitation" in row["fact_risks"]
    assert "Sample size" in row["fact_risks"]
    assert "Availability" in row["fact_risks"]
    assert "Author-acknowledged limitations" in row["fact_risks"]


def test_literature_row_normalization_backfills_training_testing_sample_size() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper A", "source": "paper-a.pdf", "sample_size": "unclear"}],
        [
            {
                "index": 0,
                "title": "Paper A",
                "source": "paper-a.pdf",
                "content_excerpt": "The dataset contains 250 sets, with 150 training / 100 testing sets used for the challenge.",
            }
        ],
        output_language="en",
    )

    assert "150 training / 100 testing" in rows[0]["sample_size"]
    assert "Sample-size candidate" in rows[0]["evidence_locations"]


def test_literature_candidates_use_full_evidence_source_before_excerpt() -> None:
    slim = LiteratureAnalysisWorkflow._slim_reference(
        {
            "index": 0,
            "title": "Paper Full",
            "source": "paper-full.pdf",
            "content_excerpt": "Opening excerpt without sample details.",
            "evidence_source_text": (
                "Opening excerpt without sample details. "
                "Methods. The dataset contains 250 sets, with 150 training / 100 testing sets."
            ),
        },
        excerpt_chars=40,
        include_excerpt=True,
    )

    candidates = slim["evidence_candidates"]["sample_size"]
    assert isinstance(candidates[0], dict)
    assert {"slot", "value", "snippet", "section", "page", "confidence"}.issubset(candidates[0])
    assert any("250 sets" in item["value"] for item in candidates)
    assert "evidence_source_text" not in slim


def test_literature_row_normalization_backfills_training_testing_cases() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper B", "source": "paper-b.pdf", "sample_size": "unclear"}],
        [
            {
                "index": 0,
                "title": "Paper B",
                "source": "paper-b.pdf",
                "content_excerpt": "ISLES 2018 includes 94 training / 62 testing cases for evaluation.",
            }
        ],
        output_language="en",
    )

    assert "94 training / 62 testing" in rows[0]["sample_size"]


def test_literature_keyword_windows_include_requested_chinese_and_english_terms() -> None:
    text = (
        "Opening. 样本量、病例、训练集、测试集、外部验证、消融实验、显著性、置信区间、"
        "代码公开、数据可用、局限性 are important. "
        "Sample size, cases, training set, test set, external validation, ablation, "
        "significance, confidence interval, code available, data available, limitations."
    )

    windows = LiteratureAnalysisWorkflow._generic_keyword_windows(text, max_chars=1200)

    assert "样本量" in windows
    assert "训练集" in windows
    assert "外部验证" in windows
    assert "代码公开" in windows
    assert "Sample size" in windows
    assert "external validation" in windows


def test_literature_evidence_candidates_extract_chinese_requested_terms() -> None:
    candidates = LiteratureAnalysisWorkflow._extract_evidence_candidates(
        {
            "title": "中文证据论文",
            "content_excerpt": (
                "方法. 样本量为120例病例，其中训练集80例，测试集40例。"
                "采用外部验证和消融实验。"
                "结果显示准确率 = 90%，差异具有统计学显著性，95%CI 为0.80-0.95。"
                "代码公开在 GitHub，数据可用。"
                "局限性. 本研究样本量小且为单中心，泛化能力有限。"
            ),
        }
    )

    assert "sample_size" in candidates
    assert "train_test_split" in candidates
    assert "evaluation_protocol" in candidates
    assert "statistical_evidence" in candidates
    assert "availability" in candidates
    assert "limitations" in candidates


def test_literature_row_normalization_backfills_author_limitations() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper C", "source": "paper-c.pdf", "limitations": "unclear"}],
        [
            {
                "index": 0,
                "title": "Paper C",
                "source": "paper-c.pdf",
                "content_excerpt": (
                    "Limitations. This study used a single trial cohort from DEFUSE 3. "
                    "It did not include stroke mimics or hemorrhage patients. "
                    "Only three experts provided annotations. "
                    "Stroke volume is only one factor associated with clinical outcomes."
                ),
            }
        ],
        output_language="en",
    )

    limitations = rows[0]["limitations"].casefold()
    assert "single trial cohort" in limitations
    assert "stroke mimics" in limitations
    assert "three experts" in limitations
    assert "clinical outcomes" in limitations


def test_literature_limitations_ignore_related_work_limited_by() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper Lim", "source": "paper-lim.pdf", "limitations": "unclear"}],
        [
            {
                "index": 0,
                "title": "Paper Lim",
                "source": "paper-lim.pdf",
                "content_excerpt": (
                    "Related Work. Prior methods are limited by small datasets. "
                    "Methods. We train a model. "
                    "Discussion. Our study is limited by a single-center cohort."
                ),
            }
        ],
        output_language="en",
    )

    limitations = rows[0]["limitations"]
    assert "single-center cohort" in limitations
    assert "Prior methods" not in limitations


def test_literature_unclear_recognizes_chinese_and_english_forms() -> None:
    unclear_values = [
        "未明确说明",
        "未报告",
        "未提及",
        "未在提取片段中明确",
        "not stated",
        "not clearly reported",
    ]

    assert all(not LiteratureAnalysisWorkflow._has_clear_fact(value) for value in unclear_values)


def test_literature_row_normalization_backfills_availability_candidate() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper C2", "source": "paper-c2.pdf", "availability": "unclear"}],
        [
            {
                "index": 0,
                "title": "Paper C2",
                "source": "paper-c2.pdf",
                "content_excerpt": "The trained model and dataset are available upon request from the corresponding author.",
            }
        ],
        output_language="en",
    )

    assert "available upon request" in rows[0]["availability"].casefold()
    assert "Availability candidate" in rows[0]["evidence_locations"]


def test_literature_row_normalization_supplements_multiple_metric_candidates() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [{"reference_index": 0, "title": "Paper C3", "source": "paper-c3.pdf", "metrics": "Dice"}],
        [
            {
                "index": 0,
                "title": "Paper C3",
                "source": "paper-c3.pdf",
                "content_excerpt": "Results report Dice = 0.82, IoU = 0.70, mIoU = 0.66, and F1 = 0.79 on the test set.",
            }
        ],
        output_language="en",
    )

    metric_text = rows[0]["metrics"]
    key_results = rows[0]["key_results"]
    assert "Dice" in metric_text
    assert "IoU" in metric_text
    assert "mIoU" in metric_text
    assert "F1" in metric_text
    assert "IoU = 0.70" in key_results


def test_literature_metric_summary_preserves_table_model_value_alignment() -> None:
    summary = LiteratureAnalysisWorkflow._summarize_metric_values(
        [
            (
                "Table 1. Cross validation performance of different models with training dataset "
                "Models Ours PixelNet [7] U-Net [12] Deeplab v2 [2] ICNet [15] PSPNet [16] "
                "Dice 0.421 0.409 0.419 0.373 0.387 0.319"
            )
        ]
    )

    assert "Dice: Ours = 0.421" in summary
    assert "Dice: U-Net = 0.419" in summary
    assert "Dice: PSPNet = 0.319" in summary


def test_literature_row_normalization_removes_uncertain_metric_comparison() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Adversarial Paper",
                "source": "paper.pdf",
                "metrics": "Dice",
                "key_results": (
                    "Cross-validation Dice 42.10%; test Dice 39%; "
                    "adversarial U-Net outperformed normal U-Net (Dice 41.9% vs 41.9%? actually slightly better)"
                ),
            }
        ],
        [
            {
                "index": 0,
                "title": "Adversarial Paper",
                "source": "paper.pdf",
                "evidence_source_text": (
                    "Table 1. Cross validation performance of different models with training dataset "
                    "Models Ours PixelNet [7] U-Net [12] Deeplab v2 [2] ICNet [15] PSPNet [16] "
                    "Dice 0.421 0.409 0.419 0.373 0.387 0.319. "
                    "Table 2. Performance of our model with testing dataset Dice Hausdorff Avg Distance "
                    "Precision Recall AVD 0.39 17741954.64 17741938.19 0.55 0.36 10.90."
                ),
            }
        ],
        output_language="en",
    )

    key_results = rows[0]["key_results"]
    assert "?" not in key_results
    assert "actually" not in key_results.casefold()
    assert "41.9% vs 41.9%" not in key_results
    assert "Dice: Ours = 0.421" in key_results
    assert "Dice: U-Net = 0.419" in key_results
    assert "self-correcting metric wording" in rows[0]["fact_risks"]


def test_literature_row_normalization_removes_field_inconsistent_english_residue() -> None:
    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Paper With Residue",
                "source": "paper.pdf",
                "baseline_or_comparator": (
                    "nnU-Net默认预处理流程；clinically motivated preprocessing steps and show that "
                    "the proposed pipeline results in a 38% improvement in Dice score over 10 folds "
                    "compared to a nnU-Net model trained with the baseline preprocessing"
                ),
                "evaluation_protocol": "训练/验证80/20划分；and some images were of poor contrast or shaken",
                "limitations": (
                    "作者明确说明：数据集切片数量少（2-18片）、部分图像对比度差或抖动；"
                    "As the training is similar to a min-max game and as the segmentor does not only "
                    "have its loss but also the extra loss from the discriminator, adversarial training "
                    "of any segmentation architecture gives a better result when compared to its normal training"
                ),
            }
        ],
        [{"index": 0, "title": "Paper With Residue", "source": "paper.pdf"}],
        output_language="zh",
    )

    row = rows[0]
    combined = " ".join(
        [
            row["baseline_or_comparator"],
            row["evaluation_protocol"],
            row["limitations"],
        ]
    )
    assert "clinically motivated preprocessing steps" not in combined
    assert "proposed pipeline results" not in combined
    assert "poor contrast or shaken" not in row["evaluation_protocol"]
    assert "As the training is similar" not in row["limitations"]
    assert row["baseline_or_comparator"] == "nnU-Net默认预处理流程"
    assert "field-inconsistent English evidence residue" in row["fact_risks"]


def test_literature_comparator_candidate_summarizer_keeps_comparison_objects_only() -> None:
    summary = LiteratureAnalysisWorkflow._summarize_comparator_candidates(
        [
            (
                "clinically motivated preprocessing steps and show that the proposed pipeline "
                "results in a 38% improvement in Dice score over 10 folds compared to a "
                "nnU-Net model trained with the baseline preprocessing"
            ),
            "SEAN (CNN), UNETR, SwinUNETR, UNet++, StrDiSeg and DINOv3 baseline were evaluated.",
        ]
    )

    assert "nnU-Net" in summary
    assert "SEAN" in summary
    assert "SwinUNETR" in summary
    assert "proposed pipeline results" not in summary
    assert "38% improvement" not in summary


def test_literature_candidate_fragments_do_not_enter_formal_fact_slots() -> None:
    row = {"dataset_or_material": "unclear", "fact_risks": ""}

    LiteratureAnalysisWorkflow._merge_candidate_fact(
        row,
        "dataset_or_material",
        "ation· preprocessing and he convolutional layers Non-Cont pre- dict",
        location="Dataset/material candidate",
        conflict_label="dataset_or_material",
    )

    assert row["dataset_or_material"] == "unclear"
    risks = row["fact_risks"]
    assert "fragmentary" in risks
    assert "ation" in risks


def test_literature_metric_range_audit_catches_out_of_range_values_and_public_exposes_risk() -> None:
    result = LiteratureAnalysisWorkflow._analysis_result(
        [
            {
                "title": "Paper D",
                "metric_values": "ASPECTS private dataset after TAGL mean Dice is 0.767",
                "metrics": "Dice",
                "key_results": "mean Dice 0.767",
            }
        ],
        {
            "overall_assessment": "Across papers, Dice ranges from 28.5%–63.85%.",
            "next_actions": [],
            "confidence": "Medium",
        },
        output_language="en",
    )

    audit_risks = " ".join(result["audit_summary"]["fact_risks"])
    assert "DICE range may omit values" in audit_risks
    assert "76.7%" in audit_risks
    public_text = " ".join(
        [
            result["summary"]["overall_assessment"],
            result["summary"]["confidence"],
            " ".join(result["summary"]["next_actions"]),
            " ".join(result["summary"]["fact_risks"]),
        ]
    )
    assert "Based on the extracted structured evidence" not in public_text
    assert "Fact consistency checks" not in public_text
    assert result["summary"]["fact_risks"]
    assert "DICE range may omit values" in " ".join(result["summary"]["fact_risks"])
    assert "ASPECTS dataset Dice=0.767" in result["summary"]["overall_assessment"]


def test_literature_high_information_excerpt_uses_generic_sections_and_windows() -> None:
    text = (
        "Intro " * 500
        + "Abstract This paper studies a general research question. "
        + "Methods We compare a baseline comparator with a proposed method. "
        + "Results Participants included n=42 subjects and Accuracy, F1, and confidence interval are reported. "
        + "Table 1 reports samples and p-value. "
        + "Limitations The authors note that data access is restricted. "
        + "Conclusion The result should be verified externally. "
        + "Tail " * 500
    )

    excerpt = LiteratureAnalysisWorkflow._high_information_excerpt(text, max_chars=1500)

    assert "Participants included n=42 subjects" in excerpt
    assert "baseline comparator" in excerpt
    assert "Table 1" in excerpt
    assert "confidence interval" in excerpt
    assert "Limitations" in excerpt


def test_frontend_analysis_export_includes_summary_fact_risks_but_not_row_debug_column() -> None:
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    summary_body = app_js.split("function summaryToText(summary)", 1)[1].split("function buildFactSlotTable", 1)[0]
    fact_slot_body = app_js.split("function factSlotColumns()", 1)[1].split("function debugFactSlotColumns()", 1)[0]

    assert "summary.fact_risks" in summary_body
    assert "row.fact_risks" not in fact_slot_body
    assert "debugFactSlotColumns" in app_js


def test_literature_integration_retries_summary_only_before_fallback() -> None:
    llm = FailingIntegratorThenSummaryLLM()
    workflow = LiteratureAnalysisWorkflow(llm=llm)
    references = [
        {"index": 0, "title": "Paper A", "source": "paper-a.pdf"},
        {"index": 1, "title": "Paper B", "source": "paper-b.pdf"},
    ]
    analyst_outputs = [
        {
            "analyst": "Contribution Analyst",
            "rows": [
                {
                    "reference_index": 0,
                    "title": "Paper A",
                    "source": "paper-a.pdf",
                    "methodology": "Method A",
                    "dataset": "Dataset A",
                    "modality": "NCCT",
                    "metrics": "Dice",
                    "key_results": "Dice 0.5",
                    "validation_setup": "5-fold CV",
                },
                {
                    "reference_index": 1,
                    "title": "Paper B",
                    "source": "paper-b.pdf",
                    "methodology": "Method B",
                    "dataset": "Dataset B",
                    "modality": "CTP",
                    "metrics": "Dice",
                    "key_results": "Dice 0.4",
                    "validation_setup": "test set",
                },
            ],
        }
    ]

    result = asyncio.run(workflow._integrate("stroke segmentation", references, analyst_outputs, "", "en"))

    assert len(llm.calls) == 2
    assert "fallback synthesizer" in llm.calls[1]["system_prompt"]
    assert "did not produce a reliable synthesis" not in result["summary"]["overall_assessment"]
    assert result["summary"]["confidence"].startswith("Medium")
    assert len(result["rows"]) == 2


def test_literature_analysis_final_output_runs_dedicated_chinese_translation() -> None:
    llm = TranslatingLiteratureLLM()
    workflow = LiteratureAnalysisWorkflow(llm=llm)
    references = [
        {
            "index": 0,
            "title": "Paper A",
            "source": "paper-a.pdf",
            "doi": "10.1000/test",
        }
    ]
    analyst_outputs = [
        {
            "analyst": "Contribution Analyst",
            "rows": [
                {
                    "reference_index": 0,
                    "title": "Paper A",
                    "source": "paper-a.pdf",
                    "contribution": "Contribution should be translated",
                    "methodology": "Method should be translated",
                    "dataset": "Dataset A",
                    "metrics": "Dice",
                    "key_results": "Dice 0.5",
                }
            ],
        }
    ]

    result = asyncio.run(workflow._integrate("脑梗死分割", references, analyst_outputs, "", "zh"))

    assert len(llm.calls) == 2
    assert "final integrator" in llm.calls[0]["system_prompt"]
    assert "Chinese translation post-processor" in llm.calls[1]["system_prompt"]
    assert "Protected proper nouns and identifiers" in llm.calls[1]["user_prompt"]
    assert result["rows"][0]["title"] == "Paper A"
    assert result["rows"][0]["source"] == "paper-a.pdf"
    assert result["rows"][0]["dataset"] == "Dataset A"
    assert result["rows"][0]["metrics"] == "Dice"
    assert result["rows"][0]["contribution"] == "提出一种值得核验的分割方法"
    assert result["summary"]["overall_assessment"] == "Paper A 已被纳入比较。"


def test_final_language_cleanup_removes_extraction_artifacts() -> None:
    result = {
        "rows": [
            {
                "reference_index": 0,
                "title": "Paper A",
                "source": "https://arxiv.org/abs/2508.16004",
                "evaluation_protocol": (
                    "10折交叉验证；ensities [13]；our proposed model in Table 1；"
                    "to the ISLES’24 challenge, where it achieved first place"
                ),
                "key_results": "在提取证据中多显示为Dice=0.6385；Metric candidate",
            }
        ],
        "summary": {
            "overall_assessment": "四篇文献在提取证据中多显示为聚焦卒中分割。",
            "common_strengths": ["在提取证据中多显示为采用公开数据集"],
            "references": ["Paper A. arXiv. https://arxiv.org/abs/2508.16004"],
            "citation_format": "apa",
        },
    }

    cleaned = LiteratureAnalysisWorkflow._clean_final_report_language(result)

    assert cleaned["rows"][0]["title"] == "Paper A"
    assert cleaned["rows"][0]["source"] == "https://arxiv.org/abs/2508.16004"
    assert cleaned["rows"][0]["key_results"] == "Dice=0.6385"
    assert cleaned["summary"]["overall_assessment"] == "四篇文献聚焦卒中分割。"
    assert cleaned["summary"]["common_strengths"] == ["采用公开数据集"]
    assert "ensities" not in cleaned["rows"][0]["evaluation_protocol"]
    assert "our proposed model" not in cleaned["rows"][0]["evaluation_protocol"]


def test_final_language_cleanup_preserves_metric_and_dataset_terms() -> None:
    text = "方法使用NCCT、CTA、CTP和DWI；报告Dice、IoU和AUC；TAGL用于ASPECTS。"

    assert LiteratureAnalysisWorkflow._clean_public_language_text(text) == text


def test_uploaded_pdf_reference_extracts_wrapped_title() -> None:
    pdf_path = Path("outputs/ct_infarct_segmentation_papers/08_random_expert_sampling_deep_learning_segmentation_ais_ncct_arxiv.pdf")
    if not pdf_path.exists():
        pytest.skip("sample stroke PDF is not available")

    reference = ResearchWebHandler._uploaded_file_to_reference(pdf_path.name, pdf_path.read_bytes())

    assert reference["title"] == (
        "Random Expert Sampling for Deep Learning Segmentation of Acute Ischemic Stroke on Non-contrast CT"
    )
    assert reference["title"] != pdf_path.stem


def test_pdf_extraction_skips_pages_that_pypdf_cannot_read(monkeypatch, capsys) -> None:
    pypdf = pytest.importorskip("pypdf")

    class FakePage:
        def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
            self.text = text
            self.error = error

        def extract_text(self) -> str:
            if self.error:
                raise self.error
            return self.text or ""

    class FakeReader:
        def __init__(self, stream) -> None:
            del stream
            self.metadata = {"/Title": "Synthetic PDF"}
            self.pages = [
                FakePage(error=ValueError("broken font descriptor")),
                FakePage("Readable page text"),
            ]

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    extracted = ResearchWebHandler._extract_pdf_content(b"%PDF")

    assert extracted["page_count"] == 2
    assert extracted["extracted_pages"] == 1
    assert "[Page 2]\nReadable page text" in extracted["text"]
    assert "Skipped 1 page(s)" in extracted["note"]
    assert "skipped PDF page 1" in capsys.readouterr().out


def test_pdf_extraction_uses_pymupdf_fallback_when_pypdf_fails(monkeypatch) -> None:
    pypdf = pytest.importorskip("pypdf")

    class BrokenPage:
        def extract_text(self) -> str:
            raise ValueError("broken font descriptor")

    class FakeReader:
        def __init__(self, stream) -> None:
            del stream
            self.metadata = {}
            self.pages = [BrokenPage(), BrokenPage()]

    def fake_fallback(content: bytes, *, page_limit: int | None) -> dict:
        assert content == b"%PDF"
        assert page_limit is None or page_limit >= 1
        return {
            "text": "[Page 1]\nRecovered text",
            "page_count": 2,
            "extracted_pages": 1,
        }

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    monkeypatch.setattr(
        ResearchWebHandler,
        "_extract_pdf_content_with_pymupdf",
        staticmethod(fake_fallback),
    )

    extracted = ResearchWebHandler._extract_pdf_content(b"%PDF")

    assert extracted["text"] == "[Page 1]\nRecovered text"
    assert extracted["extracted_pages"] == 1
    assert "Used PyMuPDF fallback extraction" in extracted["note"]


def test_uploaded_pdf_with_unstable_identity_is_marked_for_review(monkeypatch) -> None:
    fossil_text = (
        "[Page 1]\n"
        "LETTERS TO NATURE 6088 - V323.indd\n"
        "As AMS techniques are refined to handle smaller samples, it may become possible "
        "to date individual amino acid enantiomers by the 14C method. Older fossils may "
        "not always prove amenable to the determination of amino acid indigeneity during "
        "racemization and diagenesis.\n"
        "[Page 2]\nA neural network training set example appears in a pasted text layer."
    )

    def fake_extract_pdf_content(content: bytes) -> dict:
        assert content == b"%PDF suspicious"
        return {
            "text": fossil_text,
            "page_count": 2,
            "extracted_pages": 2,
            "metadata": {"title": "6088 - V323.indd", "author": ""},
            "note": "Extracted text from 2/2 pages. Used PyMuPDF fallback extraction.",
        }

    monkeypatch.setattr(
        ResearchWebHandler,
        "_extract_pdf_content",
        staticmethod(fake_extract_pdf_content),
    )

    reference = ResearchWebHandler._uploaded_file_to_reference(
        "6088 - V323.pdf",
        b"%PDF suspicious",
        expected_context="machine learning PDFs: random forest, support vector machine, neural networks",
    )

    assert reference["document_role"] == "review_needed"
    assert reference["is_literature_source"] is False
    assert reference["pdf_identity_status"] == "needs_review"
    assert "待复核材料" in reference["review_note"]
    assert reference["evidence_source_text"] == ""
    assert any("fossil/AMS" in reason for reason in reference["review_reasons"])

    literature, context_documents = ResearchWebHandler._split_reference_roles([reference])

    assert literature == []
    assert context_documents[0]["source"] == "6088 - V323.pdf"


def test_uploaded_pdf_ignores_indd_metadata_and_finds_later_nature_title(monkeypatch) -> None:
    nature_page = (
        "[Page 1]\n"
        "NATURE VOL. 323 9 OCTOBER 1986 LETTERS TO NATURE 533\n"
        "delineating the absolute indigeneity of amino acids in fossils.\n"
        "As AMS techniques are refined to handle smaller samples, it may also become possible "
        "to date individual amino acid enantiomers by the 14C method. Older fossils may not "
        "always prove amenable to the determination of amino acid indigeneity during racemization "
        "and diagenesis. References\n"
        "Learning representations\n"
        "by back-propagating errors\n"
        "David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams\n"
        "Institute for Cognitive Science, University of California, San Diego, USA\n"
        "Department of Computer Science, Carnegie-Mellon University, Pittsburgh, USA\n"
        "We describe a new learning procedure, back-propagation, for networks of neurone-like units. "
        "The procedure repeatedly adjusts the weights of the connections in the network."
    )

    def fake_extract_pdf_content(content: bytes) -> dict:
        assert content == b"%PDF nature page"
        return {
            "text": nature_page,
            "page_count": 1,
            "extracted_pages": 1,
            "metadata": {"title": "6088 - V323.indd", "author": ""},
            "note": "Extracted text from 1/1 pages. Used PyMuPDF fallback extraction.",
        }

    monkeypatch.setattr(
        ResearchWebHandler,
        "_extract_pdf_content",
        staticmethod(fake_extract_pdf_content),
    )

    reference = ResearchWebHandler._uploaded_file_to_reference(
        "323533a0.pdf",
        b"%PDF nature page",
        expected_context="machine learning PDFs: neural networks and back-propagation",
    )

    assert reference["title"] == "Learning representations by back-propagating errors"
    assert "Rumelhart" in ", ".join(reference["authors"])
    assert reference.get("document_role", "literature") == "literature"
    assert reference.get("is_literature_source", True) is True
    assert reference.get("pdf_identity_status", "ok") == "ok"


def test_uploaded_pdf_different_topic_is_not_review_needed_without_unstable_identity(monkeypatch) -> None:
    fossil_text = (
        "[Page 1]\n"
        "Amino acid enantiomers and AMS 14C dating in fossils\n"
        "Abstract. This paper studies fossil indigeneity, racemization, diagenesis, "
        "stable isotope evidence, and amino acid preservation."
    )

    def fake_extract_pdf_content(content: bytes) -> dict:
        assert content == b"%PDF fossil"
        return {
            "text": fossil_text,
            "page_count": 1,
            "extracted_pages": 1,
            "metadata": {"title": "Amino acid enantiomers and AMS 14C dating in fossils", "author": ""},
            "note": "Extracted text from 1/1 pages.",
        }

    monkeypatch.setattr(
        ResearchWebHandler,
        "_extract_pdf_content",
        staticmethod(fake_extract_pdf_content),
    )

    reference = ResearchWebHandler._uploaded_file_to_reference(
        "fossil-dating.pdf",
        b"%PDF fossil",
        expected_context="machine learning PDFs: random forest and support vector machine",
    )

    assert reference.get("document_role", "literature") == "literature"
    assert reference.get("is_literature_source", True) is True
    assert reference.get("pdf_identity_status", "ok") == "ok"


def test_literature_rows_preserve_uploaded_filename_when_source_changes() -> None:
    references = [
        {
            "index": 0,
            "title": "Paper With DOI",
            "source": "https://doi.org/10.1000/example",
            "uploaded_filename": "paper-upload.pdf",
            "source_origin": "user_upload",
            "document_role": "literature",
            "is_literature_source": True,
        }
    ]

    rows = LiteratureAnalysisWorkflow._normalize_rows(
        [
            {
                "reference_index": 0,
                "title": "Paper With DOI",
                "source": "https://doi.org/10.1000/example",
                "contribution": "Contribution",
            }
        ],
        references,
        "en",
    )

    assert rows[0]["source"] == "https://doi.org/10.1000/example"
    assert rows[0]["uploaded_filename"] == "paper-upload.pdf"
    assert rows[0]["source_origin"] == "user_upload"


def test_markdown_table_split_handles_escaped_pipe() -> None:
    row = ResearchWebHandler._split_markdown_table_row("| title | a\\|b |")

    assert row == ["title", "a|b"]



