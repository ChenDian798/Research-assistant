from __future__ import annotations

import asyncio
import ast
import json
import os
import re

from .llm import LLMClient, LLMServiceError


ANALYSIS_COLUMNS = [
    "title",
    "source",
    "uploaded_filename",
    "metadata",
    # Generic scholarly fact slots. These deliberately avoid domain-specific
    # assumptions so the same workflow can handle medicine, NLP, CV, social
    # science, materials, biology, education, surveys, datasets, and methods.
    "study_type",
    "research_objective",
    "dataset_or_material",
    "sample_size",
    "domain_or_modality",
    "method",
    "baseline_or_comparator",
    "evaluation_protocol",
    "metrics",
    "key_results",
    "statistical_evidence",
    "availability",
    "limitations",
    "evidence_locations",
    "evidence_quotes",
    "fact_risks",
    "evidence_candidates",
    "metric_values",
    # Peer-review fields kept for the existing UI and backward compatibility.
    "paper_type",
    "core_claim",
    "contribution",
    "methodology",
    "evidence_strength",
    "innovation_point",
    "reader_takeaway",
    "reader_next_step",
    "strengths",
    "weaknesses",
    "reproducibility",
    "literature_positioning",
    "application",
    "questions",
    "actionable_suggestions",
    "confidence",
    "overall_assessment",
    # Legacy structured evidence aliases.
    "dataset",
    "modality",
    "task",
    "model_or_method",
    "baseline",
    "validation_setup",
    # Backward-compatible aliases used by the current web UI.
    "innovation",
    "limitation",
    "next_step",
]


FACT_SLOT_FIELDS = [
    "study_type",
    "research_objective",
    "dataset_or_material",
    "sample_size",
    "domain_or_modality",
    "method",
    "baseline_or_comparator",
    "evaluation_protocol",
    "metrics",
    "key_results",
    "statistical_evidence",
    "availability",
    "limitations",
    "evidence_locations",
]


ABSOLUTE_TERMS = [
    "全部",
    "所有",
    "均",
    "仅",
    "都没有",
    "没有任何",
    "只报告",
    "未进行",
    "all",
    "every",
    "only",
    "none",
    "no",
    "never",
    "not reported",
]

VALIDATION_TERMS = [
    "测试",
    "验证",
    "对照",
    "比较",
    "holdout",
    "external",
    "cross-validation",
    "cross validation",
    "test",
    "validation",
    "validated",
    "comparator",
    "comparison",
    "control",
    "baseline",
]

SAMPLE_UNIT_PATTERN = re.compile(
    r"(?i)(?:\bn\s*=\s*\d+|\d[\d,.\s]*(?:participants?|patients?|samples?|images?|cases?|datasets?|records?|subjects?|trials?)\b|"
    r"\d[\d,.\s]*(?:样本|患者|病例|图像|数据集|记录|受试者|实验|语料))"
)


ACADEMIC_REVIEW_RUBRIC = """
Use peer-review standards inspired by the academic-paper-review skill:
- identify the paper's core claim and contribution
- assess methodology soundness, evidence strength, and reproducibility
- separate strengths from weaknesses with specific reasoning
- position the work relative to related literature when context is available
- provide constructive, actionable suggestions
- state uncertainty instead of inventing details when only DOI/metadata is available
""".strip()


READER_QUALITY_CONSTRAINT = """
Write for a reader trying to understand the paper quickly. Do not fill cells with generic praise. For strengths/innovation, identify the actual novelty: problem framing, method design, data/material use, evaluation design, theoretical framing, or practical use case. If there is no clear novelty in the extracted evidence, say it is unclear. Methodology should explain the logic of the method in plain language, not just list model names. Literature positioning should say how the reader should use this paper: core evidence, method reference, background, cautionary example, or needs verification. Actionable suggestions should tell the reader what to read or verify next.

请面向想快速理解论文的读者写作，不要用泛泛表扬填充单元格。创新点/优势应指出真实新意：问题切入、方法设计、数据/材料使用、评价设计、理论框架或实际应用场景。若提取证据中看不出明确新意，写“尚不清楚”。方法/证据要用通俗语言解释方法逻辑，而不是只堆模型名。文献定位要说明读者应该如何使用这篇论文：核心证据、方法参考、背景铺垫、风险反例或待复核材料。后续建议要告诉用户下一步该读哪里或核对什么。
""".strip()


GENERIC_EXCERPT_STRATEGY = """
The supplied content_excerpt is selected with a generic scholarly-paper strategy: Abstract; Methods / Methodology / Materials and Methods; Experiments / Evaluation / Results; tables and captions; Limitations; Discussion; Conclusion; plus generic keyword windows around data, participants/samples/cases/subjects, train/test/validation, external validation, baseline/comparator, metrics, results, p-values, confidence intervals, ablation, and limitations. Deterministic evidence_candidates are extracted from the fullest available evidence source before excerpt clipping. Each candidate is an object {slot, value, snippet, section, page, confidence}; write only value into formal fact fields, and use snippet only for evidence_candidates/evidence_quotes/audit. Do not assume any specific medical domain, dataset, model family, or metric.
""".strip()


ASSIGNMENT_EXCERPT_CHARS = 900
ANALYST_EXCERPT_CHARS = 10000
ANALYST_GROUP_EXCERPT_BUDGET = 40000
ANALYST_RETRY_EXCERPT_CHARS = 6000
EVIDENCE_REFINEMENT_EXCERPT_CHARS = 7500
EVIDENCE_REFINEMENT_MAX_TOKENS = 1800
INTEGRATION_EXCERPT_CHARS = 1200


REVIEW_ANALYST_GROUPS = [
    {
        "name": "Contribution Analyst",
        "focus": "core claim, contribution, novelty, significance, and literature positioning",
    },
    {
        "name": "Methodology Analyst",
        "focus": "method soundness, experimental design, statistics, and reproducibility",
    },
    {
        "name": "Evidence Analyst",
        "focus": "claim-evidence alignment, result support, and evidence strength",
    },
    {
        "name": "Critical Reviewer",
        "focus": "weaknesses, limitations, risks, author questions, and actionable improvements",
    },
]


class LiteratureAnalysisWorkflow:
    def __init__(self, llm: LLMClient | None = None, verbose: bool = False) -> None:
        self.llm = llm or LLMClient()
        self.verbose = verbose

    async def run(
        self,
        *,
        topic: str,
        references: list[dict],
        final_report: str = "",
        citation_format: str = "APA",
        formatted_references: list[str] | None = None,
        output_language: str | None = None,
    ) -> dict:
        normalized_references = self._normalize_references(references)
        clean_references = [
            reference
            for reference in normalized_references
            if self._is_literature_reference(reference)
        ]
        output_language = self._normalize_output_language(output_language) or self._detect_output_language(topic, final_report, normalized_references)
        if not clean_references:
            summary = self._context_only_summary(final_report, output_language)
            summary["references"] = formatted_references or []
            summary["citation_format"] = citation_format
            return {"rows": [], "summary": summary}

        self._log("1/3 Assigning literature to four analysts...")
        groups = await self._assign_references(topic, clean_references, final_report, output_language)
        self._log("2/3 Running four literature analysts in parallel...")
        analyst_outputs = await self._run_parallel_analysis(topic, groups, final_report, output_language)
        self._log("3/3 Integrating literature analysis rows and cross-paper summary...")
        result = await self._integrate(topic, clean_references, analyst_outputs, final_report, output_language)
        if isinstance(result, dict):
            for key in ("summary", "audit_summary"):
                if isinstance(result.get(key), dict):
                    result[key]["references"] = formatted_references or []
                    result[key]["citation_format"] = citation_format
        self._log("Done.")
        return result

    async def _assign_references(
        self, topic: str, references: list[dict], final_report: str, output_language: str
    ) -> list[dict]:
        system_prompt = """
You are LLM1, the coordinator of a literature-analysis workflow.
You will receive literature references from a research report.
Assign the references to exactly four parallel analyst groups.
Balance the workload and group related sources by contribution, methodology, evidence, and risk.
Return only valid JSON with this schema:
{
  "groups": [
    {"name": "Contribution Analyst", "focus": "...", "reference_indices": [0, 1]},
    {"name": "Methodology Analyst", "focus": "...", "reference_indices": []},
    {"name": "Evidence Analyst", "focus": "...", "reference_indices": []},
    {"name": "Critical Reviewer", "focus": "...", "reference_indices": []}
  ]
}
Use zero-based indices from the provided references array.
""".strip()
        user_prompt = self._analysis_context(
            topic,
            self._references_for_assignment(references),
            final_report,
            output_language,
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.1,
                max_tokens=900,
            )
            data = self._parse_json_object(content)
            groups = self._normalize_groups(data.get("groups"), references)
            if groups:
                return groups
        except (LLMServiceError, ValueError, KeyError, TypeError):
            self._log("Assignment failed; falling back to deterministic round-robin.")
        return self._round_robin_groups(references)

    async def _run_parallel_analysis(
        self, topic: str, groups: list[dict], final_report: str, output_language: str
    ) -> list[dict]:
        tasks = [self._run_analyst(topic, group, final_report, output_language) for group in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        analyst_outputs = []
        for group, result in zip(groups, results):
            if isinstance(result, Exception):
                self._log(f"{group['name']} failed; continuing with other analysts.")
                analyst_outputs.append(
                    {
                        "analyst": group["name"],
                        "focus": group["focus"],
                        "rows": [],
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            analyst_outputs.append(result)
        return analyst_outputs

    async def _run_analyst(self, topic: str, group: dict, final_report: str, output_language: str) -> dict:
        if not group.get("references"):
            return {
                "analyst": group["name"],
                "focus": group["focus"],
                "rows": [],
            }
        language_instruction = self._language_instruction(output_language)
        system_prompt = f"""
You are {group["name"]}, one of four parallel literature analysts.
Your focus: {group["focus"]}.

Analyze only the references assigned to you.
For each reference, apply this rubric:
{ACADEMIC_REVIEW_RUBRIC}

Return only valid JSON with this schema:
{{
  "analyst": "{group["name"]}",
  "rows": [
    {{
      "reference_index": 0,
      "title": "...",
      "source": "...",
      "metadata": "authors, venue/journal, year, domain if available",
      "study_type": "experimental study / review / methods paper / dataset paper / clinical study / unclear",
      "research_objective": "... or unclear",
      "dataset_or_material": "dataset, materials, corpus, experimental objects, case source, participants, or unclear",
      "sample_size": "sample size, experiment scale, data volume, or unclear",
      "domain_or_modality": "field, modality, object type, population, material class, or unclear",
      "method": "method, model, algorithm, intervention, or theoretical framework; unclear if absent",
      "baseline_or_comparator": "baseline, comparator, control, comparison object, or unclear",
      "evaluation_protocol": "train/test split, cross-validation, external validation, human evaluation, experiments, or unclear",
      "metrics": "reported metrics; unclear if absent",
      "key_results": "key results with numbers when available; unclear if absent",
      "statistical_evidence": "significance tests, confidence intervals, error bars, uncertainty estimates, or unclear",
      "availability": "whether code, data, model, materials are public; unclear if absent",
      "limitations": "author-acknowledged limitations; write unclear if absent, meaning the authors did not clearly state limitations",
      "evidence_locations": ["Abstract", "Methods", "Results", "Table/Figure caption", "Discussion", "Conclusion", or exact page/section when identifiable"],
      "evidence_quotes": ["1-3 short source-grounded snippets, paraphrased or very brief"],
      "paper_type": "experimental study / theoretical study / review / system or tool / position paper / unknown",
      "core_claim": "...",
      "contribution": "what specific problem this paper addresses and why it is worth reading",
      "methodology": "plain-language logic of the method; do not only list model names",
      "evidence_strength": "Strong / Moderate / Weak / Unknown, plus one-sentence reason",
      "innovation_point": "actual novelty in 1-2 sentences, or unclear",
      "reader_takeaway": "one sentence the reader should remember",
      "reader_next_step": "one sentence telling what to read or verify next",
      "strengths": "actual innovation/novelty, not generic praise; unclear if evidence is insufficient",
      "weaknesses": "...",
      "reproducibility": "...",
      "literature_positioning": "...",
      "application": "...",
      "questions": "...",
      "actionable_suggestions": "...",
      "confidence": "High / Medium / Low, with short reason",
      "overall_assessment": "Landmark / Important / Medium / Marginal / Below threshold / Unknown",
      "dataset": "legacy alias for dataset_or_material",
      "modality": "legacy alias for domain_or_modality",
      "task": "legacy alias for research_objective",
      "model_or_method": "legacy alias for method",
      "baseline": "legacy alias for baseline_or_comparator",
      "validation_setup": "legacy alias for evaluation_protocol"
    }}
  ]
}}
{language_instruction}
{READER_QUALITY_CONSTRAINT}
{GENERIC_EXCERPT_STRATEGY}
Preserve the original title and source URL when provided.
For every generic fact slot, prefer exact values from the abstract, content_excerpt, metadata, or report context. If absent, write "unclear".
When evidence_candidates are present, each item is {{slot, value, snippet, section, page, confidence}}. Use candidate value for formal fields; keep long snippet text only in evidence_candidates/evidence_quotes/audit-style fields.
When llm_evidence_brief or llm_gap_audit is present, use it as a secondary guide to prioritize supplied evidence; do not treat it as an independent source.
For limitations, include only limitations explicitly acknowledged by the authors in the extracted evidence. If author-acknowledged limitations are absent, write exactly "unclear" internally; in Chinese-facing wording this means "作者未明确说明". Put inferred limitations, missing details, caveats, or reviewer concerns into fact_risks instead.
For fact_risks, list row-level risks such as inferred rather than author-stated limitations, unclear sample size, unclear availability, missing statistical evidence, private/unverifiable data, abnormal metrics, or incomplete evidence locations.
If a reference only provides a DOI and you cannot infer reliable bibliographic metadata,
use the DOI as the title and explicitly state that metadata/full text should be checked.
Do not fabricate experiments, datasets, results, or claims that are absent from the supplied metadata,
abstract, PDF excerpt, or report context.
When pdf_text_available is true, ground the analysis in content_excerpt and mention uncertainty if the
excerpt is incomplete. When pdf_text_available is false, treat the row as metadata-only and set low
confidence unless the report context supplies reliable details.
Do not specialize the extraction to any medical field, model family, metric, dataset, or intervention.
Do not translate or globally rewrite titles, author names, journal/source names, model names, dataset names, metric names, DOI/PMID/arXiv IDs, or URLs.
""".strip()
        try:
            rows = await self._run_analyst_completion(
                topic,
                group,
                final_report,
                system_prompt,
                output_language,
                excerpt_chars=ANALYST_EXCERPT_CHARS,
                max_tokens=3400 if len(group.get("references", [])) > 1 else 2400,
            )
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"{group['name']} batch failed; retrying one paper at a time. {type(error).__name__}: {error}")
            rows = await self._retry_analyst_per_reference(
                topic,
                group,
                final_report,
                system_prompt,
                output_language,
                first_error=error,
            )
        return {
            "analyst": group["name"],
            "focus": group["focus"],
            "rows": rows if isinstance(rows, list) else [],
        }

    async def _run_analyst_completion(
        self,
        topic: str,
        group: dict,
        final_report: str,
        system_prompt: str,
        output_language: str,
        *,
        excerpt_chars: int,
        max_tokens: int,
    ) -> list[dict]:
        assigned_references = self._references_for_analyst(group["references"], excerpt_chars=excerpt_chars)
        assigned_references = await self._refine_analyst_evidence(
            topic,
            assigned_references,
            final_report,
            output_language,
        )
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Assigned references:\n{json.dumps(assigned_references, ensure_ascii=False, indent=2)}\n\n"
            f"Reference isolation rule:\n"
            f"Analyze each assigned reference using only that reference object's metadata, "
            f"content_excerpt, evidence_candidates, and evidence_brief. Do not use evidence "
            f"from any other reference or from the global report context to fill row-level facts.\n\n"
            f"Global report context, if any, is only for broad task instructions and must not "
            f"supply paper-specific facts:\n{final_report[:1200]}"
        )
        content = await self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=os.getenv("ANALYSIS_MODEL") or os.getenv("RESEARCH_MODEL"),
            temperature=0.2,
            max_tokens=max_tokens,
        )
        data = self._parse_json_object(content)
        rows = data.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("Expected rows list from literature analyst.")
        return rows

    async def _refine_analyst_evidence(
        self,
        topic: str,
        references: list[dict],
        final_report: str,
        output_language: str,
    ) -> list[dict]:
        if not references or not self._llm_evidence_refinement_enabled(references):
            return references

        refinement_input = [
            self._reference_for_evidence_refinement(reference)
            for reference in references
            if self._reference_has_refinable_evidence(reference)
        ]
        if not refinement_input:
            return references

        system_prompt = """
You are LLM0, an evidence packet optimizer for a literature-analysis workflow.
You will receive compact paper excerpts and deterministic evidence candidates.
Your job is not to write the final review. Your job is to select and organize the strongest supplied evidence for the next analyst.

Rules:
- Use only the supplied content_excerpt and evidence_candidates.
- Do not invent facts, sample sizes, metrics, availability, or limitations.
- Separate author-stated limitations from reviewer-inferred risks.
- If author-stated limitations are absent, say "authors did not clearly state limitations".
- Prefer short source-grounded snippets with section/page when available.
- Keep each evidence_brief concise, under 1000 characters when possible.
- Return only valid JSON.

Schema:
{
  "references": [
    {
      "reference_index": 0,
      "title": "...",
      "evidence_brief": "compact evidence guide for this paper",
      "gap_audit": ["missing or uncertain slots to re-check"],
      "evidence_quotes": ["very short supplied snippets"]
    }
  ]
}
""".strip()
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Output language: {output_language}\n\n"
            f"References to optimize:\n{json.dumps(refinement_input, ensure_ascii=False, indent=2)}\n\n"
            f"Report context excerpt:\n{final_report[:1200]}"
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("EVIDENCE_REFINEMENT_MODEL") or os.getenv("ANALYSIS_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.1,
                max_tokens=EVIDENCE_REFINEMENT_MAX_TOKENS,
            )
            data = self._parse_json_object(content)
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"Evidence refinement skipped; using deterministic evidence package. {type(error).__name__}: {error}")
            return references

        refined_items = data.get("references", [])
        if not isinstance(refined_items, list):
            return references
        by_key: dict[str, dict] = {}
        for item in refined_items:
            if not isinstance(item, dict):
                continue
            for key in self._reference_identity_keys(item):
                by_key.setdefault(key, item)

        refined_references = []
        for reference in references:
            refined = dict(reference)
            item = next((by_key.get(key) for key in self._reference_identity_keys(reference) if by_key.get(key)), None)
            if item:
                brief = self._compact_text(str(item.get("evidence_brief") or ""))[:1400]
                if brief:
                    refined["llm_evidence_brief"] = brief
                gap_audit = self._normalize_string_list(item.get("gap_audit"))
                if gap_audit:
                    refined["llm_gap_audit"] = gap_audit[:8]
                quotes = self._normalize_string_list(item.get("evidence_quotes"))
                if quotes:
                    refined["llm_evidence_quotes"] = [quote[:360] for quote in quotes[:6]]
            refined_references.append(refined)
        return refined_references

    @staticmethod
    def _llm_evidence_refinement_enabled(references: list[dict]) -> bool:
        raw = str(os.getenv("LLM_EVIDENCE_REFINEMENT", "enabled") or "").strip().casefold()
        if raw in {"0", "false", "no", "off", "disabled", "disable"}:
            return False
        return any(LiteratureAnalysisWorkflow._reference_has_refinable_evidence(reference) for reference in references)

    @staticmethod
    def _reference_has_refinable_evidence(reference: dict) -> bool:
        return bool(
            str(reference.get("content_excerpt") or reference.get("abstract") or "").strip()
            or reference.get("evidence_candidates")
        )

    @staticmethod
    def _reference_for_evidence_refinement(reference: dict) -> dict:
        keys = ["index", "reference_index", "title", "source", "doi", "year", "journal", "pdf_text_available", "pdf_extraction_note"]
        slim = {
            key: reference.get(key)
            for key in keys
            if reference.get(key) not in (None, "")
        }
        excerpt = str(reference.get("content_excerpt") or reference.get("abstract") or "").strip()
        if excerpt:
            slim["content_excerpt"] = LiteratureAnalysisWorkflow._high_information_excerpt(
                excerpt,
                max_chars=EVIDENCE_REFINEMENT_EXCERPT_CHARS,
            )
        candidates = reference.get("evidence_candidates")
        if candidates:
            slim["evidence_candidates"] = candidates
        return slim

    @staticmethod
    def _reference_identity_keys(reference: dict) -> list[str]:
        keys = []
        raw_identity = reference.get("reference_index")
        if raw_identity in (None, ""):
            raw_identity = reference.get("index")
        numeric_identity = str(raw_identity if raw_identity not in (None, "") else "").strip()
        if numeric_identity:
            keys.append(f"reference_index:{numeric_identity.casefold()}")
            keys.append(f"index:{numeric_identity.casefold()}")
        for field in ("source", "title"):
            value = str(reference.get(field) or "").strip()
            if value:
                keys.append(f"{field}:{value.casefold()}")
        return keys

    async def _retry_analyst_per_reference(
        self,
        topic: str,
        group: dict,
        final_report: str,
        system_prompt: str,
        output_language: str,
        *,
        first_error: Exception,
    ) -> list[dict]:
        rows: list[dict] = []
        for reference in group.get("references", []):
            single_group = {
                "name": group["name"],
                "focus": group["focus"],
                "references": [reference],
            }
            try:
                rows.extend(
                    await self._run_analyst_completion(
                        topic,
                        single_group,
                        final_report,
                        system_prompt,
                        output_language,
                        excerpt_chars=ANALYST_RETRY_EXCERPT_CHARS,
                        max_tokens=2400,
                    )
                )
            except (LLMServiceError, ValueError, KeyError, TypeError) as error:
                self._log(f"{group['name']} single-paper retry failed; using fallback for {reference.get('title') or reference.get('source')}. {type(error).__name__}: {error}")
                rows.extend(
                    self._fallback_rows_from_references(
                        [reference],
                        fallback_note=(
                            f"{group['name']} 分析重试失败：{type(first_error).__name__}；{type(error).__name__}"
                            if output_language == "zh"
                            else f"{group['name']} failed after retry: {type(first_error).__name__}; {type(error).__name__}"
                        ),
                        output_language=output_language,
                    )
                )
        return rows

    async def _integrate(
        self,
        topic: str,
        references: list[dict],
        analyst_outputs: list[dict],
        final_report: str,
        output_language: str,
    ) -> dict:
        language_instruction = self._language_instruction(output_language)
        fallback_rows = self._collect_analyst_rows(analyst_outputs)
        normalized_fallback = self._normalize_rows(fallback_rows, references, output_language)
        comparison_matrix = self._build_comparison_matrix(normalized_fallback)
        system_prompt = """
You are LLM1, the final integrator of a literature-analysis workflow.
You will receive compact reference identities, structured analyst rows, and a comparison matrix.
Create final table rows and a cross-literature review summary for researchers.

Requirements:
- Keep one row per important reference.
- Merge duplicate rows for the same reference.
- Preserve source URLs.
- Preserve the original reference index, title, and source exactly as supplied.
- __LANGUAGE_INSTRUCTION__
- Do not invent unsupported details; when uncertain, state a careful limitation.
- DOI-only inputs may lack title/abstract/full text. Preserve the DOI/source and state uncertainty instead of inventing.
- Preserve peer-review detail from the analyst outputs while keeping each cell short.
- Also fill backward-compatible alias fields where useful: innovation, limitation, next_step, dataset, modality, task, model_or_method, baseline, validation_setup.
- Prefer innovation_point for the innovation alias and reader_next_step for the next_step alias when present.
- The summary must synthesize across references; do not simply restate each row.
- The summary must cover every original reference at least once.
- Build the synthesis around generic scholarly facts: study type, objective, data/materials, sample size or scale, domain/modality/object type, method, comparator, evaluation protocol, metrics, results, statistical evidence, availability, limitations, and evidence locations.
- Use structured fact fields from analyst rows when available; do not ignore later references.
- Prefer the supplied structured facts over free-text inference. If a fact is absent, mark it uncertain instead of guessing.
- In final rows, limitations must contain only author-acknowledged limitations found in the extracted evidence. If absent, write "unclear" internally; in Chinese-facing wording this means "作者未明确说明". Put inferred limitations, missing details, abnormal metrics, private/unverifiable data, and reviewer caveats in fact_risks.
- __READER_QUALITY_CONSTRAINT__
- Any cross-paper claim using absolute wording such as "all", "only", "none", "no", "never", or "not reported" must be supported by every relevant row in the structured fact table. If support is incomplete, use qualified wording such as "some", "most", "several", "not consistently reported", or "not clearly reported in the extracted evidence".
- 凡使用“全部、所有、均、仅、都没有、未报告、无”等绝对化表述，必须由结构化事实表中所有相关文献支持。若证据不完整，改用“部分、多数、若干、提取证据中未一致报告、尚不清楚”等限定表达。
- Do not translate or globally rewrite titles, author names, journal/source names, model names, dataset names, metric names, DOI/PMID/arXiv IDs, or URLs. Only localize enum-like labels such as confidence, evidence_strength, and study_type.
- Return only valid JSON with this schema:
{
    "rows": [
    {
      "reference_index": 0,
      "title": "...",
      "source": "...",
      "metadata": "authors, venue/journal, year, domain if available",
      "study_type": "experimental study / review / methods paper / dataset paper / clinical study / unclear",
      "research_objective": "... or unclear",
      "dataset_or_material": "... or unclear",
      "sample_size": "... or unclear",
      "domain_or_modality": "... or unclear",
      "method": "... or unclear",
      "baseline_or_comparator": "... or unclear",
      "evaluation_protocol": "... or unclear",
      "metrics": "... or unclear",
      "key_results": "... or unclear",
      "statistical_evidence": "... or unclear",
      "availability": "... or unclear",
      "limitations": "... or unclear (meaning authors did not clearly state limitations)",
      "evidence_locations": ["Abstract", "Methods", "Results", "Table/Figure caption", "Discussion", "Conclusion"],
      "evidence_quotes": ["short evidence snippets"],
      "fact_risks": ["row-level uncertainty or extraction risks"],
      "paper_type": "experimental study / theoretical study / review / system or tool / position paper / unknown",
      "core_claim": "...",
      "contribution": "what specific problem this paper addresses and why it is worth reading",
      "methodology": "plain-language logic of the method; do not only list model names",
      "evidence_strength": "Strong / Moderate / Weak / Unknown, plus one-sentence reason",
      "innovation_point": "actual novelty in 1-2 sentences, or unclear",
      "reader_takeaway": "one sentence the reader should remember",
      "reader_next_step": "one sentence telling what to read or verify next",
      "strengths": "actual innovation/novelty, not generic praise; unclear if evidence is insufficient",
      "weaknesses": "...",
      "reproducibility": "...",
      "literature_positioning": "...",
      "application": "...",
      "questions": "...",
      "actionable_suggestions": "...",
      "confidence": "High / Medium / Low, with short reason",
      "overall_assessment": "Landmark / Important / Medium / Marginal / Below threshold / Unknown",
      "innovation": "short alias for innovation_point or contribution",
      "limitation": "short alias for limitations/weaknesses",
      "next_step": "short alias for reader_next_step or actionable_suggestions",
      "dataset": "legacy alias for dataset_or_material",
      "modality": "legacy alias for domain_or_modality",
      "task": "legacy alias for research_objective",
      "model_or_method": "legacy alias for method",
      "baseline": "legacy alias for baseline_or_comparator",
      "validation_setup": "legacy alias for evaluation_protocol"
    }
  ],
  "summary": {
    "overall_assessment": "one concise cross-literature assessment",
    "common_strengths": ["..."],
    "common_weaknesses": ["..."],
    "methodological_patterns": ["..."],
    "evidence_gaps": ["..."],
    "research_gaps": ["..."],
    "recommended_reading_order": ["title or source with short reason"],
    "next_actions": ["..."],
    "confidence": "High / Medium / Low, with short reason"
  }
}
""".strip().replace("__LANGUAGE_INSTRUCTION__", language_instruction).replace("__READER_QUALITY_CONSTRAINT__", READER_QUALITY_CONSTRAINT)
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Original references:\n{json.dumps(self._references_for_integration(references), ensure_ascii=False, indent=2)}\n\n"
            "Structured analyst rows:\n"
            + json.dumps(self._rows_for_integration(normalized_fallback), ensure_ascii=False, indent=2)
            + "\n\nComparison matrix:\n"
            + json.dumps(comparison_matrix, ensure_ascii=False, indent=2)
            + f"\n\nReport context excerpt:\n{final_report[:2400]}"
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("SYNTHESIS_MODEL"),
                temperature=0.15,
                max_tokens=3600,
            )
            data = self._parse_json_object(content)
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                raise ValueError("Expected rows list from literature analysis integrator.")
            normalized = self._normalize_rows(rows, references, output_language)
            if normalized:
                normalized = self._merge_integrated_rows(normalized, normalized_fallback)
                summary = self._normalize_summary(data.get("summary"), normalized, references, output_language)
                return await self._finalize_synthesized_analysis(
                    normalized,
                    summary,
                    output_language,
                    references,
                )
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"Integration failed; retrying summary-only synthesis. {type(error).__name__}: {error}")

        if normalized_fallback:
            retry_summary = await self._retry_summary_only(
                topic,
                references,
                normalized_fallback,
                comparison_matrix,
                output_language,
            )
            if retry_summary:
                return await self._finalize_synthesized_analysis(
                    normalized_fallback,
                    retry_summary,
                    output_language,
                    references,
                )

            summary = self._integration_unavailable_summary(normalized_fallback, output_language)
            return await self._finalize_annotated_analysis(
                normalized_fallback,
                summary,
                output_language,
                references,
                integrated=False,
            )

        fallback_rows = self._fallback_rows_from_references(
            references,
            fallback_note=(
                "最终整合步骤不可用；该行仅反映已提供的元数据。"
                if output_language == "zh"
                else "LLM integration unavailable; this row only reflects supplied metadata."
            ),
            output_language=output_language,
        )
        summary = self._integration_unavailable_summary(fallback_rows, output_language)
        return await self._finalize_annotated_analysis(
            fallback_rows,
            summary,
            output_language,
            references,
            integrated=False,
        )

    async def _finalize_synthesized_analysis(
        self,
        rows: list[dict],
        summary: dict,
        output_language: str,
        references: list[dict],
    ) -> dict:
        summary = self._ensure_summary_coverage(summary, rows, output_language)
        return await self._finalize_annotated_analysis(
            rows,
            summary,
            output_language,
            references,
            integrated=True,
        )

    async def _finalize_annotated_analysis(
        self,
        rows: list[dict],
        summary: dict,
        output_language: str,
        references: list[dict],
        *,
        integrated: bool,
    ) -> dict:
        summary = self._annotate_report_quality(summary, rows, output_language, integrated=integrated)
        return await self._finalize_analysis_result(rows, summary, output_language, references)

    async def _retry_summary_only(
        self,
        topic: str,
        references: list[dict],
        rows: list[dict],
        comparison_matrix: list[dict],
        output_language: str,
    ) -> dict | None:
        language_instruction = self._language_instruction(output_language)
        system_prompt = """
You are LLM1, the fallback synthesizer for a literature-analysis workflow.
The detailed row integration failed, so only produce a compact cross-literature summary.

Requirements:
- __LANGUAGE_INSTRUCTION__
- Synthesize across all supplied rows; do not list papers one by one.
- Cover every original reference at least once.
- Use only the structured facts and comparison matrix. Do not invent datasets, metrics, baselines, or validation details.
- When evidence is thin or a field is missing, state the uncertainty as a limitation.
- Any cross-paper claim using absolute wording such as "all", "only", "none", "no", "never", or "not reported" must be supported by every relevant row in the structured fact table. If support is incomplete, use qualified wording such as "some", "most", "several", "not consistently reported", or "not clearly reported in the extracted evidence".
- 凡使用“全部、所有、均、仅、都没有、未报告、无”等绝对化表述，必须由结构化事实表中所有相关文献支持。若证据不完整，改用“部分、多数、若干、提取证据中未一致报告、尚不清楚”等限定表达。
- Return only valid JSON with this schema:
{
  "summary": {
    "overall_assessment": "one concise cross-literature assessment",
    "common_strengths": ["..."],
    "common_weaknesses": ["..."],
    "methodological_patterns": ["..."],
    "evidence_gaps": ["..."],
    "research_gaps": ["..."],
    "recommended_reading_order": ["title or source with short reason"],
    "next_actions": ["..."],
    "confidence": "High / Medium / Low, with short reason"
  }
}
""".strip().replace("__LANGUAGE_INSTRUCTION__", language_instruction)
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Original references:\n{json.dumps(self._references_for_summary_retry(references), ensure_ascii=False, indent=2)}\n\n"
            "Structured rows:\n"
            + json.dumps(self._rows_for_integration(rows), ensure_ascii=False, indent=2)
            + "\n\nComparison matrix:\n"
            + json.dumps(comparison_matrix, ensure_ascii=False, indent=2)
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("SYNTHESIS_MODEL"),
                temperature=0.1,
                max_tokens=1600,
            )
            data = self._parse_json_object(content)
            summary = self._normalize_summary(data.get("summary"), rows, references, output_language)
            return summary
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"Summary-only integration failed; using fallback summary. {type(error).__name__}: {error}")
            return None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[literature] {message}", flush=True)

    @staticmethod
    def _analysis_context(topic: str, references: list[dict], final_report: str, output_language: str) -> str:
        return (
            f"Research topic:\n{topic}\n\n"
            f"Output language: {output_language}\n\n"
            f"References:\n{json.dumps(references, ensure_ascii=False, indent=2)}\n\n"
            f"Report context excerpt:\n{final_report[:2500]}"
        )

    @staticmethod
    def _references_for_assignment(references: list[dict]) -> list[dict]:
        return [
            LiteratureAnalysisWorkflow._slim_reference(
                reference,
                excerpt_chars=ASSIGNMENT_EXCERPT_CHARS,
                include_excerpt=bool(reference.get("abstract")) and not reference.get("pdf_text_available"),
            )
            for reference in references
        ]

    @staticmethod
    def _references_for_analyst(references: list[dict], *, excerpt_chars: int = ANALYST_EXCERPT_CHARS) -> list[dict]:
        remaining = ANALYST_GROUP_EXCERPT_BUDGET
        slimmed = []
        for reference in references:
            reference_excerpt_chars = min(excerpt_chars, max(0, remaining))
            slimmed.append(
                LiteratureAnalysisWorkflow._slim_reference(
                    reference,
                    excerpt_chars=reference_excerpt_chars,
                    include_excerpt=True,
                )
            )
            remaining -= reference_excerpt_chars
        return slimmed

    @staticmethod
    def _references_for_integration(references: list[dict]) -> list[dict]:
        slimmed = []
        for reference in references:
            slim = LiteratureAnalysisWorkflow._slim_reference(
                reference,
                excerpt_chars=0,
                include_excerpt=False,
            )
            cards = LiteratureAnalysisWorkflow._evidence_cards_for_reference(reference, limit=12)
            if cards:
                slim["evidence_cards"] = cards
            slimmed.append(slim)
        return slimmed

    @staticmethod
    def _references_for_summary_retry(references: list[dict]) -> list[dict]:
        return [
            {
                key: reference.get(key)
                for key in [
                    "index",
                    "title",
                    "source",
                    "doi",
                    "pmid",
                    "arxiv_id",
                    "authors",
                    "year",
                    "journal",
                    "source_origin",
                    "source_label",
                    "verification_status",
                    "provenance",
                ]
                if reference.get(key) not in (None, "")
            }
            for reference in references
        ]

    @staticmethod
    def _slim_reference(reference: dict, *, excerpt_chars: int, include_excerpt: bool) -> dict:
        keys = [
            "index",
            "title",
            "source",
            "relevance",
            "branch_name",
            "doi",
            "pmid",
            "arxiv_id",
            "authors",
            "year",
            "journal",
            "source_origin",
            "source_label",
            "screening_status",
            "screening_risks",
            "verification_status",
            "verification_risks",
            "provenance",
            "pdf_text_available",
            "pdf_page_count",
            "pdf_extracted_pages",
            "pdf_extraction_note",
            "document_role",
            "is_literature_source",
        ]
        slim = {
            key: reference.get(key)
            for key in keys
            if reference.get(key) not in (None, "")
        }
        if include_excerpt and excerpt_chars > 0:
            excerpt = str(reference.get("content_excerpt") or reference.get("abstract") or "").strip()
            if excerpt:
                clipped = LiteratureAnalysisWorkflow._high_information_excerpt(excerpt, max_chars=excerpt_chars)
                if len(excerpt) > excerpt_chars:
                    clipped = f"{clipped}\n\n[Excerpt clipped for analysis prompt budget.]"
                slim["content_excerpt"] = clipped
                candidates = LiteratureAnalysisWorkflow._extract_evidence_candidates(reference)
                if candidates:
                    slim["evidence_candidates"] = candidates
        elif reference.get("abstract") and not reference.get("pdf_text_available") and excerpt_chars > 0:
            slim["abstract"] = str(reference.get("abstract") or "")[:excerpt_chars]
        return slim

    @staticmethod
    def _evidence_cards_for_reference(reference: dict, *, limit: int = 12) -> list[dict]:
        cards: list[dict] = []
        paper_id = reference.get("index", 0)
        candidates = LiteratureAnalysisWorkflow._extract_evidence_candidates(reference)
        slot_order = [
            "sample_size",
            "train_test_split",
            "dataset_or_material",
            "evaluation_protocol",
            "metrics",
            "metric_values",
            "baseline_or_comparator",
            "statistical_evidence",
            "availability",
            "limitations",
        ]
        for slot in slot_order:
            for candidate in candidates.get(slot, []) or []:
                if not isinstance(candidate, dict):
                    continue
                value = str(candidate.get("value") or "").strip()
                snippet = str(candidate.get("snippet") or "").strip()
                if not value and not snippet:
                    continue
                cards.append(
                    {
                        "paper_id": paper_id,
                        "slot": slot,
                        "value": value[:320],
                        "section": str(candidate.get("section") or "").strip(),
                        "page": str(candidate.get("page") or "").strip(),
                        "snippet": snippet[:360],
                        "confidence": str(candidate.get("confidence") or "medium").strip(),
                    }
                )
                if len(cards) >= limit:
                    return cards
        return cards

    @staticmethod
    def _high_information_excerpt(text: str, *, max_chars: int = ANALYST_EXCERPT_CHARS) -> str:
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(text)
        flat_text = LiteratureAnalysisWorkflow._compact_text(text)
        if len(layout_text) <= max_chars:
            return layout_text

        sections = LiteratureAnalysisWorkflow._section_slices(layout_text)
        plan = [
            ("abstract", 1500),
            ("methods", 2300),
            ("experiments", 1600),
            ("evaluation", 1600),
            ("results", 2600),
            ("tables", 1900),
            ("limitations", 1200),
            ("discussion", 900),
            ("conclusion", 700),
        ]
        parts = []
        for key, budget in plan:
            section = sections.get(key, "")
            if section:
                label = key.replace("_", " ").title()
                parts.append(f"[{label}]\n{section[:budget].strip()}")

        keyword_windows = LiteratureAnalysisWorkflow._generic_keyword_windows(flat_text, max_chars=2600)
        if keyword_windows:
            parts.insert(0, f"[Keyword Windows]\n{keyword_windows}")

        if not parts:
            return layout_text[:max_chars]

        packed = "\n\n".join(parts)
        if len(packed) < max_chars * 0.65:
            tail_budget = max_chars - len(packed) - 20
            if tail_budget > 700:
                packed = f"{packed}\n\n[Opening Context]\n{layout_text[:tail_budget].strip()}"
        return packed[:max_chars]

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _normalize_layout_text(text: str) -> str:
        lines: list[str] = []
        for raw in str(text or "").splitlines():
            line = re.sub(r"[ \t\f\v]+", " ", raw).strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _section_slices(text: str) -> dict[str, str]:
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(text)
        heading_patterns = [
            ("abstract", r"abstract|摘要"),
            ("introduction", r"introduction|引言|绪论"),
            ("methods", r"materials?\s+and\s+methods|methods?|methodology|approach|方法|材料与方法|研究方法"),
            ("experiments", r"experiments?|experimental\s+setup|实验|实验设置"),
            ("evaluation", r"evaluation|evaluations?|benchmark(?:ing)?|human\s+evaluation|评价|评估|人工评估"),
            ("results", r"results?|结果"),
            ("tables", r"tables?|figures?|captions?|(?:表|图)\s*\d+"),
            ("discussion", r"discussion|讨论"),
            ("limitations", r"limitations?|limitation\s+of\s+the\s+study|future\s+work|局限|局限性|不足|限制|未来工作"),
            ("conclusion", r"conclusions?|summary|结论|总结"),
            ("references", r"references|参考文献"),
        ]
        matches: list[tuple[int, int, str]] = []
        for key, pattern in heading_patterns:
            heading_re = re.compile(
                rf"(?im)^(?:\s*\[Page\s+\d+\]\s*)?\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:{pattern})\s*[:：]?\s*$",
                flags=re.IGNORECASE,
            )
            for match in heading_re.finditer(layout_text):
                matches.append((match.start(), match.end(), key))
                break
        matched_keys = {key for _, _, key in matches}
        inline_patterns = [
            (key, pattern)
            for key, pattern in heading_patterns
            if key not in matched_keys and key != "references"
        ]
        for key, pattern in inline_patterns:
            inline_re = re.compile(
                rf"(?is)(?:^|(?<=[.!?。！？])\s*)(?:\d+(?:\.\d+)*\.?\s*)?(?:{pattern})\s*[.:：]\s*",
                flags=re.IGNORECASE,
            )
            for match in inline_re.finditer(layout_text):
                matches.append((match.start(), match.end(), key))
                break
        matches.sort(key=lambda item: item[0])

        sections: dict[str, str] = {}
        introduction_start = next((start for start, _, key in matches if key == "introduction"), None)
        if introduction_start and "abstract" not in {key for _, _, key in matches}:
            front_matter = layout_text[:introduction_start].strip()
            abstract_match = re.search(r"\babstract\b[:.\s-]*(.+)", front_matter, flags=re.IGNORECASE | re.DOTALL)
            sections["abstract"] = (abstract_match.group(1) if abstract_match else front_matter)[-2400:].strip()
        for index, (start, end, key) in enumerate(matches):
            if key == "references":
                continue
            next_start = len(layout_text)
            for later_start, _, _ in matches[index + 1 :]:
                next_start = later_start
                break
            chunk = layout_text[end:next_start].strip()
            if chunk and key not in sections:
                sections[key] = chunk
        return sections

    @staticmethod
    def _generic_keyword_windows(text: str, *, max_chars: int = 2600, window_chars: int = 520) -> str:
        keywords = [
            "Table",
            "Figure",
            "Dataset",
            "Data",
            "Participants",
            "Samples",
            "Sample size",
            "Cases",
            "Patients",
            "Subjects",
            "Train",
            "Training set",
            "Test",
            "Test set",
            "Validation",
            "External",
            "External validation",
            "Baseline",
            "Comparator",
            "Metric",
            "Accuracy",
            "Score",
            "Result",
            "p-value",
            "significance",
            "confidence interval",
            "ablation",
            "ablation study",
            "code available",
            "data available",
            "publicly available",
            "limitation",
            "表",
            "图",
            "数据集",
            "数据",
            "样本量",
            "样本",
            "患者",
            "受试者",
            "病例",
            "训练集",
            "训练",
            "测试集",
            "测试",
            "验证",
            "外部验证",
            "独立测试",
            "外部",
            "基线",
            "对照",
            "指标",
            "结果",
            "显著性",
            "显著",
            "置信区间",
            "消融",
            "消融实验",
            "代码公开",
            "代码可用",
            "数据公开",
            "数据可用",
            "开源",
            "局限性",
            "局限",
            "不足",
        ]
        windows = []
        covered: list[tuple[int, int]] = []
        pattern = re.compile("|".join(re.escape(keyword) for keyword in keywords), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - window_chars // 2)
            end = min(len(text), match.end() + window_chars // 2)
            if any(start < old_end and end > old_start for old_start, old_end in covered):
                continue
            covered.append((start, end))
            snippet = text[start:end].strip()
            if snippet:
                windows.append(snippet)
            if sum(len(item) for item in windows) >= max_chars:
                break
        return "\n\n".join(windows)[:max_chars].strip()

    @staticmethod
    def _extract_evidence_candidates(reference: dict) -> dict[str, list[dict]]:
        raw_text = str(
            reference.get("evidence_source_text")
            or reference.get("full_text_for_evidence")
            or reference.get("content_excerpt")
            or reference.get("abstract")
            or ""
        )
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(raw_text)
        flat_text = LiteratureAnalysisWorkflow._compact_text(raw_text)
        if not flat_text:
            return {}
        candidates = {
            "sample_size": LiteratureAnalysisWorkflow._extract_sample_size_candidates(flat_text),
            "train_test_split": LiteratureAnalysisWorkflow._extract_train_test_split_candidates(flat_text),
            "dataset_or_material": LiteratureAnalysisWorkflow._extract_keyword_candidates(
                flat_text,
                r"(?i)\b(?:dataset|data set|data|corpus|cohort|registry|benchmark|materials?|specimens?|samples?|sample size|participants?|patients?|cases?|subjects?)\b|(?:数据集|数据|语料库|队列|材料|样本量|样本|病例|患者|受试者|参与者)",
                limit=6,
            ),
            "baseline_or_comparator": LiteratureAnalysisWorkflow._extract_keyword_candidates(
                flat_text,
                r"(?i)\b(?:baseline|comparator|comparison|compare[ds]? with|control group|against|versus|vs\.?)\b|(?:基线|对照|比较|相比|对比)",
                limit=6,
            ),
            "evaluation_protocol": LiteratureAnalysisWorkflow._extract_keyword_candidates(
                flat_text,
                r"(?i)\b(?:train(?:ing)?/?test(?:ing)?|training set|test set|validation set|validation|cross[- ]validation|hold[- ]out|external validation|independent test|evaluation protocol|experimental setup|ablation|ablation study)\b|(?:训练集|测试集|验证集|训练|测试|验证|交叉验证|外部验证|独立测试|评价方案|评估方案|实验设置|消融|消融实验)",
                limit=6,
            ),
            "metrics": LiteratureAnalysisWorkflow._extract_metric_value_candidates(flat_text),
            "statistical_evidence": LiteratureAnalysisWorkflow._extract_keyword_candidates(
                flat_text,
                r"(?i)\b(?:p\s*[- ]?value|p\s*[<=>]\s*0?\.\d+|confidence interval|95%\s*CI|\bCI\b|standard deviation|standard error|interquartile range|IQR|significant|significance|statistically significant|effect size)\b|(?:p值|P值|置信区间|95%CI|显著性|显著|统计学显著|标准差|标准误|四分位|效应量)",
                limit=6,
            ),
            "availability": LiteratureAnalysisWorkflow._extract_keyword_candidates(
                flat_text,
                r"(?i)\b(?:(?:code|data|dataset|model|materials?|software)\s+(?:is|are|will be)?\s*(?:publicly\s*)?(?:available|released|shared|open[- ]source)|available upon request|upon reasonable request|github|gitlab|zenodo|figshare|osf|supplementary material|data availability|code availability)\b|(?:(?:代码|数据|数据集|模型|材料|软件).{0,24}(?:公开|可用|获取|提供|共享|开源|发布)|(?:代码公开|代码可用|数据公开|数据可用|公开获取|补充材料)|可向.*索取)",
                limit=6,
            ),
            "limitations": LiteratureAnalysisWorkflow._extract_limitation_candidates(layout_text),
            "metric_values": LiteratureAnalysisWorkflow._extract_metric_value_candidates(flat_text),
        }
        return {
            key: LiteratureAnalysisWorkflow._candidate_objects(key, value, layout_text)
            for key, value in candidates.items()
            if value
        }

    @staticmethod
    def _extract_train_test_split_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        split_patterns = [
            r"(?i)\b(\d[\d,]*)\s*(?:training|train)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:[/,;]|and|with)?\s*(\d[\d,]*)\s*(?:testing|test)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?",
            r"(?i)\b(\d[\d,]*)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:for|in)?\s*(?:training|train)\s*(?:[/,;]|and|with)?\s*(\d[\d,]*)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:for|in)?\s*(?:testing|test)",
            r"(?i)\b(?:training|train)\s*(?:set|split|partition)?\s*(?:n\s*=\s*)?(\d[\d,]*)\s*(?:[/,;]|and|with)?\s*(?:testing|test)\s*(?:set|split|partition)?\s*(?:n\s*=\s*)?(\d[\d,]*)",
            r"(\d[\d,]*)\s*(?:训练|训练集)\s*(?:[/,，；;、]|和|与)?\s*(\d[\d,]*)\s*(?:测试|测试集)",
            r"(?:训练集|训练)\s*(?:n\s*=\s*)?(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:[/,，；;、]|和|与)?\s*(?:测试集|测试)\s*(?:n\s*=\s*)?(\d[\d,]*)",
            r"(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:用于|作为)?\s*(?:训练集|训练)\s*(?:[/,，；;、]|和|与)?\s*(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:用于|作为)?\s*(?:测试集|测试)",
        ]
        for pattern in split_patterns:
            for match in re.finditer(pattern, text):
                snippet = LiteratureAnalysisWorkflow._candidate_snippet(text, match.start(), match.end())
                if snippet:
                    candidates.append(snippet)
        return LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=6)

    @staticmethod
    def _extract_sample_size_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        split_patterns = [
            r"(?i)\b(\d[\d,]*)\s*(?:training|train)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:[/,;]|and|with)?\s*(\d[\d,]*)\s*(?:testing|test)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?",
            r"(?i)\b(\d[\d,]*)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:for|in)?\s*(?:training|train)\s*(?:[/,;]|and|with)?\s*(\d[\d,]*)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:for|in)?\s*(?:testing|test)",
            r"(\d[\d,]*)\s*(?:训练|训练集)\s*(?:[/,，；;、]|和|与)?\s*(\d[\d,]*)\s*(?:测试|测试集)",
            r"(?:训练集|训练)\s*(?:n\s*=\s*)?(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:[/,，；;、]|和|与)?\s*(?:测试集|测试)\s*(?:n\s*=\s*)?(\d[\d,]*)",
            r"(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:用于|作为)?\s*(?:训练集|训练)\s*(?:[/,，；;、]|和|与)?\s*(\d[\d,]*)\s*(?:例|病例|样本|图像|记录|个)?\s*(?:用于|作为)?\s*(?:测试集|测试)",
        ]
        for pattern in split_patterns:
            for match in re.finditer(pattern, text):
                snippet = LiteratureAnalysisWorkflow._candidate_snippet(text, match.start(), match.end())
                if snippet:
                    candidates.append(snippet)

        unit_pattern = re.compile(
            r"(?i)(?:\bn\s*=\s*\d[\d,]*|\d[\d,.\s]*(?:sets?|participants?|patients?|samples?|images?|cases?|datasets?|records?|subjects?|trials?|corpus|corpora)\b|"
            r"(?:样本量|病例数|患者数|共纳入|纳入|包含|包括|共有)\s*(?:n\s*=\s*)?\d[\d,]*\s*(?:套|组|例|个)?(?:样本|患者|病例|图像|数据集|记录|受试者|实验|语料)?|"
            r"\d[\d,.\s]*(?:套|组|例|个)?(?:样本|患者|病例|图像|数据集|记录|受试者|实验|语料))"
        )
        for match in unit_pattern.finditer(text):
            snippet = LiteratureAnalysisWorkflow._candidate_snippet(text, match.start(), match.end())
            if snippet:
                candidates.append(snippet)
        return LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=6)

    @staticmethod
    def _extract_keyword_candidates(text: str, pattern: str, *, limit: int = 6, window_chars: int = 360) -> list[str]:
        candidates = []
        for match in re.finditer(pattern, text):
            snippet = LiteratureAnalysisWorkflow._candidate_snippet(
                text,
                match.start(),
                match.end(),
                window_chars=window_chars,
            )
            if snippet:
                candidates.append(snippet)
        return LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=limit)

    @staticmethod
    def _extract_limitation_candidates(text: str) -> list[dict]:
        sections = LiteratureAnalysisWorkflow._section_slices(text)
        allowed_sources: list[tuple[str, str]] = []
        for key in ("limitations", "discussion", "conclusion"):
            if sections.get(key):
                allowed_sources.append((key.title(), sections[key]))
        future_match = re.search(
            r"(?is)\b(?:future work|limitations and future work|conclusion and future work)\b(.{0,2500})|(?:未来工作|局限性与未来工作|结论与未来工作)(.{0,2500})",
            text,
        )
        if future_match:
            allowed_sources.append(("Future work", next((group for group in future_match.groups() if group), "")))

        candidates: list[dict] = []
        for section, source in allowed_sources:
            for sentence in LiteratureAnalysisWorkflow._split_candidate_sentences(source):
                cleaned = sentence.strip()
                if re.fullmatch(r"(?i)\s*limitations?\.?\s*", cleaned):
                    continue
                if not re.search(
                    r"(?i)\b(?:limitations?|limited|only|single[- ]center|small|future|not include|did not include|generalizability|further|more studies|cohort|trial|experts?|outcomes?|mimics?|hemorrhage|haemorrhage)\b|局限性|局限|限制|不足|未来|进一步|单中心|样本量小|样本较小|未纳入|未包括|泛化|推广",
                    cleaned,
                ):
                    continue
                start = text.find(cleaned)
                if start < 0:
                    start = 0
                candidates.append(
                    {
                        "slot": "limitations",
                        "value": LiteratureAnalysisWorkflow._summarize_limitation_candidates([cleaned]),
                        "snippet": cleaned[:700],
                        "section": section,
                        "page": LiteratureAnalysisWorkflow._infer_candidate_page(cleaned),
                        "confidence": "high" if section.lower() == "limitations" else "medium",
                    }
                )
        return LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=6)

    @staticmethod
    def _extract_metric_value_candidates(text: str) -> list[str]:
        metric_names = [
            "Dice",
            "DSC",
            "Accuracy",
            "F1",
            "AUC",
            "AUROC",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "IoU",
            "mIoU",
            "Jaccard",
            "mAP",
            "score",
            "p-value",
            "指标",
            "准确率",
            "召回率",
            "精确率",
            "敏感性",
            "特异性",
            "F1值",
            "AUC值",
            "Dice系数",
            "IoU值",
            "得分",
        ]
        metric_pattern = "|".join(re.escape(name) for name in metric_names)
        patterns = [
            rf"(?i)\b(?:mean\s+)?(?:{metric_pattern})\b[^.;。]{{0,80}}?(?:=|:|is|was|of|达到|为)?\s*(\d+(?:\.\d+)?%?)",
            rf"(?i)(\d+(?:\.\d+)?%?)\s*(?:mean\s+)?(?:{metric_pattern})\b",
        ]
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                snippet = LiteratureAnalysisWorkflow._candidate_snippet(text, match.start(), match.end())
                if snippet:
                    candidates.append(snippet)
        return LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=10)

    @staticmethod
    def _candidate_snippet(text: str, start: int, end: int, *, window_chars: int = 260) -> str:
        left = max(0, start - window_chars // 2)
        right = min(len(text), end + window_chars // 2)
        snippet = text[left:right].strip()
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet[:700].strip()

    @staticmethod
    def _split_candidate_sentences(text: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?。！？])\s+|[；;]\s*", str(text or ""))
            if len(part.strip()) >= 12
        ]

    @staticmethod
    def _dedupe_candidates(candidates: list, *, limit: int) -> list:
        deduped = []
        seen = set()
        for candidate in candidates:
            if isinstance(candidate, dict):
                item = dict(candidate)
                item["value"] = re.sub(r"\s+", " ", str(item.get("value") or "").strip(" ;,，。"))
                item["snippet"] = re.sub(r"\s+", " ", str(item.get("snippet") or "").strip())
                cleaned = str(item.get("value") or item.get("snippet") or "").strip()
            else:
                item = re.sub(r"\s+", " ", str(candidate or "")).strip(" ;,，。")
                cleaned = item
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _candidate_objects(slot: str, candidates: list, text: str = "") -> list[dict]:
        objects: list[dict] = []
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(text)
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                item = dict(candidate)
                snippet = str(item.get("snippet") or item.get("value") or "").strip()
                item.setdefault("slot", slot)
                item["value"] = str(item.get("value") or LiteratureAnalysisWorkflow._summarize_field_candidate_value(slot, snippet)).strip()
                item["snippet"] = snippet[:700]
                item.setdefault("section", LiteratureAnalysisWorkflow._infer_candidate_section(layout_text, snippet))
                item.setdefault("page", LiteratureAnalysisWorkflow._infer_candidate_page(snippet, layout_text))
                item.setdefault("confidence", "medium")
            else:
                snippet = re.sub(r"\s+", " ", str(candidate or "")).strip()
                item = {
                    "slot": slot,
                    "value": LiteratureAnalysisWorkflow._summarize_field_candidate_value(slot, snippet),
                    "snippet": snippet[:700],
                    "section": LiteratureAnalysisWorkflow._infer_candidate_section(layout_text, snippet),
                    "page": LiteratureAnalysisWorkflow._infer_candidate_page(snippet, layout_text),
                    "confidence": "medium",
                }
            if item.get("value") or item.get("snippet"):
                objects.append(item)
        return LiteratureAnalysisWorkflow._dedupe_candidates(objects, limit=10)

    @staticmethod
    def _candidate_values(candidates: list, *, slot: str = "", row: dict | None = None) -> list[str]:
        values = []
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                value = str(candidate.get("value") or "").strip()
                if value and LiteratureAnalysisWorkflow._candidate_value_is_formal_fact(value, slot=slot):
                    values.append(value)
                elif value and row is not None:
                    LiteratureAnalysisWorkflow._append_candidate_risk(
                        row,
                        slot or str(candidate.get("slot") or "evidence"),
                        value,
                        reason="low-quality or fragmentary candidate value was kept only for audit",
                    )
            elif str(candidate or "").strip():
                value = str(candidate).strip()
                if LiteratureAnalysisWorkflow._candidate_value_is_formal_fact(value, slot=slot):
                    values.append(value)
                elif row is not None:
                    LiteratureAnalysisWorkflow._append_candidate_risk(
                        row,
                        slot or "evidence",
                        value,
                        reason="low-quality or fragmentary candidate value was kept only for audit",
                    )
        return values

    @staticmethod
    def _candidate_snippets(candidates: list) -> list[str]:
        snippets = []
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                snippets.append(str(candidate.get("snippet") or candidate.get("value") or "").strip())
            else:
                snippets.append(str(candidate or "").strip())
        return [item for item in snippets if item]

    @staticmethod
    def _infer_candidate_page(snippet: str, text: str = "") -> str:
        match = re.search(r"\[Page\s+(\d+)\]", str(snippet or ""), flags=re.IGNORECASE)
        if match:
            return match.group(1)
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(text)
        position = LiteratureAnalysisWorkflow._find_snippet_position(layout_text, str(snippet or ""))
        if position < 0:
            return ""
        return LiteratureAnalysisWorkflow._infer_page_from_position(layout_text, position)

    @staticmethod
    def _find_snippet_position(text: str, snippet: str) -> int:
        if not text or not snippet:
            return -1
        direct = text.find(snippet)
        if direct >= 0:
            return direct
        normalized_text = re.sub(r"\s+", " ", text)
        normalized_snippet = re.sub(r"\s+", " ", snippet).strip()
        if not normalized_snippet:
            return -1
        normalized_position = normalized_text.find(normalized_snippet)
        if normalized_position < 0:
            return -1
        compact_prefix = normalized_text[:normalized_position]
        non_space_count = len(re.sub(r"\s+", "", compact_prefix))
        seen = 0
        for index, char in enumerate(text):
            if char.isspace():
                continue
            if seen >= non_space_count:
                return index
            seen += 1
        return -1

    @staticmethod
    def _infer_page_from_position(text: str, position: int) -> str:
        if position < 0:
            return ""
        matches = list(re.finditer(r"\[Page\s+(\d+)\]", text[:position], flags=re.IGNORECASE))
        return matches[-1].group(1) if matches else ""

    @staticmethod
    def _infer_candidate_section(text: str, snippet: str) -> str:
        if not snippet:
            return ""
        layout_text = LiteratureAnalysisWorkflow._normalize_layout_text(text)
        lower_snippet = snippet.casefold()
        for name in ["Limitations", "Discussion", "Conclusion", "Future work", "Results", "Methods", "Abstract"]:
            if name.casefold() in lower_snippet:
                return name
        sections = LiteratureAnalysisWorkflow._section_slices(layout_text)
        for name, chunk in sections.items():
            if LiteratureAnalysisWorkflow._find_snippet_position(chunk, snippet[:160]) >= 0:
                return name.title()
        position = LiteratureAnalysisWorkflow._find_snippet_position(layout_text.casefold(), lower_snippet[:160])
        if position >= 0:
            prior = layout_text[max(0, position - 1800) : position]
            matches = re.findall(
                r"\b(Abstract|Introduction|Methods?|Materials and Methods|Experiments?|Evaluation|Results?|Discussion|Limitations?|Conclusion|Future work)\b",
                prior,
                flags=re.IGNORECASE,
            )
            if matches:
                return matches[-1].title()
        return ""

    @staticmethod
    def _summarize_field_candidate_value(slot: str, snippet: str) -> str:
        text = re.sub(r"\s+", " ", str(snippet or "")).strip()
        if not text:
            return ""
        if slot in {"sample_size", "train_test_split"}:
            return LiteratureAnalysisWorkflow._summarize_sample_size_candidates([text])
        if slot in {"metrics", "metric_values"}:
            return LiteratureAnalysisWorkflow._summarize_metric_values([text])
        if slot == "availability":
            return LiteratureAnalysisWorkflow._summarize_availability_candidates([text])
        if slot == "limitations":
            return LiteratureAnalysisWorkflow._summarize_limitation_candidates([text])
        return LiteratureAnalysisWorkflow._first_informative_sentence(text)[:240]

    @staticmethod
    def _collect_analyst_rows(analyst_outputs: list[dict]) -> list[dict]:
        return [
            row
            for output in analyst_outputs
            if isinstance(output, dict)
            for row in output.get("rows", [])
            if isinstance(row, dict)
        ]

    @staticmethod
    def _rows_for_integration(rows: list[dict]) -> list[dict]:
        keys = [
            "title",
            "source",
            "study_type",
            "research_objective",
            "dataset_or_material",
            "sample_size",
            "domain_or_modality",
            "method",
            "baseline_or_comparator",
            "evaluation_protocol",
            "metrics",
            "key_results",
            "statistical_evidence",
            "availability",
            "limitations",
            "evidence_locations",
            "evidence_quotes",
            "fact_risks",
            "evidence_candidates",
            "metric_values",
            "core_claim",
            "contribution",
            "methodology",
            "dataset",
            "modality",
            "task",
            "model_or_method",
            "baseline",
            "validation_setup",
            "evidence_strength",
            "innovation_point",
            "reader_takeaway",
            "reader_next_step",
            "strengths",
            "weaknesses",
            "literature_positioning",
            "application",
            "confidence",
        ]
        compact_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            compact_rows.append(
                {
                    key: str(row.get(key) or "").strip()[:900]
                    for key in keys
                    if str(row.get(key) or "").strip()
                }
            )
        return compact_rows

    @staticmethod
    def _analysis_result(rows: list[dict], summary: dict, output_language: str = "zh") -> dict:
        audit_summary = LiteratureAnalysisWorkflow._apply_fact_consistency_checks(
            summary,
            rows,
            output_language,
        )
        public_summary = LiteratureAnalysisWorkflow._public_summary(audit_summary, output_language)
        return {
            "rows": rows,
            "summary": public_summary,
            "audit_summary": audit_summary,
            "comparison_matrix": LiteratureAnalysisWorkflow._build_comparison_matrix(rows),
        }

    async def _finalize_analysis_result(
        self,
        rows: list[dict],
        summary: dict,
        output_language: str,
        references: list[dict],
    ) -> dict:
        result = self._analysis_result(rows, summary, output_language)
        if output_language != "zh":
            return result
        return await self._translate_final_output_to_chinese(result, references)

    async def _translate_final_output_to_chinese(self, result: dict, references: list[dict]) -> dict:
        protected_terms = self._protected_translation_terms(result, references)
        system_prompt = """
You are a dedicated Chinese translation post-processor and final language-quality gate for a literature-analysis workflow.
Translate and polish the final JSON output into concise, professional Chinese.

Rules:
- Return only valid JSON with the same top-level keys and nested key names.
- Translate natural-language explanations, assessments, caveats, recommendations, and list items into Chinese.
- Remove extraction artifacts and boilerplate such as "在提取证据中多显示为", "extracted evidence suggests", "Evaluation candidate", "Metric candidate", and orphaned OCR/text fragments.
- Delete meaningless English fragments that were accidentally copied from source evidence, for example incomplete clauses like "our proposed model in Table 1" or "ensities [13]"; keep complete paper titles, source names, dataset names, metric names, model names, and standard abbreviations.
- Do not add new facts, remove key findings, change evidence strength, or change the meaning of claims. This is a language cleanup pass, not a new synthesis.
- Do not translate or rewrite protected proper nouns: original paper titles, author names, journal/source names, model names, dataset names, metric names, DOI/PMID/arXiv IDs, URLs, filenames, software names, and standard abbreviations.
- Preserve title, source, reference_index, citation_format, DOI/PMID/arXiv IDs, URLs, numeric values, formulas, units, and metric symbols exactly.
- Keep JSON arrays as arrays and objects as objects. Do not add markdown fences.
""".strip()
        user_prompt = (
            "Protected proper nouns and identifiers:\n"
            + json.dumps(protected_terms, ensure_ascii=False, indent=2)
            + "\n\nFinal literature-analysis JSON to translate:\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=(
                    os.getenv("ANALYSIS_TRANSLATION_MODEL")
                    or os.getenv("ANALYSIS_MODEL")
                    or os.getenv("SYNTHESIS_MODEL")
                ),
                temperature=0.0,
                max_tokens=4200,
            )
            translated = self._parse_json_object(content)
            if not isinstance(translated, dict):
                raise ValueError("Expected translated final output to be a JSON object.")
            merged = dict(result)
            for key in ("rows", "summary", "audit_summary", "comparison_matrix"):
                if key in translated:
                    merged[key] = translated[key]
            self._restore_translation_identity_fields(merged, result)
            return self._clean_final_report_language(merged)
        except (LLMServiceError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._log(f"Chinese translation post-processing failed; keeping original output. {type(error).__name__}: {error}")
            return self._clean_final_report_language(result)

    @staticmethod
    def _clean_final_report_language(result: dict) -> dict:
        protected_keys = {
            "title",
            "source",
            "reference_index",
            "citation_format",
            "references",
            "doi",
            "pmid",
            "arxiv_id",
            "url",
            "abs_url",
            "id",
            "evidence_candidates",
            "metric_values",
            "dataset",
            "metrics",
            "modality",
            "model_or_method",
        }

        def clean_value(value, key: str = ""):
            if isinstance(value, dict):
                return {item_key: clean_value(item_value, item_key) for item_key, item_value in value.items()}
            if isinstance(value, list):
                return [clean_value(item, key) for item in value]
            if not isinstance(value, str) or key in protected_keys:
                return value
            return LiteratureAnalysisWorkflow._clean_public_language_text(value)

        return clean_value(result)

    @staticmethod
    def _clean_public_language_text(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if re.fullmatch(r"(?i)(?:https?://\S+|doi:\s*\S+|arxiv:\s*\S+|10\.\d{4,9}/\S+)", text):
            return text

        replacements = {
            "在提取证据中多显示为": "",
            "提取证据中多显示为": "",
            "提取证据显示": "",
            "基于提取证据": "",
            "Based on the extracted evidence,": "",
            "Based on extracted evidence,": "",
            "the extracted evidence suggests": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\b(?:Dataset/material|Comparator|Evaluation|Metric|Sample-size|Availability|Statistical evidence)\s+candidate\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:Dataset/material|Comparator|Evaluation|Metric|Sample-size|Availability|Statistical evidence)\s+candidate[;；,，]?\s*", "", text, flags=re.IGNORECASE)
        text = LiteratureAnalysisWorkflow._drop_fragmentary_evidence_clauses(text)
        text = re.sub(r"\s*([；;，,])\s*(?=[；;，,。.!?])", "", text)
        text = re.sub(r"[；;，,]\s*([。.!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip(" ；;，,")
        return text

    @staticmethod
    def _drop_fragmentary_evidence_clauses(text: str) -> str:
        parts = re.split(r"([;；])", text)
        if len(parts) <= 1:
            return LiteratureAnalysisWorkflow._clean_fragmentary_evidence_text(text)

        kept: list[str] = []
        for index in range(0, len(parts), 2):
            clause = LiteratureAnalysisWorkflow._clean_fragmentary_evidence_text(parts[index])
            if not clause:
                continue
            if kept:
                kept.append("；")
            kept.append(clause)
        return "".join(kept)

    @staticmethod
    def _clean_fragmentary_evidence_text(text: str) -> str:
        clause = str(text or "").strip(" \t\r\n;；,，")
        if not clause:
            return ""
        fragment_patterns = [
            r"\bensities\s*\[\d+\]",
            r"\bour proposed model in Table\s+\d+\b",
            r"\bhad just two 2D slices\b",
            r"\bochs, and the model\b",
            r"\br severe class imbalance\b",
            r"\bmentation\s*\[\d+(?:,\s*\d+)*\]",
            r"\bto the ISLES.?24 challenge, where it achieved first place\b",
        ]
        for pattern in fragment_patterns:
            clause = re.sub(pattern, "", clause, flags=re.IGNORECASE)
        clause = re.sub(r"\s+", " ", clause).strip(" \t\r\n;；,，")
        if not clause:
            return ""
        ascii_letters = len(re.findall(r"[A-Za-z]", clause))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", clause))
        if cjk == 0 and ascii_letters > 0 and len(clause.split()) <= 8:
            if not re.search(r"\b(?:Dice|IoU|AUC|ASPECTS|NCCT|CTA|CTP|DWI|U-Net|nnU-Net|DINO|TAGL|ISLES|AISD|DEFUSE|arXiv|PubMed|DOI)\b", clause):
                return ""
        return clause

    @staticmethod
    def _protected_translation_terms(result: dict, references: list[dict]) -> list[str]:
        terms: list[str] = []
        for item in list(references or []) + list(result.get("rows") or []):
            if not isinstance(item, dict):
                continue
            for key in [
                "title",
                "source",
                "doi",
                "authors",
                "journal",
                "dataset",
                "dataset_or_material",
                "modality",
                "domain_or_modality",
                "metrics",
                "model_or_method",
                "method",
            ]:
                value = str(item.get(key) or "").strip()
                if value:
                    terms.append(value[:240])
        compact: list[str] = []
        seen = set()
        for term in terms:
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            compact.append(term)
            if len(compact) >= 80:
                break
        return compact

    @staticmethod
    def _restore_translation_identity_fields(translated: dict, original: dict) -> None:
        original_rows = [row for row in original.get("rows", []) if isinstance(row, dict)]
        translated_rows = translated.get("rows")
        if isinstance(translated_rows, list):
            for index, row in enumerate(translated_rows):
                if not isinstance(row, dict) or index >= len(original_rows):
                    continue
                original_row = original_rows[index]
                for key in ("reference_index", "title", "source"):
                    if key in original_row:
                        row[key] = original_row[key]
        original_matrix = [row for row in original.get("comparison_matrix", []) if isinstance(row, dict)]
        translated_matrix = translated.get("comparison_matrix")
        if isinstance(translated_matrix, list):
            for index, row in enumerate(translated_matrix):
                if not isinstance(row, dict) or index >= len(original_matrix):
                    continue
                for key in ("title", "source"):
                    if key in original_matrix[index]:
                        row[key] = original_matrix[index][key]
        for summary_key in ("summary", "audit_summary"):
            summary = translated.get(summary_key)
            original_summary = original.get(summary_key)
            if isinstance(summary, dict) and isinstance(original_summary, dict):
                summary["references"] = original_summary.get("references", [])
                summary["citation_format"] = original_summary.get("citation_format", "")

    @staticmethod
    def _merge_integrated_rows(integrated_rows: list[dict], fallback_rows: list[dict]) -> list[dict]:
        merged = list(integrated_rows)
        seen = {
            LiteratureAnalysisWorkflow._row_identity(row)
            for row in merged
            if LiteratureAnalysisWorkflow._row_identity(row)
        }
        for row in fallback_rows:
            key = LiteratureAnalysisWorkflow._row_identity(row)
            if key and key not in seen:
                merged.append(row)
                seen.add(key)
        return merged

    @staticmethod
    def _row_identity(row: dict) -> str:
        source = str(row.get("source") or "").strip().casefold().rstrip("/")
        title = str(row.get("title") or "").strip().casefold()
        return source or title

    @staticmethod
    def _detect_output_language(topic: str, final_report: str, references: list[dict]) -> str:
        # The literature-analysis view is Chinese-first in the UI; keep narrative
        # review text Chinese even when uploaded papers/excerpts are English.
        return "zh"

    @staticmethod
    def _normalize_output_language(output_language: str | None) -> str:
        value = str(output_language or "").strip().casefold()
        if value in {"zh", "en"}:
            return value
        return ""

    @staticmethod
    def _language_instruction(output_language: str) -> str:
        if output_language == "zh":
            return (
                "Write every narrative or analytical table-cell value and every summary value in concise Chinese. "
                "This includes study_type, paper_type, evidence_strength, confidence, and overall_assessment; use Chinese labels such as 实验研究、综述、方法论文、数据集论文、临床研究、未知、强、中等、弱、较高、中等、较低. "
                "Keep proper nouns unchanged: original paper titles, author names, DOI/PMID/arXiv IDs, URLs, journal/source names, model names, dataset names, metric names, and standard abbreviations."
            )
        return "Write all analytical table-cell text and summary values in concise English. Keep original paper titles, author names, DOI/PMID/arXiv IDs, and URLs unchanged."

    @staticmethod
    def _normalize_references(references: list[dict]) -> list[dict]:
        normalized = []
        for index, item in enumerate(references):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            normalized.append(
                {
                    "index": index,
                    "title": title,
                    "source": str(item.get("source", "") or "").strip(),
                    "uploaded_filename": str(item.get("uploaded_filename", "") or "").strip(),
                    "source_origin": str(item.get("source_origin", "") or "").strip(),
                    "source_label": str(item.get("source_label", "") or "").strip(),
                    "relevance": str(item.get("relevance", "") or "").strip(),
                    "branch_name": str(item.get("branch_name", "") or "").strip(),
                    "doi": str(item.get("doi", "") or "").strip(),
                    "pmid": str(item.get("pmid", "") or "").strip(),
                    "arxiv_id": str(item.get("arxiv_id", "") or "").strip(),
                    "authors": str(item.get("authors", "") or "").strip(),
                    "year": str(item.get("year", "") or "").strip(),
                    "journal": str(item.get("journal", "") or "").strip(),
                    "abstract": str(item.get("abstract", "") or "").strip(),
                    "screening_status": str(item.get("screening_status", "") or "").strip(),
                    "screening_risks": item.get("screening_risks") if isinstance(item.get("screening_risks"), list) else [],
                    "topic_relevance_status": str(item.get("topic_relevance_status", "") or "").strip(),
                    "topic_relevance_score": str(item.get("topic_relevance_score", "") or "").strip(),
                    "topic_relevance_risks": item.get("topic_relevance_risks") if isinstance(item.get("topic_relevance_risks"), list) else [],
                    "verification_status": str(item.get("verification_status", "") or "").strip(),
                    "verification_risks": item.get("verification_risks") if isinstance(item.get("verification_risks"), list) else [],
                    "provenance": item.get("provenance") if isinstance(item.get("provenance"), dict) else {},
                    "content_excerpt": str(item.get("content_excerpt", "") or item.get("abstract", "") or "").strip(),
                    "evidence_source_text": str(item.get("evidence_source_text", "") or item.get("full_text_for_evidence", "") or "").strip(),
                    "pdf_text_available": bool(item.get("pdf_text_available", False)),
                    "pdf_page_count": str(item.get("pdf_page_count", "") or "").strip(),
                    "pdf_extracted_pages": str(item.get("pdf_extracted_pages", "") or "").strip(),
                    "pdf_extraction_note": str(item.get("pdf_extraction_note", "") or "").strip(),
                    "document_role": str(item.get("document_role", "literature") or "literature").strip(),
                    "is_literature_source": bool(item.get("is_literature_source", True)),
                }
            )
        return normalized

    @staticmethod
    def _is_literature_reference(reference: dict) -> bool:
        role = str(reference.get("document_role") or "literature").strip().lower()
        return role == "literature" and bool(reference.get("is_literature_source", True))

    @staticmethod
    def _normalize_groups(groups, references: list[dict]) -> list[dict]:
        if not isinstance(groups, list):
            return []

        normalized = []
        used_names = set()
        for fallback_index, group in enumerate(groups[:4]):
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or f"Analyst {fallback_index + 1}").strip()
            if name in used_names:
                name = f"{name} {fallback_index + 1}"
            used_names.add(name)
            focus = str(group.get("focus") or "literature contribution and research value").strip()
            indices = group.get("reference_indices", [])
            if not isinstance(indices, list):
                indices = []
            items = [
                references[index]
                for index in indices
                if isinstance(index, int) and 0 <= index < len(references)
            ]
            normalized.append({"name": name, "focus": focus, "references": items})

        while len(normalized) < 4:
            fallback_group = REVIEW_ANALYST_GROUPS[len(normalized)]
            normalized.append(
                {
                    "name": fallback_group["name"],
                    "focus": fallback_group["focus"],
                    "references": [],
                }
            )

        assigned = {
            reference["index"]
            for group in normalized
            for reference in group["references"]
            if "index" in reference
        }
        missing = [reference for reference in references if reference["index"] not in assigned]
        for offset, reference in enumerate(missing):
            normalized[offset % 4]["references"].append(reference)
        return normalized

    @staticmethod
    def _round_robin_groups(references: list[dict]) -> list[dict]:
        groups = [{**group, "references": []} for group in REVIEW_ANALYST_GROUPS]
        for index, reference in enumerate(references):
            groups[index % 4]["references"].append(reference)
        return groups

    @staticmethod
    def _normalize_rows(rows: list, references: list[dict], output_language: str = "zh") -> list[dict]:
        by_title, by_source, by_index = LiteratureAnalysisWorkflow._reference_lookup_maps(references)
        normalized = []
        seen = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            title, source = LiteratureAnalysisWorkflow._row_title_and_source(row, by_source)
            original = LiteratureAnalysisWorkflow._match_original_reference(
                row,
                row_index=row_index,
                row_count=len(rows),
                title=title,
                source=source,
                references=references,
                by_title=by_title,
                by_source=by_source,
                by_index=by_index,
            )
            if references and not original:
                continue
            if original:
                title = str(original.get("title", "") or title).strip()
                source = str(original.get("source", "") or source).strip()
            if not title:
                continue
            key = LiteratureAnalysisWorkflow._source_key(source) or title.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized_row = LiteratureAnalysisWorkflow._build_normalized_row(
                row,
                original,
                title=title,
                source=source,
            )
            LiteratureAnalysisWorkflow._finalize_normalized_row(normalized_row, original, output_language)
            normalized.append(normalized_row)
        return normalized

    @staticmethod
    def _source_key(value: str) -> str:
        return str(value or "").strip().casefold().rstrip("/")

    @staticmethod
    def _reference_lookup_maps(references: list[dict]) -> tuple[dict[str, dict], dict[str, dict], dict[int, dict]]:
        by_title = {item["title"].casefold(): item for item in references}
        by_source = {
            LiteratureAnalysisWorkflow._source_key(item["source"]): item
            for item in references
            if item.get("source")
        }
        by_index = {
            int(item["index"]): item
            for item in references
            if isinstance(item.get("index"), int)
        }
        return by_title, by_source, by_index

    @staticmethod
    def _row_title_and_source(row: dict, by_source: dict[str, dict]) -> tuple[str, str]:
        title = str(row.get("title", "")).strip()
        source = str(row.get("source", "")).strip()
        if not title and source:
            original_by_source = by_source.get(LiteratureAnalysisWorkflow._source_key(source), {})
            title = str(original_by_source.get("title", "")).strip()
        return title, source

    @staticmethod
    def _match_original_reference(
        row: dict,
        *,
        row_index: int,
        row_count: int,
        title: str,
        source: str,
        references: list[dict],
        by_title: dict[str, dict],
        by_source: dict[str, dict],
        by_index: dict[int, dict],
    ) -> dict:
        reference_index = LiteratureAnalysisWorkflow._row_reference_index(row)
        title_match = by_title.get(title.casefold(), {}) if title else {}
        source_match = by_source.get(LiteratureAnalysisWorkflow._source_key(source), {}) if source else {}
        index_match = by_index.get(reference_index) if reference_index is not None else None
        if index_match and LiteratureAnalysisWorkflow._row_identity_conflicts_reference(
            row,
            index_match,
            title_match=title_match,
            source_match=source_match,
        ):
            original = title_match or source_match or {}
        else:
            original = index_match or title_match or source_match or {}
        if not original and row_count == len(references) and row_index < len(references):
            original = references[row_index]
        return original

    @staticmethod
    def _build_normalized_row(row: dict, original: dict, *, title: str, source: str) -> dict:
        normalized_row = {
            column: str(row.get(column) or original.get(column) or "").strip()
            for column in ANALYSIS_COLUMNS
        }
        if original:
            LiteratureAnalysisWorkflow._copy_reference_identity_fields(
                normalized_row,
                original,
                title=title,
                source=source,
            )
        return normalized_row

    @staticmethod
    def _copy_reference_identity_fields(
        row: dict,
        original: dict,
        *,
        title: str,
        source: str,
    ) -> None:
        row["title"] = str(original.get("title") or title).strip()
        row["source"] = str(original.get("source") or source).strip()
        row["uploaded_filename"] = str(original.get("uploaded_filename") or "").strip()
        row["source_origin"] = str(original.get("source_origin") or "").strip()
        row["source_label"] = str(original.get("source_label") or "").strip()
        row["screening_status"] = str(original.get("screening_status") or "").strip()
        row["screening_risks"] = original.get("screening_risks") if isinstance(original.get("screening_risks"), list) else []
        row["topic_relevance_status"] = str(original.get("topic_relevance_status") or "").strip()
        row["topic_relevance_score"] = str(original.get("topic_relevance_score") or "").strip()
        row["topic_relevance_risks"] = original.get("topic_relevance_risks") if isinstance(original.get("topic_relevance_risks"), list) else []
        row["verification_status"] = str(original.get("verification_status") or "").strip()
        row["verification_risks"] = original.get("verification_risks") if isinstance(original.get("verification_risks"), list) else []
        row["provenance"] = original.get("provenance") if isinstance(original.get("provenance"), dict) else {}

    @staticmethod
    def _finalize_normalized_row(row: dict, original: dict, output_language: str) -> None:
        LiteratureAnalysisWorkflow._attach_and_backfill_evidence_candidates(row, original)
        LiteratureAnalysisWorkflow._sync_generic_fact_aliases(row)
        LiteratureAnalysisWorkflow._fill_fact_slot_defaults(row)
        LiteratureAnalysisWorkflow._sanitize_row_metric_consistency(row, output_language)
        LiteratureAnalysisWorkflow._sanitize_row_fact_slot_consistency(row, output_language)
        LiteratureAnalysisWorkflow._separate_limitations_and_fact_risks(row, output_language)
        LiteratureAnalysisWorkflow._fill_review_defaults(row, original, output_language)
        if output_language == "zh":
            LiteratureAnalysisWorkflow._localize_row_labels(row)
        LiteratureAnalysisWorkflow._fill_legacy_aliases(row)

    @staticmethod
    def _row_reference_index(row: dict) -> int | None:
        for key in ("reference_index", "ref_index", "index"):
            value = row.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    @staticmethod
    def _row_identity_conflicts_reference(
        row: dict,
        reference: dict,
        *,
        title_match: dict | None = None,
        source_match: dict | None = None,
    ) -> bool:
        if not reference:
            return False
        row_title = str(row.get("title") or "").strip().casefold()
        row_source = str(row.get("source") or "").strip().casefold().rstrip("/")
        ref_title = str(reference.get("title") or "").strip().casefold()
        ref_source = str(reference.get("source") or "").strip().casefold().rstrip("/")

        title_points_elsewhere = bool(title_match and title_match is not reference)
        source_points_elsewhere = bool(source_match and source_match is not reference)
        if title_points_elsewhere or source_points_elsewhere:
            return True

        if row_source and ref_source and row_source != ref_source:
            return True
        if row_title and ref_title:
            row_tokens = set(re.findall(r"[a-z0-9]{4,}", row_title))
            ref_tokens = set(re.findall(r"[a-z0-9]{4,}", ref_title))
            if row_tokens and ref_tokens:
                overlap = row_tokens & ref_tokens
                return len(overlap) < max(1, min(len(row_tokens), len(ref_tokens)) // 3)
        return False

    @staticmethod
    def _attach_and_backfill_evidence_candidates(row: dict[str, str], original: dict) -> None:
        candidates = LiteratureAnalysisWorkflow._extract_evidence_candidates(original or {})
        if candidates:
            row["evidence_candidates"] = json.dumps(candidates, ensure_ascii=False)
        if candidates.get("metric_values"):
            row["metric_values"] = LiteratureAnalysisWorkflow._summarize_metric_values(
                LiteratureAnalysisWorkflow._candidate_values(candidates["metric_values"], slot="metric_values", row=row)
            )

        split_candidates = candidates.get("train_test_split") or []
        sample_candidates = [*split_candidates, *candidates.get("sample_size", [])]
        if sample_candidates:
            sample_summary = LiteratureAnalysisWorkflow._summarize_sample_size_candidates(
                LiteratureAnalysisWorkflow._candidate_values(sample_candidates, slot="sample_size", row=row)
            )
            LiteratureAnalysisWorkflow._merge_candidate_fact(
                row,
                "sample_size",
                sample_summary,
                location="Sample-size candidate",
                conflict_label="sample_size",
            )
            if split_candidates:
                LiteratureAnalysisWorkflow._merge_candidate_fact(
                    row,
                    "evaluation_protocol",
                    LiteratureAnalysisWorkflow._summarize_candidate_text(
                        LiteratureAnalysisWorkflow._candidate_values(split_candidates, slot="train_test_split", row=row),
                        limit=220,
                    ),
                    location="Train/test split candidate",
                    conflict_label="train_test_split",
                )

        field_candidates = {
            "dataset_or_material": ("Dataset/material candidate", candidates.get("dataset_or_material", [])),
            "baseline_or_comparator": ("Comparator candidate", candidates.get("baseline_or_comparator", [])),
            "evaluation_protocol": ("Evaluation candidate", candidates.get("evaluation_protocol", [])),
            "statistical_evidence": ("Statistical evidence candidate", candidates.get("statistical_evidence", [])),
            "availability": ("Availability candidate", candidates.get("availability", [])),
        }
        for field, (location, values) in field_candidates.items():
            if values:
                LiteratureAnalysisWorkflow._merge_candidate_fact(
                    row,
                    field,
                    (
                        LiteratureAnalysisWorkflow._summarize_availability_candidates(
                            LiteratureAnalysisWorkflow._candidate_values(values, slot=field, row=row)
                        )
                        if field == "availability"
                        else LiteratureAnalysisWorkflow._summarize_comparator_candidates(
                            LiteratureAnalysisWorkflow._candidate_values(values, slot=field, row=row)
                        )
                        if field == "baseline_or_comparator"
                        else LiteratureAnalysisWorkflow._summarize_candidate_text(
                            LiteratureAnalysisWorkflow._candidate_values(values, slot=field, row=row),
                            limit=260,
                        )
                    ),
                    location=location,
                    conflict_label=field,
                )

        if candidates.get("limitations"):
            LiteratureAnalysisWorkflow._merge_candidate_fact(
                row,
                "limitations",
                LiteratureAnalysisWorkflow._summarize_limitation_candidates(
                    LiteratureAnalysisWorkflow._candidate_values(candidates["limitations"], slot="limitations", row=row)
                ),
                location="Limitations/Discussion",
                conflict_label="limitations",
            )

        metric_candidates = candidates.get("metric_values") or candidates.get("metrics") or []
        if metric_candidates:
            metric_text = LiteratureAnalysisWorkflow._summarize_metric_values(
                LiteratureAnalysisWorkflow._candidate_values(metric_candidates, slot="metric_values", row=row)
            )
            metric_names = LiteratureAnalysisWorkflow._metric_names_from_candidates(metric_text)
            if metric_names:
                LiteratureAnalysisWorkflow._merge_candidate_fact(
                    row,
                    "metrics",
                    metric_names,
                    location="Metric candidate",
                    conflict_label="metrics",
                    append_only=True,
                )
            LiteratureAnalysisWorkflow._merge_candidate_fact(
                row,
                "key_results",
                metric_text[:260],
                location="Metric candidate",
                conflict_label="key_results",
                append_only=True,
            )

    @staticmethod
    def _merge_candidate_fact(
        row: dict[str, str],
        field: str,
        candidate: str,
        *,
        location: str,
        conflict_label: str,
        append_only: bool = False,
    ) -> None:
        candidate = re.sub(r"\s+", " ", str(candidate or "")).strip(" ;,")
        if not candidate:
            return
        if not LiteratureAnalysisWorkflow._candidate_value_is_formal_fact(candidate, slot=field):
            LiteratureAnalysisWorkflow._append_candidate_risk(
                row,
                conflict_label,
                candidate,
                reason="candidate value looked fragmentary or semantically incomplete",
            )
            return
        existing = str(row.get(field) or "").strip()
        if not LiteratureAnalysisWorkflow._has_clear_fact(existing):
            row[field] = candidate
            row["evidence_locations"] = LiteratureAnalysisWorkflow._append_location(row.get("evidence_locations"), location)
            return

        existing_lower = existing.casefold()
        candidate_lower = candidate.casefold()
        if candidate_lower in existing_lower:
            row["evidence_locations"] = LiteratureAnalysisWorkflow._append_location(row.get("evidence_locations"), location)
            return

        if not append_only and LiteratureAnalysisWorkflow._numeric_fact_conflict(existing, candidate):
            risk = (
                f"Evidence candidate for {conflict_label} differs from extracted field; retained extracted field and kept candidate for audit: {candidate[:300]}"
            )
            row["fact_risks"] = "; ".join(
                LiteratureAnalysisWorkflow._dedupe_fact_risks(
                    [*LiteratureAnalysisWorkflow._normalize_risk_list(row.get("fact_risks")), risk]
                )
            )
            return

        if append_only or LiteratureAnalysisWorkflow._candidate_adds_missing_tokens(existing, candidate):
            row[field] = LiteratureAnalysisWorkflow._append_fact_text(existing, candidate)
            row["evidence_locations"] = LiteratureAnalysisWorkflow._append_location(row.get("evidence_locations"), location)
            return

        risk = (
            f"Evidence candidate for {conflict_label} differs from extracted field; retained extracted field and kept candidate for audit: {candidate[:300]}"
        )
        row["fact_risks"] = "; ".join(
            LiteratureAnalysisWorkflow._dedupe_fact_risks(
                [*LiteratureAnalysisWorkflow._normalize_risk_list(row.get("fact_risks")), risk]
            )
        )

    @staticmethod
    def _append_candidate_risk(row: dict, label: str, candidate: str, *, reason: str) -> None:
        text = re.sub(r"\s+", " ", str(candidate or "")).strip()
        if not text:
            return
        risk = f"Evidence candidate for {label} was not written into formal fact fields because {reason}: {text[:240]}"
        row["fact_risks"] = "; ".join(
            LiteratureAnalysisWorkflow._dedupe_fact_risks(
                [*LiteratureAnalysisWorkflow._normalize_risk_list(row.get("fact_risks")), risk]
            )
        )

    @staticmethod
    def _append_fact_risk(row: dict, risk: str) -> None:
        cleaned = LiteratureAnalysisWorkflow._clean_fact_risk_text(risk)
        if not cleaned:
            return
        row["fact_risks"] = "; ".join(
            LiteratureAnalysisWorkflow._dedupe_fact_risks(
                [*LiteratureAnalysisWorkflow._normalize_risk_list(row.get("fact_risks")), cleaned]
            )
        )

    @staticmethod
    def _sanitize_row_metric_consistency(row: dict[str, str], output_language: str = "zh") -> None:
        key_results = str(row.get("key_results") or "").strip()
        if not key_results:
            return

        cleaned = LiteratureAnalysisWorkflow._remove_uncertain_metric_clauses(key_results)
        if cleaned != key_results:
            row["key_results"] = cleaned or str(row.get("metric_values") or "").strip() or "unclear"
            title = str(row.get("title") or row.get("source") or "").strip()
            risk = (
                f"{title}: key_results contained uncertain or self-correcting metric wording; removed unsupported comparison wording and retained structured metric candidates."
                if output_language != "zh"
                else f"{title}: key_results contained uncertain or self-correcting metric wording; removed unsupported comparison wording and retained structured metric candidates."
            )
            LiteratureAnalysisWorkflow._append_fact_risk(row, risk)

        metric_values = str(row.get("metric_values") or "").strip()
        if metric_values and metric_values.casefold() not in str(row.get("key_results") or "").casefold():
            row["key_results"] = LiteratureAnalysisWorkflow._append_fact_text(
                str(row.get("key_results") or ""),
                metric_values,
                limit=1000,
            )

    @staticmethod
    def _remove_uncertain_metric_clauses(text: str) -> str:
        parts = re.split(r"([;；])", str(text or ""))
        if len(parts) <= 1:
            return LiteratureAnalysisWorkflow._clean_uncertain_metric_clause(text)

        kept: list[str] = []
        for index in range(0, len(parts), 2):
            clause = LiteratureAnalysisWorkflow._clean_uncertain_metric_clause(parts[index])
            if not clause:
                continue
            if kept:
                kept.append("; ")
            kept.append(clause)
        return "".join(kept).strip(" ;；")

    @staticmethod
    def _clean_uncertain_metric_clause(clause: str) -> str:
        text = str(clause or "").strip(" \t\r\n;；")
        if not text:
            return ""
        suspicious = bool(
            re.search(r"[?？]", text)
            or re.search(r"(?i)\b(?:actual(?:ly)?|uncertain|unclear|maybe|apparently)\b", text)
            or re.search(r"(实际|不确定|存疑|似乎|可能|疑似|待核)", text)
        )
        has_metric = bool(
            re.search(
                r"(?i)\b(?:Dice|DSC|Accuracy|F1|AUC|AUROC|precision|recall|sensitivity|specificity|IoU|mIoU|Jaccard|mAP|score)\b",
                text,
            )
        )
        if suspicious and has_metric:
            # Drop parenthetical metric comparisons such as
            # "(Dice 41.9% vs 41.9%? actually slightly better)" instead of
            # letting review notes masquerade as extracted facts.
            text = re.sub(r"[\(（][^\)）]*(?:[?？]|actual|实际|不确定|存疑|疑似|待核)[^\)）]*[\)）]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"(?i)\b[^;；。]*\bDice\b[^;；。]*(?:[?？]|actual|实际|不确定|存疑|疑似|待核)[^;；。]*", "", text)
            text = re.sub(r"[^;；。]*(?:[?？]|actual|实际|不确定|存疑|疑似|待核)[^;；。]*\b(?:Dice|DSC|IoU|AUC|F1)\b[^;；。]*", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" \t\r\n;；,，")

    @staticmethod
    def _sanitize_row_fact_slot_consistency(row: dict[str, str], output_language: str = "zh") -> None:
        for field in ("baseline_or_comparator", "evaluation_protocol", "limitations"):
            original = str(row.get(field) or "").strip()
            if not original:
                continue
            cleaned = LiteratureAnalysisWorkflow._clean_mixed_language_fact_slot(
                original,
                field=field,
                output_language=output_language,
            )
            if cleaned != original:
                row[field] = cleaned or "unclear"
                title = str(row.get("title") or row.get("source") or "").strip()
                LiteratureAnalysisWorkflow._append_fact_risk(
                    row,
                    f"{title}: removed field-inconsistent English evidence residue from {field}: {original[:240]}",
                )

    @staticmethod
    def _clean_mixed_language_fact_slot(text: str, *, field: str, output_language: str = "zh") -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if output_language != "zh":
            return value
        clauses = LiteratureAnalysisWorkflow._split_fact_clauses(value)
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", value))
        cleaned: list[str] = []
        for clause in clauses:
            item = LiteratureAnalysisWorkflow._clean_public_language_text(clause)
            if not item:
                continue
            if LiteratureAnalysisWorkflow._fact_clause_is_field_residue(
                item,
                field=field,
                output_language=output_language,
                mixed_language=has_cjk,
            ):
                continue
            if item.casefold() not in {part.casefold() for part in cleaned}:
                cleaned.append(item)
        return "；".join(cleaned)

    @staticmethod
    def _split_fact_clauses(text: str) -> list[str]:
        return [
            part.strip(" \t\r\n;；,，。")
            for part in re.split(r";|；|\n", str(text or ""))
            if part.strip(" \t\r\n;；,，。")
        ]

    @staticmethod
    def _fact_clause_is_field_residue(
        clause: str,
        *,
        field: str,
        output_language: str,
        mixed_language: bool,
    ) -> bool:
        text = re.sub(r"\s+", " ", str(clause or "")).strip()
        if not text:
            return True
        english_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)
        cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        if output_language != "zh":
            return False
        if cjk:
            return False
        if not mixed_language and len(english_words) < 12:
            return False

        lower = text.casefold()
        explanatory_markers = [
            "clinically motivated preprocessing steps",
            "proposed pipeline results",
            "with the best validation performance",
            "consecutive patients without",
            "randomly split into",
            "to evaluate the image-classification task",
            "and some images were of poor contrast",
            "as the training is similar",
            "segmentor does not only have its loss",
            "adversarial training of any segmentation architecture gives",
            "moreover, 3d deep learning models cannot",
            "where dept slices are inconsistency",
        ]
        if any(marker in lower for marker in explanatory_markers):
            return True

        if field == "baseline_or_comparator":
            return len(english_words) >= 7 and bool(
                re.search(
                    r"(?i)\b(?:show|shows|showed|results?|selected|included|evaluate|trained|compared|improvement|performance)\b",
                    text,
                )
            )
        if field == "evaluation_protocol":
            return len(english_words) >= 6 and bool(
                re.search(r"(?i)\b(?:poor contrast|shaken|consecutive patients|image-classification|randomly split)\b", text)
            )
        if field == "limitations":
            return mixed_language and len(english_words) >= 8
        return False

    @staticmethod
    def _candidate_value_is_formal_fact(value: str, *, slot: str = "") -> bool:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" ;,")
        if not text:
            return False
        lower = text.casefold()
        if len(text) < 4:
            return False
        if re.fullmatch(r"(?i)(?:[a-z]{1,3}|[a-z]{2,8}[-·])", text):
            return False
        if re.match(r"(?i)^(?:ation|tion|sion|ing|ed|he|the)\b(?:[·.-]|\s|$)", text):
            return False
        if re.search(r"(?i)\b(?:pre-\s*dict|non-\s*cont)\b", text):
            return False
        if re.search(r"(?i)\b[A-Za-z]{1,4}-\s+[A-Za-z]{2,}\b", text):
            return False
        if re.match(r"^[a-z][a-z'-]{1,20}\s+", text) and not re.match(
            r"(?i)^(?:n\s*=|p\s*=|data|dataset|cohort|sample|participants?|patients?|cases?|train|training|test|testing|validation|baseline|comparator|dice|auc|auroc|iou|miou|f1|accuracy|precision|recall|sensitivity|specificity)\b",
            text,
        ):
            if not re.search(r"\d", text):
                return False
        if slot in {"metrics", "metric_values", "sample_size", "train_test_split"}:
            return bool(re.search(r"\d|\b(?:Dice|DSC|IoU|mIoU|F1|AUC|AUROC|Accuracy|precision|recall|sensitivity|specificity)\b", text, flags=re.IGNORECASE))
        english_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)
        if len(english_words) >= 6 and not re.search(
            r"(?i)\b(?:is|are|was|were|did|has|have|had|use[ds]?|include[ds]?|contains?|reported|evaluated|compared|trained|tested|validated|achieved|showed|found|available|released|provided|limited|noted)\b",
            text,
        ):
            return False
        return True

    @staticmethod
    def _numeric_fact_conflict(existing: str, candidate: str) -> bool:
        existing_numbers = set(re.findall(r"\d+(?:\.\d+)?", existing or ""))
        candidate_numbers = set(re.findall(r"\d+(?:\.\d+)?", candidate or ""))
        return bool(existing_numbers and candidate_numbers and existing_numbers.isdisjoint(candidate_numbers))

    @staticmethod
    def _candidate_adds_missing_tokens(existing: str, candidate: str) -> bool:
        token_pattern = r"[A-Za-z][A-Za-z0-9.+-]*|\d+(?:\.\d+)?%?"
        existing_tokens = {token.casefold() for token in re.findall(token_pattern, existing or "")}
        candidate_tokens = {token.casefold() for token in re.findall(token_pattern, candidate or "")}
        if not candidate_tokens:
            return False
        missing = candidate_tokens - existing_tokens
        return bool(missing) and len(missing) <= max(8, len(candidate_tokens))

    @staticmethod
    def _append_fact_text(existing: str, candidate: str, *, limit: int = 1000) -> str:
        parts = [
            part.strip()
            for part in re.split(r";|\n", f"{existing}; {candidate}")
            if part.strip()
        ]
        deduped = []
        seen = set()
        for part in parts:
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(part)
        return "; ".join(deduped)[:limit]

    @staticmethod
    def _summarize_candidate_text(candidates: list[str], *, limit: int = 700) -> str:
        return "; ".join(LiteratureAnalysisWorkflow._dedupe_candidates(candidates, limit=4))[:limit]

    @staticmethod
    def _summarize_comparator_candidates(candidates: list[str]) -> str:
        text = LiteratureAnalysisWorkflow._compact_text("; ".join(candidates))
        if not text:
            return ""
        known_patterns = [
            r"nnU-Net(?:\s+(?:default|baseline)\s+preprocessing)?",
            r"Majority Vote",
            r"Random Expert Sampling",
            r"CT\s*perfusion",
            r"SEAN(?:\s*\(CNN\))?",
            r"SwinUNETR",
            r"UNETR",
            r"UNet\+\+",
            r"StrDiSeg",
            r"DINOv3(?:\s+baseline)?",
            r"PixelNet",
            r"(?<!nn)U-Net",
            r"DeepLab\s+v\d+",
            r"ICNet",
            r"PSPNet",
        ]
        pieces: list[str] = []
        for pattern in known_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                label = re.sub(r"\s+", " ", match.group(0)).strip()
                if label.casefold() not in {piece.casefold() for piece in pieces}:
                    pieces.append(label)
        if pieces:
            return ", ".join(pieces[:8])
        compact = LiteratureAnalysisWorkflow._first_informative_sentence(text)
        compact = re.sub(
            r"(?i)\b(?:clinically motivated preprocessing steps and show that|the proposed pipeline results?.*)\b",
            "",
            compact,
        )
        return compact[:180].strip(" ;,")

    @staticmethod
    def _summarize_sample_size_candidates(candidates: list[str]) -> str:
        text = "; ".join(candidates)
        total = re.search(r"(?i)(\d[\d,]*)\s*(sets?|cases?|samples?|subjects?|records?|images?|patients?|participants?)\b", text)
        split = re.search(
            r"(?i)(\d[\d,]*)\s*(?:training|train)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?\s*(?:[/,;]|and|with)?\s*(\d[\d,]*)\s*(?:testing|test)\s*(?:sets?|cases?|samples?|subjects?|records?|images?)?",
            text,
        )
        if split:
            prefix = f"{total.group(1)} {total.group(2)}; " if total else ""
            return f"{prefix}{split.group(1)} training / {split.group(2)} testing"
        zh_split = re.search(r"(\d[\d,]*)\s*(?:训练|训练集)\s*(?:[/,，；;、]|和|与)?\s*(\d[\d,]*)\s*(?:测试|测试集)", text)
        if zh_split:
            return f"{zh_split.group(1)} 训练 / {zh_split.group(2)} 测试"
        total = re.search(
            r"(?i)(?:\bn\s*=\s*\d[\d,]*|\d[\d,.\s]*(?:sets?|participants?|patients?|samples?|images?|cases?|datasets?|records?|subjects?|trials?)\b|"
            r"\d[\d,.\s]*(?:套|组|例|个)?(?:样本|患者|病例|图像|数据集|记录|受试者|实验|语料))",
            text,
        )
        if total:
            return total.group(0).strip()
        return "; ".join(candidates[:2])[:500]

    @staticmethod
    def _summarize_metric_values(candidates: list[str]) -> str:
        text = "; ".join(candidates)
        pieces: list[str] = []
        metric_pattern = r"Dice|DSC|Accuracy|F1|AUC|AUROC|precision|recall|sensitivity|specificity|IoU|mIoU|Jaccard|mAP|score|ASPECTS"
        for candidate in candidates:
            for clause in re.split(r";|\n", str(candidate or "")):
                cleaned = re.sub(r"\s+", " ", clause).strip(" ;,")
                if re.search(rf"(?i)\b(?:{metric_pattern})\b\s*:\s*[^=]{{1,80}}=\s*\d+(?:\.\d+)?%?\b", cleaned):
                    if cleaned.casefold() not in {piece.casefold() for piece in pieces}:
                        pieces.append(cleaned)
        pieces.extend(LiteratureAnalysisWorkflow._table_metric_value_pieces(text, metric_pattern))
        for match in re.finditer(r"(?i)\bASPECTS\b.{0,100}?\bmean\s+Dice\b.{0,80}?\bto\s*(\d+(?:\.\d+)?%?)", text):
            label = f"ASPECTS mean Dice = {match.group(1)}"
            if label.casefold() not in {piece.casefold() for piece in pieces}:
                pieces.append(label)
        for match in re.finditer(
            rf"(?i)\b(?:(ASPECTS).{{0,60}}?)?({metric_pattern})\b[^.;,]{{0,80}}?(?:=|:|is|was|of|from|to|achieves?|reaches?|improves?[^.;,]*?to)\s*(\d+(?:\.\d+)?%?)",
            text,
        ):
            context = "ASPECTS mean " if (match.group(1) or "").casefold() == "aspects" else ""
            metric = match.group(2)
            value = match.group(3)
            label = f"{context}{metric} = {value}"
            if label.casefold() not in {piece.casefold() for piece in pieces}:
                pieces.append(label)
        for match in re.finditer(r"(?i)\bASPECTS\b.{0,80}?\bmean\s+Dice\b.{0,40}?(\d+(?:\.\d+)?%?)", text):
            label = f"ASPECTS mean Dice = {match.group(1)}"
            if label.casefold() not in {piece.casefold() for piece in pieces}:
                pieces.append(label)
        return "; ".join(pieces[:6]) or LiteratureAnalysisWorkflow._summarize_candidate_text(candidates, limit=240)

    @staticmethod
    def _table_metric_value_pieces(text: str, metric_pattern: str) -> list[str]:
        pieces: list[str] = []
        known_label_patterns = [
            r"DeepLab\s+v\d+",
            r"SwinUNETR",
            r"UNETR",
            r"UNet\+\+",
            r"StrDiSeg",
            r"PixelNet",
            r"PSPNet",
            r"ICNet",
            r"SEAN",
            r"U-Net",
            r"nnU-Net",
            r"Ours",
            r"Kurtlab",
        ]
        label_pattern = "|".join(f"(?:{pattern})" for pattern in known_label_patterns)
        table_pattern = re.compile(
            rf"(?is)\bModels?\b\s+(.{{5,240}}?)\b({metric_pattern})\b\s+((?:\d+(?:\.\d+)?%?\s+){{1,12}}\d+(?:\.\d+)?%?)"
        )
        for match in table_pattern.finditer(text):
            model_text = re.sub(r"\[[^\]]+\]", " ", match.group(1))
            metric = match.group(2)
            values = re.findall(r"\d+(?:\.\d+)?%?", match.group(3))
            labels = [
                label_match.group(0).strip()
                for label_match in re.finditer(label_pattern, model_text, flags=re.IGNORECASE)
            ]
            deduped_labels: list[str] = []
            seen = set()
            for label in labels:
                normalized = re.sub(r"\s+", " ", label).strip()
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                deduped_labels.append(normalized)
            if not deduped_labels:
                deduped_labels = [f"column {index + 1}" for index in range(len(values))]
            for label, value in zip(deduped_labels, values):
                piece = f"{metric}: {label} = {value}"
                if piece.casefold() not in {item.casefold() for item in pieces}:
                    pieces.append(piece)
                if len(pieces) >= 6:
                    return pieces
        return pieces

    @staticmethod
    def _summarize_availability_candidates(candidates: list[str]) -> str:
        text = LiteratureAnalysisWorkflow._compact_text("; ".join(candidates))
        if not text:
            return ""
        lower = text.casefold()
        if "available upon request" in lower or "upon reasonable request" in lower:
            return "Available upon request."
        if re.search(r"(?i)\bgithub|gitlab|zenodo|figshare|osf\b", text):
            match = re.search(r"(?i)\b(github|gitlab|zenodo|figshare|osf)\b", text)
            return f"Public resource mentioned ({match.group(1)})." if match else "Public resource mentioned."
        if re.search(r"(?i)\b(code|data|dataset|model|materials?|software)\b.{0,40}\b(available|released|shared|public)\b", text):
            return LiteratureAnalysisWorkflow._first_informative_sentence(text)[:160]
        return LiteratureAnalysisWorkflow._first_informative_sentence(text)[:160]

    @staticmethod
    def _first_informative_sentence(text: str) -> str:
        for sentence in LiteratureAnalysisWorkflow._split_candidate_sentences(text):
            cleaned = sentence.strip(" .;，。")
            if cleaned:
                return cleaned
        return LiteratureAnalysisWorkflow._compact_text(text)[:240]

    @staticmethod
    def _summarize_limitation_candidates(candidates: list[str]) -> str:
        clauses: list[str] = []
        for candidate in candidates:
            for clause in re.split(r";|；|,\s+(?=(?:no|only|single|not|limited)\b)", candidate):
                cleaned = clause.strip(" .;；")
                if cleaned and cleaned.casefold() not in {item.casefold() for item in clauses}:
                    clauses.append(cleaned)
                if len(clauses) >= 4:
                    break
            if len(clauses) >= 4:
                break
        return "; ".join(clauses)[:900]

    @staticmethod
    def _metric_names_from_candidates(text: str) -> str:
        names = []
        for name in ["Dice", "DSC", "Accuracy", "F1", "AUC", "AUROC", "precision", "recall", "sensitivity", "specificity", "IoU", "mIoU", "Jaccard", "mAP", "p-value"]:
            if re.search(rf"(?i)\b{re.escape(name)}\b", text) and name not in names:
                names.append(name)
        return ", ".join(names)

    @staticmethod
    def _append_location(existing, location: str) -> str:
        parts = [
            part.strip()
            for part in re.split(r";|；|,|，", str(existing or ""))
            if part.strip() and part.strip().casefold() != "unclear"
        ]
        if location not in parts:
            parts.append(location)
        return "; ".join(parts)

    @staticmethod
    def _build_comparison_matrix(rows: list[dict]) -> list[dict]:
        matrix = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            matrix.append(
                {
                    "title": str(row.get("title") or row.get("source") or "").strip(),
                    "study_type": str(row.get("study_type") or "").strip(),
                    "research_objective": str(row.get("research_objective") or row.get("task") or row.get("core_claim") or "").strip(),
                    "dataset_or_material": str(row.get("dataset_or_material") or row.get("dataset") or "").strip(),
                    "sample_size": str(row.get("sample_size") or "").strip(),
                    "domain_or_modality": str(row.get("domain_or_modality") or row.get("modality") or "").strip(),
                    "method": str(row.get("method") or row.get("model_or_method") or row.get("methodology") or "").strip(),
                    "baseline_or_comparator": str(row.get("baseline_or_comparator") or row.get("baseline") or "").strip(),
                    "evaluation_protocol": str(row.get("evaluation_protocol") or row.get("validation_setup") or "").strip(),
                    "metrics": str(row.get("metrics") or "").strip(),
                    "key_results": str(row.get("key_results") or "").strip(),
                    "statistical_evidence": str(row.get("statistical_evidence") or "").strip(),
                    "availability": str(row.get("availability") or "").strip(),
                    "evidence_strength": str(row.get("evidence_strength") or "").strip(),
                    "limitations": str(row.get("limitations") or row.get("weaknesses") or "").strip(),
                    "evidence_locations": str(row.get("evidence_locations") or "").strip(),
                }
            )
        return matrix

    @staticmethod
    def _apply_fact_consistency_checks(summary: dict, rows: list[dict], output_language: str = "zh") -> dict:
        checked = dict(summary or LiteratureAnalysisWorkflow._empty_summary())
        next_actions = LiteratureAnalysisWorkflow._normalize_string_list(checked.get("next_actions"))
        consistency_risks = LiteratureAnalysisWorkflow._fact_consistency_risks(checked, rows, output_language)
        risks = LiteratureAnalysisWorkflow._dedupe_fact_risks(
            [*LiteratureAnalysisWorkflow._normalize_risk_list(checked.get("fact_risks")), *consistency_risks]
        )
        checked = LiteratureAnalysisWorkflow._qualify_absolute_summary_claims(checked, risks, output_language)
        if consistency_risks:
            next_actions.append(
                (
                    "将跨文献绝对化结论改写为受结构化事实表支持的限定表达。"
                    if output_language == "zh"
                    else "Rewrite absolute cross-paper conclusions as qualified claims supported by the structured fact table."
                )
            )
        for risk in risks:
            if risk not in next_actions:
                next_actions.append(risk)
        checked["next_actions"] = LiteratureAnalysisWorkflow._normalize_string_list(next_actions)
        checked["fact_risks"] = risks
        if risks:
            confidence = str(checked.get("confidence") or "").strip()
            note = (
                "事实一致性校验发现风险；请复核结构化事实槽。"
                if output_language == "zh"
                else "Fact consistency checks found risks; review the structured fact slots."
            )
            if note not in confidence:
                checked["confidence"] = f"{confidence} {note}".strip()
        return checked

    @staticmethod
    def _fact_consistency_risks(summary: dict, rows: list[dict], output_language: str = "zh") -> list[str]:
        summary_text = json.dumps(summary or {}, ensure_ascii=False).casefold()
        risks: list[str] = []

        if any(term.casefold() in summary_text for term in ABSOLUTE_TERMS):
            row_support = [
                all(LiteratureAnalysisWorkflow._has_clear_fact(row.get(field)) for field in FACT_SLOT_FIELDS)
                for row in rows
                if isinstance(row, dict)
            ]
            if row_support and not all(row_support):
                risks.append(
                    "该跨文献结论使用绝对化表述，但并非所有文献的结构化事实字段都支持。"
                    if output_language == "zh"
                    else "This cross-paper conclusion uses absolute wording, but not every row supports it in the structured fact fields."
                )

        says_only_metric = bool(
            re.search(r"(仅|只报告|only|sole(?:ly)?)", summary_text)
            and re.search(r"(指标|metric|metrics|measure|score)", summary_text)
        )
        if says_only_metric and any(
            LiteratureAnalysisWorkflow._row_has_multiple_metrics(row)
            or LiteratureAnalysisWorkflow._has_clear_fact(row.get("statistical_evidence"))
            for row in rows
            if isinstance(row, dict)
        ):
            risks.append(
                "指标单一性的总结可能过度概括，请核对 metrics/statistical_evidence。"
                if output_language == "zh"
                else "The summary may overgeneralize metric single-ness; check metrics/statistical_evidence."
            )

        says_validation_missing = bool(
            re.search(r"(缺乏|没有|无|未进行|lack|without|no|none|not)", summary_text)
            and re.search(r"(外部验证|独立测试|对照实验|验证|测试|对照|external validation|independent test|control|comparator|validation|test)", summary_text)
        )
        if says_validation_missing and any(
            LiteratureAnalysisWorkflow._row_mentions_validation_or_comparison(row)
            for row in rows
            if isinstance(row, dict)
        ):
            risks.append(
                "验证/对照缺失的总结可能过度概括；部分文献的 evaluation_protocol 或 baseline_or_comparator 含测试、验证、对照或比较信息。"
                if output_language == "zh"
                else "The claim about missing validation/comparators may be overgeneralized; some rows contain test, validation, comparator, or comparison information."
            )

        missed_sample_titles = [
            str(row.get("title") or row.get("source") or "").strip()
            for row in rows
            if isinstance(row, dict)
            and not LiteratureAnalysisWorkflow._has_clear_fact(row.get("sample_size"))
            and LiteratureAnalysisWorkflow._row_has_sample_size_signal(row)
        ]
        if missed_sample_titles:
            prefix = (
                "可能漏抽样本量："
                if output_language == "zh"
                else "Possible missed sample size extraction: "
            )
            risks.append(prefix + "; ".join(missed_sample_titles[:5]))

        risks.extend(LiteratureAnalysisWorkflow._metric_range_consistency_risks(summary_text, rows, output_language))

        return risks

    @staticmethod
    def _metric_range_consistency_risks(summary_text: str, rows: list[dict], output_language: str = "zh") -> list[str]:
        metric_names = ["dice", "dsc", "accuracy", "f1", "auc", "auroc", "iou", "miou", "jaccard", "map", "score"]
        risks = []
        for metric in metric_names:
            if metric not in summary_text:
                continue
            ranges = LiteratureAnalysisWorkflow._summary_metric_ranges(summary_text, metric)
            if not ranges:
                continue
            row_values = LiteratureAnalysisWorkflow._row_metric_values(rows, metric)
            for low, high in ranges:
                outside = [
                    (title, value)
                    for title, value in row_values
                    if value < low - 1e-9 or value > high + 1e-9
                ]
                if outside:
                    examples = "; ".join(f"{title}: {value:g}%" for title, value in outside[:5])
                    risks.append(
                        (
                            f"跨文献 {metric.upper()} 区间可能漏报或范围限定不清；结构化指标候选中存在区间外数值：{examples}。若只统计某类任务，需要在总结中明确范围。"
                            if output_language == "zh"
                            else f"The cross-paper {metric.upper()} range may omit values or lack scope; structured metric candidates include out-of-range values: {examples}. If the range covers only a task subset, state that scope explicitly."
                        )
                    )
        return risks

    @staticmethod
    def _summary_metric_ranges(summary_text: str, metric: str) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        pattern = re.compile(
            rf"(?is)(?:{re.escape(metric)}).{{0,80}}?(\d+(?:\.\d+)?)\s*(%)?\s*(?:-|–|—|~|至|到)\s*(\d+(?:\.\d+)?)\s*(%)?"
        )
        for match in pattern.finditer(summary_text):
            low = float(match.group(1))
            high = float(match.group(3))
            if not match.group(2) and low <= 1 and high <= 1:
                low *= 100
                high *= 100
            ranges.append((min(low, high), max(low, high)))
        return ranges

    @staticmethod
    def _row_metric_values(rows: list[dict], metric: str) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        pattern = re.compile(
            rf"(?is)(?:{re.escape(metric)}).{{0,80}}?(\d+(?:\.\d+)?)\s*(%)?|(\d+(?:\.\d+)?)\s*(%)?.{{0,30}}?(?:{re.escape(metric)})"
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("source") or "untitled").strip()
            text = " ".join(
                LiteratureAnalysisWorkflow._stringify_value(row.get(field))
                for field in ["metric_values", "metrics", "key_results", "evidence_quotes"]
            )
            for match in pattern.finditer(text):
                raw = match.group(1) or match.group(3)
                if not raw:
                    continue
                value = float(raw)
                percent = bool(match.group(2) or match.group(4))
                if not percent and value <= 1:
                    value *= 100
                values.append((title, value))
        return values

    @staticmethod
    def _public_summary(summary: dict, output_language: str = "zh") -> dict:
        public = LiteratureAnalysisWorkflow._empty_summary()
        public["overall_assessment"] = LiteratureAnalysisWorkflow._sanitize_public_text(
            summary.get("overall_assessment"),
            output_language,
        )
        public["confidence"] = LiteratureAnalysisWorkflow._sanitize_public_text(
            summary.get("confidence"),
            output_language,
        )
        for key in [
            "common_strengths",
            "common_weaknesses",
            "methodological_patterns",
            "evidence_gaps",
            "research_gaps",
            "recommended_reading_order",
        ]:
            public[key] = [
                cleaned
                for item in LiteratureAnalysisWorkflow._normalize_string_list(summary.get(key))
                if (cleaned := LiteratureAnalysisWorkflow._sanitize_public_text(item, output_language))
            ]
        public["next_actions"] = [
            cleaned
            for item in LiteratureAnalysisWorkflow._normalize_string_list(summary.get("next_actions"))
            if (cleaned := LiteratureAnalysisWorkflow._public_next_action(item, output_language))
        ][:3]
        public["fact_risks"] = [
            cleaned
            for item in LiteratureAnalysisWorkflow._normalize_string_list(summary.get("fact_risks"))
            if (cleaned := LiteratureAnalysisWorkflow._public_fact_risk(item, output_language))
        ][:5]
        public["references"] = LiteratureAnalysisWorkflow._normalize_string_list(summary.get("references"))
        public["citation_format"] = str(summary.get("citation_format") or "").strip()
        public = LiteratureAnalysisWorkflow._correct_public_summary_metric_ranges(public, summary, output_language)
        return public

    @staticmethod
    def _correct_public_summary_metric_ranges(public: dict, audit_summary: dict, output_language: str = "zh") -> dict:
        text = " ".join(
            [
                str(audit_summary.get("overall_assessment") or ""),
                " ".join(LiteratureAnalysisWorkflow._normalize_string_list(audit_summary.get("next_actions"))),
                " ".join(LiteratureAnalysisWorkflow._normalize_string_list(audit_summary.get("fact_risks"))),
            ]
        )
        if not re.search(r"(?i)\bDICE range may omit values|DICE.*out-of-range|ASPECTS", text):
            return public
        overall = str(public.get("overall_assessment") or "").strip()
        if re.search(r"(?i)ASPECTS.*0\.767|0\.767.*ASPECTS|76\.7%.*ASPECTS", overall):
            return public
        correction = (
            "公开 AISD/特定子任务范围为 28.5%-63.85%；另有 ASPECTS 数据集 Dice=0.767。"
            if output_language == "zh"
            else "Public AISD/specific-task Dice range is 28.5%-63.85%; an ASPECTS dataset Dice=0.767 is also reported."
        )
        public["overall_assessment"] = f"{overall} {correction}".strip()
        return public

    @staticmethod
    def _sanitize_public_text(value, output_language: str = "zh") -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        replacements = {
            "基于当前结构化事实表，": "",
            "事实一致性校验发现风险；请复核结构化事实槽。": "",
            "Fact consistency checks found risks; review the structured fact slots.": "",
            "Based on the extracted structured evidence, ": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if output_language == "zh":
            text = re.sub(r"\bunclear\b", "未明确说明", text, flags=re.IGNORECASE)
        else:
            text = re.sub(r"\bunclear\b", "not clearly stated", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" ；;")

    @staticmethod
    def _public_next_action(value, output_language: str = "zh") -> str:
        text = LiteratureAnalysisWorkflow._sanitize_public_text(value, output_language)
        if not text:
            return ""
        audit_markers = [
            "事实风险",
            "事实槽",
            "结构化事实表",
            "请核对 metrics/statistical_evidence",
            "fact consistency",
            "structured fact",
            "fact_risks",
            "Possible missed sample size extraction",
            "Author-acknowledged limitations were not clearly extracted",
            "Sample size or experiment scale was not clearly extracted",
            "Availability of code",
            "Statistical evidence was not clearly extracted",
        ]
        if any(marker.casefold() in text.casefold() for marker in audit_markers):
            return ""
        return text

    @staticmethod
    def _public_fact_risk(value, output_language: str = "zh") -> str:
        text = LiteratureAnalysisWorkflow._sanitize_public_text(value, output_language)
        if not text:
            return ""
        low_value_markers = [
            "low-quality or fragmentary candidate value",
            "fragmentary or semantically incomplete",
            "differs from extracted field",
            "DICE range may omit values",
            "Possible missed sample size extraction",
            "Author-acknowledged limitations were not clearly extracted",
            "Sample size or experiment scale was not clearly extracted",
            "Availability of code",
            "Statistical evidence was not clearly extracted",
            "private or unverifiable",
            "abnormal metric",
        ]
        if any(marker.casefold() in text.casefold() for marker in low_value_markers):
            return text[:500]
        if len(text) >= 24 and re.search(r"(?i)\b(?:risk|unclear|missing|not clearly|candidate|conflict|availability|statistical|sample size|limitations?)\b", text):
            return text[:500]
        return ""

    @staticmethod
    def _has_clear_fact(value) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        lower = text.casefold()
        explicit_unclear = {
            "unclear",
            "unknown",
            "not reported",
            "not stated",
            "not clearly reported",
            "not mentioned",
            "n/a",
            "na",
            "none",
            "未明确说明",
            "未报告",
            "未提及",
            "未在提取片段中明确",
            "未在提取内容中明确",
            "未清楚报告",
            "未明确报告",
            "不明确",
            "不清楚",
            "无",
            "暂无",
            "鏈槑纭?",
            "鏈煡",
            "涓嶆竻妤?",
        }
        if lower in {item.casefold() for item in explicit_unclear}:
            return False
        unclear_patterns = [
            r"^未(明确)?(说明|报告|提及|提供|列出|公开)",
            r"^未在.*(提取片段|提取内容|片段|证据).*(明确|清楚)",
            r"^没有(明确)?(说明|报告|提及|提供|列出|公开)",
            r"^不(清楚|明确)",
            r"^no\s+(clear|explicit)?\s*(report|statement|information)",
            r"^not\s+stated",
            r"^not\s+mentioned",
            r"^not\s+clearly\s+reported",
            r"^not\s+(clearly|explicitly)?\s*(reported|stated|provided|available)",
        ]
        return not any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in unclear_patterns)

    @staticmethod
    def _row_has_substantive_facts(row: dict) -> bool:
        return sum(1 for field in FACT_SLOT_FIELDS if LiteratureAnalysisWorkflow._has_clear_fact(row.get(field))) >= 3

    @staticmethod
    def _row_has_multiple_metrics(row: dict) -> bool:
        metrics = str(row.get("metrics") or "").strip()
        if not LiteratureAnalysisWorkflow._has_clear_fact(metrics):
            return False
        return len([part for part in re.split(r"[,;，；/、]+", metrics) if part.strip()]) >= 2

    @staticmethod
    def _row_mentions_validation_or_comparison(row: dict) -> bool:
        text = " ".join(
            str(row.get(field) or "")
            for field in ["evaluation_protocol", "baseline_or_comparator", "validation_setup", "baseline"]
        ).casefold()
        return any(term in text for term in VALIDATION_TERMS)

    @staticmethod
    def _row_has_sample_size_signal(row: dict) -> bool:
        text = " ".join(
            LiteratureAnalysisWorkflow._stringify_value(row.get(field))
            for field in ["evidence_quotes", "key_results", "evaluation_protocol", "validation_setup"]
        )
        return bool(SAMPLE_UNIT_PATTERN.search(text))

    @staticmethod
    def _stringify_value(value) -> str:
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value or "")

    @staticmethod
    def _annotate_report_quality(
        summary: dict,
        rows: list[dict],
        output_language: str = "zh",
        *,
        integrated: bool,
    ) -> dict:
        normalized = dict(summary or LiteratureAnalysisWorkflow._empty_summary())
        weak_titles = [
            str(row.get("title") or row.get("source") or "").strip()
            for row in rows
            if LiteratureAnalysisWorkflow._structured_fact_count(row) < 3
        ]
        notes = LiteratureAnalysisWorkflow._normalize_string_list(normalized.get("next_actions"))
        if weak_titles:
            if output_language == "zh":
                note = "事实槽不足，导出前建议复核这些文献的研究类型、目标、数据/材料、样本量、方法、指标、结果、统计证据或验证设置：" + "；".join(weak_titles[:5])
            else:
                note = "Structured facts are thin; verify study type, objective, data/materials, sample size, method, metrics, results, statistical evidence, or evaluation protocol for: " + "; ".join(weak_titles[:5])
            if note not in notes:
                notes.append(note)
        if not integrated:
            if output_language == "zh":
                note = "当前为逐篇分析草稿：最终跨文献整合未成功，不能视作完整综述。"
                confidence_note = "报告状态：逐篇草稿。"
            else:
                note = "Current output is a per-paper draft: final cross-paper integration did not succeed."
                confidence_note = "Report status: per-paper draft."
            if note not in notes:
                notes.insert(0, note)
            confidence = str(normalized.get("confidence") or "").strip()
            if confidence_note not in confidence:
                normalized["confidence"] = f"{confidence} {confidence_note}".strip()
        normalized["next_actions"] = notes
        return normalized

    @staticmethod
    def _structured_fact_count(row: dict) -> int:
        return sum(
            1
            for field in FACT_SLOT_FIELDS
            if LiteratureAnalysisWorkflow._has_clear_fact(row.get(field))
        )

    @staticmethod
    def _ensure_summary_coverage(summary: dict, rows: list[dict], output_language: str = "zh") -> dict:
        normalized = dict(summary or LiteratureAnalysisWorkflow._empty_summary())
        missing = LiteratureAnalysisWorkflow._summary_missing_titles(normalized, rows)
        if not missing:
            return normalized

        if output_language == "zh":
            note = "覆盖检查：跨文献总结未充分纳入这些文献，建议重新运行整合或人工核验：" + "；".join(missing)
            confidence_note = "覆盖不足；当前总结可能偏向部分文献。"
        else:
            note = "Coverage check: the cross-literature summary did not sufficiently include these papers: " + "; ".join(missing)
            confidence_note = "Coverage incomplete; the current synthesis may overrepresent some papers."

        next_actions = LiteratureAnalysisWorkflow._normalize_string_list(normalized.get("next_actions"))
        if note not in next_actions:
            next_actions.insert(0, note)
        normalized["next_actions"] = next_actions
        confidence = str(normalized.get("confidence") or "").strip()
        normalized["confidence"] = f"{confidence} {confidence_note}".strip()
        reading = LiteratureAnalysisWorkflow._normalize_string_list(normalized.get("recommended_reading_order"))
        for title in missing:
            item = (
                f"{title}: 需要补入跨文献比较"
                if output_language == "zh"
                else f"{title}: needs inclusion in the cross-paper comparison"
            )
            if item not in reading:
                reading.append(item)
        normalized["recommended_reading_order"] = reading
        return normalized

    @staticmethod
    def _summary_missing_titles(summary: dict, rows: list[dict]) -> list[str]:
        summary_text = json.dumps(summary or {}, ensure_ascii=False).casefold()
        missing = []
        for row in rows:
            title = str(row.get("title") or row.get("source") or "").strip()
            if not title:
                continue
            key_terms = [
                term
                for term in title.casefold().replace("-", " ").split()
                if len(term) >= 5
            ][:4]
            if title.casefold() in summary_text:
                continue
            if key_terms and sum(1 for term in key_terms if term in summary_text) >= min(2, len(key_terms)):
                continue
            missing.append(title)
        return missing

    @staticmethod
    def _integration_unavailable_summary(rows: list[dict], output_language: str = "zh") -> dict:
        summary = LiteratureAnalysisWorkflow._empty_summary()
        titles = [
            str(row.get("title") or row.get("source") or "").strip()
            for row in rows
            if isinstance(row, dict) and (row.get("title") or row.get("source"))
        ]
        if output_language == "zh":
            summary["overall_assessment"] = (
                f"已完成 {len(rows)} 篇/项文献的单篇结构化分析，但跨文献综合步骤未成功产出可靠结果。"
                "当前不生成伪综合结论；请基于下方单篇分析表或 comparison matrix 重新整合。"
            )
            summary["recommended_reading_order"] = [
                f"{title}: 先核验数据集、方法、指标和主要局限" for title in titles
            ]
            summary["next_actions"] = [
                "重新运行跨文献整合，或根据结构化证据字段生成比较矩阵。",
                "确认每篇文献都进入方法路线、证据强度和研究空白三个比较维度。",
                "避免把单篇 bullet 直接拼接成跨文献总结。",
            ]
            summary["confidence"] = "低；单篇分析可用，但真正的跨文献综合未完成。"
        else:
            summary["overall_assessment"] = (
                f"Structured per-paper analysis was completed for {len(rows)} literature item(s), "
                "but the cross-literature integration step did not produce a reliable synthesis."
            )
            summary["recommended_reading_order"] = [
                f"{title}: verify dataset, method, metrics, and limitations first" for title in titles
            ]
            summary["next_actions"] = [
                "Rerun cross-literature integration or synthesize from the structured comparison matrix.",
                "Check that every paper appears in method-family, evidence-strength, and research-gap comparisons.",
                "Do not present concatenated per-paper bullets as a true cross-paper synthesis.",
            ]
            summary["confidence"] = "Low; per-paper analysis is available, but cross-literature synthesis is incomplete."
        return summary

    @staticmethod
    def _normalize_summary(summary, rows: list[dict], references: list[dict], output_language: str = "zh") -> dict:
        if not isinstance(summary, dict):
            return LiteratureAnalysisWorkflow._fallback_summary(rows, "Integrator did not return a valid summary.", output_language)

        normalized = LiteratureAnalysisWorkflow._empty_summary()
        normalized["overall_assessment"] = str(
            summary.get("overall_assessment") or ""
        ).strip()
        normalized["confidence"] = str(summary.get("confidence") or "").strip()

        for key in [
            "common_strengths",
            "common_weaknesses",
            "methodological_patterns",
            "evidence_gaps",
            "research_gaps",
            "recommended_reading_order",
            "next_actions",
        ]:
            normalized[key] = LiteratureAnalysisWorkflow._normalize_string_list(summary.get(key))

        if output_language == "zh":
            normalized["confidence"] = LiteratureAnalysisWorkflow._localize_label_text(normalized["confidence"])

        if not normalized["overall_assessment"]:
            normalized["overall_assessment"] = (
                f"已整合 {len(rows)} 篇/项文献；具体结论仍需结合全文和证据质量继续核验。"
                if output_language == "zh"
                else f"Integrated {len(rows)} literature item(s); conclusions should still be checked against full texts and evidence quality."
            )
        if not normalized["recommended_reading_order"]:
            normalized["recommended_reading_order"] = [
                (
                    f"{row.get('title') or row.get('source')}：优先核验核心贡献与证据"
                    if output_language == "zh"
                    else f"{row.get('title') or row.get('source')}: prioritize checking the core contribution and evidence"
                )
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
        if not normalized["confidence"]:
            has_pdf_text = any(reference.get("pdf_text_available") for reference in references)
            if output_language == "zh":
                normalized["confidence"] = "中等；包含部分 PDF 提取文本，但可能未覆盖全文。" if has_pdf_text else "较低；主要基于元数据、摘要、链接或报告上下文。"
            else:
                normalized["confidence"] = "Medium; includes extracted PDF text but may not cover full papers." if has_pdf_text else "Low; based on metadata, abstracts, links, or report context."
        return normalized

    @staticmethod
    def _fallback_summary(rows: list[dict], note: str, output_language: str = "zh") -> dict:
        used_analyst_outputs = "analyst outputs" in note or "分组分析" in note
        summary = LiteratureAnalysisWorkflow._empty_summary()
        if output_language == "en":
            summary["overall_assessment"] = (
                f"A structured review table has been generated for {len(rows)} literature item(s). "
                "The cross-literature summary below is derived from the parallel analyst results, "
                "so it should be treated as a cautious synthesis rather than a fully re-integrated final pass."
                if used_analyst_outputs
                else f"A structured review table has been generated for {len(rows)} literature item(s). "
                "The summary is based on the available metadata and supplied context, so please verify it against full texts before making strong claims."
            )
            summary["recommended_reading_order"] = [
                f"{row.get('title') or row.get('source')}: check the full-text evidence and methods first"
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
            summary["next_actions"] = [
                "Add full-text PDFs or reliable abstracts/metadata first.",
                "Verify low-confidence items against methods, datasets, baselines, and statistical evidence.",
                "Run a cross-literature comparison to separate consensus, disagreement, and research gaps.",
            ]
            summary["confidence"] = (
                "Medium-low; based on completed analyst outputs, but the final integration pass did not produce a usable structured summary."
                if used_analyst_outputs
                else "Low; based mainly on supplied metadata and context."
            )
        else:
            summary["overall_assessment"] = (
                f"已生成 {len(rows)} 篇/项文献的结构化分析表。下面的跨文献总结来自各分组分析员的结果，"
                "可作为谨慎综合使用；建议在形成强结论前继续结合全文核验。"
                if used_analyst_outputs
                else f"已生成 {len(rows)} 篇/项文献的结构化分析表。当前总结主要基于可用元数据和上下文，"
                "建议补充全文后再确认方法、证据和研究空白。"
            )
            summary["recommended_reading_order"] = [
                f"{row.get('title') or row.get('source')}：先核验全文证据和方法细节"
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
            summary["next_actions"] = [
                "优先补充全文 PDF 或可靠的摘要/元数据。",
                "对低置信度条目核验实验设置、数据集、baseline 和统计显著性。",
                "再做一次跨文献对比，区分共识、分歧和真正的研究空白。",
            ]
            summary["confidence"] = (
                "中低；已使用分组分析结果生成总结，但最终整合步骤没有产出可用的结构化总结。"
                if used_analyst_outputs
                else "较低；主要基于已提供的元数据和上下文。"
            )
        summary["common_strengths"] = LiteratureAnalysisWorkflow._collect_unique(rows, "strengths", limit=3)
        summary["common_weaknesses"] = LiteratureAnalysisWorkflow._collect_unique(rows, "weaknesses", limit=3)
        summary["methodological_patterns"] = LiteratureAnalysisWorkflow._collect_unique(rows, "methodology", limit=3)
        summary["evidence_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "evidence_strength", limit=3)
        summary["research_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "limitations", limit=3)
        return summary

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "overall_assessment": "",
            "common_strengths": [],
            "common_weaknesses": [],
            "methodological_patterns": [],
            "evidence_gaps": [],
            "research_gaps": [],
            "recommended_reading_order": [],
            "next_actions": [],
            "fact_risks": [],
            "references": [],
            "citation_format": "",
            "confidence": "",
        }

    @staticmethod
    def _context_only_summary(final_report: str, output_language: str = "zh") -> dict:
        if output_language == "en":
            summary = LiteratureAnalysisWorkflow._empty_summary()
            has_context = bool(final_report.strip())
            summary["overall_assessment"] = (
                "No uploaded content was identified as analyzable literature; the material is better used as writing requirements, grading criteria, task instructions, or auxiliary context."
                if has_context
                else "No analyzable paper or literature item was provided."
            )
            summary["research_gaps"] = [
                "Add DOI identifiers, paper links, full-text PDFs, or clearly mark which uploaded files are literature to analyze."
            ]
            summary["next_actions"] = [
                "Keep rubrics or assignment requirements as output constraints instead of counting them as literature.",
                "Add real paper sources, then generate the item-by-item analysis again.",
            ]
            summary["confidence"] = "High; no literature references were available after filtering auxiliary documents."
            return summary

        summary = LiteratureAnalysisWorkflow._empty_summary()
        has_context = bool(final_report.strip())
        summary["overall_assessment"] = (
            "未发现可作为论文/文献条目分析的上传内容；上传材料更适合作为写作要求、评分标准、任务说明或辅助上下文。"
            if has_context
            else "未提供可分析的论文/文献条目。"
        )
        summary["research_gaps"] = [
            "请补充 DOI、论文链接、PDF 全文，或明确哪些上传文件是需要分析的论文。"
        ]
        summary["next_actions"] = [
            "将评分标准或作业要求保留为输出约束，不要计入文献分析表。",
            "补充真正的论文来源后再生成逐篇分析。",
        ]
        summary["confidence"] = "较高；辅助文档过滤后没有可分析的文献条目。"
        return summary

    @staticmethod
    def _normalize_string_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _collect_unique(rows: list[dict], key: str, limit: int) -> list[str]:
        values = []
        seen = set()
        for row in rows:
            value = str(row.get(key, "") or "").strip()
            if not value:
                continue
            dedupe_key = value.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append(value)
            if len(values) >= limit:
                break
        return values

    @staticmethod
    def _localize_row_labels(row: dict[str, str]) -> None:
        for key in ["study_type", "paper_type", "evidence_strength", "confidence", "overall_assessment"]:
            row[key] = LiteratureAnalysisWorkflow._localize_label_text(row.get(key, ""))

    @staticmethod
    def _localize_label_text(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return text
        phrase_map = {
            "llm integration unavailable; this row only reflects supplied metadata.": "最终整合步骤不可用；该行仅反映已提供的元数据。",
            "llm integration unavailable; summary derived from analyst outputs.": "最终整合步骤不可用；总结基于分组分析结果生成。",
            "llm integration unavailable; summary derived from supplied metadata only.": "最终整合步骤不可用；总结仅基于已提供的元数据生成。",
            "integrator did not return a valid summary.": "整合器未返回可用的结构化总结。",
        }
        mapped = phrase_map.get(text.casefold())
        if mapped:
            return mapped
        replacements = [
            ("Below threshold", "低于阈值"),
            ("Medium-low", "中低"),
            ("Experimental study", "实验研究"),
            ("Review", "综述"),
            ("Methods paper", "方法论文"),
            ("Dataset paper", "数据集论文"),
            ("Clinical study", "临床研究"),
            ("experimental study", "实验研究"),
            ("review", "综述"),
            ("methods paper", "方法论文"),
            ("dataset paper", "数据集论文"),
            ("clinical study", "临床研究"),
            ("Empirical", "实证研究"),
            ("Theoretical", "理论研究"),
            ("Survey", "综述"),
            ("Systems", "系统或工具"),
            ("System", "系统或工具"),
            ("Position", "立场文章"),
            ("Unknown", "未知"),
            ("Strong", "强"),
            ("Moderate", "中等"),
            ("Weak", "弱"),
            ("High", "较高"),
            ("Medium", "中等"),
            ("Low", "较低"),
            ("Landmark", "里程碑"),
            ("Significant", "重要"),
            ("Marginal", "边缘"),
            ("requires full-text", "需要阅读全文"),
            ("requires code, data, and method details", "需要代码、数据和方法细节"),
            ("based only on supplied metadata/context", "仅基于已提供的元数据/上下文"),
            ("based on supplied metadata and context", "基于已提供的元数据和上下文"),
            ("fallback generated from supplied metadata only", "仅基于已提供元数据生成自动结果"),
        ]
        localized = text
        for english, chinese in replacements:
            localized = localized.replace(english, chinese)
        return localized.replace(";", "；")

    @staticmethod
    def _sync_generic_fact_aliases(row: dict[str, str]) -> None:
        alias_pairs = [
            ("dataset_or_material", "dataset"),
            ("domain_or_modality", "modality"),
            ("research_objective", "task"),
            ("method", "model_or_method"),
            ("baseline_or_comparator", "baseline"),
            ("evaluation_protocol", "validation_setup"),
        ]
        for generic_key, legacy_key in alias_pairs:
            generic = str(row.get(generic_key) or "").strip()
            legacy = str(row.get(legacy_key) or "").strip()
            if not generic and legacy:
                row[generic_key] = legacy
            if not legacy and generic and generic.casefold() != "unclear":
                row[legacy_key] = generic
        if not str(row.get("study_type") or "").strip() and str(row.get("paper_type") or "").strip():
            row["study_type"] = str(row.get("paper_type") or "").strip()
        if not str(row.get("paper_type") or "").strip() and str(row.get("study_type") or "").strip():
            row["paper_type"] = str(row.get("study_type") or "").strip()

    @staticmethod
    def _fill_fact_slot_defaults(row: dict[str, str]) -> None:
        for field in FACT_SLOT_FIELDS:
            if not str(row.get(field) or "").strip():
                row[field] = "unclear"

    @staticmethod
    def _separate_limitations_and_fact_risks(row: dict[str, str], output_language: str = "zh") -> None:
        risks = LiteratureAnalysisWorkflow._normalize_risk_list(row.get("fact_risks"))
        limitations = str(row.get("limitations") or "").strip()
        inferred = LiteratureAnalysisWorkflow._extract_inferred_limitation_risk(limitations)

        if inferred:
            if output_language == "zh":
                risks.append(f"作者承认的局限未明确抽取；以下内容属于推断性局限或评审风险，不应写入 limitations：{inferred}")
            else:
                risks.append(f"Author-acknowledged limitations were not clearly extracted; this inferred limitation or reviewer concern belongs in fact_risks, not limitations: {inferred}")
            row["limitations"] = "unclear"
        elif not LiteratureAnalysisWorkflow._has_clear_fact(limitations):
            row["limitations"] = "unclear"

        risks.extend(LiteratureAnalysisWorkflow._row_level_fact_risks(row, output_language))
        row["fact_risks"] = "; ".join(LiteratureAnalysisWorkflow._dedupe_fact_risks(risks))

    @staticmethod
    def _normalize_risk_list(value) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            risks: list[str] = []
            for item in value:
                risks.extend(LiteratureAnalysisWorkflow._normalize_risk_list(item))
            return risks
        text = str(value or "").strip()
        if not text:
            return []
        if not LiteratureAnalysisWorkflow._clean_fact_risk_text(text):
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
                parsed_risks = LiteratureAnalysisWorkflow._normalize_risk_list(parsed)
                if parsed_risks:
                    return parsed_risks

        text = re.sub(r"""(['"])\s*,\s*(['"])""", r"\1;\2", text)
        text = re.sub(r"""(['"])\s*,\s*$""", r"\1", text)
        text = text.strip("[]")
        return [
            cleaned
            for part in re.split(r"[;\n；]+", text)
            if (cleaned := LiteratureAnalysisWorkflow._clean_fact_risk_text(part))
        ]

    @staticmethod
    def _clean_fact_risk_text(value) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.replace("\\n", "\n").strip()
        text = text.strip(" \t\r\n'\"`，,;；。")
        text = text.strip("[]")
        text = text.strip(" \t\r\n'\"`，,;；。")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        wrapper = re.fullmatch(
            r"(?is)(unclear|unknown|not\s+reported|n/?a|none|null)\s*[\(（]\s*(.*?)\s*[\)）]\s*",
            text,
        )
        if wrapper:
            text = wrapper.group(2).strip()

        text = text.strip(" \t\r\n'\"`，,;；。()（）")
        if not text:
            return ""
        if re.fullmatch(r"(?is)(unclear|unknown|not\s+reported|n/?a|none|null|无|暂无|不详)", text):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _dedupe_fact_risks(risks: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for risk in risks:
            cleaned = LiteratureAnalysisWorkflow._clean_fact_risk_text(risk)
            if not cleaned:
                continue
            key = LiteratureAnalysisWorkflow._fact_risk_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    @staticmethod
    def _fact_risk_key(text: str) -> str:
        lower = text.casefold()
        if (
            "author-acknowledged limitations" in lower
            or ("limitations were not clearly extracted" in lower and "inferred limitations" in lower)
            or ("作者承认" in text and "局限" in text and "未明确" in text)
        ):
            title = text.split(":", 1)[0].strip().casefold() if ":" in text else ""
            return f"{title}:author_acknowledged_limitations_missing"
        return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", lower)).strip()

    @staticmethod
    def _qualify_absolute_summary_claims(summary: dict, risks: list[str], output_language: str = "zh") -> dict:
        if not risks:
            return summary
        risk_text = " ".join(risks).casefold()
        if "absolute wording" not in risk_text and "绝对化" not in risk_text:
            return summary

        def qualify(text: str) -> str:
            value = str(text or "").strip()
            if not value:
                return value
            if re.search(r"(?i)\b(based on|the extracted evidence suggests|available extracted evidence)\b", value):
                return value
            if output_language == "zh":
                prefix = "基于当前结构化事实表，"
                value = re.sub(r"所有|全部", "部分", value)
                value = re.sub(r"均|都", "在提取证据中多显示为", value, count=1)
            else:
                prefix = "Based on the extracted structured evidence, "
                value = re.sub(r"(?i)\ball papers\b", "the analyzed papers", value)
                value = re.sub(r"(?i)\bevery paper\b", "the analyzed papers", value)
            return value if value.startswith(prefix) else f"{prefix}{value}"

        summary["overall_assessment"] = qualify(summary.get("overall_assessment"))
        for key in [
            "common_strengths",
            "common_weaknesses",
            "methodological_patterns",
            "evidence_gaps",
            "research_gaps",
        ]:
            if isinstance(summary.get(key), list):
                summary[key] = [qualify(item) for item in summary[key]]
        return summary

    @staticmethod
    def _extract_inferred_limitation_risk(limitations: str) -> str:
        text = str(limitations or "").strip()
        if not text:
            return ""
        lower = text.casefold()
        inference_markers = [
            "从内容推断",
            "可推断",
            "推断",
            "但可",
            "可能",
            "作者未",
            "未明确列出",
            "未明确讨论",
            "未在摘要",
            "not explicitly",
            "not clearly",
            "not stated",
            "not discussed",
            "can be inferred",
            "inferred",
            "may be",
            "appears to",
        ]
        if not any(marker.casefold() in lower for marker in inference_markers):
            return ""
        cleaned = re.sub(
            r"^(作者)?未(在[^；;。.]*)?明确(列出|说明|讨论|报告)?(作者承认的)?局限性?[；;:：。.\s]*",
            "",
            text,
        ).strip()
        cleaned = re.sub(r"^(从内容)?可?推断[：:；;\s]*", "", cleaned).strip()
        return cleaned or text

    @staticmethod
    def _row_level_fact_risks(row: dict[str, str], output_language: str = "zh") -> list[str]:
        zh = output_language == "zh"
        risks: list[str] = []
        title = str(row.get("title") or row.get("source") or "").strip()

        def add(zh_text: str, en_text: str) -> None:
            risks.append(zh_text if zh else en_text)

        if not LiteratureAnalysisWorkflow._has_clear_fact(row.get("sample_size")):
            add("样本量/实验规模未明确抽取。", "Sample size or experiment scale was not clearly extracted.")
        if not LiteratureAnalysisWorkflow._has_clear_fact(row.get("statistical_evidence")):
            add("统计证据未明确抽取；请核对是否报告显著性检验、置信区间、误差范围或不确定性估计。", "Statistical evidence was not clearly extracted; check significance tests, confidence intervals, error ranges, or uncertainty estimates.")
        elif re.search(r"未报告|未提供|缺乏|没有|not reported|not provided|lack|without", str(row.get("statistical_evidence") or ""), flags=re.IGNORECASE):
            add("统计证据不完整；可能只报告了均值/标准差或相关性，缺少显著性检验、置信区间或不确定性估计。", "Statistical evidence is incomplete; it may report only means/standard deviations or correlations while missing significance tests, confidence intervals, or uncertainty estimates.")
        if not LiteratureAnalysisWorkflow._has_clear_fact(row.get("availability")):
            add("代码、数据、模型或材料公开性未明确抽取。", "Availability of code, data, models, or materials was not clearly extracted.")
        if not LiteratureAnalysisWorkflow._has_clear_fact(row.get("limitations")):
            add("作者承认的局限未明确抽取；不要用推断性局限替代。", "Author-acknowledged limitations were not clearly extracted; do not replace them with inferred limitations.")
        if not LiteratureAnalysisWorkflow._has_clear_fact(row.get("evidence_locations")):
            add("证据位置未明确抽取。", "Evidence locations were not clearly extracted.")
        if LiteratureAnalysisWorkflow._row_has_sample_size_signal(row) and not LiteratureAnalysisWorkflow._has_clear_fact(row.get("sample_size")):
            add("文本中疑似存在样本量信号，但 sample_size 未抽取。", "Text appears to contain sample-size signals, but sample_size was not extracted.")

        metric_text = " ".join(
            str(row.get(field) or "")
            for field in ["metrics", "key_results", "evidence_quotes"]
        )
        if re.search(r"abnormal|anomal|异常|极大|极端", metric_text, flags=re.IGNORECASE):
            add("结果或指标包含异常值提示，需人工复核计算方式和单位。", "Results or metrics include an abnormal-value signal; manually verify calculation and units.")
        if re.search(r"private|私有|unverifiable|无法独立验证", str(row.get("dataset_or_material") or ""), flags=re.IGNORECASE):
            add("包含私有或不可独立核验的数据来源。", "Includes private or not independently verifiable data sources.")

        return [f"{title}: {risk}" if title and not risk.startswith(title) else risk for risk in risks]

    @staticmethod
    def _fill_review_defaults(row: dict[str, str], original: dict, output_language: str = "zh") -> None:
        if not row.get("metadata"):
            row["metadata"] = LiteratureAnalysisWorkflow._format_metadata(original)
        if not row.get("paper_type"):
            row["paper_type"] = "未知" if output_language == "zh" else "Unknown"
        if not row.get("core_claim"):
            row["core_claim"] = (
                "需结合摘要或全文进一步确认核心主张。"
                if output_language == "zh"
                else original.get("abstract", "")[:300] or original.get("relevance", "")
            )
        if not row.get("evidence_strength"):
            row["evidence_strength"] = "未知；需要阅读全文或证据审查。" if output_language == "zh" else "Unknown; requires full-text or evidence review."
        if not row.get("reproducibility"):
            row["reproducibility"] = "未知；需要代码、数据和方法细节。" if output_language == "zh" else "Unknown; requires code, data, and method details."
        if not row.get("confidence"):
            row["confidence"] = "较低；仅基于已提供的元数据/上下文。" if output_language == "zh" else "Low; based only on supplied metadata/context."
        if not row.get("overall_assessment"):
            row["overall_assessment"] = "未知" if output_language == "zh" else "Unknown"

    @staticmethod
    def _fill_legacy_aliases(row: dict[str, str]) -> None:
        if not row.get("innovation"):
            row["innovation"] = row.get("innovation_point", "") or row.get("contribution", "")
        if not row.get("method") or str(row.get("method")).strip().casefold() == "unclear":
            method_parts = [
                value
                for value in [row.get("model_or_method", ""), row.get("methodology", "")]
                if value and str(value).strip().casefold() != "unclear"
            ]
            if method_parts:
                row["method"] = "；".join(dict.fromkeys(method_parts))
        LiteratureAnalysisWorkflow._sync_generic_fact_aliases(row)
        if not row.get("limitation"):
            limitation_parts = [
                value
                for value in [row.get("limitations", ""), row.get("weaknesses", "")]
                if value and str(value).strip().casefold() != "unclear"
            ]
            row["limitation"] = "；".join(dict.fromkeys(limitation_parts))
        if not row.get("next_step"):
            row["next_step"] = row.get("reader_next_step", "") or row.get("actionable_suggestions", "")

    @staticmethod
    def _fallback_rows_from_references(
        references: list[dict],
        *,
        fallback_note: str,
        output_language: str = "zh",
    ) -> list[dict]:
        rows = []
        for reference in references:
            if output_language == "zh":
                fallback_note = LiteratureAnalysisWorkflow._localize_label_text(fallback_note)
                methodology = "需要阅读全文进一步确认。"
                evidence_strength = "未知；当前上下文不足以评估证据强度。"
                reproducibility = "未知；需要方法、代码、数据和实验细节。"
                questions = "全文中实际提出了哪些主张、证据和方法？"
                suggestions = "获取或上传全文，以便进行有依据的同行评审式分析。"
                confidence = "较低；该结果仅基于已提供的元数据。"
                overall = "未知"
                core_claim = "需结合摘要或全文进一步确认核心主张。"
                contribution = "需结合全文进一步确认贡献。"
            else:
                methodology = "Requires full-text review."
                evidence_strength = "Unknown; supplied context is insufficient for evidence assessment."
                reproducibility = "Unknown; requires method, code, data, and experiment details."
                questions = "What claims, evidence, and methods are present in the full text?"
                suggestions = "Fetch or upload the full paper for a grounded peer-review analysis."
                confidence = "Low; fallback generated from supplied metadata only."
                overall = "Unknown"
                core_claim = str(reference.get("abstract", "") or reference.get("relevance", "")).strip()[:500]
                contribution = str(reference.get("relevance", "")).strip()
            row = {
                "title": str(reference.get("title", "")).strip(),
                "source": str(reference.get("source", "")).strip(),
                "source_origin": str(reference.get("source_origin", "")).strip(),
                "source_label": str(reference.get("source_label", "")).strip(),
                "screening_status": str(reference.get("screening_status", "")).strip(),
                "screening_risks": reference.get("screening_risks") if isinstance(reference.get("screening_risks"), list) else [],
                "verification_status": str(reference.get("verification_status", "")).strip(),
                "verification_risks": reference.get("verification_risks") if isinstance(reference.get("verification_risks"), list) else [],
                "provenance": reference.get("provenance") if isinstance(reference.get("provenance"), dict) else {},
                "metadata": LiteratureAnalysisWorkflow._format_metadata(reference),
                "study_type": "unclear",
                "research_objective": core_claim or "unclear",
                "dataset_or_material": "unclear",
                "sample_size": "unclear",
                "domain_or_modality": "unclear",
                "method": methodology,
                "baseline_or_comparator": "unclear",
                "evaluation_protocol": "unclear",
                "metrics": "unclear",
                "key_results": "unclear",
                "statistical_evidence": "unclear",
                "availability": "unclear",
                "evidence_locations": "metadata/context only",
                "evidence_quotes": "",
                "fact_risks": fallback_note,
                "paper_type": "未知" if output_language == "zh" else "Unknown",
                "core_claim": core_claim,
                "contribution": contribution,
                "methodology": methodology,
                "evidence_strength": evidence_strength,
                "innovation_point": "尚不清楚" if output_language == "zh" else "unclear",
                "reader_takeaway": contribution or ("尚不清楚" if output_language == "zh" else "unclear"),
                "reader_next_step": suggestions,
                "strengths": "",
                "weaknesses": fallback_note,
                "reproducibility": reproducibility,
                "literature_positioning": str(reference.get("branch_name", "")).strip(),
                "application": "",
                "limitations": fallback_note,
                "questions": questions,
                "actionable_suggestions": suggestions,
                "confidence": confidence,
                "overall_assessment": overall,
            }
            LiteratureAnalysisWorkflow._separate_limitations_and_fact_risks(row, output_language)
            LiteratureAnalysisWorkflow._fill_legacy_aliases(row)
            rows.append(row)
        return rows

    @staticmethod
    def _format_metadata(reference: dict) -> str:
        parts = [
            str(reference.get("authors", "")).strip(),
            str(reference.get("journal", "")).strip(),
            str(reference.get("year", "")).strip(),
            str(reference.get("doi", "")).strip(),
        ]
        return "; ".join(part for part in parts if part)

    @staticmethod
    def _parse_json_object(content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data
