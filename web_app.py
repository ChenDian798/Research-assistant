from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
import re
import sys
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.research_agent.citations import format_references, normalize_citation_format
from src.research_agent.doi import enrich_references_with_doi_metadata, extract_arxiv_id, extract_doi, extract_pmid
from src.research_agent.literature_workflow import LiteratureAnalysisWorkflow
from src.research_agent.llm import LLMClient, LLMServiceError
from src.research_agent.novelty_check import NoveltyCheckWorkflow
from src.research_agent.novelty_planner import build_novelty_plan
from src.research_agent.novelty_search import run_novelty_search_plan
from src.research_agent.paper_search import PaperSearchError, search_papers
from src.research_agent.reference_relevance import apply_relevance_gate
from src.research_agent.reference_screening import screen_references
from src.research_agent.reference_verification import verify_references
from src.research_agent.web_analysis_routes import AnalysisRouteService
from src.research_agent.web_history import (
    apply_history_entry_update,
    history_entry_summary,
    history_references,
    read_history_data,
    write_history_data,
)
from src.research_agent import web_document_identity
from src.research_agent import web_pdf_extract
from src.research_agent.web_jobs import load_persisted_job_log, persist_job_log
from src.research_agent.web_pdf_export import markdown_to_pdf_bytes, normalize_pdf_symbols
from src.research_agent.web_response import (
    content_type_for_path,
    frontend_dist_file,
    frontend_index_path,
    send_binary,
    send_file,
    send_json,
)
from src.research_agent.web_runtime import ProcessFileLock, enable_auto_file_logging
from src.research_agent.web_search_routes import SearchRouteService
from src.research_agent.web_uploads import (
    extract_docx_text,
    multipart_fields,
    normalize_extracted_text,
    read_multipart_uploads,
)
from src.research_agent.web_utils import (
    bounded_int,
    markdown_has_wide_table,
    normalize_output_language,
    parse_byte_range,
    split_markdown_table_row,
    truthy,
)


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
ANNOTATION_RECORD_PATH = ROOT / "检索标注记录.md"
HISTORY_PATH = ROOT / "history_records.json"
MAX_PDF_UPLOAD_MB = 30
MAX_PDF_UPLOAD_BYTES = MAX_PDF_UPLOAD_MB * 1024 * 1024
MAX_PDF_UPLOAD_FILES = 4
PDF_EXTRACT_PAGE_LIMIT = 120
PDF_REFERENCE_EXCERPT_CHARS = 10000
CONTEXT_DOCUMENT_EXCERPT_CHARS = 4000
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
HISTORY_LOCK_PATH = ROOT / "history_records.lock"


HISTORY_LOCK = ProcessFileLock(HISTORY_LOCK_PATH)


def _enable_auto_file_logging(port: int | str = "") -> tuple[Path, Path] | None:
    return enable_auto_file_logging(LOG_DIR, port=port)


class ResearchWebHandler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentWeb/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(frontend_index_path(WEB_DIR, WEB_DIST_DIR), "text/html; charset=utf-8")
            return

        if not path.startswith("/api/"):
            dist_file = frontend_dist_file(WEB_DIST_DIR, path)
            if dist_file:
                self._send_file(dist_file, content_type_for_path(dist_file))
                return

        if path == "/styles.css":
            self._send_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return

        if path == "/app.js":
            self._send_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
            return

        if path == "/health":
            self._send_json({"ok": True})
            return

        if path == "/api/history":
            self._send_json({"history": self._history_entries(summary=True)})
            return

        if path.startswith("/api/history/"):
            history_id = path.rsplit("/", 1)[-1]
            entry = self._history_entry(history_id)
            if not entry:
                self._send_json({"error": "History entry not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(entry)
            return

        if path.startswith("/api/literature-search/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        if path.startswith("/api/literature-analysis/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        if path.startswith("/api/novelty-check/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._send_json({"error": "Job not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/export/pdf":
            self._handle_pdf_export()
            return

        if path == "/api/literature-analysis":
            self._handle_literature_analysis()
            return

        if path == "/api/literature-search":
            self._handle_literature_search()
            return

        if path == "/api/novelty-check":
            self._handle_novelty_check()
            return

        if path == "/api/literature-analysis/pdf":
            self._handle_literature_pdf_analysis()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/history/"):
            history_id = path.rsplit("/", 1)[-1]
            if self._delete_history_entry(history_id):
                self._send_json({"ok": True, "history_id": history_id})
                return
            self._send_json({"error": "History entry not found."}, HTTPStatus.NOT_FOUND)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _search_route_service(self) -> SearchRouteService:
        return SearchRouteService(self, JOBS, JOBS_LOCK)

    def _handle_literature_search(self) -> None:
        return self._search_route_service()._handle_literature_search()

    def _run_literature_search_job(self, job_id: str, history_id: str, request: dict) -> None:
        return self._search_route_service()._run_literature_search_job(job_id, history_id, request)

    def _finish_literature_search_job_with_error(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        message: str,
    ) -> None:
        return self._search_route_service()._finish_literature_search_job_with_error(
            job_id,
            history_id,
            port,
            message,
        )

    def _handle_novelty_check(self) -> None:
        return self._search_route_service()._handle_novelty_check()

    def _run_novelty_check_job(self, job_id: str, history_id: str, request: dict) -> None:
        return self._search_route_service()._run_novelty_check_job(job_id, history_id, request)

    def _set_novelty_job(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        *,
        status: str,
        stage: str,
    ) -> None:
        return self._search_route_service()._set_novelty_job(
            job_id,
            history_id,
            port,
            status=status,
            stage=stage,
        )

    def _finish_novelty_check_job_with_error(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        message: str,
    ) -> None:
        return self._search_route_service()._finish_novelty_check_job_with_error(
            job_id,
            history_id,
            port,
            message,
        )

    def _analysis_route_service(self) -> AnalysisRouteService:
        return AnalysisRouteService(self)

    def _handle_literature_analysis(self) -> None:
        return self._analysis_route_service()._handle_literature_analysis()

    def _handle_literature_pdf_analysis(self) -> None:
        return self._analysis_route_service()._handle_literature_pdf_analysis()

    def _handle_pdf_export(self) -> None:
        try:
            payload = self._read_json()
            title = str(payload.get("title", "") or "Research report").strip()
            markdown = str(payload.get("markdown", "") or "").strip()
            if not markdown:
                self._send_json({"error": "Markdown content cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return
            pdf = markdown_to_pdf_bytes(title, markdown)
            filename = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", title).strip("_")[:80] or "research_report"
            send_binary(
                self,
                pdf,
                "application/pdf",
                f'{filename}.pdf',
            )
        except BrokenPipeError:
            print("[web] client disconnected before PDF export response was sent", flush=True)
        except RuntimeError as error:
            self._send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before PDF export error response was sent", flush=True)

    def _run_literature_analysis_job(
        self,
        job_id: str,
        topic: str,
        references: list[dict],
        final_report: str,
        citation_format: str = "APA",
        include_audit: bool = False,
        port: int | str = "",
        history_id: str = "",
        history_slot: str = "result",
        output_language: str = "zh",
    ) -> None:
        return self._analysis_route_service()._run_literature_analysis_job(
            job_id,
            topic,
            references,
            final_report,
            citation_format=citation_format,
            include_audit=include_audit,
            port=port,
            history_id=history_id,
            history_slot=history_slot,
            output_language=output_language,
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}", flush=True)

    @staticmethod
    def _persist_job_log(job_id: str, job: dict, *, port: int | str = "") -> None:
        persist_job_log(LOG_DIR, job_id, job, port=port)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Request body is not valid UTF-8 JSON. If you are uploading a file, "
                "please submit it as PDF or DOCX through the upload control."
            ) from error
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _server_port(self) -> int | str:
        server = getattr(self, "server", None)
        value = getattr(server, "server_port", "")
        if value:
            return value
        return os.getenv("WEB_PORT", "")

    @staticmethod
    def _truthy(value) -> bool:
        return truthy(value)

    @staticmethod
    def _normalize_output_language(value) -> str:
        return normalize_output_language(value)

    @staticmethod
    def _paper_search_enabled() -> bool:
        return ResearchWebHandler._truthy(os.getenv("PAPER_SEARCH_ENABLED", "false"))

    @staticmethod
    def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
        return bounded_int(value, default=default, minimum=minimum, maximum=maximum)

    @staticmethod
    def _split_verified_search_candidates(
        qualified: list[dict],
        needs_review: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        return SearchRouteService._split_verified_search_candidates(qualified, needs_review)

    @staticmethod
    def _dedupe_final_search_candidates(
        qualified: list[dict],
        needs_review: list[dict],
        screened: dict,
    ) -> tuple[list[dict], list[dict]]:
        return SearchRouteService._dedupe_final_search_candidates(qualified, needs_review, screened)

    @staticmethod
    def _apply_final_search_limits(
        qualified: list[dict],
        needs_review: list[dict],
        *,
        requested_sources,
        max_results_per_source: int,
        include_needs_review: bool,
    ) -> tuple[list[dict], list[dict]]:
        return SearchRouteService._apply_final_search_limits(
            qualified,
            needs_review,
            requested_sources=requested_sources,
            max_results_per_source=max_results_per_source,
            include_needs_review=include_needs_review,
        )

    @staticmethod
    def _write_search_audit_log(
        *,
        query: str,
        result: dict,
        screened: dict,
        qualified_references: list[dict],
        needs_review_references: list[dict],
        requested_sources: str,
        year: str,
        port: int | str = "",
        append_annotation_record: bool = True,
    ) -> tuple[Path | None, Path | None]:
        audit = {
            "status": "done",
            "kind": "literature_search_audit",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "port": port,
            "query": result.get("query", query),
            "search_mode": result.get("search_mode", ""),
            "requested_search_mode": result.get("requested_search_mode", ""),
            "mode_inference_status": result.get("mode_inference_status", ""),
            "mode_inference_error": result.get("mode_inference_error", ""),
            "mode_inference_rationale": result.get("mode_inference_rationale", ""),
            "mode_inference_confidence": result.get("mode_inference_confidence", ""),
            "requested_sources": requested_sources,
            "sources_used": result.get("sources_used", []),
            "year": year,
            "backend_query": result.get("backend_query", ""),
            "rules_fallback_query": result.get("rules_fallback_query", ""),
            "llm_search_query": result.get("llm_search_query", ""),
            "llm_pubmed_query": result.get("llm_pubmed_query", ""),
            "llm_error": result.get("llm_error", ""),
            "llm_raw_response": result.get("llm_raw_response", ""),
            "query_rewrite_status": result.get("query_rewrite_status", "rules"),
            "query_plan": result.get("query_plan", {}),
            "queries_by_source": result.get("queries_by_source", {}),
            "source_results": result.get("source_results", {}),
            "internal_source_results": result.get("internal_source_results", {}),
            "channel_results": result.get("channel_results", {}),
            "errors": result.get("errors", {}),
            "raw_count": result.get("raw_count", 0),
            "counts": {
                "qualified": len(qualified_references),
                "needs_review": len(needs_review_references),
                "rejected": len(screened.get("rejected", [])),
            },
            "qualified_references": ResearchWebHandler._audit_references(qualified_references),
            "needs_review_references": ResearchWebHandler._audit_references(needs_review_references),
            "rejected_references": ResearchWebHandler._audit_references(screened.get("rejected", [])),
        }
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            port_part = f"_port{port}" if str(port or "").strip() else ""
            path = LOG_DIR / f"search_audit{port_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            annotation_path = ResearchWebHandler._append_search_annotation_record(audit, path) if append_annotation_record else None
            return path, annotation_path
        except OSError as error:
            print(f"[web] failed to write search audit log: {error}", flush=True)
            return None, None

    @staticmethod
    def _write_novelty_audit_log(
        *,
        innovation_text: str,
        plan: dict,
        search_payload: dict,
        requested_sources: str,
        year: str,
        port: int | str = "",
    ) -> Path | None:
        audit = {
            "status": "done",
            "kind": "novelty_search_audit",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "port": port,
            "innovation_text": innovation_text,
            "requested_sources": requested_sources,
            "year": year,
            "plan": plan,
            "diagnostics": search_payload.get("diagnostics", {}),
            "source_results": search_payload.get("source_results", {}),
            "errors": search_payload.get("errors", {}),
            "raw_count": search_payload.get("raw_count", 0),
            "deduped_count": search_payload.get("deduped_count", 0),
            "strong_candidates": ResearchWebHandler._audit_references(search_payload.get("strong_candidates") or []),
            "weak_candidates": ResearchWebHandler._audit_references(search_payload.get("weak_candidates") or []),
            "source_noise": ResearchWebHandler._audit_references(search_payload.get("source_noise") or []),
        }
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            port_part = f"_port{port}" if str(port or "").strip() else ""
            path = LOG_DIR / f"novelty_audit{port_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            return path
        except OSError as error:
            print(f"[web] failed to write novelty audit log: {error}", flush=True)
            return None

    @staticmethod
    def _append_search_annotation_record(audit: dict, audit_path: Path) -> Path | None:
        try:
            ANNOTATION_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not ANNOTATION_RECORD_PATH.exists():
                ANNOTATION_RECORD_PATH.write_text(
                    "# 检索标注记录\n\n"
                    "说明：本文件由文献检索自动追加候选文献，并给出机器预标注。"
                    "请人工确认或修正“人工判断”“错误类型”“备注”。\n\n"
                    "人工判断建议值：`应收`、`待复核`、`应拒`。\n\n"
                    "错误类型建议值：`query_too_broad`、`query_too_narrow`、`missing_synonym`、`wrong_source`、"
                    "`off_topic_passed`、`relevant_rejected`、`metadata_noise`、`translation_error`、"
                    "`concept_dropped`、`modality_mismatch`、`task_mismatch`、`disease_mismatch`。\n",
                    encoding="utf-8",
                )
            with ANNOTATION_RECORD_PATH.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write("\n\n")
                handle.write(ResearchWebHandler._search_annotation_markdown(audit, audit_path))
            return ANNOTATION_RECORD_PATH
        except OSError as error:
            print(f"[web] failed to write search annotation record: {error}", flush=True)
            return None

    @staticmethod
    def _search_annotation_markdown(audit: dict, audit_path: Path) -> str:
        lines = [
            f"## port{ResearchWebHandler._markdown_inline(audit.get('port'))} {ResearchWebHandler._markdown_inline(audit.get('created_at'))} {ResearchWebHandler._markdown_inline(audit.get('query'))}",
            "",
            f"- port：{ResearchWebHandler._markdown_inline(audit.get('port'))}",
            f"- search_audit：{ResearchWebHandler._markdown_inline(str(audit_path))}",
            f"- 原始中文 query：{ResearchWebHandler._markdown_inline(audit.get('query'))}",
            f"- 规则 fallback query：{ResearchWebHandler._markdown_inline(audit.get('rules_fallback_query'))}",
            f"- LLM search query：{ResearchWebHandler._markdown_inline(audit.get('llm_search_query'))}",
            f"- LLM PubMed query：{ResearchWebHandler._markdown_inline(audit.get('llm_pubmed_query'))}",
            f"- LLM error：{ResearchWebHandler._markdown_inline(audit.get('llm_error'))}",
            f"- LLM raw response：{ResearchWebHandler._markdown_inline(audit.get('llm_raw_response'))}",
            f"- mode inference status：{ResearchWebHandler._markdown_inline(audit.get('mode_inference_status'))}",
            f"- mode inference error：{ResearchWebHandler._markdown_inline(audit.get('mode_inference_error'))}",
            f"- query rewrite status：{ResearchWebHandler._markdown_inline(audit.get('query_rewrite_status'))}",
            f"- sources used：{ResearchWebHandler._markdown_inline(', '.join(audit.get('sources_used') or []))}",
            f"- queries by source：{ResearchWebHandler._markdown_inline(json.dumps(audit.get('queries_by_source', {}), ensure_ascii=False))}",
            f"- counts：{ResearchWebHandler._markdown_inline(json.dumps(audit.get('counts', {}), ensure_ascii=False))}",
            "",
        ]
        include_rejected = ResearchWebHandler._truthy(os.getenv("SEARCH_ANNOTATION_INCLUDE_REJECTED", "false"))
        max_annotations = ResearchWebHandler._bounded_int(
            os.getenv("SEARCH_ANNOTATION_MAX_REFERENCES"),
            default=30,
            minimum=1,
            maximum=500,
        )
        candidates = [
            ("qualified", audit.get("qualified_references") or []),
            ("needs_review", audit.get("needs_review_references") or []),
        ]
        if include_rejected:
            candidates.append(("rejected", audit.get("rejected_references") or []))
        index = 1
        for group, references in candidates:
            for reference in references:
                if index > max_annotations:
                    break
                if not isinstance(reference, dict):
                    continue
                if ResearchWebHandler._is_example_reference(reference):
                    continue
                title = reference.get("title") or "(untitled)"
                audit_summary = reference.get("audit_summary") if isinstance(reference.get("audit_summary"), dict) else {}
                identifier = ResearchWebHandler._reference_identifier(reference)
                suggestion = ResearchWebHandler._annotation_suggestion(group, reference, audit_summary, audit)
                lines.extend(
                    [
                        f"### {index}. [{group}] {ResearchWebHandler._markdown_inline(title)}",
                        "",
                        f"- 标识：{ResearchWebHandler._markdown_inline(identifier)}",
                        f"- 来源：{ResearchWebHandler._markdown_inline(reference.get('source'))}",
                        f"- 系统判断：{ResearchWebHandler._markdown_inline(audit_summary.get('screening_status') or group)}",
                        f"- 相关性：{ResearchWebHandler._markdown_inline(audit_summary.get('topic_relevance_status'))} / {ResearchWebHandler._markdown_inline(audit_summary.get('topic_relevance_score'))}",
                        f"- 命中 concepts/keywords：{ResearchWebHandler._markdown_inline('; '.join(audit_summary.get('matched_concepts_or_keywords') or []))}",
                        f"- 风险/缺失 concepts：{ResearchWebHandler._markdown_inline('; '.join(audit_summary.get('missing_or_risk_concepts') or []))}",
                        f"- 筛选原因：{ResearchWebHandler._markdown_inline('; '.join(audit_summary.get('screening_reasons') or []))}",
                        f"- 验证状态：{ResearchWebHandler._markdown_inline(audit_summary.get('verification_status'))}",
                        f"- 建议人工判断：{ResearchWebHandler._markdown_inline(suggestion.get('judgment'))}",
                        f"- 建议错误类型：{ResearchWebHandler._markdown_inline(suggestion.get('error_type'))}",
                        f"- 建议依据：{ResearchWebHandler._markdown_inline(suggestion.get('reason'))}",
                        f"- 人工判断：{ResearchWebHandler._markdown_inline(suggestion.get('judgment'))}",
                        f"- 错误类型：{ResearchWebHandler._markdown_inline(suggestion.get('error_type'))}",
                        "- 备注：",
                        "",
                    ]
                )
                index += 1
            if index > max_annotations:
                break
        if index == 1:
            lines.extend(["### 无候选文献", "", "- 备注：本次检索没有生成可标注文献。", ""])
        elif not include_rejected and audit.get("rejected_references"):
            lines.extend(
                [
                    "### 未展开的 rejected 文献",
                    "",
                    f"- 数量：{len(audit.get('rejected_references') or [])}",
                    "- 说明：默认不把 rejected 全量写入 Markdown，避免标注量爆炸；完整 rejected 明细在 search_audit JSON 中。",
                    "- 如需展开 rejected，设置环境变量 `SEARCH_ANNOTATION_INCLUDE_REJECTED=true` 后重新检索。",
                    "",
                ]
            )
        if index > max_annotations:
            lines.extend(
                [
                    "### 已截断的候选文献",
                    "",
                    f"- 本次 Markdown 标注最多写入：{max_annotations} 条。",
                    "- 完整候选明细在 search_audit JSON 中。",
                    "- 如需增加数量，设置环境变量 `SEARCH_ANNOTATION_MAX_REFERENCES`。",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _annotation_suggestion(group: str, reference: dict, audit_summary: dict, audit: dict) -> dict:
        status = str(audit_summary.get("screening_status") or group or "").strip().casefold()
        topic_status = str(audit_summary.get("topic_relevance_status") or "").strip().casefold()
        verification_status = str(audit_summary.get("verification_status") or "").strip().casefold()
        risks = [
            str(item)
            for item in [
                *(audit_summary.get("missing_or_risk_concepts") or []),
                *(audit_summary.get("screening_risks") or []),
                *(audit_summary.get("verification_risks") or []),
            ]
            if item
        ]
        reasons = [str(item) for item in (audit_summary.get("screening_reasons") or []) if item]
        risk_text = " ".join([*risks, *reasons]).casefold()

        if status == "rejected" or topic_status == "off_topic" or group == "rejected":
            return {
                "judgment": "应拒",
                "error_type": ResearchWebHandler._suggested_error_type_from_risks(risk_text, default=""),
                "reason": "系统已判为 rejected/off_topic；通常只需抽查是否存在相关文献被误拒。",
            }

        if status == "needs_review" or topic_status == "borderline" or verification_status == "needs_review":
            return {
                "judgment": "待复核",
                "error_type": ResearchWebHandler._suggested_error_type_from_risks(risk_text, default="metadata_noise" if verification_status == "needs_review" else ""),
                "reason": "系统存在边界相关性、元数据冲突或验证风险，需要人工确认。",
            }

        if topic_status == "relevant" and status == "qualified":
            return {
                "judgment": "应收",
                "error_type": "",
                "reason": "系统判为 qualified/relevant，且未发现强制复核信号。",
            }

        return {
            "judgment": "待复核",
            "error_type": ResearchWebHandler._suggested_error_type_from_risks(risk_text, default=""),
            "reason": "系统信号不完整，建议快速人工确认。",
        }

    @staticmethod
    def _suggested_error_type_from_risks(risk_text: str, *, default: str = "") -> str:
        text = str(risk_text or "").casefold()
        if "verification" in text or "metadata" in text or "doi" in text:
            return "metadata_noise"
        if "missing_required" in text or "concept" in text:
            return "concept_dropped"
        if "modality" in text or "ct" in text or "mri" in text or "x-ray" in text:
            return "modality_mismatch"
        if "task" in text or "segmentation" in text or "classification" in text or "diagnosis" in text:
            return "task_mismatch"
        if "condition" in text or "disease" in text or "negative_topic_hint" in text:
            return "disease_mismatch"
        if "low_topic_overlap" in text:
            return "query_too_broad"
        return default

    @staticmethod
    def _is_example_reference(reference: dict) -> bool:
        identifier_text = " ".join(
            str(reference.get(key) or "")
            for key in ("id", "doi", "source", "title")
        ).casefold()
        return "10.1000/example" in identifier_text

    @staticmethod
    def _reference_identifier(reference: dict) -> str:
        for key in ("doi", "pmid", "arxiv_id", "id"):
            value = str(reference.get(key) or "").strip()
            if value:
                return f"{key}:{value}"
        return ""

    @staticmethod
    def _markdown_inline(value) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text.replace("|", "\\|")

    @staticmethod
    def _audit_references(references: list[dict]) -> list[dict]:
        audited = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            public = {
                key: value
                for key, value in reference.items()
                if key not in {"evidence_source_text", "full_text_for_evidence"}
            }
            public["audit_summary"] = {
                "screening_status": reference.get("screening_status", ""),
                "screening_reasons": reference.get("screening_reasons", []),
                "screening_risks": reference.get("screening_risks", []),
                "topic_relevance_status": reference.get("topic_relevance_status", ""),
                "topic_relevance_score": reference.get("topic_relevance_score", ""),
                "matched_concepts_or_keywords": reference.get("topic_relevance_reasons", []),
                "missing_or_risk_concepts": reference.get("topic_relevance_risks", []),
                "verification_status": reference.get("verification_status", ""),
                "verification_risks": reference.get("verification_risks", []),
            }
            audited.append(public)
        return audited

    @staticmethod
    def _pdf_extract_page_limit() -> int | None:
        raw = str(os.getenv("PDF_EXTRACT_PAGE_LIMIT", str(PDF_EXTRACT_PAGE_LIMIT)) or "").strip()
        if raw.casefold() in {"all", "full", "none", "unlimited", "0"}:
            return None
        try:
            return max(1, int(raw))
        except ValueError:
            return PDF_EXTRACT_PAGE_LIMIT

    @staticmethod
    def _pdf_parser_mode() -> str:
        raw = str(os.getenv("PDF_PARSER", "auto") or "").strip().casefold()
        if raw in {"basic", "default", "pypdf", "pymupdf"}:
            return "basic"
        if raw in {"opendataloader", "open-dataloader", "odl", "enhanced"}:
            return "opendataloader"
        return "auto"

    @staticmethod
    def _pdf_opendataloader_min_chars() -> int:
        return ResearchWebHandler._bounded_int(
            os.getenv("PDF_OPENDATALOADER_MIN_CHARS"),
            default=2000,
            minimum=0,
            maximum=100000,
        )

    @staticmethod
    def _pdf_opendataloader_min_page_ratio() -> float:
        raw = str(os.getenv("PDF_OPENDATALOADER_MIN_PAGE_RATIO", "0.5") or "").strip()
        try:
            parsed = float(raw)
        except ValueError:
            parsed = 0.5
        return max(0.0, min(parsed, 1.0))

    @staticmethod
    def _normalize_extracted_text(value: str) -> str:
        return normalize_extracted_text(value)

    def _read_pdf_uploads_with_fields(
        self,
        *,
        allow_empty: bool = False,
    ) -> tuple[list[tuple[str, bytes]], list[dict], dict[str, str]]:
        return read_multipart_uploads(
            headers=self.headers,
            rfile=self.rfile,
            max_upload_bytes=MAX_PDF_UPLOAD_BYTES,
            max_upload_mb=MAX_PDF_UPLOAD_MB,
            allow_empty=allow_empty,
        )

    @staticmethod
    def _multipart_fields(form) -> dict[str, str]:
        return multipart_fields(form)

    def _augment_references_with_llm(
        self,
        references: list[dict],
        raw_text: str,
        *,
        purpose: str,
    ) -> tuple[list[dict], list[dict]]:
        clean_references = [dict(item) for item in references if isinstance(item, dict)]
        if not self._should_use_llm_reference_fallback(raw_text):
            return clean_references, []
        try:
            extracted = asyncio.run(
                self._llm_extract_reference_candidates(
                    raw_text=raw_text,
                    references=clean_references,
                    purpose=purpose,
                )
            )
        except Exception as error:
            print(f"[web] LLM reference fallback skipped: {type(error).__name__}: {error}", flush=True)
            return clean_references, []

        all_items = clean_references + extracted
        return self._split_reference_roles(self._dedupe_reference_candidates(all_items))

    @staticmethod
    def _should_use_llm_reference_fallback(raw_text: str) -> bool:
        text = str(raw_text or "").strip()
        if not text:
            return False
        link_like = re.search(
            r"https?://|www\.|doi\b|10\.\d{4,9}/|arxiv|pmid|pubmed|ncbi|[\w.-]+\.(?:com|org|net|edu|gov|io|cn|uk|de|jp|fr|au|ca)\b",
            text,
            flags=re.IGNORECASE,
        )
        return bool(link_like)

    async def _llm_extract_reference_candidates(
        self,
        *,
        raw_text: str,
        references: list[dict],
        purpose: str,
    ) -> list[dict]:
        system_prompt = """
You extract scholarly reference candidates from messy user input.
Use deterministic parsing as primary; you are only the fallback for ambiguous or missed items.
Return only valid JSON, no markdown.
Schema:
{
  "items": [
    {
      "role": "literature | instructions | background",
      "title": "short title or identifier",
      "source": "URL, DOI URL, PubMed URL, arXiv URL, or empty",
      "identifier_type": "doi | pmid | arxiv | url | unknown",
      "identifier": "normalized DOI/PMID/arXiv id/URL if available",
      "note": "short reason"
    }
  ]
}
Rules:
- Include only items that are present in the user's text. Do not invent papers, URLs, DOIs, PMIDs, arXiv IDs, authors, journals, or titles.
- Treat writing requirements, rubrics, assignment prompts, style rules, and grading criteria as instructions, not literature.
- Prefer official-looking source strings: DOI as https://doi.org/<doi>, PMID as https://pubmed.ncbi.nlm.nih.gov/<pmid>/, arXiv as https://arxiv.org/abs/<id>.
- If an existing parsed reference already covers an item, omit it unless the role should be corrected to instructions/background.
- Return at most 12 items.
""".strip()
        user_prompt = (
            f"Purpose:\n{purpose}\n\n"
            f"Already parsed references:\n{json.dumps(references, ensure_ascii=False, indent=2)[:5000]}\n\n"
            f"Raw user text:\n{raw_text[:8000]}"
        )
        content = await LLMClient().complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1200,
        )
        data = self._parse_llm_json_object(content)
        items = data.get("items", [])
        if not isinstance(items, list):
            return []
        references_out = []
        for item in items[:12]:
            if not isinstance(item, dict):
                continue
            candidate = self._llm_item_to_reference(item)
            if candidate:
                references_out.append(candidate)
        return references_out

    @staticmethod
    def _parse_llm_json_object(content: str) -> dict:
        cleaned = str(content or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _llm_item_to_reference(item: dict) -> dict:
        role = str(item.get("role") or "literature").strip().lower()
        if role not in {"literature", "instructions", "background"}:
            role = "literature"
        identifier_type = str(item.get("identifier_type") or "").strip().lower()
        identifier = str(item.get("identifier") or "").strip()
        source = str(item.get("source") or "").strip()
        title = str(item.get("title") or "").strip()
        note = str(item.get("note") or "").strip()

        if identifier_type == "doi" and identifier:
            identifier = identifier.removeprefix("https://doi.org/").removeprefix("doi:").strip()
            source = f"https://doi.org/{identifier}"
            title = title or f"DOI: {identifier}"
        elif identifier_type == "pmid" and identifier:
            match = re.search(r"\d{6,9}", identifier)
            if match:
                identifier = match.group(0)
                source = f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/"
                title = title or f"PMID: {identifier}"
        elif identifier_type == "arxiv" and identifier:
            identifier = re.sub(r"^(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)", "", identifier, flags=re.IGNORECASE)
            identifier = re.sub(r"\.pdf$", "", identifier.strip(), flags=re.IGNORECASE)
            source = f"https://arxiv.org/abs/{identifier}"
            title = title or f"arXiv: {identifier}"
        elif identifier_type == "url" and identifier and not source:
            source = identifier

        if not source and not title and not note:
            return {}
        title = title or source or "LLM extracted item"
        return {
            "title": title[:500],
            "source": source[:1000],
            "relevance": note or "LLM fallback extracted this item from user-provided link/context.",
            "branch_name": "LLM link fallback",
            "abstract": note,
            "content_excerpt": note,
            "document_role": "literature" if role == "literature" else role,
            "is_literature_source": role == "literature",
            "source_origin": "llm_fallback",
            "source_label": "LLM fallback",
        }

    @staticmethod
    def _dedupe_reference_candidates(references: list[dict]) -> list[dict]:
        deduped = []
        seen = set()
        for reference in references:
            if not isinstance(reference, dict):
                continue
            source = str(reference.get("source") or "").strip().lower().rstrip("/")
            title = str(reference.get("title") or "").strip().casefold()
            key = source or title
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(reference)
        return deduped

    @staticmethod
    def _uploaded_file_to_reference(filename: str, content: bytes, *, expected_context: str = "") -> dict:
        return web_document_identity._uploaded_file_to_reference(
            filename,
            content,
            expected_context=expected_context,
            extract_pdf_content=ResearchWebHandler._extract_pdf_content,
            extract_docx_text=extract_docx_text,
            build_high_information_package=lambda text: LiteratureAnalysisWorkflow._high_information_excerpt(
                text,
                max_chars=PDF_REFERENCE_EXCERPT_CHARS,
            ),
            extract_doi_func=extract_doi,
            extract_arxiv_id_func=extract_arxiv_id,
            extract_pmid_func=extract_pmid,
        )

    @staticmethod
    def _split_reference_roles(references: list[dict]) -> tuple[list[dict], list[dict]]:
        return web_document_identity._split_reference_roles(references)

    @staticmethod
    def _build_uploaded_context(documents: list[dict]) -> str:
        return web_document_identity._build_uploaded_context(
            documents,
            build_context_document_package=lambda text: LiteratureAnalysisWorkflow._high_information_excerpt(
                text,
                max_chars=CONTEXT_DOCUMENT_EXCERPT_CHARS,
            ),
        )

    @staticmethod
    def _extract_pdf_content(content: bytes) -> dict:
        return web_pdf_extract._extract_pdf_content(
            content,
            pdf_parser_mode=ResearchWebHandler._pdf_parser_mode,
            extract_pdf_content_basic=ResearchWebHandler._extract_pdf_content_basic,
            should_try_opendataloader_pdf=lambda extracted: web_pdf_extract._should_try_opendataloader_pdf(
                extracted,
                pdf_opendataloader_min_chars=ResearchWebHandler._pdf_opendataloader_min_chars,
                pdf_opendataloader_min_page_ratio=ResearchWebHandler._pdf_opendataloader_min_page_ratio,
            ),
            extract_pdf_content_with_opendataloader=ResearchWebHandler._extract_pdf_content_with_opendataloader,
        )

    @staticmethod
    def _extract_pdf_content_basic(content: bytes) -> dict:
        return web_pdf_extract._extract_pdf_content_basic(
            content,
            pdf_extract_page_limit=ResearchWebHandler._pdf_extract_page_limit,
            clean_pdf_metadata=ResearchWebHandler._clean_pdf_metadata,
            clean_pdf_page_text=ResearchWebHandler._clean_pdf_page_text,
            extract_pdf_content_with_pymupdf=lambda value: ResearchWebHandler._extract_pdf_content_with_pymupdf(
                value,
                page_limit=ResearchWebHandler._pdf_extract_page_limit(),
            ),
        )

    @staticmethod
    def _extract_pdf_content_with_opendataloader(content: bytes) -> dict | None:
        return web_pdf_extract._extract_pdf_content_with_opendataloader(
            content,
            opendataloader_convert_function=web_pdf_extract._opendataloader_convert_function,
            run_opendataloader_convert=web_pdf_extract._run_opendataloader_convert,
            opendataloader_result_markdown=web_pdf_extract._opendataloader_result_markdown,
            read_opendataloader_markdown=web_pdf_extract._read_opendataloader_markdown,
            opendataloader_result_json=web_pdf_extract._opendataloader_result_json,
            read_opendataloader_json=web_pdf_extract._read_opendataloader_json,
            text_from_opendataloader_json=web_pdf_extract._text_from_opendataloader_json,
            clean_pdf_page_text=ResearchWebHandler._clean_pdf_page_text,
            page_count_from_opendataloader_json=web_pdf_extract._page_count_from_opendataloader_json,
            page_count_from_marked_text=web_pdf_extract._page_count_from_marked_text,
        )

    @staticmethod
    def _clean_pdf_page_text(raw_text: str) -> str:
        return web_pdf_extract._clean_pdf_page_text(
            raw_text,
            normalize_extracted_text=normalize_extracted_text,
        )

    @staticmethod
    def _extract_pdf_content_with_pymupdf(content: bytes, *, page_limit: int | None) -> dict | None:
        return web_pdf_extract._extract_pdf_content_with_pymupdf(
            content,
            page_limit=page_limit,
            clean_pdf_page_text=ResearchWebHandler._clean_pdf_page_text,
        )

    @staticmethod
    def _clean_pdf_metadata(metadata) -> dict:
        return web_pdf_extract._clean_pdf_metadata(
            metadata,
            normalize_extracted_text=normalize_extracted_text,
        )

    @staticmethod
    def _build_pdf_context(references: list[dict]) -> str:
        sections = []
        for reference in references:
            title = reference.get("title", "Uploaded PDF")
            source = reference.get("source", "")
            excerpt = reference.get("content_excerpt") or reference.get("abstract", "")
            extraction_note = reference.get("pdf_extraction_note", "")
            metadata = reference.get("pdf_metadata") or {}
            metadata_lines = []
            if metadata:
                metadata_lines.append("PDF metadata:")
                metadata_lines.extend(
                    f"- {key}: {value}"
                    for key, value in metadata.items()
                    if value
                )
            if extraction_note:
                metadata_lines.append(f"Extraction note: {extraction_note}")
            metadata_block = "\n".join(metadata_lines)
            sections.append(
                f"## {title}\n\n"
                f"Source: {source}\n\n"
                f"{metadata_block}\n\n"
                "Content excerpt for grounded paper review:\n"
                f"{excerpt}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _public_references(references: list[dict]) -> list[dict]:
        internal_keys = {"evidence_source_text", "full_text_for_evidence", "raw_source_record"}
        return [
            {key: value for key, value in reference.items() if key not in internal_keys}
            for reference in references
            if isinstance(reference, dict)
        ]

    @staticmethod
    def _history_entries(limit: int = 200, *, summary: bool = False) -> list[dict]:
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            if ResearchWebHandler._reconcile_history_jobs_unlocked(data):
                ResearchWebHandler._write_history_data_unlocked(data)
            entries = data.get("items", [])
            if not isinstance(entries, list):
                return []
            valid_entries = [entry for entry in entries if isinstance(entry, dict)]
            ordered_entries = sorted(
                enumerate(valid_entries),
                key=lambda item: (str(item[1].get("updated_at") or item[1].get("created_at") or ""), item[0]),
                reverse=True,
            )
            entries = [entry for _, entry in ordered_entries][:limit]
            if summary:
                return [history_entry_summary(entry) for entry in entries]
            return entries

    @staticmethod
    def _history_entry(history_id: str) -> dict | None:
        history_id = str(history_id or "").strip()
        if not history_id:
            return None
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            if ResearchWebHandler._reconcile_history_jobs_unlocked(data):
                ResearchWebHandler._write_history_data_unlocked(data)
            for entry in data.get("items", []):
                if isinstance(entry, dict) and entry.get("id") == history_id:
                    return dict(entry)
        return None

    @staticmethod
    def _delete_history_entry(history_id: str) -> bool:
        history_id = str(history_id or "").strip()
        if not history_id:
            return False
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            items = data.get("items", [])
            if not isinstance(items, list):
                return False
            next_items = [entry for entry in items if not (isinstance(entry, dict) and entry.get("id") == history_id)]
            if len(next_items) == len(items):
                return False
            data["items"] = next_items
            ResearchWebHandler._write_history_data_unlocked(data)
        return True

    @staticmethod
    def _create_history_entry(
        *,
        kind: str,
        source: str,
        title: str,
        status: str,
        request: dict | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        job_id: str = "",
        error: str = "",
    ) -> str:
        history_id = uuid.uuid4().hex
        now = datetime.now().isoformat(timespec="seconds")
        request = request or {}
        counts = counts or {}
        entry = {
            "id": history_id,
            "kind": kind,
            "source": source,
            "title": ResearchWebHandler._history_display_title(
                kind=kind,
                source=source,
                title=title,
                request=request,
                counts=counts,
            ),
            "status": status,
            "created_at": now,
            "updated_at": now,
            "job_id": job_id,
            "request": request,
            "result": result or {},
            "counts": counts,
        }
        if error:
            entry["error"] = error
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            items = data.setdefault("items", [])
            if not isinstance(items, list):
                data["items"] = items = []
            items.append(entry)
            ResearchWebHandler._write_history_data_unlocked(data)
        return history_id

    @staticmethod
    def _history_display_title(
        *,
        kind: str,
        source: str,
        title: str,
        request: dict | None = None,
        counts: dict | None = None,
    ) -> str:
        raw_title = str(title or "").strip()
        if kind not in {"direct_analysis", "search_analysis"}:
            return raw_title[:500] or "Untitled task"

        request = request if isinstance(request, dict) else {}
        counts = counts if isinstance(counts, dict) else {}
        references = request.get("references")
        references = references if isinstance(references, list) else []
        reference_count = ResearchWebHandler._first_positive_int(
            request.get("reference_count"),
            counts.get("references"),
            len(references),
        )
        item_text = f"{reference_count}篇" if reference_count else "资料"
        prefix = "检索流分析" if source == "search" or kind == "search_analysis" else "直接分析"
        subject = ResearchWebHandler._history_subject(raw_title, references)
        return f"{prefix} · {item_text} · {subject}"[:80]

    @staticmethod
    def _first_positive_int(*values: object) -> int:
        for value in values:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
        return 0

    @staticmethod
    def _history_subject(title: str, references: list[dict]) -> str:
        candidates = [str(title or "").strip()]
        for reference in references[:4]:
            if isinstance(reference, dict):
                candidates.append(str(reference.get("title") or reference.get("source") or "").strip())
        combined = " ".join(part for part in candidates if part)
        subject = ResearchWebHandler._known_history_subject(combined)
        if subject:
            return subject

        for candidate in candidates:
            cleaned = ResearchWebHandler._clean_history_subject(candidate)
            if cleaned:
                return cleaned
        return "文献分析"

    @staticmethod
    def _known_history_subject(text: str) -> str:
        normalized = str(text or "").casefold()
        normalized = re.sub(r"\.(?:pdf|docx?)\b", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\b(?:arxiv|pubmed|pmid|doi)\b", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\b10\.\d{4,9}/\S+", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"https?://\S+", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[_\-.]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        checks = [
            (("stroke", "segmentation"), "卒中分割"),
            (("ischemic stroke",), "缺血性卒中"),
            (("medical imaging", "segmentation"), "医学影像分割"),
            (("deep learning",), "深度学习"),
            (("machine learning",), "机器学习"),
            (("ct", "segmentation"), "CT分割"),
            (("mri", "segmentation"), "MRI分割"),
        ]
        for needles, label in checks:
            if all(needle in normalized for needle in needles):
                return label
        return ""

    @staticmethod
    def _clean_history_subject(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\.(?:pdf|docx?)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:arxiv|pubmed|pmid|doi)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b10\.\d{4,9}/\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(^|\s)\d{1,3}[_\-\s]+", " ", cleaned)
        cleaned = re.sub(r"[_\-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ·,;；，。")
        if not cleaned:
            return ""
        sentence = re.split(r"[。！？!?；;\n\r]", cleaned, maxsplit=1)[0].strip()
        if sentence:
            cleaned = sentence
        generic = {
            "current research",
            "literature analysis",
            "literature-analysis",
            "user provided literature links and pdf analysis",
            "user-provided literature links and pdf analysis",
        }
        if cleaned.casefold() in generic:
            return ""
        return cleaned[:28].strip()

    @staticmethod
    def _update_history_entry(
        history_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        error: str | None = None,
    ) -> None:
        history_id = str(history_id or "").strip()
        if not history_id:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            items = data.get("items", [])
            if not isinstance(items, list):
                return
            for entry in items:
                if not isinstance(entry, dict) or entry.get("id") != history_id:
                    continue
                apply_history_entry_update(
                    entry,
                    now,
                    status=status,
                    stage=stage,
                    result=result,
                    counts=counts,
                    error=error,
                )
                ResearchWebHandler._write_history_data_unlocked(data)
                return

    @staticmethod
    def _start_history_analysis(history_id: str, job_id: str, request: dict) -> None:
        return AnalysisRouteService(ResearchWebHandler)._start_history_analysis(history_id, job_id, request)

    @staticmethod
    def _update_analysis_history(
        history_id: str,
        history_slot: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        error: str | None = None,
    ) -> None:
        return AnalysisRouteService(ResearchWebHandler)._update_analysis_history(
            history_id,
            history_slot,
            status=status,
            stage=stage,
            result=result,
            counts=counts,
            error=error,
        )

    @staticmethod
    def _update_history_analysis(
        history_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        error: str | None = None,
    ) -> None:
        return AnalysisRouteService(ResearchWebHandler)._update_history_analysis(
            history_id,
            status=status,
            stage=stage,
            result=result,
            counts=counts,
            error=error,
        )

    @staticmethod
    def _read_history_data_unlocked() -> dict:
        return read_history_data(HISTORY_PATH)

    @staticmethod
    def _write_history_data_unlocked(data: dict) -> None:
        write_history_data(HISTORY_PATH, data)

    @staticmethod
    def _reconcile_history_jobs_unlocked(data: dict) -> bool:
        items = data.get("items", [])
        if not isinstance(items, list):
            return False
        changed = False
        now = datetime.now().isoformat(timespec="seconds")
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "done" and entry.get("stage") == "Task interrupted":
                entry.pop("stage", None)
                entry.pop("error", None)
                entry["updated_at"] = now
                changed = True
            analysis = entry.get("analysis") if isinstance(entry.get("analysis"), dict) else None
            if analysis and analysis.get("status") == "done" and analysis.get("stage") == "Task interrupted":
                analysis.pop("stage", None)
                analysis.pop("error", None)
                entry["analysis"] = analysis
                entry["updated_at"] = now
                changed = True
            if entry.get("status") not in {"queued", "running"}:
                continue
            job_id = str(entry.get("job_id") or "").strip()
            if not job_id:
                continue
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if job:
                changed = ResearchWebHandler._sync_history_entry_from_job(entry, job, now) or changed
                continue
            persisted_job = ResearchWebHandler._load_persisted_job_log(job_id)
            if persisted_job:
                changed = ResearchWebHandler._sync_history_entry_from_job(entry, persisted_job, now) or changed
                continue
            entry["status"] = "error"
            entry["stage"] = "Task interrupted"
            entry["error"] = "任务已中断：后台任务不在当前服务进程中，可能是服务重启或进程退出导致。请重新提交。"
            entry["updated_at"] = now
            changed = True
        return changed

    @staticmethod
    def _sync_history_entry_from_job(entry: dict, job: dict, now: str) -> bool:
        changed = False
        status = str(job.get("status") or "").strip()
        if job.get("history_slot") == "analysis":
            analysis = entry.get("analysis") if isinstance(entry.get("analysis"), dict) else {}
            if status and analysis.get("status") != status:
                analysis["status"] = status
                entry["status"] = status
                changed = True
            stage = job.get("stage")
            if stage and analysis.get("stage") != stage:
                analysis["stage"] = stage
                entry["stage"] = stage
                changed = True
            error = job.get("error")
            if error and analysis.get("error") != error:
                analysis["error"] = error
                entry["error"] = error
                changed = True
            if status == "done":
                result = {
                    "rows": job.get("rows", []),
                    "summary": job.get("summary", {}),
                    "references": job.get("references", []),
                    "citation_format": job.get("citation_format", ""),
                }
                if analysis.get("result") != result:
                    analysis["result"] = result
                    changed = True
                counts = analysis.get("counts") if isinstance(analysis.get("counts"), dict) else {}
                row_count = len(job.get("rows", [])) if isinstance(job.get("rows"), list) else 0
                if counts.get("rows") != row_count:
                    counts["rows"] = row_count
                    analysis["counts"] = counts
                    changed = True
            if changed:
                entry["kind"] = "search_flow"
                entry["analysis"] = analysis
                entry["updated_at"] = now
            return changed
        if status and entry.get("status") != status:
            entry["status"] = status
            changed = True
        stage = job.get("stage")
        if stage and entry.get("stage") != stage:
            entry["stage"] = stage
            changed = True
        error = job.get("error")
        if error and entry.get("error") != error:
            entry["error"] = error
            changed = True
        if status == "done":
            kind = entry.get("kind")
            if kind in {"direct_analysis", "search_analysis"}:
                result = {
                    "rows": job.get("rows", []),
                    "summary": job.get("summary", {}),
                    "references": job.get("references", []),
                    "citation_format": job.get("citation_format", ""),
                }
                if entry.get("result") != result:
                    entry["result"] = result
                    changed = True
                counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else {}
                row_count = len(job.get("rows", [])) if isinstance(job.get("rows"), list) else 0
                if counts.get("rows") != row_count:
                    counts["rows"] = row_count
                    entry["counts"] = counts
                    changed = True
            elif kind in {"literature_search", "search_flow"}:
                if entry.get("result") != job:
                    entry["result"] = job
                    changed = True
            elif kind == "novelty_check":
                result = {
                    "status": job.get("status", ""),
                    "innovation_text": job.get("innovation_text", ""),
                    "innovation_claims": job.get("innovation_claims", []),
                    "overall": job.get("overall", {}),
                    "comparisons": job.get("comparisons", []),
                    "next_steps": job.get("next_steps", []),
                    "counts": job.get("counts", {}),
                    "references": job.get("references", []),
                    "search": job.get("search", {}),
                    "plan": job.get("plan", {}),
                    "diagnostics": job.get("diagnostics", {}),
                    "llm_assessment": job.get("llm_assessment", {}),
                    "closest_prior_work": job.get("closest_prior_work", []),
                    "novelty_dimensions": job.get("novelty_dimensions", {}),
                    "source_results": job.get("source_results", {}),
                    "errors": job.get("errors", {}),
                    "raw_count": job.get("raw_count", 0),
                    "search_audit_log": job.get("search_audit_log", ""),
                }
                if entry.get("result") != result:
                    entry["result"] = result
                    changed = True
                counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else {}
                job_counts = job.get("counts") if isinstance(job.get("counts"), dict) else {}
                if job_counts and counts != job_counts:
                    entry["counts"] = dict(job_counts)
                    changed = True
        if changed:
            entry["updated_at"] = now
        return changed

    @staticmethod
    def _load_persisted_job_log(job_id: str) -> dict | None:
        return load_persisted_job_log(LOG_DIR, job_id)

    @staticmethod
    def _history_references(references: list[dict]) -> list[dict]:
        return history_references(references)

    def _send_file(self, path: Path, content_type: str) -> None:
        send_file(self, path, content_type)

    @staticmethod
    def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
        return parse_byte_range(range_header, file_size)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        send_json(self, payload, status)

    @staticmethod
    def _normalize_pdf_symbols(value: str) -> str:
        return normalize_pdf_symbols(value)

    @staticmethod
    def _markdown_has_wide_table(markdown: str) -> bool:
        return markdown_has_wide_table(markdown)

    @staticmethod
    def _split_markdown_table_row(line: str) -> list[str]:
        return split_markdown_table_row(line)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    env_path = ROOT / ".env"
    example_env_path = ROOT / ".env.example"
    load_dotenv(env_path if env_path.exists() else example_env_path)
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.environ["WEB_PORT"] = str(port)
    log_paths = _enable_auto_file_logging(port)
    if log_paths:
        stdout_path, stderr_path = log_paths
        print(f"Research Agent Web stdout log: {stdout_path}", flush=True)
        print(f"Research Agent Web stderr log: {stderr_path}", flush=True)
    server = ThreadingHTTPServer((host, port), ResearchWebHandler)
    print(f"Research Agent Web is listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
