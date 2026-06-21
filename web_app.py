from __future__ import annotations

import asyncio
import cgi
from datetime import datetime
import hashlib
import html
import io
import json
import mimetypes
import os
import re
import sys
import threading
import traceback
import unicodedata
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse
from xml.etree import ElementTree

from dotenv import load_dotenv

from src.research_agent.citations import format_references, normalize_citation_format
from src.research_agent.doi import enrich_references_with_doi_metadata, extract_arxiv_id, extract_doi, extract_pmid
from src.research_agent.literature_workflow import LiteratureAnalysisWorkflow
from src.research_agent.llm import LLMClient, LLMServiceError
from src.research_agent.paper_search import PaperSearchError, search_papers
from src.research_agent.reference_relevance import apply_relevance_gate
from src.research_agent.reference_screening import screen_references
from src.research_agent.reference_verification import verify_references


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
HISTORY_LOCK = threading.Lock()


class _TeeStream:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "replace")

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def _enable_auto_file_logging(port: int | str = "") -> tuple[Path, Path] | None:
    raw = os.getenv("WEB_AUTO_LOGS", "1").strip().casefold()
    if raw in {"0", "false", "no", "off"}:
        return None
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    port_part = f"_{port}" if str(port or "").strip() else ""
    stdout_path = LOG_DIR / f"web_backend{port_part}_{stamp}.out.log"
    stderr_path = LOG_DIR / f"web_backend{port_part}_{stamp}.err.log"
    stdout_log = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_log = stderr_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, stdout_log)
    sys.stderr = _TeeStream(sys.stderr, stderr_log)
    return stdout_path, stderr_path


class ResearchWebHandler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentWeb/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(self._frontend_index_path(), "text/html; charset=utf-8")
            return

        if not path.startswith("/api/"):
            dist_file = self._frontend_dist_file(path)
            if dist_file:
                self._send_file(dist_file, self._content_type_for_path(dist_file))
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
            self._send_json({"history": self._history_entries()})
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

    def _handle_literature_search(self) -> None:
        try:
            if not self._paper_search_enabled():
                self._send_json(
                    {
                        "error": "Academic search is not enabled. Set PAPER_SEARCH_ENABLED=true and install paper-search-mcp.",
                        "search_enabled": False,
                    },
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            payload = self._read_json()
            request = self._search_request_from_payload(payload)
            if not request["query"]:
                self._send_json({"error": "Search query cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            if self._truthy(payload.get("run_async")):
                job_id = uuid.uuid4().hex
                history_id = self._create_history_entry(
                    kind="search_flow",
                    source="search",
                    title=request["query"],
                    status="queued",
                    job_id=job_id,
                    request=self._search_history_request(request),
                    counts={},
                )
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "status": "queued",
                        "kind": "literature_search",
                        "port": self._server_port(),
                        "history_id": history_id,
                    }
                thread = threading.Thread(
                    target=self._run_literature_search_job,
                    args=(job_id, history_id, request),
                    daemon=True,
                )
                thread.start()
                self._send_json(
                    {
                        "job_id": job_id,
                        "history_id": history_id,
                        "status": "queued",
                        "search_enabled": True,
                    },
                    HTTPStatus.ACCEPTED,
                )
                return

            response_payload, history_payload = self._run_literature_search_pipeline(request)
            history_id = self._create_history_entry(
                kind="search_flow",
                source="search",
                title=response_payload["query"] or request["query"],
                status="done",
                request=self._search_history_request(request),
                result=response_payload,
                counts=history_payload["counts"],
            )
            response_payload["history_id"] = history_id
            self._send_json(response_payload)
        except PaperSearchError as error:
            history_id = self._create_history_entry(
                kind="search_flow",
                source="search",
                title=locals().get("request", {}).get("query") or "Literature search",
                status="error",
                request=self._search_history_request(locals().get("request", {})),
                error=str(error),
            )
            self._send_json(
                {
                    "error": str(error),
                    "search_enabled": True,
                    "qualified_references": [],
                    "needs_review_references": [],
                    "rejected_count": 0,
                    "history_id": history_id,
                },
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            self._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _run_literature_search_job(self, job_id: str, history_id: str, request: dict) -> None:
        port = self._server_port()
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "kind": "literature_search",
                "port": port,
                "stage": "Searching literature...",
                "history_id": history_id,
            }
        self._update_history_entry(history_id, status="running", stage="Searching literature...")

        try:
            response_payload, history_payload = self._run_literature_search_pipeline(request)
            response_payload["history_id"] = history_id
            response_payload["job_id"] = job_id
            done_job = {
                **response_payload,
                "status": "done",
                "kind": "literature_search",
                "port": port,
                "history_id": history_id,
            }
            with JOBS_LOCK:
                JOBS[job_id] = done_job
            self._update_history_entry(
                history_id,
                status="done",
                stage="Search complete",
                result=response_payload,
                counts=history_payload["counts"],
            )
        except PaperSearchError as error:
            self._finish_literature_search_job_with_error(job_id, history_id, port, str(error))
        except Exception as error:
            traceback.print_exc()
            self._finish_literature_search_job_with_error(job_id, history_id, port, f"{type(error).__name__}: {error}")

    def _finish_literature_search_job_with_error(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        message: str,
    ) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "error",
                "kind": "literature_search",
                "port": port,
                "error": message,
                "history_id": history_id,
            }
        self._update_history_entry(history_id, status="error", error=message)

    def _run_literature_search_pipeline(self, request: dict) -> tuple[dict, dict]:
        query = str(request.get("query") or "").strip()
        sources = str(request.get("sources") or "").strip()
        search_mode = str(request.get("search_mode") or "auto").strip().lower() or "auto"
        year = str(request.get("year") or "").strip()
        max_results_per_source = int(request.get("max_results_per_source") or 5)
        timeout_seconds = int(request.get("timeout_seconds") or 45)
        include_needs_review = bool(request.get("include_needs_review", True))
        append_annotation_record = bool(request.get("append_annotation_record", True))

        result = search_papers(
            query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            year=year,
            timeout_seconds=timeout_seconds,
            search_mode=search_mode,
        )
        max_total = self._bounded_int(os.getenv("PAPER_SEARCH_MAX_TOTAL"), default=40, minimum=1, maximum=200)
        screened = screen_references(result.get("papers", [])[:max_total])
        screened = apply_relevance_gate(query, screened, query_plan=result.get("query_plan"))
        verified_qualified = verify_references(screened["qualified"])
        verified_needs_review = verify_references(screened["needs_review"])
        qualified_references, needs_review_references = self._split_verified_search_candidates(
            verified_qualified,
            verified_needs_review,
        )
        qualified_references, needs_review_references = self._dedupe_final_search_candidates(
            qualified_references,
            needs_review_references,
            screened,
        )
        audit_log, annotation_record = self._write_search_audit_log(
            query=query,
            result=result,
            screened=screened,
            qualified_references=qualified_references,
            needs_review_references=needs_review_references,
            requested_sources=sources,
            year=year,
            port=self._server_port(),
            append_annotation_record=append_annotation_record,
        )
        response_payload = {
            "status": "done",
            "port": self._server_port(),
            "query": result.get("query", query),
            "search_mode": result.get("search_mode", search_mode),
            "requested_search_mode": result.get("requested_search_mode", search_mode),
            "backend_query": result.get("backend_query", ""),
            "rules_fallback_query": result.get("rules_fallback_query", ""),
            "llm_search_query": result.get("llm_search_query", ""),
            "llm_pubmed_query": result.get("llm_pubmed_query", ""),
            "llm_error": result.get("llm_error", ""),
            "llm_raw_response": result.get("llm_raw_response", ""),
            "query_rewrite_status": result.get("query_rewrite_status", "rules"),
            "query_plan": result.get("query_plan", {}),
            "queries_by_source": result.get("queries_by_source", {}),
            "sources_used": result.get("sources_used", []),
            "qualified_references": self._public_references(qualified_references),
            "needs_review_references": self._public_references(needs_review_references) if include_needs_review else [],
            "rejected_count": len(screened["rejected"]),
            "rejected_references": self._public_references(screened["rejected"]),
            "source_results": result.get("source_results", {}),
            "errors": result.get("errors", {}),
            "raw_count": result.get("raw_count", 0),
            "search_audit_log": str(audit_log) if audit_log else "",
            "annotation_record": str(annotation_record) if annotation_record else "",
            "annotation_record_enabled": append_annotation_record,
            "search_enabled": True,
        }
        return response_payload, {
            "counts": {
                "qualified": len(qualified_references),
                "needs_review": len(needs_review_references) if include_needs_review else 0,
                "rejected": len(screened["rejected"]),
            }
        }

    def _search_request_from_payload(self, payload: dict) -> dict:
        return {
            "query": str(payload.get("query", "") or "").strip(),
            "sources": str(payload.get("sources") or os.getenv("PAPER_SEARCH_DEFAULT_SOURCES") or "arxiv,pubmed,semantic").strip(),
            "max_results_per_source": self._bounded_int(
                payload.get("max_results_per_source"),
                default=self._bounded_int(os.getenv("PAPER_SEARCH_MAX_RESULTS_PER_SOURCE"), default=5, minimum=1, maximum=50),
                minimum=1,
                maximum=50,
            ),
            "timeout_seconds": self._bounded_int(
                os.getenv("PAPER_SEARCH_TIMEOUT_SECONDS"),
                default=45,
                minimum=1,
                maximum=180,
            ),
            "include_needs_review": self._truthy(payload.get("include_needs_review", True)),
            "append_annotation_record": self._truthy(payload.get("append_annotation_record", True)),
            "year": str(payload.get("year", "") or "").strip(),
            "search_mode": str(payload.get("search_mode") or "auto").strip().lower() or "auto",
        }

    @staticmethod
    def _search_history_request(request: dict) -> dict:
        return {
            "query": str(request.get("query") or "").strip(),
            "sources": str(request.get("sources") or "").strip(),
            "search_mode": str(request.get("search_mode") or "auto").strip().lower() or "auto",
            "year": str(request.get("year") or "").strip(),
            "max_results_per_source": request.get("max_results_per_source") or 5,
            "include_needs_review": bool(request.get("include_needs_review", True)),
        }

    def _handle_literature_analysis(self) -> None:
        try:
            payload = self._read_json()
            references = payload.get("references", [])
            final_report = str(payload.get("final_report", "") or "")
            if not isinstance(references, list):
                self._send_json(
                    {"error": "References must be a list."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            references, context_documents = self._augment_references_with_llm(
                references,
                final_report,
                purpose="literature analysis",
            )
            if context_documents:
                context_block = self._build_uploaded_context(context_documents)
                final_report = f"{final_report}\n\n{context_block}".strip()
            if not references and not final_report.strip():
                self._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            topic = str(payload.get("topic", "") or "current research").strip()
            citation_format = str(payload.get("citation_format", "APA") or "APA").strip()
            include_audit = self._truthy(payload.get("include_audit"))
            history_source = str(payload.get("history_source") or "direct").strip().lower()
            if history_source not in {"direct", "search"}:
                history_source = "direct"
            job_id = uuid.uuid4().hex
            port = self._server_port()
            analysis_history_request = {
                "topic": topic,
                "references": self._history_references(references),
                "reference_count": len(references),
                "has_context": bool(final_report.strip()),
                "citation_format": citation_format,
            }
            existing_history_id = str(payload.get("history_id") or "").strip()
            history_slot = "result"
            if history_source == "search" and existing_history_id and self._history_entry(existing_history_id):
                history_id = existing_history_id
                history_slot = "analysis"
                self._start_history_analysis(history_id, job_id, analysis_history_request)
            else:
                history_id = self._create_history_entry(
                    kind="search_analysis" if history_source == "search" else "direct_analysis",
                    source=history_source,
                    title=topic,
                    status="queued",
                    job_id=job_id,
                    request=analysis_history_request,
                    counts={"references": len(references)},
                )
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "queued",
                    "kind": "literature_analysis",
                    "port": port,
                    "history_id": history_id,
                    "history_slot": history_slot,
                }

            thread = threading.Thread(
                target=self._run_literature_analysis_job,
                args=(job_id, topic, references, final_report, citation_format, include_audit, port, history_id, history_slot),
                daemon=True,
            )
            thread.start()
            self._send_json({"job_id": job_id, "history_id": history_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before literature response was sent", flush=True)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before literature error response was sent", flush=True)

    def _handle_literature_pdf_analysis(self) -> None:
        try:
            files, link_references, fields = self._read_pdf_uploads_with_fields(allow_empty=True)
            if len(files) > MAX_PDF_UPLOAD_FILES:
                self._send_json(
                    {
                        "error": (
                            f"文献分析每次最多上传 {MAX_PDF_UPLOAD_FILES} 个 PDF/DOCX 文件。"
                            "请清空后重新选择，或分批上传分析。"
                        )
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            references = list(link_references)
            user_context = fields.get("user_context", "").strip()
            topic = str(fields.get("topic", "") or "").strip() or "user-provided literature links and PDF analysis"
            # The default topic for file-only analysis is generated from uploaded
            # filenames. Do not run reference extraction over that synthetic text:
            # names like "*_arxiv.pdf" can otherwise be misread as an extra paper.
            llm_text = user_context if files else "\n\n".join(part for part in [topic, user_context] if part)
            references, llm_context_documents = self._augment_references_with_llm(
                references,
                llm_text,
                purpose="literature analysis upload",
            )
            expected_context = "\n\n".join(part for part in [topic, user_context] if part)
            uploaded_references = [
                self._uploaded_file_to_reference(
                    filename,
                    content,
                    expected_context=expected_context,
                )
                for filename, content in files
            ]
            references, context_documents = self._split_reference_roles(
                references + uploaded_references + llm_context_documents
            )
            review_needed_documents = [
                document
                for document in context_documents
                if str(document.get("document_role") or "").strip().lower() == "review_needed"
            ]
            if not references and not user_context and not context_documents:
                self._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            final_report = self._build_pdf_context(references)
            if context_documents:
                context_block = self._build_uploaded_context(context_documents)
                final_report = f"{final_report}\n\n{context_block}".strip()
            if user_context:
                final_report = (
                    f"{final_report}\n\n" if final_report else ""
                ) + f"User-provided text context or instructions:\n{user_context}"
            citation_format = str(fields.get("citation_format", "APA") or "APA").strip()
            include_audit = self._truthy(fields.get("include_audit"))
            history_source = str(fields.get("history_source") or "direct").strip().lower()
            if history_source not in {"direct", "search"}:
                history_source = "direct"
            job_id = uuid.uuid4().hex
            port = self._server_port()
            history_id = self._create_history_entry(
                kind="search_analysis" if history_source == "search" else "direct_analysis",
                source=history_source,
                title=topic,
                status="queued",
                job_id=job_id,
                request={
                    "topic": topic,
                    "references": self._history_references(references),
                    "review_needed_documents": self._history_references(review_needed_documents),
                    "reference_count": len(references),
                    "file_count": len(files),
                    "has_context": bool(user_context.strip() or context_documents),
                    "citation_format": citation_format,
                },
                counts={
                    "references": len(references),
                    "files": len(files),
                    "review_needed": len(review_needed_documents),
                },
            )
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "queued",
                    "kind": "literature_analysis",
                    "port": port,
                    "history_id": history_id,
                }

            thread = threading.Thread(
                target=self._run_literature_analysis_job,
                args=(job_id, topic, references, final_report, citation_format, include_audit, port, history_id),
                daemon=True,
            )
            thread.start()
            self._send_json(
                {
                    "job_id": job_id,
                    "history_id": history_id,
                    "status": "queued",
                    "references": self._public_references(references),
                    "review_needed_documents": self._public_references(review_needed_documents),
                },
                HTTPStatus.ACCEPTED,
            )
        except BrokenPipeError:
            print("[web] client disconnected before document literature response was sent", flush=True)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            try:
                self._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before document literature error response was sent", flush=True)

    def _handle_pdf_export(self) -> None:
        try:
            payload = self._read_json()
            title = str(payload.get("title", "") or "Research report").strip()
            markdown = str(payload.get("markdown", "") or "").strip()
            if not markdown:
                self._send_json({"error": "Markdown content cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return
            pdf = self._markdown_to_pdf_bytes(title, markdown)
            filename = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", title).strip("_")[:80] or "research_report"
            self._send_binary(
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
    ) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "kind": "literature_analysis",
                "port": port,
                "stage": "Starting literature analysis...",
                "history_id": history_id,
                "history_slot": history_slot,
            }
        self._update_analysis_history(history_id, history_slot, status="running", stage="Starting literature analysis...")

        try:
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "running",
                    "kind": "literature_analysis",
                    "port": port,
                    "stage": "Resolving DOI metadata...",
                    "history_id": history_id,
                    "history_slot": history_slot,
                }
            self._update_analysis_history(history_id, history_slot, status="running", stage="Resolving DOI metadata...")
            references = enrich_references_with_doi_metadata(references)
            citation_format = normalize_citation_format(citation_format)
            formatted_references = format_references(references, citation_format)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "running",
                    "kind": "literature_analysis",
                    "port": port,
                    "stage": "Running LLM literature analysis...",
                    "history_id": history_id,
                    "history_slot": history_slot,
                }
            self._update_analysis_history(history_id, history_slot, status="running", stage="Running LLM literature analysis...")
            analysis_result = asyncio.run(
                LiteratureAnalysisWorkflow(verbose=True).run(
                    topic=topic,
                    references=references,
                    final_report=final_report,
                    citation_format=citation_format,
                    formatted_references=formatted_references,
                )
            )
            if isinstance(analysis_result, dict):
                rows = analysis_result.get("rows", [])
                summary = analysis_result.get("summary", {})
                audit_summary = analysis_result.get("audit_summary", {})
            else:
                rows = analysis_result
                summary = {}
                audit_summary = {}
            if isinstance(summary, dict):
                summary.setdefault("references", formatted_references)
                summary.setdefault("citation_format", citation_format)
            done_job = {
                "status": "done",
                "kind": "literature_analysis",
                "port": port,
                "rows": rows,
                "summary": summary,
                "references": formatted_references,
                "citation_format": citation_format,
                "history_id": history_id,
                "history_slot": history_slot,
            }
            if include_audit and isinstance(audit_summary, dict):
                done_job["audit_summary"] = audit_summary
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                port_part = f"_port{port}" if str(port or "").strip() else ""
                (LOG_DIR / f"last_job{port_part}_{job_id}.json").write_text(
                    json.dumps(done_job, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as error:
                print(f"[web] failed to write last job log: {error}", flush=True)
            with JOBS_LOCK:
                JOBS[job_id] = done_job
            if history_slot == "analysis":
                self._update_history_analysis(
                    history_id,
                    status="done",
                    stage="Analysis complete",
                    result={
                        "rows": rows,
                        "summary": summary,
                        "references": formatted_references,
                        "citation_format": citation_format,
                    },
                    counts={"rows": len(rows) if isinstance(rows, list) else 0},
                )
            else:
                self._update_history_entry(
                    history_id,
                    status="done",
                    stage="Analysis complete",
                    result={
                        "rows": rows,
                        "summary": summary,
                        "references": formatted_references,
                        "citation_format": citation_format,
                    },
                    counts={"rows": len(rows) if isinstance(rows, list) else 0},
                )
        except LLMServiceError as error:
            print(f"[web] literature LLM service error: {error}", flush=True)
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "literature_analysis",
                    "port": port,
                    "error": str(error),
                    "history_id": history_id,
                    "history_slot": history_slot,
                }
            self._update_analysis_history(history_id, history_slot, status="error", error=str(error))
        except Exception as error:
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "kind": "literature_analysis",
                    "port": port,
                    "error": f"{type(error).__name__}: {error}",
                    "history_id": history_id,
                    "history_slot": history_slot,
                }
            self._update_analysis_history(history_id, history_slot, status="error", error=f"{type(error).__name__}: {error}")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}", flush=True)

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
        if isinstance(value, bool):
            return value
        return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _paper_search_enabled() -> bool:
        return ResearchWebHandler._truthy(os.getenv("PAPER_SEARCH_ENABLED", "false"))

    @staticmethod
    def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _split_verified_search_candidates(
        qualified: list[dict],
        needs_review: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        final_qualified = []
        final_needs_review = list(needs_review)
        for reference in qualified:
            if reference.get("verification_status") == "needs_review":
                reference["screening_status"] = "needs_review"
                risks = list(reference.get("screening_risks") or [])
                risks.append("verification_metadata_conflict")
                reference["screening_risks"] = list(dict.fromkeys(risks))
                final_needs_review.append(reference)
            else:
                final_qualified.append(reference)
        return final_qualified, final_needs_review

    @staticmethod
    def _dedupe_final_search_candidates(
        qualified: list[dict],
        needs_review: list[dict],
        screened: dict,
    ) -> tuple[list[dict], list[dict]]:
        seen = set()
        final_qualified, duplicate_qualified = ResearchWebHandler._dedupe_reference_list_for_output(
            qualified,
            seen,
        )
        final_needs_review, duplicate_needs_review = ResearchWebHandler._dedupe_reference_list_for_output(
            needs_review,
            seen,
        )
        duplicates = duplicate_qualified + duplicate_needs_review
        if duplicates:
            rejected = screened.setdefault("rejected", [])
            rejected.extend(duplicates)
        return final_qualified, final_needs_review

    @staticmethod
    def _dedupe_reference_list_for_output(
        references: list[dict],
        seen: set[str],
    ) -> tuple[list[dict], list[dict]]:
        kept = []
        duplicates = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            keys = ResearchWebHandler._final_output_reference_keys(reference)
            duplicate_key = next((key for key in keys if key in seen), "")
            if duplicate_key:
                duplicate = dict(reference)
                duplicate["screening_status"] = "rejected"
                duplicate["duplicate_key"] = duplicate_key
                reasons = list(duplicate.get("screening_reasons") or [])
                risks = list(duplicate.get("screening_risks") or [])
                reasons.append("duplicate")
                risks.append("duplicate_final_output")
                duplicate["screening_reasons"] = list(dict.fromkeys(reason for reason in reasons if reason))
                duplicate["screening_risks"] = list(dict.fromkeys(risk for risk in risks if risk))
                duplicates.append(duplicate)
                continue
            seen.update(keys)
            kept.append(reference)
        return kept, duplicates

    @staticmethod
    def _final_output_reference_keys(reference: dict) -> list[str]:
        keys = []
        for field in ("doi", "pmid", "arxiv_id", "source", "id"):
            value = str(reference.get(field) or "").strip().rstrip("/")
            if value:
                keys.append(f"{field}:{value.casefold()}")
        title_key = ResearchWebHandler._final_output_title_key(reference.get("title"))
        if title_key:
            keys.append(f"title:{title_key}")
        return list(dict.fromkeys(keys))

    @staticmethod
    def _final_output_title_key(title) -> str:
        text = unicodedata.normalize("NFKC", str(title or "")).casefold()
        text = re.sub(r"\W+", " ", text).strip()
        if len(text) < 24 or len(text.split()) < 4:
            return ""
        return text

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
    def _normalize_extracted_text(value: str) -> str:
        text = ResearchWebHandler._repair_mojibake(str(value or ""))
        text = unicodedata.normalize("NFKC", text)
        text = ResearchWebHandler._normalize_pdf_text_symbols(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = text.replace("\u00ad", "")
        lines = [
            re.sub(r"[ \t\f\v]+", " ", line).strip()
            for line in text.splitlines()
        ]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line)).strip()

    @staticmethod
    def _repair_mojibake(value: str) -> str:
        text = str(value or "")
        if not text:
            return ""

        def badness(candidate: str) -> int:
            private_use = sum(1 for char in candidate if "\ue000" <= char <= "\uf8ff")
            controls = sum(1 for char in candidate if "\x80" <= char <= "\x9f")
            replacement = candidate.count("\ufffd")
            latin_mojibake = len(re.findall(r"[\u00c2\u00c3\u00e2][\u0080-\u00ff]?", candidate))
            return replacement * 8 + private_use * 5 + controls * 3 + latin_mojibake * 2

        best = text
        best_score = badness(text)
        for encoding in ("latin-1", "cp1252", "gb18030"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            score = badness(repaired)
            if score < best_score:
                best = repaired
                best_score = score
        return best

    @staticmethod
    def _extract_docx_text(content: bytes) -> str:
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as error:
            raise ValueError("DOCX file is invalid or corrupted.") from error

        parts = []
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        xml_names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or name.startswith("word/header")
            or name.startswith("word/footer")
        ]
        for xml_name in xml_names:
            try:
                root = ElementTree.fromstring(archive.read(xml_name))
            except (ElementTree.ParseError, UnicodeDecodeError):
                continue
            for paragraph in root.findall(".//w:p", namespaces):
                runs = [
                    node.text or ""
                    for node in paragraph.findall(".//w:t", namespaces)
                    if node.text
                ]
                line = "".join(runs).strip()
                if line:
                    parts.append(line)
        return ResearchWebHandler._normalize_extracted_text("\n".join(parts))

    def _read_pdf_uploads(self) -> tuple[list[tuple[str, bytes]], list[dict]]:
        files, references, _ = self._read_pdf_uploads_with_fields()
        return files, references

    def _read_pdf_uploads_with_fields(
        self,
        *,
        allow_empty: bool = False,
    ) -> tuple[list[tuple[str, bytes]], list[dict], dict[str, str]]:
        content_type = self.headers.get("Content-Type", "")
        media_type, _ = cgi.parse_header(content_type)
        if media_type != "multipart/form-data":
            raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Upload body cannot be empty.")
        if content_length > MAX_PDF_UPLOAD_BYTES:
            raise ValueError(f"Upload is too large. Please keep it under {MAX_PDF_UPLOAD_MB} MB. Try uploading fewer files at a time.")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
        )
        items = form["pdf"] if "pdf" in form else []
        if not isinstance(items, list):
            items = [items]

        references = self._multipart_references(form)
        fields = self._multipart_fields(form)
        files: list[tuple[str, bytes]] = []
        for item in items:
            filename = Path(item.filename or "uploaded-document").name
            suffix = Path(filename).suffix.lower()
            if suffix == ".doc":
                raise ValueError(
                    f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
                )
            if suffix not in {".pdf", ".docx"}:
                raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")
            content = item.file.read()
            if not content:
                continue
            files.append((filename, content))

        if not allow_empty and not files and not references:
            raise ValueError("Please upload at least one non-empty PDF/DOCX file or provide references.")
        return files, references, fields

    @staticmethod
    def _multipart_references(form: cgi.FieldStorage) -> list[dict]:
        raw = form.getvalue("references", "[]")
        if isinstance(raw, list):
            raw = raw[0] if raw else "[]"
        try:
            data = json.loads(str(raw or "[]"))
        except json.JSONDecodeError as error:
            raise ValueError("References field must be valid JSON.") from error
        if not isinstance(data, list):
            raise ValueError("References field must be a JSON list.")
        return [dict(item) for item in data if isinstance(item, dict) and str(item.get("title", "")).strip()]

    @staticmethod
    def _multipart_fields(form: cgi.FieldStorage) -> dict[str, str]:
        fields: dict[str, str] = {}
        for key in ["topic", "max_results", "category", "start_date", "end_date", "citation_format", "source_mode", "user_context"]:
            value = form.getvalue(key, "")
            if isinstance(value, list):
                value = value[0] if value else ""
            fields[key] = str(value or "")
        return fields

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
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            extracted = ResearchWebHandler._extract_pdf_content(content)
            text = extracted["text"]
            has_text = bool(text)
            if not has_text:
                text = "未能从 PDF 中提取到可读文本，可能是扫描件或受保护文档。"
            excerpt = ResearchWebHandler._build_high_information_package(text)
            metadata = extracted["metadata"]
            bibliographic = ResearchWebHandler._infer_pdf_bibliographic_metadata(
                filename=filename,
                text=text,
                metadata=metadata,
            )
            bibliographic_identity = ResearchWebHandler._build_pdf_bibliographic_identity(
                filename=filename,
                text=text if has_text else "",
                metadata=metadata,
            )
            identity_gate = ResearchWebHandler._pdf_identity_review_gate(
                filename=filename,
                title=bibliographic.get("title") or Path(filename).stem or filename,
                text=text if has_text else "",
                metadata=metadata,
                expected_context=expected_context,
            )
            reference = {
                "title": bibliographic.get("title") or Path(filename).stem or filename,
                "source": filename,
                "uploaded_filename": filename,
                "relevance": (
                    "用户上传 PDF，系统已提取正文片段用于分析。"
                    if has_text
                    else "用户上传 PDF，但系统未提取到可读正文；需基于元数据谨慎分析。"
                ),
                "branch_name": "文件上传",
                "abstract": excerpt,
                "content_excerpt": excerpt,
                "evidence_source_text": text if has_text else "",
                "full_text_for_evidence": text if has_text else "",
                "pdf_text_available": has_text,
                "pdf_page_count": extracted["page_count"],
                "pdf_extracted_pages": extracted["extracted_pages"],
                "pdf_extraction_note": extracted["note"],
                "pdf_content_sha256": hashlib.sha256(content).hexdigest(),
                "authors": bibliographic.get("authors", []),
                "year": bibliographic.get("year", ""),
                "journal": bibliographic.get("journal", ""),
                "bibliographic_identity": bibliographic_identity,
                "pdf_metadata": metadata,
                "document_type": "PDF",
                "source_origin": "user_upload",
                "source_label": "Uploaded PDF",
            }
            if identity_gate:
                reference.update(identity_gate)
                reference["relevance"] = identity_gate["review_note"]
                reference["branch_name"] = "Uploaded PDF needs review"
                reference["abstract"] = (
                    f"{identity_gate['review_note']}\n\nExtracted text preview:\n"
                    f"{ResearchWebHandler._debug_preview(text, 1200)}"
                ).strip()
                reference["content_excerpt"] = reference["abstract"]
                reference["evidence_source_text"] = ""
                reference["full_text_for_evidence"] = ""
            ResearchWebHandler._log_uploaded_pdf_reference_debug(
                filename=filename,
                content=content,
                extracted=extracted,
                bibliographic=bibliographic,
                bibliographic_identity=bibliographic_identity,
                text=text if has_text else "",
                reference=reference,
            )
            doi = extract_doi(reference)
            arxiv_id = extract_arxiv_id(reference)
            pmid = extract_pmid(reference)
            if doi:
                reference["doi"] = doi
            if arxiv_id:
                reference["arxiv_id"] = arxiv_id
            if pmid:
                reference["pmid"] = pmid
            return reference

        if suffix == ".docx":
            text = ResearchWebHandler._extract_docx_text(content)
            has_text = bool(text)
            if not has_text:
                text = "未能从 DOCX 中提取到可读文本。"
            excerpt = ResearchWebHandler._build_high_information_package(text)
            return {
                "title": Path(filename).stem or filename,
                "source": filename,
                "uploaded_filename": filename,
                "relevance": (
                    "用户上传 DOCX，系统已提取正文片段用于分析。"
                    if has_text
                    else "用户上传 DOCX，但系统未提取到可读正文；需谨慎分析。"
                ),
                "branch_name": "文件上传",
                "abstract": excerpt,
                "content_excerpt": excerpt,
                "evidence_source_text": text if has_text else "",
                "full_text_for_evidence": text if has_text else "",
                "pdf_text_available": has_text,
                "pdf_page_count": 0,
                "pdf_extracted_pages": 0,
                "pdf_extraction_note": "Extracted text from DOCX." if has_text else "No readable text extracted from DOCX.",
                "authors": "",
                "year": "",
                "journal": "",
                "pdf_metadata": {},
                "document_type": "DOCX",
                "source_origin": "user_upload",
                "source_label": "Uploaded DOCX",
            }

        if suffix == ".doc":
            raise ValueError(
                f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
            )
        raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")

    @staticmethod
    def _pdf_identity_review_gate(
        *,
        filename: str,
        title: str,
        text: str,
        metadata: dict | None = None,
        expected_context: str = "",
    ) -> dict:
        metadata = metadata or {}
        identity_text = "\n".join(
            [
                filename,
                title,
                str(metadata.get("title", "") or ""),
                str(metadata.get("subject", "") or ""),
            ]
        ).casefold()
        body = str(text or "")[:12000].casefold()
        compact_body = re.sub(r"\s+", " ", body)
        expected = str(expected_context or "").casefold()

        production_title = bool(
            re.search(r"\b(?:indd|v\d{2,}|proof|galley|layout|typeset)\b", identity_text)
            or re.search(r"\b\d{3,}\s*-\s*v\d+\b", identity_text)
        )
        metadata_title = str(metadata.get("title", "") or "").strip().casefold()
        selected_title = str(title or "").strip()
        selected_title_is_stable = bool(
            selected_title
            and ResearchWebHandler._clean_bibliographic_title(selected_title)
            and not ResearchWebHandler._looks_like_pdf_body_sentence(selected_title)
            and selected_title.casefold() != metadata_title
            and selected_title.casefold() in compact_body
        )
        fossil_topic_hits = sum(
            1
            for term in [
                "ams",
                "14c",
                "amino acid",
                "enantiomer",
                "fossil",
                "indigeneity",
                "racemization",
                "diagenesis",
                "stable isotope",
                "letters to nature",
            ]
            if term in body
        )
        ml_expected_hits = sum(
            1
            for term in [
                "machine learning",
                "deep learning",
                "neural",
                "random forest",
                "support vector",
                "classification",
                "classifier",
                "regression",
                "dataset",
                "training",
            ]
            if term in expected
        )
        ml_body_hits = sum(
            1
            for term in [
                "machine learning",
                "deep learning",
                "neural network",
                "random forest",
                "support-vector",
                "support vector",
                "classifier",
                "training set",
                "test set",
            ]
            if term in body
        )

        reasons = []
        context_conflict = fossil_topic_hits >= 3 and ml_expected_hits >= 1
        mixed_body_topics = fossil_topic_hits >= 3 and ml_body_hits >= 1 and not selected_title_is_stable

        if production_title:
            reasons.append("PDF title/metadata looks like a production or layout filename, not a stable paper title")
        if mixed_body_topics:
            reasons.append("extracted text mixes fossil/AMS/amino-acid content with machine-learning text inside the same PDF")
        if production_title and context_conflict and not selected_title_is_stable:
            reasons.append("first-page evidence suggests an unrelated fossil/AMS article under an unstable title")
        if selected_title_is_stable and reasons == ["PDF title/metadata looks like a production or layout filename, not a stable paper title"]:
            return {}
        if context_conflict and not (production_title or mixed_body_topics):
            return {}
        if not reasons:
            return {}

        note = (
            "待复核材料：PDF 文本与用户预期主题或文献身份不一致，请确认上传文件、拼接页、隐藏文本/OCR 层或 PDF 元数据。"
        )
        return {
            "document_role": "review_needed",
            "is_literature_source": False,
            "pdf_identity_status": "needs_review",
            "review_note": note,
            "review_reasons": reasons,
        }

    @staticmethod
    def _debug_preview(value: str, limit: int = 800) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:limit]

    @staticmethod
    def _log_uploaded_pdf_reference_debug(
        *,
        filename: str,
        content: bytes,
        extracted: dict,
        bibliographic: dict,
        bibliographic_identity: str,
        text: str,
        reference: dict,
    ) -> None:
        debug_payload = {
            "filename": filename,
            "content_length": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "pdf_page_count": extracted.get("page_count"),
            "pdf_extracted_pages": extracted.get("extracted_pages"),
            "pdf_extraction_note": extracted.get("note"),
            "metadata_title": (extracted.get("metadata") or {}).get("title", ""),
            "metadata_author": (extracted.get("metadata") or {}).get("author", ""),
            "inferred_title": bibliographic.get("title", ""),
            "document_role": reference.get("document_role", "literature"),
            "pdf_identity_status": reference.get("pdf_identity_status", "ok"),
            "review_reasons": reference.get("review_reasons", []),
            "first_800_chars": ResearchWebHandler._debug_preview(text, 800),
            "bibliographic_identity_first_800_chars": ResearchWebHandler._debug_preview(
                bibliographic_identity,
                800,
            ),
        }
        print(
            "[web] uploaded PDF reference debug "
            + json.dumps(debug_payload, ensure_ascii=False),
            flush=True,
        )

    @staticmethod
    def _infer_pdf_bibliographic_metadata(filename: str, text: str, metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        metadata_title = ResearchWebHandler._clean_bibliographic_title(str(metadata.get("title", "") or ""))
        metadata_author = ResearchWebHandler._clean_pdf_author_string(str(metadata.get("author", "") or ""))
        first_page = ResearchWebHandler._first_pdf_page_text(text)
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in first_page.splitlines()
            if re.sub(r"\s+", " ", line).strip()
        ][:80]

        title = metadata_title
        if not title:
            title = ResearchWebHandler._infer_title_from_pdf_lines(lines)

        authors = ResearchWebHandler._split_pdf_authors(metadata_author)
        if not authors:
            authors = ResearchWebHandler._infer_authors_from_pdf_lines(lines, title)

        year = ""
        for source in [first_page, str(metadata.get("subject", "") or ""), filename]:
            match = re.search(r"\b(19|20)\d{2}\b", source)
            if match:
                year = match.group(0)
                break

        return {
            "title": title or Path(filename).stem or filename,
            "authors": authors,
            "year": year,
            "journal": "",
        }

    @staticmethod
    def _first_pdf_page_text(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"\[Page\s+1\]\s*(.*?)(?=\n\n\[Page\s+2\]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text[:5000]

    @staticmethod
    def _build_pdf_bibliographic_identity(filename: str, text: str, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        first_page = ResearchWebHandler._first_pdf_page_text(text)
        first_page = re.split(r"\b(?:references|bibliography|works cited)\b", first_page, maxsplit=1, flags=re.IGNORECASE)[0]
        parts = [
            filename,
            str(metadata.get("title", "") or ""),
            str(metadata.get("author", "") or ""),
            str(metadata.get("subject", "") or ""),
            first_page[:4000],
        ]
        return "\n".join(part for part in parts if str(part).strip())

    @staticmethod
    def _clean_bibliographic_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title or "").strip()
        if not title:
            return ""
        if title.lower() in {"untitled", "document", "paper"}:
            return ""
        if re.fullmatch(r"\[?\s*page\s+\d+\s*\]?", title, flags=re.IGNORECASE):
            return ""
        if re.fullmatch(r"[\w\s.-]+\.(?:pdf|docx?|indd|qxd)", title, flags=re.IGNORECASE):
            return ""
        if re.search(r"\b(?:indd|proof|galley|layout|typeset)\b", title, flags=re.IGNORECASE):
            return ""
        if re.fullmatch(r"(?:[a-f0-9]{8,}|\d{3,}|[\d\s_-]*v\d+[\d\s_-]*)", title, flags=re.IGNORECASE):
            return ""
        if re.search(r"\b(?:vol\.?|volume)\s*\d+|\b(?:letters\s+to\s+nature|nature\s+vol)\b", title, flags=re.IGNORECASE):
            return ""
        if re.fullmatch(r"\d{3,}\s*-\s*v\d+", title, flags=re.IGNORECASE):
            return ""
        if re.search(r"^microsoft word|^latex|^overleaf", title, flags=re.IGNORECASE):
            return ""
        return title[:300]

    @staticmethod
    def _clean_pdf_author_string(author: str) -> str:
        author = re.sub(r"\s+", " ", author or "").strip()
        if re.search(r"^(unknown|anonymous|uploaded|user|pdf)$", author, flags=re.IGNORECASE):
            return ""
        return author

    @staticmethod
    def _infer_title_from_pdf_lines(lines: list[str]) -> str:
        best = ResearchWebHandler._best_scored_title_from_pdf_lines(lines)
        if best:
            return best

        skip = re.compile(
            r"^(?:arxiv|abstract|keywords?|introduction|preprint|submitted|accepted|published|doi\b|"
            r"journal|conference|proceedings|\[?\s*page\s+\d+\s*\]?|\d+|"
            r"letters\s+to\s+nature|nature\s+vol|medical imaging with deep learning|midl\b|short paper\b)",
            flags=re.IGNORECASE,
        )
        candidates = []
        for line in lines[:16]:
            clean = ResearchWebHandler._normalize_pdf_text_symbols(line).strip(" -")
            if not clean or skip.search(clean):
                continue
            if not ResearchWebHandler._clean_bibliographic_title(clean):
                continue
            if ResearchWebHandler._looks_like_pdf_body_sentence(clean):
                continue
            if re.search(r"@|www\.|https?://", clean, flags=re.IGNORECASE):
                break
            if re.search(r"\b(?:abstract|university|department|institute|hospital|school|center|centre|corresponding author|word count|short title)\b", clean, flags=re.IGNORECASE):
                break
            if ResearchWebHandler._looks_like_pdf_author_line(clean):
                break
            if len(clean) < 3 and not candidates:
                continue
            if len(clean) > 180 and not candidates:
                return ResearchWebHandler._trim_pdf_title_noise(clean)
            candidates.append(clean)
            if len(candidates) >= 5:
                break
        if not candidates:
            return ""
        title = " ".join(candidates)
        title = re.sub(r"\s+", " ", title).strip()
        return ResearchWebHandler._trim_pdf_title_noise(title)[:300]

    @staticmethod
    def _best_scored_title_from_pdf_lines(lines: list[str]) -> str:
        scored: list[tuple[int, int, str]] = []
        for index, raw_line in enumerate(lines[:80]):
            first = ResearchWebHandler._normalize_pdf_text_symbols(raw_line).strip(" -")
            if not ResearchWebHandler._is_plausible_pdf_title_line(first):
                continue
            candidate_lines = [first]
            for next_line in lines[index + 1 : min(len(lines), index + 5)]:
                clean_next = ResearchWebHandler._normalize_pdf_text_symbols(next_line).strip(" -")
                if not clean_next:
                    continue
                if ResearchWebHandler._looks_like_pdf_author_line(clean_next):
                    break
                if ResearchWebHandler._looks_like_pdf_affiliation_line(clean_next):
                    break
                if not ResearchWebHandler._is_plausible_pdf_title_continuation(clean_next):
                    break
                candidate_lines.append(clean_next)
            candidate = ResearchWebHandler._trim_pdf_title_noise(" ".join(candidate_lines))
            if not candidate or len(candidate) < 8:
                continue
            score = ResearchWebHandler._score_pdf_title_candidate(candidate, lines, index, len(candidate_lines))
            if score >= 18:
                scored.append((score, -index, candidate[:300]))
        if not scored:
            return ""
        scored.sort(reverse=True)
        return scored[0][2]

    @staticmethod
    def _is_plausible_pdf_title_line(value: str) -> bool:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text or len(text) < 4 or len(text) > 180:
            return False
        if ResearchWebHandler._clean_bibliographic_title(text) == "":
            return False
        if ResearchWebHandler._looks_like_pdf_author_line(text):
            return False
        if ResearchWebHandler._looks_like_pdf_affiliation_line(text):
            return False
        if re.search(r"@|https?://|www\.|^\[?page\s+\d+\]?$", text, flags=re.IGNORECASE):
            return False
        if re.search(
            r"^(?:abstract|keywords?|introduction|references|bibliography|received|accepted|editor:|"
            r"journal|conference|proceedings|figure|table)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return False
        if re.search(r"\b(?:nature publishing group|kluwer academic publishers|manufactured in)\b", text, flags=re.IGNORECASE):
            return False
        if re.fullmatch(r"[\d\s.,;:()/-]+", text):
            return False
        return True

    @staticmethod
    def _is_plausible_pdf_title_continuation(value: str) -> bool:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not ResearchWebHandler._is_plausible_pdf_title_line(text):
            return False
        if len(text.split()) > 12:
            return False
        if text.endswith(".") and not re.search(r"\b(?:vs\.|etc\.)$", text, flags=re.IGNORECASE):
            return False
        if re.match(r"^(?:we|this|the|as|in|for|there|older|finally)\b", text, flags=re.IGNORECASE):
            return False
        return True

    @staticmethod
    def _looks_like_pdf_body_sentence(value: str) -> bool:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return False
        if re.match(r"^(?:as|older|finally|also|there|if|to|this|the|we describe|we report)\b", text, flags=re.IGNORECASE):
            return True
        if text.endswith(".") and len(text.split()) > 10:
            return True
        return False

    @staticmethod
    def _score_pdf_title_candidate(candidate: str, lines: list[str], start_index: int, line_count: int) -> int:
        text = re.sub(r"\s+", " ", candidate or "").strip()
        score = 0
        words = re.findall(r"[A-Za-z][A-Za-z-]+", text)
        if 2 <= len(words) <= 18:
            score += 10
        if text and text[0].isupper():
            score += 5
        if not text.endswith("."):
            score += 4
        if line_count >= 2:
            score += 5
        if re.search(r"\b(?:learning|network|neural|representation|classification|regression|segmentation|model|algorithm)\b", text, flags=re.IGNORECASE):
            score += 4

        following = [
            ResearchWebHandler._normalize_pdf_text_symbols(line).strip()
            for line in lines[start_index + line_count : min(len(lines), start_index + line_count + 8)]
        ]
        if any(ResearchWebHandler._looks_like_pdf_author_line(line) for line in following[:4]):
            score += 35
        if any(ResearchWebHandler._looks_like_pdf_affiliation_line(line) for line in following[:6]):
            score += 8
        if any(re.match(r"^(?:abstract\b|we\s+describe\b|this\s+paper\b)", line, flags=re.IGNORECASE) for line in following):
            score += 6

        if text and text[0].islower():
            score -= 18
        if text.endswith("."):
            score -= 4
        if re.match(r"^(?:as|the|this|we|older|finally|also|there|if|to)\b", text, flags=re.IGNORECASE):
            score -= 10
        if re.search(r"\b(?:grant|acknowledge|support|received|accepted|references?)\b", text, flags=re.IGNORECASE):
            score -= 12
        return score

    @staticmethod
    def _normalize_pdf_text_symbols(value: str) -> str:
        return str(value or "").translate(
            str.maketrans(
                {
                    "\ufb00": "ff",
                    "\ufb01": "fi",
                    "\ufb02": "fl",
                    "\ufb03": "ffi",
                    "\ufb04": "ffl",
                }
            )
        )

    @staticmethod
    def _looks_like_pdf_author_line(value: str) -> bool:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return False
        if re.search(r"\b[A-Z][a-zA-Z.-]+\s+[A-Z]\.\s+[A-Z][a-zA-Z.-]+\b", text):
            return True
        if re.search(r"\s*&\s*[A-Z][a-zA-Z.-]+", text) and len(re.findall(r"\b[A-Z][a-zA-Z.-]+\b", text)) >= 3:
            return True
        if re.search(r"\b(?:MD|PhD|MSc|MS|Dr\.|Prof\.)\b", text):
            return True
        if len(re.findall(r"\b[A-Z]\.", text)) >= 2:
            return True
        if "," in text and len(re.findall(r"\b[A-Z][a-zA-Z.-]{2,}\b", text)) >= 3:
            return True
        return False

    @staticmethod
    def _looks_like_pdf_affiliation_line(value: str) -> bool:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return False
        return bool(
            re.search(
                r"\b(?:university|department|institute|laborator(?:y|ies)|hospital|school|center|centre|college|"
                r"faculty|academy|at&t|carnegie|mellon|california|pittsburgh)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _trim_pdf_title_noise(value: str) -> str:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return ""
        text = re.sub(r"^arXiv:\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w+\s+\d{4}\s+", "", text, flags=re.IGNORECASE)
        stop_patterns = [
            r"\s+\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:MD|PhD|MSc|MS)\b",
            r"\s+\b[A-Z]\.[A-Z]\.\s*[A-Z][a-zA-Z-]+\d*",
            r"\s+\b[A-Z][a-zA-Z-]+\s+[A-Z][a-zA-Z-]+\s+\d",
            r"\s+\bAbstract\b",
            r"\s+\bShort Title\b",
            r"\s+\bWord Count\b",
            r"\s+\b\d[A-Z]?[A-Za-z ]*(?:University|Department|Institute|Hospital|School|Center|Centre)\b",
        ]
        cut = len(text)
        for pattern in stop_patterns:
            match = re.search(pattern, text)
            if match:
                cut = min(cut, match.start())
        return text[:cut].strip(" ,;:-")

    @staticmethod
    def _infer_authors_from_pdf_lines(lines: list[str], title: str) -> list[str]:
        if not lines or not title:
            return []
        title_words = set(re.findall(r"[A-Za-z]{4,}", title.casefold()))
        title_key = re.sub(r"\s+", " ", title.casefold()).strip()
        start = 0
        for index in range(min(len(lines), 80)):
            parts = []
            for end_index in range(index, min(len(lines), index + 5)):
                parts.append(re.sub(r"\s+", " ", lines[end_index]).strip())
                if re.sub(r"\s+", " ", " ".join(parts).casefold()).strip() == title_key:
                    start = end_index + 1
                    break
            if start:
                break

        scan_lines = lines[start : min(len(lines), start + 10)] if start else lines[:16]
        for line in scan_lines:
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean or clean == title:
                continue
            if not ResearchWebHandler._clean_bibliographic_title(clean) and re.search(
                r"\b(?:nature|vol|letters)\b",
                clean,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(r"abstract|keywords?|doi|arxiv|university|department|institute|hospital|@|www\.|https?://", clean, flags=re.IGNORECASE):
                continue
            words = re.findall(r"[A-Za-z]{2,}", clean)
            if len(words) < 2 or len(words) > 24:
                continue
            overlap = sum(1 for word in words if word.casefold() in title_words)
            if overlap >= max(2, len(words) // 3):
                continue
            authors = ResearchWebHandler._split_pdf_authors(clean)
            if authors:
                return authors[:20]
        return []

    @staticmethod
    def _split_pdf_authors(value: str) -> list[str]:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return []
        text = re.sub(r"\b(corresponding author|affiliations?|authors?)\b:?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\d+|\*|†|‡|§", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;")
        if not text:
            return []
        if ";" in text:
            parts = text.split(";")
        elif re.search(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE):
            parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
        elif "," in text and not re.search(r"[A-Z][a-z]+,\s*[A-Z]\.", text):
            parts = text.split(",")
        else:
            parts = [text]
        authors = []
        for part in parts:
            author = re.sub(r"\([^)]*\)", "", part).strip(" ,")
            if not author or len(author) > 80:
                continue
            if re.search(r"\b(university|department|institute|hospital|center|centre|school|college|lab)\b", author, flags=re.IGNORECASE):
                continue
            if len(re.findall(r"[A-Za-z]", author)) < 3:
                continue
            authors.append(author)
        return authors

    @staticmethod
    def _infer_uploaded_document_role(
        *,
        filename: str,
        text: str,
        metadata: dict | None = None,
        document_type: str = "",
    ) -> str:
        metadata = metadata or {}
        sample = f"{filename}\n{metadata.get('title', '')}\n{text[:6000]}".casefold()
        requirement_terms = [
            "assignment",
            "rubric",
            "grading",
            "marking criteria",
            "assessment",
            "writing requirement",
            "format requirement",
            "style guide",
            "instructions",
            "brief",
            "word count",
            "submission",
            "deadline",
            "课程要求",
            "作业要求",
            "写作要求",
            "评分标准",
            "评分细则",
            "格式要求",
            "字数",
            "提交",
            "老师要求",
            "导师要求",
        ]
        scholarly_terms = [
            "abstract",
            "introduction",
            "methodology",
            "methods",
            "results",
            "discussion",
            "conclusion",
            "references",
            "doi:",
            "关键词",
            "摘要",
            "引言",
            "研究方法",
            "实验",
            "结果",
            "讨论",
            "参考文献",
        ]
        requirement_score = sum(1 for term in requirement_terms if term in sample)
        scholarly_score = sum(1 for term in scholarly_terms if term in sample)
        requirement_score += sum(
            1
            for term in [
                "\u8bfe\u7a0b\u8981\u6c42",
                "\u4f5c\u4e1a\u8981\u6c42",
                "\u5199\u4f5c\u8981\u6c42",
                "\u8bc4\u5206\u6807\u51c6",
                "\u8bc4\u5206\u7ec6\u5219",
                "\u683c\u5f0f\u8981\u6c42",
                "\u5b57\u6570",
                "\u63d0\u4ea4",
                "\u8001\u5e08\u8981\u6c42",
                "\u5bfc\u5e08\u8981\u6c42",
            ]
            if term in sample
        )
        scholarly_score += sum(
            1
            for term in [
                "\u5173\u952e\u8bcd",
                "\u6458\u8981",
                "\u5f15\u8a00",
                "\u7814\u7a76\u65b9\u6cd5",
                "\u5b9e\u9a8c",
                "\u7ed3\u679c",
                "\u8ba8\u8bba",
                "\u53c2\u8003\u6587\u732e",
            ]
            if term in sample
        )
        has_identifier = bool(re.search(r"\bdoi\b|10\.\d{4,9}/|arxiv", sample, re.IGNORECASE))
        has_pdf_title_metadata = document_type == "PDF" and bool(str(metadata.get("title", "")).strip())

        if requirement_score >= 2 and scholarly_score < 4:
            return "instructions"
        if document_type == "DOCX" and requirement_score >= 1 and not has_identifier:
            return "instructions"
        if has_identifier or has_pdf_title_metadata or scholarly_score >= 3:
            return "literature"
        if requirement_score >= 1:
            return "instructions"
        return "literature"

    @staticmethod
    def _split_reference_roles(references: list[dict]) -> tuple[list[dict], list[dict]]:
        literature = []
        context_documents = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            item = dict(reference)
            role = str(item.get("document_role") or "").strip().lower()
            if not role and item.get("document_type"):
                role = ResearchWebHandler._infer_uploaded_document_role(
                    filename=str(item.get("source") or item.get("title") or ""),
                    text=str(item.get("content_excerpt") or item.get("abstract") or ""),
                    metadata=item.get("pdf_metadata") if isinstance(item.get("pdf_metadata"), dict) else {},
                    document_type=str(item.get("document_type") or ""),
                )
                item["document_role"] = role
                item["is_literature_source"] = role == "literature"
            if role and role != "literature":
                context_documents.append(item)
            else:
                literature.append(item)
        return literature, context_documents

    @staticmethod
    def _build_uploaded_context_topic(topic: str, documents: list[dict]) -> str:
        context = ResearchWebHandler._build_uploaded_context(documents)
        return (
            f"{topic}\n\n"
            "User-uploaded auxiliary documents are provided below. First infer their role. "
            "Treat writing requirements, rubrics, assignment prompts, and style constraints "
            "as instructions for the review, not as papers or evidence.\n\n"
            f"{context}"
        ).strip()

    @staticmethod
    def _build_uploaded_context(documents: list[dict]) -> str:
        sections = [
            "Uploaded auxiliary documents. These may be writing requirements, rubrics, assignment prompts, style constraints, background notes, or source evidence.",
            "Use them as source evidence only when they clearly contain substantive research content.",
        ]
        for index, document in enumerate(documents, start=1):
            title = document.get("title") or document.get("source") or f"Uploaded document {index}"
            source = document.get("source", "")
            role = document.get("document_role", "unknown")
            excerpt = ResearchWebHandler._build_context_document_package(
                document.get("content_excerpt") or document.get("abstract") or ""
            )
            sections.append(
                f"## Auxiliary document {index}: {title}\n\n"
                f"Source: {source}\n\n"
                f"Detected role: {role}\n\n"
                "Use note: classify this document before using it; if it is a rubric or writing requirement, apply it as task constraints rather than literature evidence.\n\n"
                f"Extracted excerpt:\n{excerpt}"
            )
        return "\n\n".join(section.strip() for section in sections if section).strip()

    @staticmethod
    def _extract_pdf_content(content: bytes) -> dict:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF extraction requires the pypdf package. Run: pip install pypdf") from error

        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        metadata = ResearchWebHandler._clean_pdf_metadata(reader.metadata)
        parts = []
        extracted_pages = 0
        skipped_page_notes = []
        page_limit = ResearchWebHandler._pdf_extract_page_limit()
        pages_to_scan = reader.pages if page_limit is None else reader.pages[:page_limit]
        for page_number, page in enumerate(pages_to_scan, start=1):
            try:
                page_text = ResearchWebHandler._clean_pdf_page_text(page.extract_text() or "")
            except Exception as error:
                skipped_page_notes.append(
                    f"page {page_number}: {type(error).__name__}: {error}"
                )
                print(
                    f"[web] skipped PDF page {page_number} during text extraction: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            if not page_text:
                continue
            extracted_pages += 1
            parts.append(f"[Page {page_number}]\n{page_text}")

        text = "\n\n".join(parts).strip()
        fallback_note = ""
        if skipped_page_notes or not text:
            fallback = ResearchWebHandler._extract_pdf_content_with_pymupdf(
                content,
                page_limit=page_limit,
            )
            if fallback and fallback["extracted_pages"] > extracted_pages:
                text = fallback["text"]
                extracted_pages = fallback["extracted_pages"]
                page_count = max(page_count, fallback["page_count"])
                fallback_note = " Used PyMuPDF fallback extraction."
                print(
                    f"[web] used PyMuPDF fallback PDF extraction; extracted "
                    f"{extracted_pages}/{page_count} pages.",
                    flush=True,
                )
        if text:
            scan_note = "all pages scanned" if page_limit is None else f"first {min(page_count, page_limit)} pages scanned"
            note = (
                f"Extracted text from {extracted_pages}/{page_count} pages "
                f"({scan_note}).{fallback_note}"
            )
        else:
            note = "No readable text extracted; the PDF may be scanned, image-only, or protected."
        if skipped_page_notes:
            note = (
                f"{note} Skipped {len(skipped_page_notes)} page(s) because pypdf "
                "could not extract text from them."
            )
        return {
            "text": text,
            "page_count": page_count,
            "extracted_pages": extracted_pages,
            "metadata": metadata,
            "note": note,
        }

    @staticmethod
    def _clean_pdf_page_text(raw_text: str) -> str:
        page_text = ResearchWebHandler._normalize_extracted_text(raw_text or "")
        page_lines = [
            re.sub(r"[ \t\f\v]+", " ", line).strip()
            for line in page_text.splitlines()
        ]
        return "\n".join(line for line in page_lines if line).strip()

    @staticmethod
    def _extract_pdf_content_with_pymupdf(content: bytes, *, page_limit: int | None) -> dict | None:
        try:
            import fitz
        except ImportError:
            return None

        try:
            document = fitz.open(stream=content, filetype="pdf")
        except Exception as error:
            print(
                f"[web] PyMuPDF fallback PDF extraction failed to open document: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return None

        try:
            page_count = int(document.page_count)
            scan_count = page_count if page_limit is None else min(page_count, page_limit)
            parts = []
            extracted_pages = 0
            for page_index in range(scan_count):
                try:
                    page = document.load_page(page_index)
                    page_text = ResearchWebHandler._clean_pdf_page_text(page.get_text("text") or "")
                except Exception as error:
                    print(
                        f"[web] PyMuPDF skipped PDF page {page_index + 1}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                if not page_text:
                    continue
                extracted_pages += 1
                parts.append(f"[Page {page_index + 1}]\n{page_text}")
            return {
                "text": "\n\n".join(parts).strip(),
                "page_count": page_count,
                "extracted_pages": extracted_pages,
            }
        finally:
            document.close()

    @staticmethod
    def _build_high_information_package(text: str, *, max_chars: int = PDF_REFERENCE_EXCERPT_CHARS) -> str:
        return LiteratureAnalysisWorkflow._high_information_excerpt(text, max_chars=max_chars)

    @staticmethod
    def _build_context_document_package(text: str, *, max_chars: int = CONTEXT_DOCUMENT_EXCERPT_CHARS) -> str:
        return LiteratureAnalysisWorkflow._high_information_excerpt(text, max_chars=max_chars)

    @staticmethod
    def _clean_pdf_metadata(metadata) -> dict:
        if not metadata:
            return {}
        cleaned = {}
        for key, value in dict(metadata).items():
            normalized_key = str(key).lstrip("/").lower()
            if normalized_key in {"title", "author", "subject", "creator", "producer"}:
                cleaned[normalized_key] = ResearchWebHandler._normalize_extracted_text(str(value or ""))[:500]
        return cleaned

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
    def _history_entries(limit: int = 200) -> list[dict]:
        with HISTORY_LOCK:
            data = ResearchWebHandler._read_history_data_unlocked()
            if ResearchWebHandler._reconcile_history_jobs_unlocked(data):
                ResearchWebHandler._write_history_data_unlocked(data)
            entries = data.get("items", [])
            if not isinstance(entries, list):
                return []
            return sorted(
                [entry for entry in entries if isinstance(entry, dict)],
                key=lambda entry: str(entry.get("updated_at") or entry.get("created_at") or ""),
                reverse=True,
            )[:limit]

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
                if status is not None:
                    entry["status"] = status
                if stage is not None:
                    entry["stage"] = stage
                if result is not None:
                    entry["result"] = result
                if counts is not None:
                    existing_counts = entry.get("counts")
                    if not isinstance(existing_counts, dict):
                        existing_counts = {}
                    existing_counts.update(counts)
                    entry["counts"] = existing_counts
                if error is not None:
                    entry["error"] = error
                entry["updated_at"] = now
                ResearchWebHandler._write_history_data_unlocked(data)
                return

    @staticmethod
    def _start_history_analysis(history_id: str, job_id: str, request: dict) -> None:
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
                entry["kind"] = "search_flow"
                entry["status"] = "running"
                entry["stage"] = "Running LLM literature analysis..."
                entry["job_id"] = job_id
                entry["analysis"] = {
                    "status": "queued",
                    "stage": "Queued literature analysis",
                    "job_id": job_id,
                    "request": request,
                    "result": {},
                    "counts": {"references": request.get("reference_count", 0)},
                }
                entry["updated_at"] = now
                ResearchWebHandler._write_history_data_unlocked(data)
                return

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
        if history_slot != "analysis":
            ResearchWebHandler._update_history_entry(
                history_id,
                status=status,
                stage=stage,
                result=result,
                counts=counts,
                error=error,
            )
            return
        ResearchWebHandler._update_history_analysis(
            history_id,
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
                analysis = entry.get("analysis") if isinstance(entry.get("analysis"), dict) else {}
                if status is not None:
                    analysis["status"] = status
                    entry["status"] = status
                if stage is not None:
                    analysis["stage"] = stage
                    entry["stage"] = stage
                if result is not None:
                    analysis["result"] = result
                if counts is not None:
                    existing_counts = analysis.get("counts") if isinstance(analysis.get("counts"), dict) else {}
                    existing_counts.update(counts)
                    analysis["counts"] = existing_counts
                if error is not None:
                    analysis["error"] = error
                    entry["error"] = error
                entry["kind"] = "search_flow"
                entry["analysis"] = analysis
                entry["updated_at"] = now
                ResearchWebHandler._write_history_data_unlocked(data)
                return

    @staticmethod
    def _read_history_data_unlocked() -> dict:
        if not HISTORY_PATH.exists():
            return {"version": 1, "items": []}
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8-sig") or "{}")
        except (OSError, json.JSONDecodeError) as error:
            print(f"[web] failed to read history file: {error}", flush=True)
            return {"version": 1, "items": []}
        if isinstance(data, list):
            return {"version": 1, "items": data}
        if not isinstance(data, dict):
            return {"version": 1, "items": []}
        items = data.get("items")
        if not isinstance(items, list):
            data["items"] = []
        data.setdefault("version", 1)
        return data

    @staticmethod
    def _write_history_data_unlocked(data: dict) -> None:
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            temp_path = HISTORY_PATH.with_suffix(f"{HISTORY_PATH.suffix}.tmp")
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
            temp_path.replace(HISTORY_PATH)
        except OSError as error:
            print(f"[web] failed to write history file: {error}", flush=True)

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
        if changed:
            entry["updated_at"] = now
        return changed

    @staticmethod
    def _load_persisted_job_log(job_id: str) -> dict | None:
        try:
            matches = sorted(LOG_DIR.glob(f"last_job*_{job_id}.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            return None
        for path in matches:
            try:
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _history_references(references: list[dict]) -> list[dict]:
        heavy_keys = {
            "abstract",
            "content_excerpt",
            "evidence_source_text",
            "full_text_for_evidence",
            "raw_source_record",
            "pdf_metadata",
            "bibliographic_identity",
        }
        return [
            {key: value for key, value in reference.items() if key not in heavy_keys}
            for reference in references
            if isinstance(reference, dict)
        ]

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _frontend_index_path() -> Path:
        dist_index = WEB_DIST_DIR / "index.html"
        return dist_index if dist_index.exists() else WEB_DIR / "index.html"

    @staticmethod
    def _frontend_dist_file(request_path: str) -> Path | None:
        if not WEB_DIST_DIR.exists():
            return None
        relative = request_path.lstrip("/")
        if not relative:
            return None
        candidate = (WEB_DIST_DIR / relative).resolve()
        try:
            candidate.relative_to(WEB_DIST_DIR.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    @staticmethod
    def _content_type_for_path(path: Path) -> str:
        if path.suffix == ".js":
            return "application/javascript; charset=utf-8"
        if path.suffix == ".css":
            return "text/css; charset=utf-8"
        content_type, _ = mimetypes.guess_type(str(path))
        return content_type or "application/octet-stream"

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, body: bytes, content_type: str, filename: str) -> None:
        fallback_filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename).strip("_") or "download.pdf"
        encoded_filename = quote(filename)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename=\"{fallback_filename}\"; filename*=UTF-8''{encoded_filename}",
        )
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _markdown_to_pdf_bytes(title: str, markdown: str) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib import colors
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as error:
            raise RuntimeError("PDF export requires reportlab. Run: pip install -r requirements.txt") from error

        font_name = "Helvetica"
        for font_path in [
            Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]:
            if not font_path.exists():
                continue
            try:
                pdfmetrics.registerFont(TTFont("ResearchFont", str(font_path)))
                font_name = "ResearchFont"
                break
            except Exception:
                continue

        styles = getSampleStyleSheet()
        normal = ParagraphStyle(
            "ResearchNormal",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=15,
            spaceAfter=6,
        )
        heading1 = ParagraphStyle(
            "ResearchHeading1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=18,
            leading=23,
            spaceBefore=8,
            spaceAfter=8,
        )
        heading2 = ParagraphStyle(
            "ResearchHeading2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=14,
            leading=19,
            spaceBefore=8,
            spaceAfter=6,
        )
        heading3 = ParagraphStyle(
            "ResearchHeading3",
            parent=styles["Heading3"],
            fontName=font_name,
            fontSize=12,
            leading=17,
            spaceBefore=6,
            spaceAfter=4,
        )

        pagesize = landscape(A4) if ResearchWebHandler._markdown_has_wide_table(markdown) else A4
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=pagesize,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title,
        )
        table_normal = ParagraphStyle(
            "ResearchTableNormal",
            parent=normal,
            fontSize=8.5,
            leading=12,
            spaceAfter=0,
        )
        story = [Paragraph(ResearchWebHandler._pdf_inline_text(title), heading1), Spacer(1, 4)]
        pending_list = []

        def flush_list() -> None:
            nonlocal pending_list
            if not pending_list:
                return
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(item, normal)) for item in pending_list],
                    bulletType="bullet",
                    leftIndent=14,
                )
            )
            pending_list = []

        def append_table(table_lines: list[str]) -> None:
            rows = [ResearchWebHandler._split_markdown_table_row(line) for line in table_lines]
            if len(rows) < 2:
                return
            data = [
                [Paragraph(ResearchWebHandler._pdf_inline_text(cell), table_normal) for cell in row]
                for row in [rows[0], *rows[2:]]
                if row
            ]
            if not data:
                return
            column_count = max(len(row) for row in data)
            for row in data:
                while len(row) < column_count:
                    row.append(Paragraph("", normal))
            col_widths = ResearchWebHandler._pdf_table_column_widths(doc.width, column_count)
            table = Table(
                data,
                colWidths=col_widths,
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_name),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7dee8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 6))

        lines = markdown.splitlines()
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.strip()
            if not line:
                flush_list()
                story.append(Spacer(1, 4))
                index += 1
                continue
            if ResearchWebHandler._is_markdown_table_start(lines, index):
                flush_list()
                table_lines = []
                while index < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                append_table(table_lines)
                continue
            if line.startswith("### "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[4:]), heading3))
            elif line.startswith("## "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[3:]), heading2))
            elif line.startswith("# "):
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line[2:]), heading1))
            elif re.match(r"^[-*]\s+", line):
                pending_list.append(ResearchWebHandler._pdf_inline_text(re.sub(r"^[-*]\s+", "", line)))
            else:
                flush_list()
                story.append(Paragraph(ResearchWebHandler._pdf_inline_text(line), normal))
            index += 1
        flush_list()
        doc.build(story)
        return buffer.getvalue()

    @staticmethod
    def _pdf_inline_text(value: str) -> str:
        text = ResearchWebHandler._normalize_pdf_symbols(str(value or ""))
        text = html.escape(text)
        text = re.sub(r"&lt;br\s*/?&gt;", "<br/>", text, flags=re.IGNORECASE)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
        return text

    @staticmethod
    def _normalize_pdf_symbols(value: str) -> str:
        superscripts = str.maketrans({
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁻": "-",
            "⁺": "+",
        })
        unicode_superscripts = str.maketrans({
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁻": "-",
            "⁺": "+",
        })
        escaped_superscripts = str.maketrans({
            "\u2070": "0",
            "\u00b9": "1",
            "\u00b2": "2",
            "\u00b3": "3",
            "\u2074": "4",
            "\u2075": "5",
            "\u2076": "6",
            "\u2077": "7",
            "\u2078": "8",
            "\u2079": "9",
            "\u207b": "-",
            "\u207a": "+",
        })
        text = value.translate(superscripts).translate(unicode_superscripts).translate(escaped_superscripts)
        text = re.sub(r"(?<=\d)-(?=\d)", "^-", text)
        text = text.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u00d7", "x")
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        text = text.replace("≤", "<=").replace("≥", ">=").replace("×", "x")
        text = text.replace("\ufffd", "").replace("□", "")
        text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
        text = text.replace("≤", "<=").replace("≥", ">=").replace("×", "x")
        return text

    @staticmethod
    def _markdown_has_wide_table(markdown: str) -> bool:
        lines = markdown.splitlines()
        for index, line in enumerate(lines):
            if not ResearchWebHandler._is_markdown_table_start(lines, index):
                continue
            columns = len(ResearchWebHandler._split_markdown_table_row(line))
            if columns >= 5:
                return True
        return False

    @staticmethod
    def _pdf_table_column_widths(total_width: float, column_count: int) -> list[float]:
        if column_count == 6:
            weights = [1.35, 1.15, 1.35, 1.5, 1.2, 1.2]
        elif column_count == 5:
            weights = [1.35, 1.15, 1.4, 1.45, 1.2]
        else:
            weights = [1.0] * column_count
        total_weight = sum(weights)
        return [total_width * weight / total_weight for weight in weights]

    @staticmethod
    def _is_markdown_table_start(lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        current = lines[index]
        separator = lines[index + 1]
        return bool(
            re.match(r"^\s*\|.*\|\s*$", current)
            and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", separator)
        )

    @staticmethod
    def _split_markdown_table_row(line: str) -> list[str]:
        text = str(line or "").strip()
        if text.startswith("|"):
            text = text[1:]
        if text.endswith("|"):
            text = text[:-1]
        cells: list[str] = []
        cell = []
        for index, char in enumerate(text):
            if char == "|" and (index == 0 or text[index - 1] != "\\"):
                cells.append("".join(cell).replace("\\|", "|").strip())
                cell = []
            else:
                cell.append(char)
        cells.append("".join(cell).replace("\\|", "|").strip())
        return cells

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
