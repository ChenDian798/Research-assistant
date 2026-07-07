from __future__ import annotations

import asyncio
import sys
import threading
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus

from src.research_agent.citations import format_references, normalize_citation_format
from src.research_agent.doi import enrich_references_with_doi_metadata
from src.research_agent.literature_workflow import LiteratureAnalysisWorkflow
from src.research_agent.llm import LLMServiceError
from src.research_agent.web_history import apply_history_analysis_update
from src.research_agent.web_jobs import set_job_error, set_job_status


class AnalysisRouteService:
    def __init__(self, handler) -> None:
        self.handler = handler

    def _module(self):
        module_name = self.handler.__module__ if isinstance(self.handler, type) else self.handler.__class__.__module__
        return sys.modules.get(module_name)

    def _dependency(self, name: str, fallback):
        module = self._module()
        if module is None:
            return fallback
        return getattr(module, name, fallback)

    @property
    def jobs(self) -> dict[str, dict[str, object]]:
        return self._dependency("JOBS", {})

    @property
    def jobs_lock(self):
        return self._dependency("JOBS_LOCK", threading.Lock())

    def _handle_literature_analysis(self) -> None:
        handler = self.handler
        try:
            payload = handler._read_json()
            references = payload.get("references", [])
            final_report = str(payload.get("final_report", "") or "")
            if not isinstance(references, list):
                handler._send_json(
                    {"error": "References must be a list."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            references, context_documents = handler._augment_references_with_llm(
                references,
                final_report,
                purpose="literature analysis",
            )
            if context_documents:
                context_block = handler._build_uploaded_context(context_documents)
                final_report = f"{final_report}\n\n{context_block}".strip()
            if not references and not final_report.strip():
                handler._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            topic = str(payload.get("topic", "") or "current research").strip()
            citation_format = str(payload.get("citation_format", "APA") or "APA").strip()
            include_audit = handler._truthy(payload.get("include_audit"))
            output_language = handler._normalize_output_language(payload.get("output_language"))
            history_source = str(payload.get("history_source") or "direct").strip().lower()
            if history_source not in {"direct", "search"}:
                history_source = "direct"
            job_id = uuid.uuid4().hex
            port = handler._server_port()
            analysis_history_request = {
                "topic": topic,
                "references": handler._history_references(references),
                "reference_count": len(references),
                "has_context": bool(final_report.strip()),
                "citation_format": citation_format,
                "output_language": output_language,
            }
            existing_history_id = str(payload.get("history_id") or "").strip()
            history_slot = "result"
            if history_source == "search" and existing_history_id and handler._history_entry(existing_history_id):
                history_id = existing_history_id
                history_slot = "analysis"
                handler._start_history_analysis(history_id, job_id, analysis_history_request)
            else:
                history_id = handler._create_history_entry(
                    kind="search_analysis" if history_source == "search" else "direct_analysis",
                    source=history_source,
                    title=topic,
                    status="queued",
                    job_id=job_id,
                    request=analysis_history_request,
                    counts={"references": len(references)},
                )
            set_job_status(
                self.jobs,
                self.jobs_lock,
                job_id,
                {
                    "status": "queued",
                    "kind": "literature_analysis",
                    "port": port,
                    "history_id": history_id,
                    "history_slot": history_slot,
                },
            )

            thread_module = self._dependency("threading", threading)
            thread = thread_module.Thread(
                target=handler._run_literature_analysis_job,
                args=(job_id, topic, references, final_report, citation_format, include_audit, port, history_id, history_slot, output_language),
                daemon=True,
            )
            thread.start()
            handler._send_json({"job_id": job_id, "history_id": history_id, "status": "queued"}, HTTPStatus.ACCEPTED)
        except BrokenPipeError:
            print("[web] client disconnected before literature response was sent", flush=True)
        except Exception as error:
            traceback.print_exc()
            try:
                handler._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before literature error response was sent", flush=True)

    def _handle_literature_pdf_analysis(self) -> None:
        handler = self.handler
        max_pdf_upload_files = self._dependency("MAX_PDF_UPLOAD_FILES", 4)
        try:
            files, link_references, fields = handler._read_pdf_uploads_with_fields(allow_empty=True)
            if len(files) > max_pdf_upload_files:
                handler._send_json(
                    {
                        "error": (
                            f"文献分析每次最多上传 {max_pdf_upload_files} 个 PDF/DOCX 文件。"
                            "请清空后重新选择，或分批上传分析。"
                        )
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            references = list(link_references)
            user_context = fields.get("user_context", "").strip()
            topic = str(fields.get("topic", "") or "").strip() or "user-provided literature links and PDF analysis"
            llm_text = user_context if files else "\n\n".join(part for part in [topic, user_context] if part)
            references, llm_context_documents = handler._augment_references_with_llm(
                references,
                llm_text,
                purpose="literature analysis upload",
            )
            expected_context = "\n\n".join(part for part in [topic, user_context] if part)
            uploaded_references = [
                handler._uploaded_file_to_reference(
                    filename,
                    content,
                    expected_context=expected_context,
                )
                for filename, content in files
            ]
            references, context_documents = handler._split_reference_roles(
                references + uploaded_references + llm_context_documents
            )
            review_needed_documents = [
                document
                for document in context_documents
                if str(document.get("document_role") or "").strip().lower() == "review_needed"
            ]
            if not references and not user_context and not context_documents:
                handler._send_json(
                    {"error": "Please provide references, uploaded files, or text context."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            final_report = handler._build_pdf_context(references)
            if context_documents:
                context_block = handler._build_uploaded_context(context_documents)
                final_report = f"{final_report}\n\n{context_block}".strip()
            if user_context:
                final_report = (
                    f"{final_report}\n\n" if final_report else ""
                ) + f"User-provided text context or instructions:\n{user_context}"
            citation_format = str(fields.get("citation_format", "APA") or "APA").strip()
            include_audit = handler._truthy(fields.get("include_audit"))
            output_language = handler._normalize_output_language(fields.get("output_language"))
            history_source = str(fields.get("history_source") or "direct").strip().lower()
            if history_source not in {"direct", "search"}:
                history_source = "direct"
            job_id = uuid.uuid4().hex
            port = handler._server_port()
            history_id = handler._create_history_entry(
                kind="search_analysis" if history_source == "search" else "direct_analysis",
                source=history_source,
                title=topic,
                status="queued",
                job_id=job_id,
                request={
                    "topic": topic,
                    "references": handler._history_references(references),
                    "review_needed_documents": handler._history_references(review_needed_documents),
                    "reference_count": len(references),
                    "file_count": len(files),
                    "has_context": bool(user_context.strip() or context_documents),
                    "citation_format": citation_format,
                    "output_language": output_language,
                },
                counts={
                    "references": len(references),
                    "files": len(files),
                    "review_needed": len(review_needed_documents),
                },
            )
            set_job_status(
                self.jobs,
                self.jobs_lock,
                job_id,
                {
                    "status": "queued",
                    "kind": "literature_analysis",
                    "port": port,
                    "history_id": history_id,
                },
            )

            thread_module = self._dependency("threading", threading)
            thread = thread_module.Thread(
                target=handler._run_literature_analysis_job,
                args=(job_id, topic, references, final_report, citation_format, include_audit, port, history_id, "result", output_language),
                daemon=True,
            )
            thread.start()
            handler._send_json(
                {
                    "job_id": job_id,
                    "history_id": history_id,
                    "status": "queued",
                    "references": handler._public_references(references),
                    "review_needed_documents": handler._public_references(review_needed_documents),
                },
                HTTPStatus.ACCEPTED,
            )
        except BrokenPipeError:
            print("[web] client disconnected before document literature response was sent", flush=True)
        except ValueError as error:
            handler._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            try:
                handler._send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except BrokenPipeError:
                print("[web] client disconnected before document literature error response was sent", flush=True)

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
        handler = self.handler
        output_language = handler._normalize_output_language(output_language)
        self._set_analysis_job_stage(job_id, port, history_id, history_slot, "Starting literature analysis...")
        handler._update_analysis_history(history_id, history_slot, status="running", stage="Starting literature analysis...")

        enrich_references = self._dependency("enrich_references_with_doi_metadata", enrich_references_with_doi_metadata)
        normalize_citation = self._dependency("normalize_citation_format", normalize_citation_format)
        format_reference_list = self._dependency("format_references", format_references)
        workflow_cls = self._dependency("LiteratureAnalysisWorkflow", LiteratureAnalysisWorkflow)
        llm_service_error = self._dependency("LLMServiceError", LLMServiceError)

        try:
            self._set_analysis_job_stage(job_id, port, history_id, history_slot, "Resolving DOI metadata...")
            handler._update_analysis_history(history_id, history_slot, status="running", stage="Resolving DOI metadata...")
            references = enrich_references(references)
            citation_format = normalize_citation(citation_format)
            formatted_references = format_reference_list(references, citation_format)
            self._set_analysis_job_stage(job_id, port, history_id, history_slot, "Running LLM literature analysis...")
            handler._update_analysis_history(history_id, history_slot, status="running", stage="Running LLM literature analysis...")
            analysis_result = asyncio.run(
                workflow_cls(verbose=True).run(
                    topic=topic,
                    references=references,
                    final_report=final_report,
                    citation_format=citation_format,
                    formatted_references=formatted_references,
                    output_language=output_language,
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
            done_job = self._build_analysis_done_job(
                port=port,
                rows=rows,
                summary=summary,
                references=formatted_references,
                citation_format=citation_format,
                output_language=output_language,
                history_id=history_id,
                history_slot=history_slot,
            )
            if include_audit and isinstance(audit_summary, dict):
                done_job["audit_summary"] = audit_summary
            handler._persist_job_log(job_id, done_job, port=port)
            set_job_status(self.jobs, self.jobs_lock, job_id, done_job)
            if history_slot == "analysis":
                handler._update_history_analysis(
                    history_id,
                    status="done",
                    stage="Analysis complete",
                    result={
                        "rows": rows,
                        "summary": summary,
                        "references": formatted_references,
                        "citation_format": citation_format,
                        "output_language": output_language,
                    },
                    counts={"rows": len(rows) if isinstance(rows, list) else 0},
                )
            else:
                handler._update_history_entry(
                    history_id,
                    status="done",
                    stage="Analysis complete",
                    result={
                        "rows": rows,
                        "summary": summary,
                        "references": formatted_references,
                        "citation_format": citation_format,
                        "output_language": output_language,
                    },
                    counts={"rows": len(rows) if isinstance(rows, list) else 0},
                )
        except llm_service_error as error:
            print(f"[web] literature LLM service error: {error}", flush=True)
            set_job_error(
                self.jobs,
                self.jobs_lock,
                job_id,
                "literature_analysis",
                port,
                str(error),
                history_id=history_id,
                history_slot=history_slot,
            )
            handler._update_analysis_history(history_id, history_slot, status="error", error=str(error))
        except Exception as error:
            traceback.print_exc()
            set_job_error(
                self.jobs,
                self.jobs_lock,
                job_id,
                "literature_analysis",
                port,
                f"{type(error).__name__}: {error}",
                history_id=history_id,
                history_slot=history_slot,
            )
            handler._update_analysis_history(history_id, history_slot, status="error", error=f"{type(error).__name__}: {error}")

    def _set_analysis_job_stage(
        self,
        job_id: str,
        port: int | str,
        history_id: str,
        history_slot: str,
        stage: str,
    ) -> None:
        set_job_status(
            self.jobs,
            self.jobs_lock,
            job_id,
            {
                "status": "running",
                "kind": "literature_analysis",
                "port": port,
                "stage": stage,
                "history_id": history_id,
                "history_slot": history_slot,
            },
        )

    @staticmethod
    def _build_analysis_done_job(
        *,
        port: int | str,
        rows,
        summary,
        references,
        citation_format: str,
        output_language: str,
        history_id: str,
        history_slot: str,
    ) -> dict:
        return {
            "status": "done",
            "kind": "literature_analysis",
            "port": port,
            "rows": rows,
            "summary": summary,
            "references": references,
            "citation_format": citation_format,
            "output_language": output_language,
            "history_id": history_id,
            "history_slot": history_slot,
        }

    def _start_history_analysis(self, history_id: str, job_id: str, request: dict) -> None:
        handler = self.handler
        history_id = str(history_id or "").strip()
        if not history_id:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self._dependency("HISTORY_LOCK", threading.Lock()):
            data = handler._read_history_data_unlocked()
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
                handler._write_history_data_unlocked(data)
                return

    def _update_analysis_history(
        self,
        history_id: str,
        history_slot: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        error: str | None = None,
    ) -> None:
        handler = self.handler
        if history_slot != "analysis":
            handler._update_history_entry(
                history_id,
                status=status,
                stage=stage,
                result=result,
                counts=counts,
                error=error,
            )
            return
        handler._update_history_analysis(
            history_id,
            status=status,
            stage=stage,
            result=result,
            counts=counts,
            error=error,
        )

    def _update_history_analysis(
        self,
        history_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: dict | None = None,
        counts: dict | None = None,
        error: str | None = None,
    ) -> None:
        handler = self.handler
        history_id = str(history_id or "").strip()
        if not history_id:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self._dependency("HISTORY_LOCK", threading.Lock()):
            data = handler._read_history_data_unlocked()
            items = data.get("items", [])
            if not isinstance(items, list):
                return
            for entry in items:
                if not isinstance(entry, dict) or entry.get("id") != history_id:
                    continue
                apply_history_analysis_update(
                    entry,
                    now,
                    status=status,
                    stage=stage,
                    result=result,
                    counts=counts,
                    error=error,
                )
                handler._write_history_data_unlocked(data)
                return
