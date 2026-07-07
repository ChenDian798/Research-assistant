from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import traceback
import unicodedata
import uuid
from http import HTTPStatus
from urllib.parse import urlparse

from src.research_agent.llm import LLMServiceError
from src.research_agent.novelty_check import NoveltyCheckWorkflow
from src.research_agent.novelty_planner import build_novelty_plan
from src.research_agent.novelty_search import run_novelty_search_plan
from src.research_agent.paper_search import PaperSearchError, search_papers
from src.research_agent.reference_relevance import apply_relevance_gate
from src.research_agent.reference_screening import screen_references
from src.research_agent.reference_verification import verify_references
from src.research_agent.web_jobs import set_job_error, set_job_status


class SearchRouteService:
    def __init__(self, handler, jobs: dict[str, dict[str, object]], jobs_lock) -> None:
        self.handler = handler
        self.jobs = jobs
        self.jobs_lock = jobs_lock

    def _dependency(self, name: str, fallback):
        if self.handler is None:
            return fallback
        module = sys.modules.get(self.handler.__class__.__module__)
        if module is None:
            return fallback
        return getattr(module, name, fallback)

    def _handle_literature_search(self) -> None:
        handler = self.handler
        try:
            if not handler._paper_search_enabled():
                handler._send_json(
                    {
                        "error": "Academic search is not enabled. Set PAPER_SEARCH_ENABLED=true and install paper-search-mcp.",
                        "search_enabled": False,
                    },
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            payload = handler._read_json()
            request = self._search_request_from_payload(payload)
            if not request["query"]:
                handler._send_json({"error": "Search query cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            if handler._truthy(payload.get("run_async")):
                job_id = uuid.uuid4().hex
                history_id = handler._create_history_entry(
                    kind="search_flow",
                    source="search",
                    title=request["query"],
                    status="queued",
                    job_id=job_id,
                    request=self._search_history_request(request),
                    counts={},
                )
                set_job_status(
                    self.jobs,
                    self.jobs_lock,
                    job_id,
                    {
                        "status": "queued",
                        "kind": "literature_search",
                        "port": handler._server_port(),
                        "history_id": history_id,
                    },
                )
                thread = threading.Thread(
                    target=handler._run_literature_search_job,
                    args=(job_id, history_id, request),
                    daemon=True,
                )
                thread.start()
                handler._send_json(
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
            history_id = handler._create_history_entry(
                kind="search_flow",
                source="search",
                title=response_payload["query"] or request["query"],
                status="done",
                request=self._search_history_request(request),
                result=response_payload,
                counts=history_payload["counts"],
            )
            response_payload["history_id"] = history_id
            handler._send_json(response_payload)
        except PaperSearchError as error:
            history_id = handler._create_history_entry(
                kind="search_flow",
                source="search",
                title=locals().get("request", {}).get("query") or "Literature search",
                status="error",
                request=self._search_history_request(locals().get("request", {})),
                error=str(error),
            )
            handler._send_json(
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
            handler._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            handler._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _run_literature_search_job(self, job_id: str, history_id: str, request: dict) -> None:
        handler = self.handler
        port = handler._server_port()
        set_job_status(
            self.jobs,
            self.jobs_lock,
            job_id,
            {
                "status": "running",
                "kind": "literature_search",
                "port": port,
                "stage": "Searching literature...",
                "history_id": history_id,
            },
        )
        handler._update_history_entry(history_id, status="running", stage="Searching literature...")

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
            set_job_status(self.jobs, self.jobs_lock, job_id, done_job)
            handler._update_history_entry(
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
        set_job_error(
            self.jobs,
            self.jobs_lock,
            job_id,
            "literature_search",
            port,
            message,
            history_id=history_id,
        )
        self.handler._update_history_entry(history_id, status="error", error=message)

    def _run_literature_search_pipeline(self, request: dict) -> tuple[dict, dict]:
        handler = self.handler
        query = str(request.get("query") or "").strip()
        sources = str(request.get("sources") or "").strip()
        search_mode = str(request.get("search_mode") or "auto").strip().lower() or "auto"
        year = str(request.get("year") or "").strip()
        max_results_per_source = int(request.get("max_results_per_source") or 5)
        timeout_seconds = int(request.get("timeout_seconds") or 45)
        include_needs_review = bool(request.get("include_needs_review", True))
        append_annotation_record = bool(request.get("append_annotation_record", True))

        search_papers_func = self._dependency("search_papers", search_papers)
        screen_references_func = self._dependency("screen_references", screen_references)
        apply_relevance_gate_func = self._dependency("apply_relevance_gate", apply_relevance_gate)
        verify_references_func = self._dependency("verify_references", verify_references)

        result = search_papers_func(
            query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            year=year,
            timeout_seconds=timeout_seconds,
            search_mode=search_mode,
        )
        max_total = handler._bounded_int(os.getenv("PAPER_SEARCH_MAX_TOTAL"), default=40, minimum=1, maximum=200)
        screened = screen_references_func(result.get("papers", [])[:max_total])
        screened = apply_relevance_gate_func(query, screened, query_plan=result.get("query_plan"))
        verified_qualified = verify_references_func(screened["qualified"])
        verified_needs_review = verify_references_func(screened["needs_review"])
        qualified_references, needs_review_references = self._split_verified_search_candidates(
            verified_qualified,
            verified_needs_review,
        )
        qualified_references, needs_review_references = self._dedupe_final_search_candidates(
            qualified_references,
            needs_review_references,
            screened,
        )
        qualified_references, needs_review_references = self._apply_final_search_limits(
            qualified_references,
            needs_review_references,
            requested_sources=result.get("sources_used") or sources,
            max_results_per_source=max_results_per_source,
            include_needs_review=include_needs_review,
        )
        audit_log, annotation_record = handler._write_search_audit_log(
            query=query,
            result=result,
            screened=screened,
            qualified_references=qualified_references,
            needs_review_references=needs_review_references,
            requested_sources=sources,
            year=year,
            port=handler._server_port(),
            append_annotation_record=append_annotation_record,
        )
        response_payload = self._build_search_response_payload(
            result=result,
            query=query,
            search_mode=search_mode,
            qualified_references=qualified_references,
            needs_review_references=needs_review_references,
            screened=screened,
            include_needs_review=include_needs_review,
            audit_log=audit_log,
            annotation_record=annotation_record,
            append_annotation_record=append_annotation_record,
        )
        return response_payload, {
            "counts": self._build_search_counts(
                qualified_references,
                needs_review_references,
                screened,
                include_needs_review=include_needs_review,
            )
        }

    def _build_search_response_payload(
        self,
        *,
        result: dict,
        query: str,
        search_mode: str,
        qualified_references: list[dict],
        needs_review_references: list[dict],
        screened: dict,
        include_needs_review: bool,
        audit_log,
        annotation_record,
        append_annotation_record: bool,
    ) -> dict:
        handler = self.handler
        return {
            "status": "done",
            "port": handler._server_port(),
            "query": result.get("query", query),
            "search_mode": result.get("search_mode", search_mode),
            "requested_search_mode": result.get("requested_search_mode", search_mode),
            "mode_inference_status": result.get("mode_inference_status", ""),
            "mode_inference_error": result.get("mode_inference_error", ""),
            "mode_inference_rationale": result.get("mode_inference_rationale", ""),
            "mode_inference_confidence": result.get("mode_inference_confidence", ""),
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
            "qualified_references": handler._public_references(qualified_references),
            "needs_review_references": handler._public_references(needs_review_references) if include_needs_review else [],
            "rejected_count": len(screened["rejected"]),
            "rejected_references": handler._public_references(screened["rejected"]),
            "source_results": result.get("source_results", {}),
            "internal_source_results": result.get("internal_source_results", {}),
            "channel_results": result.get("channel_results", {}),
            "errors": result.get("errors", {}),
            "raw_count": result.get("raw_count", 0),
            "search_audit_log": str(audit_log) if audit_log else "",
            "annotation_record": str(annotation_record) if annotation_record else "",
            "annotation_record_enabled": append_annotation_record,
            "search_enabled": True,
        }

    @staticmethod
    def _build_search_counts(
        qualified_references: list[dict],
        needs_review_references: list[dict],
        screened: dict,
        *,
        include_needs_review: bool,
    ) -> dict:
        return {
            "qualified": len(qualified_references),
            "needs_review": len(needs_review_references) if include_needs_review else 0,
            "rejected": len(screened["rejected"]),
        }

    def _search_request_from_payload(self, payload: dict) -> dict:
        handler = self.handler
        return {
            "query": str(payload.get("query", "") or "").strip(),
            "sources": str(payload.get("sources") or os.getenv("PAPER_SEARCH_DEFAULT_SOURCES") or "arxiv,pubmed,semantic").strip(),
            "max_results_per_source": handler._bounded_int(
                payload.get("max_results_per_source"),
                default=handler._bounded_int(os.getenv("PAPER_SEARCH_MAX_RESULTS_PER_SOURCE"), default=5, minimum=1, maximum=50),
                minimum=1,
                maximum=50,
            ),
            "timeout_seconds": handler._bounded_int(
                os.getenv("PAPER_SEARCH_TIMEOUT_SECONDS"),
                default=45,
                minimum=1,
                maximum=180,
            ),
            "include_needs_review": handler._truthy(payload.get("include_needs_review", True)),
            "append_annotation_record": handler._truthy(payload.get("append_annotation_record", True)),
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

    def _handle_novelty_check(self) -> None:
        handler = self.handler
        try:
            if not handler._paper_search_enabled():
                handler._send_json(
                    {
                        "error": "Academic search is not enabled. Set PAPER_SEARCH_ENABLED=true and install paper-search-mcp.",
                        "search_enabled": False,
                    },
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            payload = handler._read_json()
            request = self._novelty_request_from_payload(payload)
            if not request["innovation_text"]:
                handler._send_json({"error": "Innovation description cannot be empty."}, HTTPStatus.BAD_REQUEST)
                return

            job_id = uuid.uuid4().hex
            history_id = handler._create_history_entry(
                kind="novelty_check",
                source="novelty",
                title=request["innovation_text"],
                status="queued",
                job_id=job_id,
                request=self._novelty_history_request(request),
                counts={},
            )
            set_job_status(
                self.jobs,
                self.jobs_lock,
                job_id,
                {
                    "status": "queued",
                    "kind": "novelty_check",
                    "port": handler._server_port(),
                    "history_id": history_id,
                },
            )
            thread = threading.Thread(
                target=handler._run_novelty_check_job,
                args=(job_id, history_id, request),
                daemon=True,
            )
            thread.start()
            handler._send_json(
                {
                    "job_id": job_id,
                    "history_id": history_id,
                    "status": "queued",
                    "search_enabled": True,
                },
                HTTPStatus.ACCEPTED,
            )
        except ValueError as error:
            handler._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            traceback.print_exc()
            handler._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _run_novelty_check_job(self, job_id: str, history_id: str, request: dict) -> None:
        handler = self.handler
        port = handler._server_port()
        self._set_novelty_job(
            job_id,
            history_id,
            port,
            status="running",
            stage="Planning novelty search...",
        )
        handler._update_history_entry(history_id, status="running", stage="Planning novelty search...")
        try:
            build_novelty_plan_func = self._dependency("build_novelty_plan", build_novelty_plan)
            run_novelty_search_plan_func = self._dependency("run_novelty_search_plan", run_novelty_search_plan)
            verify_references_func = self._dependency("verify_references", verify_references)
            novelty_workflow_cls = self._dependency("NoveltyCheckWorkflow", NoveltyCheckWorkflow)

            plan = build_novelty_plan_func(request["innovation_text"], request["search_mode"])
            self._set_novelty_job(
                job_id,
                history_id,
                port,
                status="running",
                stage="Searching literature...",
            )
            handler._update_history_entry(history_id, status="running", stage="Searching literature...")
            search_payload = run_novelty_search_plan_func(
                plan,
                request["sources"],
                year=request["year"],
                max_results_per_source=request["max_results_per_source"],
                timeout_seconds=request["timeout_seconds"],
            )
            search_payload["query"] = request["innovation_text"]
            search_payload["search_mode"] = plan.get("domain") or request["search_mode"]
            search_payload["requested_search_mode"] = request["search_mode"]
            search_payload["sources_used"] = [
                source for source in str(request.get("sources") or "").split(",") if source.strip()
            ]
            references = list(search_payload.get("candidates") or [])
            if request["include_filtered_references"]:
                references.extend((search_payload.get("source_noise") or [])[:10])
            references = verify_references_func(references)
            verified_by_index = {index: reference for index, reference in enumerate(references)}
            primary_count = len(search_payload.get("candidates") or [])
            search_payload["candidates"] = [verified_by_index[index] for index in range(primary_count) if index in verified_by_index]
            if request["include_filtered_references"]:
                search_payload["source_noise"] = [
                    verified_by_index[index]
                    for index in range(primary_count, len(references))
                    if index in verified_by_index
                ]
            audit_log = handler._write_novelty_audit_log(
                innovation_text=request["innovation_text"],
                plan=plan,
                search_payload=search_payload,
                requested_sources=request["sources"],
                year=request["year"],
                port=port,
            )

            self._set_novelty_job(
                job_id,
                history_id,
                port,
                status="running",
                stage="Assessing novelty overlap...",
            )
            handler._update_history_entry(history_id, status="running", stage="Assessing novelty overlap...")
            novelty_result = asyncio.run(
                novelty_workflow_cls().run(
                    innovation_text=plan.get("clean_innovation_text") or request["innovation_text"],
                    references=references[: request["max_assessment_references"]],
                    search_payload=search_payload,
                )
            )
            novelty_result["plan"] = plan
            novelty_result["diagnostics"] = search_payload.get("diagnostics", {})
            novelty_result["search_audit_log"] = str(audit_log) if audit_log else ""
            novelty_result["source_results"] = search_payload.get("source_results", {})
            novelty_result["errors"] = search_payload.get("errors", {})
            novelty_result["raw_count"] = search_payload.get("raw_count", 0)
            novelty_result["history_id"] = history_id
            novelty_result["job_id"] = job_id
            novelty_result["port"] = port
            novelty_result["kind"] = "novelty_check"
            novelty_result["search_enabled"] = True
            done_job = self._build_novelty_done_job(novelty_result, port=port, history_id=history_id)
            handler._persist_job_log(job_id, done_job, port=port)
            set_job_status(self.jobs, self.jobs_lock, job_id, done_job)
            handler._update_history_entry(
                history_id,
                status="done",
                stage="Novelty check complete",
                result=novelty_result,
                counts=novelty_result.get("counts", {}),
            )
        except (PaperSearchError, LLMServiceError) as error:
            self._finish_novelty_check_job_with_error(job_id, history_id, port, str(error))
        except Exception as error:
            traceback.print_exc()
            self._finish_novelty_check_job_with_error(job_id, history_id, port, f"{type(error).__name__}: {error}")

    @staticmethod
    def _build_novelty_done_job(novelty_result: dict, *, port: int | str, history_id: str) -> dict:
        return {
            **novelty_result,
            "status": "done",
            "kind": "novelty_check",
            "port": port,
            "history_id": history_id,
        }

    def _set_novelty_job(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        *,
        status: str,
        stage: str,
    ) -> None:
        set_job_status(
            self.jobs,
            self.jobs_lock,
            job_id,
            {
                "status": status,
                "kind": "novelty_check",
                "port": port,
                "stage": stage,
                "history_id": history_id,
            },
        )

    def _finish_novelty_check_job_with_error(
        self,
        job_id: str,
        history_id: str,
        port: int | str,
        message: str,
    ) -> None:
        set_job_error(
            self.jobs,
            self.jobs_lock,
            job_id,
            "novelty_check",
            port,
            message,
            history_id=history_id,
        )
        self.handler._update_history_entry(history_id, status="error", error=message)

    def _novelty_request_from_payload(self, payload: dict) -> dict:
        handler = self.handler
        return {
            "innovation_text": str(payload.get("innovation_text") or payload.get("query") or "").strip(),
            "sources": str(payload.get("sources") or os.getenv("PAPER_SEARCH_DEFAULT_SOURCES") or "arxiv,pubmed,semantic").strip(),
            "search_mode": str(payload.get("search_mode") or "auto").strip().lower() or "auto",
            "year": str(payload.get("year") or "").strip(),
            "max_results_per_source": handler._bounded_int(
                payload.get("max_results_per_source"),
                default=handler._bounded_int(os.getenv("PAPER_SEARCH_MAX_RESULTS_PER_SOURCE"), default=5, minimum=1, maximum=50),
                minimum=1,
                maximum=50,
            ),
            "timeout_seconds": handler._bounded_int(
                os.getenv("PAPER_SEARCH_TIMEOUT_SECONDS"),
                default=45,
                minimum=1,
                maximum=180,
            ),
            "append_annotation_record": handler._truthy(payload.get("append_annotation_record", False)),
            "include_filtered_references": handler._truthy(payload.get("include_filtered_references", False)),
            "max_assessment_references": handler._bounded_int(
                payload.get("max_assessment_references"),
                default=handler._bounded_int(os.getenv("NOVELTY_CHECK_MAX_ASSESSMENT_REFERENCES"), default=30, minimum=1, maximum=80),
                minimum=1,
                maximum=80,
            ),
        }

    @staticmethod
    def _novelty_history_request(request: dict) -> dict:
        return {
            "innovation_text": str(request.get("innovation_text") or "").strip(),
            "sources": str(request.get("sources") or "").strip(),
            "search_mode": str(request.get("search_mode") or "auto").strip().lower() or "auto",
            "year": str(request.get("year") or "").strip(),
            "max_results_per_source": request.get("max_results_per_source") or 5,
            "include_filtered_references": bool(request.get("include_filtered_references", False)),
        }

    @staticmethod
    def _split_verified_search_candidates(
        qualified: list[dict],
        needs_review: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        final_qualified = []
        final_needs_review = list(needs_review)
        for reference in qualified:
            verification_status = str(reference.get("verification_status") or "").strip().casefold()
            if verification_status in {"needs_review", "unverified", "partial", ""}:
                reference["screening_status"] = "needs_review"
                risks = list(reference.get("screening_risks") or [])
                if verification_status == "needs_review":
                    risks.append("verification_metadata_conflict")
                elif verification_status == "unverified":
                    risks.append("verification_lookup_failed")
                elif verification_status == "partial":
                    risks.append("verification_partial")
                else:
                    risks.append("verification_missing")
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
        final_qualified, duplicate_qualified = SearchRouteService._dedupe_reference_list_for_output(
            qualified,
            seen,
        )
        final_needs_review, duplicate_needs_review = SearchRouteService._dedupe_reference_list_for_output(
            needs_review,
            seen,
        )
        duplicates = duplicate_qualified + duplicate_needs_review
        if duplicates:
            rejected = screened.setdefault("rejected", [])
            rejected.extend(duplicates)
        return final_qualified, final_needs_review

    @staticmethod
    def _apply_final_search_limits(
        qualified: list[dict],
        needs_review: list[dict],
        *,
        requested_sources,
        max_results_per_source: int,
        include_needs_review: bool,
    ) -> tuple[list[dict], list[dict]]:
        source_names = SearchRouteService._normalized_requested_sources(requested_sources)
        per_source_cap = max(1, int(max_results_per_source or 5))
        global_cap = max(1, len(source_names) or 1) * per_source_cap
        counts = {source: 0 for source in source_names}
        qualified_out = []
        for reference in qualified:
            source = SearchRouteService._reference_source_bucket(reference, source_names)
            if counts.get(source, 0) >= per_source_cap or len(qualified_out) >= global_cap:
                continue
            counts[source] = counts.get(source, 0) + 1
            qualified_out.append(reference)

        needs_review_out = []
        if include_needs_review:
            for reference in needs_review:
                if len(qualified_out) + len(needs_review_out) >= global_cap:
                    break
                source = SearchRouteService._reference_source_bucket(reference, source_names)
                if counts.get(source, 0) >= per_source_cap:
                    continue
                counts[source] = counts.get(source, 0) + 1
                needs_review_out.append(reference)
        return qualified_out, needs_review_out

    @staticmethod
    def _normalized_requested_sources(requested_sources) -> list[str]:
        if isinstance(requested_sources, (list, tuple)):
            raw_items = requested_sources
        else:
            raw_items = str(requested_sources or "").split(",")
        sources = []
        for item in raw_items:
            source = str(item or "").strip().lower().replace(" ", "")
            if source == "semantic-scholar":
                source = "semantic"
            if source and source not in sources:
                sources.append(source)
        return sources or ["arxiv", "pubmed", "semantic"]

    @staticmethod
    def _reference_source_bucket(reference: dict, source_names: list[str]) -> str:
        candidates = [
            reference.get("retrieved_from"),
            reference.get("raw_source_record", {}).get("source") if isinstance(reference.get("raw_source_record"), dict) else "",
            reference.get("source_label"),
        ]
        source_url = str(reference.get("source") or "")
        parsed = urlparse(source_url)
        host = parsed.netloc.lower()
        if host.endswith("arxiv.org"):
            candidates.append("arxiv")
        elif host.endswith("pubmed.ncbi.nlm.nih.gov") or host.endswith("ncbi.nlm.nih.gov"):
            candidates.append("pubmed")
        elif host.endswith("doi.org"):
            candidates.extend(["crossref", "doi"])
        elif "semanticscholar" in host:
            candidates.append("semantic")
        elif "openalex" in host:
            candidates.append("openalex")
        for candidate in candidates:
            source = str(candidate or "").strip().lower().replace(" ", "")
            if source in {"semanticscholar", "semantic-scholar"}:
                source = "semantic"
            if source in source_names:
                return source
        return source_names[0] if source_names else "unknown"

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
            keys = SearchRouteService._final_output_reference_keys(reference)
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
        title_key = SearchRouteService._final_output_title_key(reference.get("title"))
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
