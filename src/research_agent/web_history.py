from __future__ import annotations

import json
from pathlib import Path


def read_history_data(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "items": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
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


def write_history_data(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        temp_path.replace(path)
    except OSError as error:
        print(f"[web] failed to write history file: {error}", flush=True)


def history_entry_summary(entry: dict) -> dict:
    summary_keys = {
        "id",
        "kind",
        "source",
        "title",
        "status",
        "created_at",
        "updated_at",
        "job_id",
        "stage",
        "error",
        "counts",
    }
    summary = {key: entry.get(key) for key in summary_keys if key in entry}
    summary["is_summary"] = True
    request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
    request_summary = history_request_summary(request)
    if request_summary:
        summary["request"] = request_summary
    analysis = entry.get("analysis") if isinstance(entry.get("analysis"), dict) else None
    if analysis:
        summary["analysis"] = history_analysis_summary(analysis)
    return summary


def history_request_summary(request: dict) -> dict:
    summary_keys = {
        "query",
        "topic",
        "search_mode",
        "sources",
        "year",
        "reference_count",
        "file_count",
        "output_language",
        "innovation_text",
    }
    return {key: request.get(key) for key in summary_keys if key in request}


def history_analysis_summary(analysis: dict) -> dict:
    summary_keys = {
        "id",
        "status",
        "stage",
        "job_id",
        "error",
        "counts",
    }
    return {key: analysis.get(key) for key in summary_keys if key in analysis}


def history_references(references: list[dict]) -> list[dict]:
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


def apply_history_entry_update(
    entry: dict,
    now: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    result: dict | None = None,
    counts: dict | None = None,
    error: str | None = None,
) -> None:
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


def apply_history_analysis_update(
    entry: dict,
    now: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    result: dict | None = None,
    counts: dict | None = None,
    error: str | None = None,
) -> None:
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
