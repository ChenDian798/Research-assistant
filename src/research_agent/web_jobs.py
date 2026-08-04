from __future__ import annotations

import json
from pathlib import Path

from src.research_agent.web_security import CURRENT_USER_ID, active_store


def set_job_status(jobs: dict[str, dict[str, object]], jobs_lock, job_id: str, payload: dict) -> dict:
    """Persist a state transition before updating the compatibility cache.

    `jobs` is intentionally only a short-lived cache for older in-process callers;
    API reads and worker recovery use the database record exclusively.
    """
    previous: dict[str, object] = {}
    with jobs_lock:
        previous = dict(jobs.get(job_id, {}))
        owner_user_id = str(payload.get("owner_user_id") or previous.get("owner_user_id") or CURRENT_USER_ID.get() or "")
        if owner_user_id:
            payload = {**payload, "owner_user_id": owner_user_id}
        jobs[job_id] = payload
    store = active_store()
    if store and owner_user_id and store.user_exists(owner_user_id):
        payload = store.save_job(owner_user_id, job_id, payload)
    return payload


def set_job_error(
    jobs: dict[str, dict[str, object]],
    jobs_lock,
    job_id: str,
    kind: str,
    port: int | str,
    message: str,
    **extra,
) -> dict:
    payload = {
        "status": "error",
        "kind": kind,
        "port": port,
        "error": message,
        **extra,
    }
    return set_job_status(jobs, jobs_lock, job_id, payload)


def persist_job_log(log_dir: Path, job_id: str, job: dict, *, port: int | str = "") -> None:
    """Write a diagnostic snapshot only; it is never used to restore job state."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        port_part = f"_port{port}" if str(port or "").strip() else ""
        (log_dir / f"last_job{port_part}_{job_id}.json").write_text(
            json.dumps(job, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as error:
        print(f"[web] failed to write last job log: {error}", flush=True)


def load_persisted_job_log(log_dir: Path, job_id: str) -> dict | None:
    """Deprecated compatibility helper for pre-database job logs."""
    try:
        matches = sorted(
            log_dir.glob(f"last_job*_{job_id}.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
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
