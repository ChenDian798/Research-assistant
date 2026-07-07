from __future__ import annotations

import json
from pathlib import Path


def set_job_status(jobs: dict[str, dict[str, object]], jobs_lock, job_id: str, payload: dict) -> dict:
    with jobs_lock:
        jobs[job_id] = payload
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
