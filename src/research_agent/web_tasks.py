"""Celery entry point for durable research jobs.

The HTTP process only creates a database job and sends its id to Redis.  Workers
re-load all input from PostgreSQL, so broker redelivery and process restarts do
not lose the task's state.
"""
from __future__ import annotations

import os
import threading


def _queue_mode() -> str:
    # A clean checkout can still run locally without Redis.  Production explicitly
    # sets TASK_QUEUE_MODE=celery in its environment template.
    return os.getenv("TASK_QUEUE_MODE", "thread").strip().casefold()


def _celery_app():
    try:
        from celery import Celery  # type: ignore
    except ImportError as error:  # pragma: no cover - deployment configuration guard
        raise RuntimeError("Celery is required for durable background jobs. Install requirements.txt.") from error

    broker = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND", broker)
    app = Celery("research_agent", broker=broker, backend=backend)
    app.conf.update(
        task_default_queue=os.getenv("CELERY_QUEUE", "research-jobs"),
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT_SECONDS", "1800")),
        task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT_SECONDS", "1740")),
        broker_connection_retry_on_startup=True,
        beat_schedule={
            "recover-stalled-research-jobs": {
                "task": "research_agent.recover_jobs",
                "schedule": float(os.getenv("CELERY_RECOVERY_INTERVAL_SECONDS", "60")),
            },
        },
    )
    return app


celery_app = _celery_app()


@celery_app.task(bind=True, name="research_agent.execute_job")
def execute_job_task(self, job_id: str) -> None:
    # Import lazily so a Celery worker and the HTTP process configure the exact
    # same environment-backed store without an import cycle at module load time.
    import web_app

    try:
        web_app.execute_persisted_job(job_id, task_id=self.request.id, attempt=self.request.retries + 1)
    except Exception as error:
        maximum = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        if self.request.retries < maximum:
            web_app.mark_job_for_retry(job_id, error, attempt=self.request.retries + 1)
            raise self.retry(exc=error, countdown=min(60, 2 ** self.request.retries))
        web_app.mark_job_failed(job_id, error)
        raise


@celery_app.task(name="research_agent.recover_jobs")
def recover_jobs_task() -> int:
    """Re-enqueue database jobs that survived an API or worker interruption."""
    import web_app

    stale_after = int(os.getenv("JOB_STALE_AFTER_SECONDS", "300"))
    count = 0
    for job_id in web_app.DATA_STORE.recoverable_job_ids(stale_after_seconds=stale_after):
        execute_job_task.delay(job_id)
        count += 1
    return count


def enqueue_job(job_id: str) -> str:
    """Submit a job id to Celery; thread mode is an explicit local-only fallback."""
    if _queue_mode() == "thread":
        import web_app

        def run_local() -> None:
            try:
                web_app.execute_persisted_job(job_id, task_id="local-thread", attempt=1)
            except Exception as error:
                web_app.mark_job_failed(job_id, error)

        thread = threading.Thread(target=run_local, daemon=True)
        thread.start()
        return "local-thread"
    result = execute_job_task.delay(job_id)
    return str(result.id)
