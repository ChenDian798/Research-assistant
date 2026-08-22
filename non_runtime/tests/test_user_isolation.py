from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import web_app
from src.research_agent.web_security import ObjectStorage, UserDataStore, configure_store, decode_json_value


@pytest.fixture
def isolated_store(monkeypatch, tmp_path: Path):
    store = UserDataStore(tmp_path, f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setattr(web_app, "DATA_STORE", store)
    configure_store(store)
    return store


def _handler(path: str, cookie: str = "", csrf: str = ""):
    handler = object.__new__(web_app.ResearchWebHandler)
    handler.path = path
    handler.headers = {"Cookie": cookie, "X-CSRF-Token": csrf}
    sent = {}
    handler._send_json = lambda payload, status=200: sent.update(payload=payload, status=status)
    return handler, sent


def test_postgres_jsonb_values_are_accepted_without_redecoding() -> None:
    payload = {"status": "queued", "request": {"topic": "stroke"}}

    assert decode_json_value(payload, {}) is payload
    assert decode_json_value('{"status": "queued"}', {}) == {"status": "queued"}


def test_claiming_retry_clears_stale_terminal_metadata(isolated_store) -> None:
    user = isolated_store.provision_user("retry-user")
    job, _created = isolated_store.create_job(user["id"], "literature_analysis", {})
    isolated_store.save_job(
        user["id"],
        job["id"],
        {
            "status": "queued",
            "stage": "Retry scheduled",
            "error": "previous attempt failed",
            "finished_at": "2026-01-01T00:00:00+00:00",
        },
    )

    claimed = isolated_store.claim_job(job["id"], task_id="retry-task", attempt=2)

    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["stage"] == "Worker accepted job"
    assert claimed["progress"] == 1
    assert claimed["error"] is None
    assert claimed["finished_at"] is None
    assert claimed["started_at"] is not None


def test_minio_upload_does_not_request_unconfigured_server_side_encryption(monkeypatch, tmp_path: Path) -> None:
    calls = []
    client = SimpleNamespace(put_object=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *_args, **_kwargs: client))
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET", "private-bucket")
    monkeypatch.delenv("S3_SERVER_SIDE_ENCRYPTION", raising=False)
    monkeypatch.delenv("S3_SSE_KMS_KEY_ID", raising=False)

    storage = ObjectStorage(tmp_path)
    storage.put("user", "uploads", "paper.pdf", b"pdf")

    assert calls[0]["Bucket"] == "private-bucket"
    assert "ServerSideEncryption" not in calls[0]
    assert "SSEKMSKeyId" not in calls[0]


def _user_session(store: UserDataStore, subject: str):
    user = store.provision_user(subject, f"{subject}@example.test", subject)
    token, csrf = store.create_session(user["id"])
    return user, token, csrf


def test_unauthenticated_history_is_rejected(isolated_store):
    handler, sent = _handler("/api/history")
    handler.do_GET()
    assert sent["status"] == 401


def test_users_cannot_read_delete_or_poll_each_others_resources(isolated_store):
    user_a, token_a, csrf_a = _user_session(isolated_store, "oidc-a")
    user_b, token_b, csrf_b = _user_session(isolated_store, "oidc-b")
    history_id = "history-a"
    isolated_store.create_history(user_a["id"], {"id": history_id, "status": "done", "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"})
    isolated_store.save_job(user_a["id"], "job-a", {"id": "job-a", "status": "running"})

    handler, sent = _handler(f"/api/history/{history_id}", f"research_session={token_b}")
    handler.do_GET()
    assert sent["status"] == 404

    handler, sent = _handler(f"/api/history/{history_id}", f"research_session={token_b}", csrf_b)
    handler.do_DELETE()
    assert sent["status"] == 404

    handler, sent = _handler("/api/literature-analysis/job-a", f"research_session={token_b}")
    handler.do_GET()
    assert sent["status"] == 404

    handler, sent = _handler(f"/api/history/{history_id}", f"research_session={token_a}")
    handler.do_GET()
    assert sent["status"] == 200
    assert sent["payload"]["owner_user_id"] == user_a["id"]


def test_cookie_flags_and_csrf_protection(isolated_store, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
    _user, token, csrf = _user_session(isolated_store, "oidc-a")
    handler, sent = _handler("/api/export/pdf", f"research_session={token}")
    handler.do_POST()
    assert sent["status"] == 403

    captured = []
    handler.send_header = lambda name, value: captured.append((name, value))
    handler._set_session_cookie("opaque")
    cookie = dict(captured)["Set-Cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie

    handler, _sent = _handler("/api/history", f"research_session={token}", csrf)
    assert handler._require_csrf(handler._authenticated_user()) is True


def test_legacy_history_migration_retains_records_and_backup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LEGACY_ADMIN_OIDC_SUBJECT", "legacy-admin")
    history = tmp_path / "history_records.json"
    history.write_text(json.dumps({"items": [{"id": "one", "title": "first"}, {"id": "two", "title": "second"}]}), encoding="utf-8")
    store = UserDataStore(tmp_path, f"sqlite:///{tmp_path / 'migration.db'}")
    store.migrate_legacy_history(history)
    admin = store.provision_user("legacy-admin")
    assert {item["id"] for item in store.histories(admin["id"], 10)} == {"one", "two"}
    assert history.with_suffix(".json.pre-db-migration.bak").read_text(encoding="utf-8") == history.read_text(encoding="utf-8")


def test_user_deletion_removes_owned_records_jobs_and_files(isolated_store):
    user, _token, _csrf = _user_session(isolated_store, "oidc-delete")
    isolated_store.create_history(user["id"], {"id": "history-delete", "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"})
    isolated_store.save_job(user["id"], "job-delete", {"status": "done"})
    file_path = isolated_store.store_file(user["id"], "uploads", "paper.pdf", b"pdf", 30)
    isolated_store.delete_user(user["id"])
    assert isolated_store.history(user["id"], "history-delete") is None
    assert isolated_store.job(user["id"], "job-delete") is None
    assert not file_path.exists()


def test_local_user_login_and_disabled_status(isolated_store):
    user = isolated_store.create_local_user("local@example.test", "password-123", "Local User")
    authenticated = isolated_store.authenticate_local_user("LOCAL@example.test", "password-123")
    assert authenticated["id"] == user["id"]
    assert authenticated["role"] == "user"

    assert isolated_store.authenticate_local_user("local@example.test", "wrong-password") is None
    isolated_store.update_user_admin(user["id"], user["id"], status="disabled")
    with pytest.raises(PermissionError):
        isolated_store.authenticate_local_user("local@example.test", "password-123")


def test_admin_user_management_and_last_admin_guard(isolated_store):
    admin = isolated_store.create_local_user("admin@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("user@example.test", "password-123", "User")

    updated = isolated_store.update_user_admin(admin["id"], user["id"], role="admin")
    assert updated["role"] == "admin"
    assert isolated_store.admin_user_count() == 2

    isolated_store.update_user_admin(admin["id"], user["id"], role="user")
    with pytest.raises(ValueError):
        isolated_store.update_user_admin(admin["id"], admin["id"], role="user")


def test_admin_history_requires_admin_and_writes_detail_audit(isolated_store):
    admin = isolated_store.create_local_user("history-admin@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("history-user@example.test", "password-123", "History User")
    other = isolated_store.create_local_user("other-user@example.test", "password-123", "Other User")
    admin_token, _admin_csrf = isolated_store.create_session(admin["id"])
    user_token, _user_csrf = isolated_store.create_session(user["id"])
    isolated_store.create_history(
        user["id"],
        {
            "id": "history-owned",
            "kind": "literature_search",
            "status": "done",
            "title": "Stroke segmentation",
            "job_id": "job-owned",
            "request": {"query": "stroke segmentation", "sources": "pubmed", "search_mode": "auto", "year": "2026"},
            "counts": {"candidates": 12, "qualified": 3, "needs_review": 2, "rejected": 7},
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        },
    )
    isolated_store.create_job(user["id"], "literature_search", {}, job_id="job-owned")
    isolated_store.save_job(
        user["id"],
        "job-owned",
        {
            "status": "done",
            "started_at": "2026-01-02T00:00:02+00:00",
            "finished_at": "2026-01-02T00:00:12+00:00",
        },
    )
    isolated_store.create_history(
        other["id"],
        {
            "id": "history-other",
            "kind": "novelty_check",
            "status": "error",
            "title": "Other topic",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-03T00:00:00+00:00",
        },
    )

    handler, sent = _handler("/api/admin/history")
    handler.do_GET()
    assert sent["status"] == 401

    handler, sent = _handler("/api/admin/history", f"research_session={user_token}")
    handler.do_GET()
    assert sent["status"] == 403

    handler, sent = _handler(f"/api/admin/history?owner_user_id={user['id']}", f"research_session={admin_token}")
    handler.do_GET()
    assert sent["status"] == 200
    assert [item["id"] for item in sent["payload"]["history"]] == ["history-owned"]
    item = sent["payload"]["history"][0]
    assert item["owner_user_id"] == user["id"]
    assert item["owner_email"] == "history-user@example.test"
    assert item["request"]["query"] == "stroke segmentation"
    assert item["counts"]["qualified"] == 3

    handler, sent = _handler("/api/admin/history/history-owned", f"research_session={admin_token}")
    handler.do_GET()
    assert sent["status"] == 200
    assert sent["payload"]["owner_user_id"] == user["id"]
    assert sent["payload"]["owner_email"] == "history-user@example.test"
    assert sent["payload"]["admin_summary"]["counts"]["qualified"] == 3
    assert sent["payload"]["admin_summary"]["queue_seconds"] is not None
    assert sent["payload"]["admin_summary"]["run_seconds"] == 10
    audit = isolated_store._fetchone(
        "SELECT action, actor_user_id, resource_type, resource_id, detail FROM audit_logs WHERE action=? AND resource_id=?",
        ("history.admin_viewed", "history-owned"),
    )
    assert audit is not None
    assert audit["actor_user_id"] == admin["id"]
    assert audit["resource_type"] == "history"
    assert json.loads(audit["detail"])["owner_user_id"] == user["id"]


def test_admin_history_limit_is_capped_at_300(isolated_store):
    admin = isolated_store.create_local_user("limit-admin@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("limit-user@example.test", "password-123", "Limit User")
    admin_token, _csrf = isolated_store.create_session(admin["id"])
    for index in range(305):
        isolated_store.create_history(
            user["id"],
            {
                "id": f"history-limit-{index:03d}",
                "kind": "literature_search",
                "status": "done",
                "title": f"History {index}",
                "created_at": f"2026-01-01T00:00:{index % 60:02d}+00:00",
                "updated_at": f"2026-01-01T00:00:{index % 60:02d}+00:00",
            },
        )

    handler, sent = _handler("/api/admin/history?limit=999", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    assert len(sent["payload"]["history"]) == 300


def test_admin_light_metrics_aggregates_by_user_and_rejects_non_admin(isolated_store):
    admin = isolated_store.create_local_user("metrics-admin@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("metrics-user@example.test", "password-123", "Metrics User")
    other = isolated_store.create_local_user("metrics-other@example.test", "password-123", "Other User")
    disabled = isolated_store.create_local_user("metrics-disabled@example.test", "password-123", "Disabled")
    isolated_store.update_user_admin(admin["id"], disabled["id"], status="disabled")
    admin_token, _admin_csrf = isolated_store.create_session(admin["id"])
    user_token, _user_csrf = isolated_store.create_session(user["id"])
    now = datetime.now(timezone.utc)
    recent = now.isoformat(timespec="seconds")
    older = (now - timedelta(days=2)).isoformat(timespec="seconds")

    isolated_store.create_history(
        user["id"],
        {
            "id": "metrics-history-recent",
            "kind": "literature_search",
            "status": "done",
            "title": "Recent search",
            "request": {"query": "recent"},
            "counts": {"candidates": 10, "qualified": 4, "needs_review": 1, "rejected": 5},
            "created_at": recent,
            "updated_at": recent,
        },
    )
    isolated_store.create_history(
        user["id"],
        {
            "id": "metrics-history-older",
            "kind": "direct_analysis",
            "status": "done",
            "title": "Older analysis",
            "counts": {"references": 2},
            "created_at": older,
            "updated_at": older,
        },
    )
    isolated_store.create_history(
        other["id"],
        {
            "id": "metrics-history-error",
            "kind": "novelty_check",
            "status": "error",
            "title": "Failed novelty",
            "error": "timeout",
            "counts": {"raw_count": 6, "rejected_count": 6},
            "created_at": recent,
            "updated_at": recent,
        },
    )
    done_job, _created = isolated_store.create_job(user["id"], "literature_search", {})
    isolated_store.save_job(user["id"], done_job["id"], {"status": "done"})
    error_job, _created = isolated_store.create_job(other["id"], "novelty_check", {})
    isolated_store.save_job(other["id"], error_job["id"], {"status": "error", "error": "timeout"})
    running_job, _created = isolated_store.create_job(user["id"], "literature_analysis", {})
    isolated_store.save_job(user["id"], running_job["id"], {"status": "running"})
    isolated_store.store_file(user["id"], "uploads", "paper.pdf", b"%PDF-1.7\n", 30)

    handler, sent = _handler("/api/admin/metrics?days=7", f"research_session={user_token}")
    handler.do_GET()
    assert sent["status"] == 403

    handler, sent = _handler("/api/admin/metrics?days=7", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    payload = sent["payload"]
    assert payload["range_days"] == 7
    assert payload["users"] == {"total": 4, "active": 3, "disabled": 1}
    assert payload["usage"]["history_total"] == 3
    assert payload["usage"]["literature_search"] == 1
    assert payload["usage"]["direct_analysis"] == 1
    assert payload["usage"]["novelty_check"] == 1
    assert payload["jobs"]["done"] == 1
    assert payload["jobs"]["error"] == 1
    assert payload["jobs"]["running"] == 1
    assert payload["storage"]["file_count"] == 1
    assert payload["quality"]["candidate_total"] == 18
    assert payload["quality"]["qualified_total"] == 4
    assert payload["quality"]["needs_review_total"] == 1
    assert payload["quality"]["rejected_total"] == 11
    by_owner = {item["owner_user_id"]: item for item in payload["users_usage"]}
    assert by_owner[user["id"]]["email"] == "metrics-user@example.test"
    assert by_owner[user["id"]]["history_total"] == 2
    assert by_owner[user["id"]]["job_done"] == 1
    assert by_owner[user["id"]]["file_count"] == 1
    assert by_owner[other["id"]]["job_error"] == 1
    assert payload["recent_errors"][0]["history_id"] == "metrics-history-error"
    assert payload["recent_errors"][0]["owner_user_id"] == other["id"]
    assert payload["recent_errors"][0]["user"] == "metrics-other@example.test"

    handler, sent = _handler("/api/admin/metrics?days=1", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    assert sent["payload"]["range_days"] == 1
    assert sent["payload"]["usage"]["history_total"] == 2
    assert sent["payload"]["usage"]["direct_analysis"] == 0


def test_admin_light_metrics_empty_data_returns_zeroes(isolated_store):
    admin = isolated_store.create_local_user("empty-metrics-admin@example.test", "password-123", "Admin", role="admin")
    admin_token, _csrf = isolated_store.create_session(admin["id"])

    handler, sent = _handler("/api/admin/metrics?days=30", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    assert sent["payload"]["range_days"] == 30
    assert sent["payload"]["users"]["total"] == 1
    assert sent["payload"]["usage"]["history_total"] == 0
    assert sent["payload"]["jobs"]["done"] == 0
    assert sent["payload"]["storage"]["file_count"] == 0
    assert sent["payload"]["users_usage"] == []
    assert sent["payload"]["recent_errors"] == []


def test_admin_created_user_can_login(isolated_store):
    admin = isolated_store.create_local_user("admin-create@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("invited@example.test", "initial-123", "Invited")
    isolated_store.audit(admin["id"], "user.admin_created", "user", user["id"])

    authenticated = isolated_store.authenticate_local_user("invited@example.test", "initial-123")
    assert authenticated["id"] == user["id"]
    assert authenticated["status"] == "active"


def test_admin_audit_logs_endpoint_and_sensitive_user_actions(isolated_store, monkeypatch):
    admin = isolated_store.create_local_user("audit-admin@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("audit-user@example.test", "password-123", "User")
    admin_token, admin_csrf = isolated_store.create_session(admin["id"])
    user_token, _user_csrf = isolated_store.create_session(user["id"])

    handler, sent = _handler("/api/admin/audit-logs", f"research_session={user_token}")
    handler.do_GET()
    assert sent["status"] == 403

    handler, sent = _handler("/api/admin/users", f"research_session={admin_token}", admin_csrf)
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {
            "email": "created-by-admin@example.test",
            "password": "password-123",
            "display_name": "Created",
            "role": "user",
        },
    )
    handler.do_POST()
    assert sent["status"] == 201
    created_user_id = sent["payload"]["user"]["id"]

    handler, sent = _handler(f"/api/admin/users/{created_user_id}", f"research_session={admin_token}", admin_csrf)
    monkeypatch.setattr(handler, "_read_json", lambda: {"status": "disabled"})
    handler.do_PATCH()
    assert sent["status"] == 200

    handler, sent = _handler(f"/api/admin/users/{created_user_id}", f"research_session={admin_token}", admin_csrf)
    handler.do_DELETE()
    assert sent["status"] == 200

    handler, sent = _handler("/api/admin/audit-logs?limit=30", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    logs = sent["payload"]["audit_logs"]
    assert len(logs) <= 30
    actions = [item["action"] for item in logs]
    assert "user.admin_created" in actions
    assert "user.admin_updated" in actions
    assert "user.deleted" in actions
    admin_delete_log = next(item for item in logs if item["action"] == "user.deleted" and item["actor_user_id"] == admin["id"])
    assert admin_delete_log["resource_id"] == created_user_id
    assert admin_delete_log["actor_email"] == "audit-admin@example.test"
    assert admin_delete_log["detail"]["source"] == "admin"

    isolated_store.audit(user["id"], "history.created", "history", "user-history")
    handler, sent = _handler(f"/api/admin/audit-logs?limit=30&actor_user_id={user['id']}", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    filtered_logs = sent["payload"]["audit_logs"]
    assert filtered_logs
    assert {item["actor_user_id"] for item in filtered_logs} == {user["id"]}
    assert filtered_logs[0]["actor_email"] == "audit-user@example.test"


def test_admin_audit_logs_limit_defaults_to_recent_30(isolated_store):
    admin = isolated_store.create_local_user("audit-limit-admin@example.test", "password-123", "Admin", role="admin")
    admin_token, _csrf = isolated_store.create_session(admin["id"])
    for index in range(35):
        isolated_store.audit(admin["id"], f"audit.event_{index:02d}", "test", str(index))

    handler, sent = _handler("/api/admin/audit-logs?limit=999", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    assert 30 < len(sent["payload"]["audit_logs"]) <= 100

    handler, sent = _handler("/api/admin/audit-logs", f"research_session={admin_token}")
    handler.do_GET()

    assert sent["status"] == 200
    assert len(sent["payload"]["audit_logs"]) == 30


def test_beta_retention_and_server_protection_defaults_are_explicit(monkeypatch):
    monkeypatch.delenv("DATA_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("UPLOAD_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("EXPORT_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("PAPER_SEARCH_MAX_RESULTS_PER_SOURCE_CAP", "5")

    policy = web_app.ResearchWebHandler._upload_security_policy()
    assert policy.max_total_mb == 30
    assert policy.max_file_mb == 15
    assert policy.max_file_count == 4
    assert web_app.ResearchWebHandler._search_result_cap() == 5
    env_text = Path(".env.example").read_text(encoding="utf-8")
    assert "DATA_RETENTION_DAYS=30" in env_text
    assert "UPLOAD_RETENTION_DAYS=14" in env_text
    assert "EXPORT_RETENTION_DAYS=7" in env_text
    assert "PAPER_SEARCH_MAX_RESULTS_PER_SOURCE_CAP=5" in env_text


def test_evaluation_run_persists_items_and_stage_events(isolated_store):
    admin = isolated_store.create_local_user("eval-admin@example.test", "password-123", "Eval Admin", role="admin")
    run_id = isolated_store.create_evaluation_run(
        admin["id"],
        "smoke evaluation",
        "literature_search",
        {"sources": "arxiv", "concurrency": 1},
        ["query one", "query two"],
    )
    items = isolated_store.evaluation_items_for_run(run_id)
    assert len(items) == 2

    isolated_store.update_evaluation_item(items[0]["id"], status="done", result={"timings": {"planning_seconds": 0.1}}, total_duration_ms=150, started=True, finished=True)
    isolated_store.add_evaluation_stage_event(items[0]["id"], "planning", duration_ms=100, output_count=2)
    isolated_store.update_evaluation_run(run_id, status="running")
    detail = isolated_store.evaluation_run_detail(admin["id"], run_id)

    assert detail["completed_count"] == 1
    assert detail["items"][0]["status"] == "done"
    assert detail["items"][0]["events"][0]["stage"] == "planning"
