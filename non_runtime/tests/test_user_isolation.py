from __future__ import annotations

import json
from pathlib import Path

import pytest

import web_app
from src.research_agent.web_security import UserDataStore, configure_store


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


def test_admin_created_user_can_login(isolated_store):
    admin = isolated_store.create_local_user("admin-create@example.test", "password-123", "Admin", role="admin")
    user = isolated_store.create_local_user("invited@example.test", "initial-123", "Invited")
    isolated_store.audit(admin["id"], "user.admin_created", "user", user["id"])

    authenticated = isolated_store.authenticate_local_user("invited@example.test", "initial-123")
    assert authenticated["id"] == user["id"]
    assert authenticated["status"] == "active"


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
