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


def test_cookie_flags_and_csrf_protection(isolated_store):
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
