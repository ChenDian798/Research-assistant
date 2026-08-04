"""OIDC-backed sessions and user-owned persistence for the web application.

The module deliberately contains no password handling.  Identities originate only
from an OpenID Connect provider, while this service owns short-lived opaque
server-side sessions.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


CURRENT_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user_id", default=None)
_STORE: "UserDataStore | None" = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_store(store: "UserDataStore") -> None:
    global _STORE
    _STORE = store


def active_store() -> "UserDataStore | None":
    return _STORE


class ObjectStorage:
    """Private object storage with a deliberately local-only development fallback."""

    def __init__(self, root: Path) -> None:
        self.mode = os.getenv("OBJECT_STORAGE_BACKEND", "local").strip().casefold()
        self.root = root
        self.bucket = os.getenv("S3_BUCKET", "")
        self._client = None
        if self.mode == "s3":
            if not self.bucket:
                raise RuntimeError("S3_BUCKET is required when OBJECT_STORAGE_BACKEND=s3")
            try:
                import boto3  # type: ignore
            except ImportError as error:  # pragma: no cover - deployment guard
                raise RuntimeError("boto3 is required for S3/MinIO object storage") from error
            self._client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL") or None)

    def put(self, owner: str, kind: str, filename: str, content: bytes) -> str:
        key = f"private/{owner}/{kind}/{uuid.uuid4().hex}_{Path(filename).name or 'file'}"
        if self.mode == "s3":
            self._client.put_object(Bucket=self.bucket, Key=key, Body=content, ServerSideEncryption="AES256")
            return f"s3://{self.bucket}/{key}"
        target = (self.root / owner / kind / Path(key).name).resolve()
        if self.root not in target.parents:
            raise ValueError("Invalid user storage path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    def delete(self, object_key: str) -> None:
        if object_key.startswith("s3://"):
            _, _, bucket_and_key = object_key.partition("s3://")
            bucket, _, key = bucket_and_key.partition("/")
            if bucket and key and self._client:
                self._client.delete_object(Bucket=bucket, Key=key)
            return
        Path(object_key).unlink(missing_ok=True)

    def signed_download_url(self, object_key: str, expires_seconds: int = 300) -> str:
        if not object_key.startswith("s3://") or not self._client:
            raise RuntimeError("Signed downloads require S3/MinIO object storage")
        _, _, bucket_and_key = object_key.partition("s3://")
        bucket, _, key = bucket_and_key.partition("/")
        return self._client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_seconds)


class _ClosingConnection:
    """DB-API context wrapper that also releases SQLite file handles on Windows."""

    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        self.connection.__enter__()
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        try:
            return self.connection.__exit__(exc_type, exc, tb)
        finally:
            self.connection.close()


class UserDataStore:
    """Small DB-API data layer supporting SQLite and PostgreSQL (psycopg)."""

    def __init__(self, root: Path, database_url: str | None = None) -> None:
        self.root = root
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.file_root = Path(os.getenv("USER_FILE_ROOT", str(root / "user_data"))).resolve()
        self.file_root.mkdir(parents=True, exist_ok=True)
        self.object_storage = ObjectStorage(self.file_root)
        self._postgres = self.database_url.startswith(("postgres://", "postgresql://"))
        self._sqlite_path = Path(self.database_url.removeprefix("sqlite:///") or (root / "data" / "research.db"))
        if not self._postgres:
            self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self):
        if not self._postgres:
            connection = sqlite3.connect(self._sqlite_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return _ClosingConnection(connection)
        try:
            import psycopg  # type: ignore
            return _ClosingConnection(psycopg.connect(self.database_url))
        except ImportError:
            try:
                import psycopg2  # type: ignore
                return _ClosingConnection(psycopg2.connect(self.database_url))
            except ImportError as error:
                raise RuntimeError("PostgreSQL requires psycopg or psycopg2; install it in the production image.") from error

    def _execute(self, connection, sql: str, params: tuple = ()):
        if self._postgres:
            sql = sql.replace("?", "%s")
        cursor = connection.cursor()
        cursor.execute(sql, params)
        return cursor

    def initialize(self) -> None:
        identity = "TEXT" if not self._postgres else "TEXT"
        json_type = "JSONB" if self._postgres else "TEXT"
        with self._connect() as connection:
            for statement in (
                f"CREATE TABLE IF NOT EXISTS users (id {identity} PRIMARY KEY, oidc_subject {identity} UNIQUE NOT NULL, email {identity}, display_name {identity}, created_at {identity} NOT NULL, deleted_at {identity})",
                f"CREATE TABLE IF NOT EXISTS sessions (id {identity} PRIMARY KEY, user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, token_hash {identity} UNIQUE NOT NULL, csrf_token {identity} NOT NULL, expires_at {identity} NOT NULL, created_at {identity} NOT NULL)",
                f"CREATE TABLE IF NOT EXISTS histories (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, payload {identity} NOT NULL, created_at {identity} NOT NULL, updated_at {identity} NOT NULL)",
                f"CREATE INDEX IF NOT EXISTS histories_owner_updated ON histories(owner_user_id, updated_at DESC)",
                f"CREATE TABLE IF NOT EXISTS jobs (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, payload {identity} NOT NULL, created_at {identity} NOT NULL, updated_at {identity} NOT NULL)",
                f"CREATE INDEX IF NOT EXISTS jobs_owner_updated ON jobs(owner_user_id, updated_at DESC)",
                f"CREATE TABLE IF NOT EXISTS files (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, kind {identity} NOT NULL, storage_path {identity} NOT NULL, original_name {identity}, created_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE TABLE IF NOT EXISTS audit_log (id {identity} PRIMARY KEY, actor_user_id {identity}, action {identity} NOT NULL, resource_type {identity} NOT NULL, resource_id {identity}, detail {identity}, created_at {identity} NOT NULL)",
                f"CREATE TABLE IF NOT EXISTS schema_migrations (name {identity} PRIMARY KEY, applied_at {identity} NOT NULL)",
                # New tables deliberately keep their public names.  The old `histories`
                # and `files` tables are retained only so upgrades from the previous
                # local-development build are non-destructive.
                f"CREATE TABLE IF NOT EXISTS history_entries (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, job_id {identity}, kind {identity} NOT NULL, status {identity} NOT NULL, title {identity}, payload {json_type} NOT NULL, created_at {identity} NOT NULL, updated_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE INDEX IF NOT EXISTS history_entries_owner_updated ON history_entries(owner_user_id, updated_at DESC)",
                f"CREATE TABLE IF NOT EXISTS job_events (id {identity} PRIMARY KEY, job_id {identity} NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, status {identity}, stage {identity}, progress INTEGER, detail {json_type}, created_at {identity} NOT NULL)",
                f"CREATE INDEX IF NOT EXISTS job_events_job_created ON job_events(job_id, created_at)",
                f"CREATE TABLE IF NOT EXISTS documents (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, original_name {identity} NOT NULL, content_type {identity}, sha256 {identity} NOT NULL, object_key {identity} NOT NULL, scan_status {identity} NOT NULL, created_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE INDEX IF NOT EXISTS documents_owner_created ON documents(owner_user_id, created_at DESC)",
                f"CREATE TABLE IF NOT EXISTS artifacts (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, job_id {identity} REFERENCES jobs(id) ON DELETE SET NULL, kind {identity} NOT NULL, object_key {identity} NOT NULL, content_type {identity}, created_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE TABLE IF NOT EXISTS audit_logs (id {identity} PRIMARY KEY, actor_user_id {identity}, action {identity} NOT NULL, resource_type {identity} NOT NULL, resource_id {identity}, detail {json_type}, created_at {identity} NOT NULL)",
            ):
                self._execute(connection, statement)
            self._ensure_job_columns(connection, identity, json_type)
            self._migrate_legacy_tables(connection)
            connection.commit()

    def _ensure_job_columns(self, connection, identity: str, json_type: str) -> None:
        """Add explicit job fields without invalidating existing SQLite databases."""
        columns = {
            "kind": identity,
            "status": identity,
            "stage": identity,
            "progress": "INTEGER",
            "request_json": json_type,
            "result_json": json_type,
            "error": identity,
            "idempotency_key": identity,
            "task_id": identity,
            "attempt": "INTEGER NOT NULL DEFAULT 0",
            "started_at": identity,
            "finished_at": identity,
            "expires_at": identity,
        }
        if self._postgres:
            for name, definition in columns.items():
                self._execute(connection, f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {name} {definition}")
        else:
            existing = {row[1] for row in self._execute(connection, "PRAGMA table_info(jobs)").fetchall()}
            for name, definition in columns.items():
                if name not in existing:
                    self._execute(connection, f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
        self._execute(connection, "CREATE INDEX IF NOT EXISTS jobs_owner_status ON jobs(owner_user_id, status, updated_at DESC)")
        # NULL keys are intentionally allowed: clients that do not send an idempotency
        # key still receive independent jobs.
        self._execute(connection, "CREATE UNIQUE INDEX IF NOT EXISTS jobs_owner_idempotency ON jobs(owner_user_id, idempotency_key)")

    def _migrate_legacy_tables(self, connection) -> None:
        """Copy previous DB-backed history once; this is separate from JSON import."""
        marker = "legacy_histories_table_v1"
        seen = self._execute(connection, "SELECT name FROM schema_migrations WHERE name=?", (marker,)).fetchone()
        if seen:
            return
        rows = self._execute(connection, "SELECT id, owner_user_id, payload, created_at, updated_at FROM histories").fetchall()
        for row in rows:
            history_id, owner, payload, created_at, updated_at = row[0], row[1], row[2], row[3], row[4]
            entry = json.loads(payload)
            self._execute(
                connection,
                "INSERT INTO history_entries (id, owner_user_id, job_id, kind, status, title, payload, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (history_id, owner, entry.get("job_id"), entry.get("kind", "legacy"), entry.get("status", "done"), entry.get("title", ""), json.dumps(entry, ensure_ascii=False), created_at, updated_at, None),
            )
        self._execute(connection, "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)", (marker, utc_now()))

    def _row(self, row):
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            return dict(row)
        columns = [description[0] for description in getattr(row, "cursor_description", [])]
        return dict(zip(columns, row)) if columns else row

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._connect() as connection:
            cursor = self._execute(connection, sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            if isinstance(row, sqlite3.Row):
                return dict(row)
            return dict(zip([item[0] for item in cursor.description], row))

    def audit(self, actor: str | None, action: str, resource_type: str, resource_id: str = "", detail: dict | None = None) -> None:
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO audit_logs (id, actor_user_id, action, resource_type, resource_id, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, actor, action, resource_type, resource_id, json.dumps(detail or {}, ensure_ascii=False), utc_now()))
            connection.commit()

    def provision_user(self, subject: str, email: str = "", display_name: str = "") -> dict:
        existing = self._fetchone("SELECT * FROM users WHERE oidc_subject = ? AND deleted_at IS NULL", (subject,))
        if existing:
            return existing
        user = {"id": uuid.uuid4().hex, "oidc_subject": subject, "email": email, "display_name": display_name, "created_at": utc_now()}
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO users (id, oidc_subject, email, display_name, created_at, deleted_at) VALUES (?, ?, ?, ?, ?, NULL)", tuple(user.values()))
            connection.commit()
        self.audit(user["id"], "user.provisioned", "user", user["id"])
        return user

    def user_exists(self, user_id: str | None) -> bool:
        return bool(user_id and self._fetchone("SELECT id FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)))

    def create_session(self, user_id: str) -> tuple[str, str]:
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = datetime.fromtimestamp(time.time() + int(os.getenv("SESSION_TTL_SECONDS", "28800")), timezone.utc).isoformat(timespec="seconds")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO sessions (id, user_id, token_hash, csrf_token, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, user_id, digest, csrf, expires, utc_now()))
            connection.commit()
        return token, csrf

    def session_user(self, token: str | None) -> dict | None:
        if not token:
            return None
        row = self._fetchone("SELECT s.csrf_token, s.expires_at, u.id, u.email, u.display_name FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND u.deleted_at IS NULL", (hashlib.sha256(token.encode()).hexdigest(),))
        if not row or row["expires_at"] <= utc_now():
            return None
        return row

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as connection:
            self._execute(connection, "DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))
            connection.commit()

    def create_history(self, owner: str, entry: dict) -> str:
        history_id, now = entry["id"], utc_now()
        entry = {**entry, "owner_user_id": owner}
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO history_entries (id, owner_user_id, job_id, kind, status, title, payload, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (history_id, owner, entry.get("job_id"), entry.get("kind", "legacy"), entry.get("status", "queued"), entry.get("title", ""), json.dumps(entry, ensure_ascii=False), entry.get("created_at", now), entry.get("updated_at", now), entry.get("expires_at")))
            connection.commit()
        self.audit(owner, "history.created", "history", history_id)
        return history_id

    def history(self, owner: str, history_id: str) -> dict | None:
        row = self._fetchone("SELECT payload FROM history_entries WHERE id=? AND owner_user_id=?", (history_id, owner))
        return json.loads(row["payload"]) if row else None

    def histories(self, owner: str, limit: int) -> list[dict]:
        with self._connect() as connection:
            rows = self._execute(connection, "SELECT payload FROM history_entries WHERE owner_user_id=? ORDER BY updated_at DESC LIMIT ?", (owner, limit)).fetchall()
            return [json.loads(row[0]) for row in rows]

    def update_history_internal(self, history_id: str, mutate) -> bool:
        with self._connect() as connection:
            row = self._execute(connection, "SELECT owner_user_id, payload FROM history_entries WHERE id=?", (history_id,)).fetchone()
            if not row:
                return False
            owner, payload = row[0], json.loads(row[1])
            mutate(payload)
            self._execute(connection, "UPDATE history_entries SET job_id=?, kind=?, status=?, title=?, payload=?, updated_at=? WHERE id=?", (payload.get("job_id"), payload.get("kind", "legacy"), payload.get("status", "queued"), payload.get("title", ""), json.dumps(payload, ensure_ascii=False), payload.get("updated_at", utc_now()), history_id))
            connection.commit()
        return True

    def delete_history(self, owner: str, history_id: str) -> bool:
        with self._connect() as connection:
            cursor = self._execute(connection, "DELETE FROM history_entries WHERE id=? AND owner_user_id=?", (history_id, owner))
            connection.commit()
        if cursor.rowcount:
            self.audit(owner, "history.deleted", "history", history_id)
            return True
        return False

    def save_job(self, owner: str, job_id: str, payload: dict) -> dict:
        """Upsert the authoritative job record; never delete/reinsert a running job."""
        return self.upsert_job(owner, job_id, payload)

    def create_job(self, owner: str, kind: str, request: dict, *, job_id: str | None = None, idempotency_key: str = "", expires_at: str | None = None) -> tuple[dict, bool]:
        """Create one queued job, or return the existing job for an idempotency key."""
        if idempotency_key:
            existing = self._fetchone("SELECT * FROM jobs WHERE owner_user_id=? AND idempotency_key=?", (owner, idempotency_key))
            if existing:
                return self._hydrate_job(existing), False
        job_id = job_id or uuid.uuid4().hex
        now = utc_now()
        payload = {"id": job_id, "owner_user_id": owner, "kind": kind, "status": "queued", "request": request, "created_at": now}
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO jobs (id, owner_user_id, payload, created_at, updated_at, kind, status, stage, progress, request_json, result_json, error, idempotency_key, task_id, attempt, started_at, finished_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (job_id, owner, json.dumps(payload, ensure_ascii=False), now, now, kind, "queued", "Queued", 0, json.dumps(request, ensure_ascii=False), json.dumps({}, ensure_ascii=False), None, idempotency_key or None, None, 0, None, None, expires_at))
            self._insert_job_event(connection, job_id, "queued", "Queued", 0, {"event": "created"})
            connection.commit()
        self.audit(owner, "job.created", "job", job_id, {"kind": kind})
        return self.job(owner, job_id) or payload, True

    def upsert_job(self, owner: str, job_id: str, payload: dict) -> dict:
        existing = self._fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        previous = self._hydrate_job(existing) if existing else {}
        merged = {**previous, **payload, "id": job_id, "owner_user_id": owner}
        now = utc_now()
        status = str(merged.get("status") or previous.get("status") or "queued")
        stage = str(merged.get("stage") or previous.get("stage") or "Queued")
        progress = int(merged.get("progress") or 0)
        request = merged.get("request") if isinstance(merged.get("request"), dict) else previous.get("request", {})
        result = merged.get("result") if isinstance(merged.get("result"), dict) else self._job_result_from_payload(merged)
        if not isinstance(result, dict):
            result = {}
        # Client-facing result fields remain flattened for backwards compatibility,
        # while result_json is the durable result index for new consumers.
        protected = {"id", "owner_user_id", "kind", "status", "stage", "progress", "request", "error", "created_at", "started_at", "finished_at", "expires_at", "idempotency_key", "task_id", "attempt"}
        result = {**result, **{key: value for key, value in merged.items() if key not in protected and key not in {"result", "port"}}}
        merged["result"] = result
        merged["request"] = request
        merged.setdefault("created_at", previous.get("created_at") or now)
        if status == "running" and not merged.get("started_at"):
            merged["started_at"] = now
        if status in {"done", "failed", "error", "cancelled"} and not merged.get("finished_at"):
            merged["finished_at"] = now
        with self._connect() as connection:
            if existing:
                self._execute(connection, "UPDATE jobs SET owner_user_id=?, payload=?, updated_at=?, kind=?, status=?, stage=?, progress=?, request_json=?, result_json=?, error=?, idempotency_key=?, task_id=?, attempt=?, started_at=?, finished_at=?, expires_at=? WHERE id=?", (owner, json.dumps(merged, ensure_ascii=False), now, merged.get("kind", ""), status, stage, progress, json.dumps(request, ensure_ascii=False), json.dumps(result, ensure_ascii=False), merged.get("error"), merged.get("idempotency_key"), merged.get("task_id"), int(merged.get("attempt") or 0), merged.get("started_at"), merged.get("finished_at"), merged.get("expires_at"), job_id))
            else:
                self._execute(connection, "INSERT INTO jobs (id, owner_user_id, payload, created_at, updated_at, kind, status, stage, progress, request_json, result_json, error, idempotency_key, task_id, attempt, started_at, finished_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (job_id, owner, json.dumps(merged, ensure_ascii=False), merged["created_at"], now, merged.get("kind", ""), status, stage, progress, json.dumps(request, ensure_ascii=False), json.dumps(result, ensure_ascii=False), merged.get("error"), merged.get("idempotency_key"), merged.get("task_id"), int(merged.get("attempt") or 0), merged.get("started_at"), merged.get("finished_at"), merged.get("expires_at")))
            if (previous.get("status"), previous.get("stage"), previous.get("progress")) != (status, stage, progress):
                self._insert_job_event(connection, job_id, status, stage, progress, {"error": merged.get("error", "")})
            connection.commit()
        return merged

    def _insert_job_event(self, connection, job_id: str, status: str, stage: str, progress: int, detail: dict) -> None:
        self._execute(connection, "INSERT INTO job_events (id, job_id, status, stage, progress, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, job_id, status, stage, progress, json.dumps(detail, ensure_ascii=False), utc_now()))

    @staticmethod
    def _job_result_from_payload(payload: dict) -> dict:
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def _hydrate_job(self, row: dict | None) -> dict:
        if not row:
            return {}
        payload = json.loads(row.get("payload") or "{}")
        payload.update({key: row.get(key) for key in ("id", "owner_user_id", "kind", "status", "stage", "progress", "error", "idempotency_key", "task_id", "attempt", "created_at", "started_at", "finished_at", "expires_at") if row.get(key) is not None})
        for key, column in (("request", "request_json"), ("result", "result_json")):
            value = row.get(column)
            if value:
                payload[key] = json.loads(value) if isinstance(value, str) else value
        payload.update(payload.get("result") or {})
        return payload

    def job(self, owner: str, job_id: str) -> dict | None:
        row = self._fetchone("SELECT * FROM jobs WHERE id=? AND owner_user_id=?", (job_id, owner))
        return self._hydrate_job(row) if row else None

    def job_by_id(self, job_id: str) -> dict | None:
        """Worker-only lookup. HTTP handlers must use owner-scoped `job`."""
        return self._hydrate_job(self._fetchone("SELECT * FROM jobs WHERE id=?", (job_id,)))

    def claim_job(self, job_id: str, *, task_id: str = "", attempt: int = 0, stale_after_seconds: int = 300) -> dict | None:
        """Atomically claim queued/stale work so broker redelivery cannot double-run it."""
        now = utc_now()
        cutoff = datetime.fromtimestamp(time.time() - stale_after_seconds, timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                "UPDATE jobs SET status=?, stage=?, progress=?, task_id=?, attempt=?, started_at=COALESCE(started_at, ?), updated_at=? WHERE id=? AND (status='queued' OR (status='running' AND updated_at < ?))",
                ("running", "Worker accepted job", 1, task_id or None, attempt, now, now, job_id, cutoff),
            )
            if not cursor.rowcount:
                connection.commit()
                return None
            self._insert_job_event(connection, job_id, "running", "Worker accepted job", 1, {"event": "claimed", "task_id": task_id, "attempt": attempt})
            connection.commit()
        return self.job_by_id(job_id)

    def job_events(self, owner: str, job_id: str) -> list[dict]:
        if not self.job(owner, job_id):
            return []
        with self._connect() as connection:
            rows = self._execute(connection, "SELECT status, stage, progress, detail, created_at FROM job_events WHERE job_id=? ORDER BY created_at", (job_id,)).fetchall()
            return [{"status": row[0], "stage": row[1], "progress": row[2], "detail": json.loads(row[3] or "{}"), "created_at": row[4]} for row in rows]

    def recoverable_job_ids(self, *, stale_after_seconds: int = 300, limit: int = 100) -> list[str]:
        """IDs that can be safely re-enqueued after a broker/API/worker outage."""
        cutoff = datetime.fromtimestamp(time.time() - stale_after_seconds, timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            rows = self._execute(
                connection,
                "SELECT id FROM jobs WHERE status='queued' OR (status='running' AND updated_at < ?) ORDER BY created_at LIMIT ?",
                (cutoff, limit),
            ).fetchall()
            return [str(row[0]) for row in rows]

    def store_file(self, owner: str, kind: str, filename: str, content: bytes, retention_days: int) -> Path:
        """Local-development object store adapter.

        Production storage is selected with OBJECT_STORAGE_BACKEND=s3 and handled by
        `ObjectStorage`; keeping this return type makes the old upload handlers work
        unchanged during the transition.
        """
        safe_name = Path(filename).name or "file"
        object_key = self.object_storage.put(owner, kind, safe_name, content)
        target = Path(object_key)
        expiry = datetime.fromtimestamp(time.time() + retention_days * 86400, timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO files (id, owner_user_id, kind, storage_path, original_name, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, owner, kind, object_key, safe_name, utc_now(), expiry))
            if kind == "uploads":
                self._execute(connection, "INSERT INTO documents (id, owner_user_id, original_name, content_type, sha256, object_key, scan_status, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, owner, safe_name, "application/octet-stream", hashlib.sha256(content).hexdigest(), object_key, "passed", utc_now(), expiry))
            elif kind == "exports":
                self._execute(connection, "INSERT INTO artifacts (id, owner_user_id, job_id, kind, object_key, content_type, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, owner, None, "pdf", object_key, "application/pdf", utc_now(), expiry))
            connection.commit()
        return target

    def delete_user(self, owner: str) -> None:
        with self._connect() as connection:
            objects = self._execute(connection, "SELECT storage_path FROM files WHERE owner_user_id=?", (owner,)).fetchall()
            self._execute(connection, "DELETE FROM users WHERE id=?", (owner,))
            connection.commit()
        for row in objects:
            self.object_storage.delete(row[0])
        shutil.rmtree(self.file_root / owner, ignore_errors=True)
        self.audit(owner, "user.deleted", "user", owner)

    def enforce_retention(self, days: int) -> None:
        cutoff = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            self._execute(connection, "DELETE FROM history_entries WHERE updated_at < ?", (cutoff,))
            self._execute(connection, "DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
            files = self._execute(connection, "SELECT storage_path FROM files WHERE expires_at < ?", (utc_now(),)).fetchall()
            self._execute(connection, "DELETE FROM files WHERE expires_at < ?", (utc_now(),))
            self._execute(connection, "DELETE FROM documents WHERE expires_at < ?", (utc_now(),))
            self._execute(connection, "DELETE FROM artifacts WHERE expires_at < ?", (utc_now(),))
            connection.commit()
        for row in files:
            self.object_storage.delete(row[0])

    def artifact_download_url(self, owner: str, artifact_id: str, *, expires_seconds: int = 300) -> str | None:
        row = self._fetchone("SELECT object_key, expires_at FROM artifacts WHERE id=? AND owner_user_id=?", (artifact_id, owner))
        if not row or (row.get("expires_at") and row["expires_at"] <= utc_now()):
            return None
        return self.object_storage.signed_download_url(row["object_key"], expires_seconds)

    def migrate_legacy_history(self, history_path: Path) -> None:
        marker = "legacy_history_records_v1"
        if self._fetchone("SELECT name FROM schema_migrations WHERE name=?", (marker,)):
            return
        backup = history_path.with_suffix(history_path.suffix + ".pre-db-migration.bak")
        try:
            raw = history_path.read_text(encoding="utf-8-sig") if history_path.exists() else ""
            source = json.loads(raw or "{}")
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot migrate legacy history safely: {error}") from error
        items = source if isinstance(source, list) else source.get("items", []) if isinstance(source, dict) else []
        if history_path.exists() and not backup.exists():
            shutil.copy2(history_path, backup)
        admin = self.provision_user(os.getenv("LEGACY_ADMIN_OIDC_SUBJECT", "legacy-history-admin"), display_name="Legacy history administrator")
        with self._connect() as connection:
            for item in items:
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                entry["id"] = str(entry.get("id") or uuid.uuid4().hex)
                entry["owner_user_id"] = admin["id"]
                now = utc_now()
                self._execute(connection, "INSERT INTO history_entries (id, owner_user_id, job_id, kind, status, title, payload, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING", (entry["id"], admin["id"], entry.get("job_id"), entry.get("kind", "legacy"), entry.get("status", "done"), entry.get("title", ""), json.dumps(entry, ensure_ascii=False), entry.get("created_at", now), entry.get("updated_at", now), None))
            self._execute(connection, "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)", (marker, utc_now()))
            connection.commit()
        self.audit(admin["id"], "history.legacy_migrated", "history", detail={"count": len(items), "backup": str(backup)})


class OIDCClient:
    def __init__(self) -> None:
        self.issuer = os.getenv("OIDC_ISSUER", "").rstrip("/")
        self.client_id = os.getenv("OIDC_CLIENT_ID", "")
        self.client_secret = os.getenv("OIDC_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("OIDC_REDIRECT_URI", "")

    @property
    def enabled(self) -> bool:
        return bool(self.issuer and self.client_id and self.redirect_uri)

    def discovery(self) -> dict:
        if not self.enabled:
            raise RuntimeError("OIDC is not configured")
        with urllib.request.urlopen(self.issuer + "/.well-known/openid-configuration", timeout=10) as response:
            return json.loads(response.read())

    def authorization_url(self, state: str, nonce: str) -> str:
        metadata = self.discovery()
        query = urllib.parse.urlencode({"response_type": "code", "client_id": self.client_id, "redirect_uri": self.redirect_uri, "scope": "openid profile email", "state": state, "nonce": nonce})
        return metadata["authorization_endpoint"] + "?" + query

    def complete(self, code: str) -> dict:
        metadata = self.discovery()
        body = urllib.parse.urlencode({"grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_uri, "client_id": self.client_id, "client_secret": self.client_secret}).encode()
        request = urllib.request.Request(metadata["token_endpoint"], data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(request, timeout=15) as response:
            tokens = json.loads(response.read())
        userinfo_request = urllib.request.Request(metadata["userinfo_endpoint"], headers={"Authorization": "Bearer " + tokens["access_token"]})
        with urllib.request.urlopen(userinfo_request, timeout=15) as response:
            claims = json.loads(response.read())
        if not claims.get("sub"):
            raise RuntimeError("OIDC provider returned no subject claim")
        return claims
