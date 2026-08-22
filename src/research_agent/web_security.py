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


def decode_json_value(value, default):
    """Accept SQLite JSON text and psycopg-decoded PostgreSQL JSONB values."""
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


PASSWORD_HASH_VERSION = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 310_000


def normalize_email(value: str) -> str:
    return str(value or "").strip().casefold()


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return "$".join(
        (
            PASSWORD_HASH_VERSION,
            str(PASSWORD_HASH_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        version, iterations_text, salt_text, digest_text = str(encoded or "").split("$", 3)
        if version != PASSWORD_HASH_VERSION:
            return False
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4))
    except (ValueError, TypeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


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
        self.server_side_encryption = os.getenv("S3_SERVER_SIDE_ENCRYPTION", "").strip()
        self.sse_kms_key_id = os.getenv("S3_SSE_KMS_KEY_ID", "").strip()
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
            put_options = {"Bucket": self.bucket, "Key": key, "Body": content}
            if self.server_side_encryption:
                put_options["ServerSideEncryption"] = self.server_side_encryption
            if self.sse_kms_key_id:
                put_options["SSEKMSKeyId"] = self.sse_kms_key_id
            self._client.put_object(**put_options)
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
                f"CREATE TABLE IF NOT EXISTS reference_feedback (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, history_id {identity}, reference_key {identity} NOT NULL, vote {identity} NOT NULL, title {identity}, source_label {identity}, doi {identity}, pmid {identity}, arxiv_id {identity}, created_at {identity} NOT NULL, updated_at {identity} NOT NULL)",
                f"CREATE UNIQUE INDEX IF NOT EXISTS reference_feedback_owner_history_ref ON reference_feedback(owner_user_id, history_id, reference_key)",
                f"CREATE INDEX IF NOT EXISTS reference_feedback_owner_updated ON reference_feedback(owner_user_id, updated_at DESC)",
                f"CREATE TABLE IF NOT EXISTS job_events (id {identity} PRIMARY KEY, job_id {identity} NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, status {identity}, stage {identity}, progress INTEGER, detail {json_type}, created_at {identity} NOT NULL)",
                f"CREATE INDEX IF NOT EXISTS job_events_job_created ON job_events(job_id, created_at)",
                f"CREATE TABLE IF NOT EXISTS documents (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, original_name {identity} NOT NULL, content_type {identity}, sha256 {identity} NOT NULL, object_key {identity} NOT NULL, scan_status {identity} NOT NULL, created_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE INDEX IF NOT EXISTS documents_owner_created ON documents(owner_user_id, created_at DESC)",
                f"CREATE TABLE IF NOT EXISTS artifacts (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, job_id {identity} REFERENCES jobs(id) ON DELETE SET NULL, kind {identity} NOT NULL, object_key {identity} NOT NULL, content_type {identity}, created_at {identity} NOT NULL, expires_at {identity})",
                f"CREATE TABLE IF NOT EXISTS audit_logs (id {identity} PRIMARY KEY, actor_user_id {identity}, action {identity} NOT NULL, resource_type {identity} NOT NULL, resource_id {identity}, detail {json_type}, created_at {identity} NOT NULL)",
                f"CREATE TABLE IF NOT EXISTS evaluation_runs (id {identity} PRIMARY KEY, owner_user_id {identity} NOT NULL REFERENCES users(id) ON DELETE CASCADE, name {identity} NOT NULL, kind {identity} NOT NULL, status {identity} NOT NULL, request_json {json_type} NOT NULL, total_count INTEGER NOT NULL, completed_count INTEGER NOT NULL DEFAULT 0, created_at {identity} NOT NULL, updated_at {identity} NOT NULL, finished_at {identity})",
                f"CREATE INDEX IF NOT EXISTS evaluation_runs_owner_updated ON evaluation_runs(owner_user_id, updated_at DESC)",
                f"CREATE TABLE IF NOT EXISTS evaluation_items (id {identity} PRIMARY KEY, run_id {identity} NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE, query {identity} NOT NULL, status {identity} NOT NULL, total_duration_ms INTEGER, result_json {json_type}, error {identity}, created_at {identity} NOT NULL, started_at {identity}, finished_at {identity})",
                f"CREATE INDEX IF NOT EXISTS evaluation_items_run_created ON evaluation_items(run_id, created_at)",
                f"CREATE TABLE IF NOT EXISTS evaluation_stage_events (id {identity} PRIMARY KEY, item_id {identity} NOT NULL REFERENCES evaluation_items(id) ON DELETE CASCADE, stage {identity} NOT NULL, source {identity}, duration_ms INTEGER, input_count INTEGER, output_count INTEGER, detail_json {json_type}, created_at {identity} NOT NULL)",
                f"CREATE INDEX IF NOT EXISTS evaluation_stage_events_item_created ON evaluation_stage_events(item_id, created_at)",
            ):
                self._execute(connection, statement)
            self._ensure_user_columns(connection, identity)
            self._ensure_job_columns(connection, identity, json_type)
            self._migrate_legacy_tables(connection)
            connection.commit()

    def _ensure_user_columns(self, connection, identity: str) -> None:
        columns = {
            "password_hash": identity,
            "role": f"{identity} NOT NULL DEFAULT 'user'",
            "status": f"{identity} NOT NULL DEFAULT 'active'",
            "email_verified": "INTEGER NOT NULL DEFAULT 1",
            "last_login_at": identity,
        }
        if self._postgres:
            for name, definition in columns.items():
                self._execute(connection, f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {name} {definition}")
        else:
            existing = {row[1] for row in self._execute(connection, "PRAGMA table_info(users)").fetchall()}
            for name, definition in columns.items():
                if name not in existing:
                    self._execute(connection, f"ALTER TABLE users ADD COLUMN {name} {definition}")
        self._execute(connection, "CREATE UNIQUE INDEX IF NOT EXISTS users_active_email_unique ON users(email) WHERE deleted_at IS NULL AND email IS NOT NULL AND email <> ''")

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
            entry = decode_json_value(payload, {})
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

    def admin_recent_audit_logs(self, limit: int = 30, actor_user_id: str = "") -> list[dict]:
        limit = max(1, min(int(limit or 30), 100))
        where = ""
        params: list[object] = []
        if actor_user_id:
            where = "WHERE a.actor_user_id = ?"
            params.append(actor_user_id)
        params.append(limit)
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                f"""
                SELECT
                    a.actor_user_id, u.email AS actor_email, u.display_name AS actor_display_name,
                    a.action, a.resource_type, a.resource_id, a.detail, a.created_at
                FROM audit_logs a
                LEFT JOIN users u ON u.id = a.actor_user_id
                {where}
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description]
        logs = []
        for row in rows:
            item = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(columns, row))
            item["detail"] = decode_json_value(item.get("detail"), {})
            logs.append(item)
        return logs

    def provision_user(self, subject: str, email: str = "", display_name: str = "") -> dict:
        existing = self._fetchone("SELECT * FROM users WHERE oidc_subject = ? AND deleted_at IS NULL", (subject,))
        if existing:
            return existing
        user = {"id": uuid.uuid4().hex, "oidc_subject": subject, "email": normalize_email(email), "display_name": display_name, "created_at": utc_now()}
        with self._connect() as connection:
            self._execute(connection, "INSERT INTO users (id, oidc_subject, email, display_name, created_at, deleted_at, role, status, email_verified) VALUES (?, ?, ?, ?, ?, NULL, 'user', 'active', 1)", tuple(user.values()))
            connection.commit()
        self.audit(user["id"], "user.provisioned", "user", user["id"])
        return user

    def create_local_user(self, email: str, password: str, display_name: str = "", *, role: str = "user") -> dict:
        email = normalize_email(email)
        display_name = str(display_name or "").strip() or email.split("@", 1)[0]
        role = "admin" if role == "admin" else "user"
        if not email or "@" not in email:
            raise ValueError("请输入有效邮箱。")
        if len(password or "") < 8:
            raise ValueError("密码至少需要 8 位。")
        user = {
            "id": uuid.uuid4().hex,
            "oidc_subject": f"local:{email}",
            "email": email,
            "display_name": display_name,
            "created_at": utc_now(),
            "password_hash": password_hash(password),
            "role": role,
            "status": "active",
        }
        with self._connect() as connection:
            existing = self._execute(connection, "SELECT id FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
            if existing:
                raise ValueError("该邮箱已注册。")
            self._execute(
                connection,
                "INSERT INTO users (id, oidc_subject, email, display_name, created_at, deleted_at, password_hash, role, status, email_verified) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, 1)",
                (user["id"], user["oidc_subject"], user["email"], user["display_name"], user["created_at"], user["password_hash"], user["role"], user["status"]),
            )
            connection.commit()
        self.audit(user["id"], "user.created", "user", user["id"], {"role": role, "source": "local"})
        return self.public_user(user)

    def authenticate_local_user(self, email: str, password: str) -> dict | None:
        email = normalize_email(email)
        row = self._fetchone("SELECT * FROM users WHERE email=? AND deleted_at IS NULL", (email,))
        if not row or not row.get("password_hash") or not verify_password(password or "", str(row.get("password_hash") or "")):
            return None
        if row.get("status") != "active":
            raise PermissionError("该账号已被禁用，请联系管理员。")
        now = utc_now()
        with self._connect() as connection:
            self._execute(connection, "UPDATE users SET last_login_at=? WHERE id=?", (now, row["id"]))
            connection.commit()
        row["last_login_at"] = now
        self.audit(row["id"], "auth.login", "user", row["id"])
        return self.public_user(row)

    def public_user(self, user: dict) -> dict:
        return {
            "id": user.get("id", ""),
            "email": user.get("email", ""),
            "display_name": user.get("display_name", ""),
            "role": user.get("role") or "user",
            "status": user.get("status") or "active",
            "created_at": user.get("created_at", ""),
            "last_login_at": user.get("last_login_at", ""),
        }

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
        row = self._fetchone("SELECT s.csrf_token, s.expires_at, u.id, u.email, u.display_name, u.role, u.status FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND u.deleted_at IS NULL", (hashlib.sha256(token.encode()).hexdigest(),))
        if not row or row["expires_at"] <= utc_now() or row.get("status") != "active":
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
        return decode_json_value(row["payload"], {}) if row else None

    def histories(self, owner: str, limit: int) -> list[dict]:
        with self._connect() as connection:
            rows = self._execute(connection, "SELECT payload FROM history_entries WHERE owner_user_id=? ORDER BY updated_at DESC LIMIT ?", (owner, limit)).fetchall()
            return [decode_json_value(row[0], {}) for row in rows]

    def record_reference_feedback(self, owner: str, payload: dict) -> dict:
        now = utc_now()
        history_id = str(payload.get("history_id") or "").strip()
        reference = payload.get("reference") if isinstance(payload.get("reference"), dict) else {}
        reference_key = str(
            payload.get("reference_key")
            or reference.get("candidate_id")
            or reference.get("dedupe_key")
            or reference.get("doi")
            or reference.get("pmid")
            or reference.get("arxiv_id")
            or reference.get("source")
            or reference.get("title")
            or ""
        ).strip()
        vote = str(payload.get("vote") or "").strip().lower()
        if vote not in {"yes", "no"}:
            raise ValueError("Feedback vote must be yes or no.")
        if not reference_key:
            raise ValueError("Feedback reference key is required.")
        item = {
            "id": uuid.uuid4().hex,
            "owner_user_id": owner,
            "history_id": history_id,
            "reference_key": reference_key[:500],
            "vote": vote,
            "title": str(reference.get("title") or "")[:500],
            "source_label": str(reference.get("source_label") or reference.get("retrieved_from") or "")[:160],
            "doi": str(reference.get("doi") or "")[:255],
            "pmid": str(reference.get("pmid") or "")[:64],
            "arxiv_id": str(reference.get("arxiv_id") or "")[:64],
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as connection:
            existing = self._execute(
                connection,
                "SELECT id, created_at FROM reference_feedback WHERE owner_user_id=? AND history_id=? AND reference_key=?",
                (owner, item["history_id"], item["reference_key"]),
            ).fetchone()
            if existing:
                item["id"] = existing[0]
                item["created_at"] = existing[1]
                self._execute(
                    connection,
                    "UPDATE reference_feedback SET vote=?, title=?, source_label=?, doi=?, pmid=?, arxiv_id=?, updated_at=? WHERE id=?",
                    (item["vote"], item["title"], item["source_label"], item["doi"], item["pmid"], item["arxiv_id"], item["updated_at"], item["id"]),
                )
            else:
                self._execute(
                    connection,
                    "INSERT INTO reference_feedback (id, owner_user_id, history_id, reference_key, vote, title, source_label, doi, pmid, arxiv_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (item["id"], owner, item["history_id"], item["reference_key"], item["vote"], item["title"], item["source_label"], item["doi"], item["pmid"], item["arxiv_id"], item["created_at"], item["updated_at"]),
                )
            connection.commit()
        return item

    def admin_histories(
        self,
        limit: int = 100,
        owner_user_id: str = "",
        owner_keyword: str = "",
        kind: str = "",
        status: str = "",
    ) -> list[dict]:
        limit = max(1, min(int(limit or 100), 300))
        conditions = ["u.deleted_at IS NULL"]
        params: list[object] = []
        owner_user_id = str(owner_user_id or "").strip()
        owner_keyword = str(owner_keyword or "").strip().casefold()
        kind = str(kind or "").strip()
        status = str(status or "").strip()
        if owner_user_id:
            conditions.append("h.owner_user_id = ?")
            params.append(owner_user_id)
        if owner_keyword:
            conditions.append("(LOWER(COALESCE(u.email, '')) LIKE ? OR LOWER(COALESCE(u.display_name, '')) LIKE ? OR LOWER(h.owner_user_id) LIKE ?)")
            keyword = f"%{owner_keyword}%"
            params.extend([keyword, keyword, keyword])
        if kind:
            conditions.append("h.kind = ?")
            params.append(kind)
        if status:
            conditions.append("h.status = ?")
            params.append(status)
        params.append(limit)
        sql = f"""
            SELECT
                h.id, h.owner_user_id, u.email AS owner_email, u.display_name AS owner_display_name,
                h.kind, h.status, h.title, h.job_id, h.payload, h.created_at, h.updated_at
            FROM history_entries h
            JOIN users u ON u.id = h.owner_user_id
            WHERE {' AND '.join(conditions)}
            ORDER BY h.updated_at DESC
            LIMIT ?
        """
        with self._connect() as connection:
            cursor = self._execute(connection, sql, tuple(params))
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description]
        return [self._admin_history_summary(dict(row) if isinstance(row, sqlite3.Row) else dict(zip(columns, row))) for row in rows]

    def admin_history_detail(self, history_id: str) -> dict | None:
        row = self._fetchone(
            """
            SELECT
                h.id, h.owner_user_id, u.email AS owner_email, u.display_name AS owner_display_name,
                h.kind, h.status, h.title, h.job_id, h.payload, h.created_at, h.updated_at
            FROM history_entries h
            JOIN users u ON u.id = h.owner_user_id
            WHERE h.id=? AND u.deleted_at IS NULL
            """,
            (str(history_id or "").strip(),),
        )
        if not row:
            return None
        detail = decode_json_value(row.get("payload"), {})
        if not isinstance(detail, dict):
            detail = {}
        result = {
            **detail,
            "id": row.get("id", detail.get("id", "")),
            "owner_user_id": row.get("owner_user_id", detail.get("owner_user_id", "")),
            "owner_email": row.get("owner_email", ""),
            "owner_display_name": row.get("owner_display_name", ""),
            "kind": row.get("kind", detail.get("kind", "")),
            "status": row.get("status", detail.get("status", "")),
            "title": row.get("title", detail.get("title", "")),
            "job_id": row.get("job_id", detail.get("job_id", "")),
            "created_at": row.get("created_at", detail.get("created_at", "")),
            "updated_at": row.get("updated_at", detail.get("updated_at", "")),
        }
        job_id = str(result.get("job_id") or "").strip()
        job = self.job_by_id(job_id) if job_id else None
        result["admin_summary"] = self._admin_history_observability_summary(result, job or {})
        return result

    @staticmethod
    def _admin_history_summary(row: dict) -> dict:
        payload = decode_json_value(row.get("payload"), {})
        payload = payload if isinstance(payload, dict) else {}
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        request_keys = {"query", "topic", "sources", "search_mode", "year", "reference_count", "file_count"}
        return {
            "id": row.get("id", ""),
            "owner_user_id": row.get("owner_user_id", ""),
            "owner_email": row.get("owner_email", ""),
            "owner_display_name": row.get("owner_display_name", ""),
            "kind": row.get("kind") or payload.get("kind", ""),
            "status": row.get("status") or payload.get("status", ""),
            "title": row.get("title") or payload.get("title", ""),
            "job_id": row.get("job_id") or payload.get("job_id", ""),
            "counts": payload.get("counts") if isinstance(payload.get("counts"), dict) else {},
            "created_at": row.get("created_at") or payload.get("created_at", ""),
            "updated_at": row.get("updated_at") or payload.get("updated_at", ""),
            "request": {key: request.get(key) for key in request_keys if key in request},
        }

    @staticmethod
    def _admin_history_observability_summary(detail: dict, job: dict) -> dict:
        result = detail.get("result") if isinstance(detail.get("result"), dict) else {}
        counts = detail.get("counts") if isinstance(detail.get("counts"), dict) else {}
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        source_results = result.get("source_results") if isinstance(result.get("source_results"), dict) else {}
        if not source_results and isinstance(result.get("internal_source_results"), dict):
            source_results = result.get("internal_source_results")
        diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
        error = str(detail.get("error") or job.get("error") or result.get("error") or "")
        errors = []
        if error:
            errors.append(error)
        result_errors = result.get("errors")
        if isinstance(result_errors, dict):
            errors.extend(f"{key}: {value}" for key, value in result_errors.items() if value)
        elif isinstance(result_errors, list):
            errors.extend(str(item) for item in result_errors if item)
        warnings = diagnostics.get("warnings") if isinstance(diagnostics.get("warnings"), list) else []
        created_at = str(detail.get("created_at") or "")
        updated_at = str(detail.get("updated_at") or "")
        job_created_at = str(job.get("created_at") or "")
        started_at = str(job.get("started_at") or "")
        finished_at = str(job.get("finished_at") or "")
        return {
            "status": detail.get("status") or job.get("status") or "",
            "stage": detail.get("stage") or job.get("stage") or "",
            "history_created_at": created_at,
            "history_updated_at": updated_at,
            "job_created_at": job_created_at,
            "job_started_at": started_at,
            "job_finished_at": finished_at,
            "history_elapsed_seconds": UserDataStore._seconds_between(created_at, updated_at),
            "queue_seconds": UserDataStore._seconds_between(job_created_at, started_at),
            "run_seconds": UserDataStore._seconds_between(started_at or job_created_at, finished_at or updated_at),
            "total_seconds": UserDataStore._seconds_between(job_created_at or created_at, finished_at or updated_at),
            "counts": counts,
            "timings": timings,
            "source_results": source_results,
            "warnings": [str(item) for item in warnings[:8]],
            "errors": errors[:8],
        }

    @staticmethod
    def _seconds_between(start: str, end: str) -> float | None:
        if not start or not end:
            return None
        try:
            start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0.0, round((end_dt - start_dt).total_seconds(), 3))

    def admin_light_metrics(self, days: int = 7) -> dict:
        days = int(days or 7)
        if days not in {1, 7, 30}:
            days = 7
        cutoff = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).isoformat(timespec="seconds")
        metrics = {
            "range_days": days,
            "users": {"total": 0, "active": 0, "disabled": 0},
            "usage": {
                "history_total": 0,
                "literature_search": 0,
                "novelty_check": 0,
                "direct_analysis": 0,
                "search_analysis": 0,
            },
            "jobs": {"queued": 0, "running": 0, "done": 0, "error": 0},
            "quality": {
                "candidate_total": 0,
                "qualified_total": 0,
                "needs_review_total": 0,
                "rejected_total": 0,
            },
            "storage": {"file_count": 0},
            "users_usage": [],
            "recent_errors": [],
        }
        with self._connect() as connection:
            user_rows = self._execute(
                connection,
                "SELECT status, COUNT(*) AS count FROM users WHERE deleted_at IS NULL GROUP BY status",
            ).fetchall()
            for row in user_rows:
                status, count = str(row[0] or "active"), int(row[1] or 0)
                metrics["users"]["total"] += count
                if status == "disabled":
                    metrics["users"]["disabled"] += count
                else:
                    metrics["users"]["active"] += count

            history_rows = self._execute(
                connection,
                """
                SELECT h.owner_user_id, u.email, u.display_name, h.id, h.kind, h.status, h.title, h.payload, h.updated_at
                FROM history_entries h
                JOIN users u ON u.id = h.owner_user_id
                WHERE u.deleted_at IS NULL AND h.updated_at >= ?
                ORDER BY h.updated_at DESC
                """,
                (cutoff,),
            ).fetchall()
            job_rows = self._execute(
                connection,
                """
                SELECT j.owner_user_id, u.email, u.display_name, j.status, j.updated_at
                FROM jobs j
                JOIN users u ON u.id = j.owner_user_id
                WHERE u.deleted_at IS NULL AND j.updated_at >= ?
                """,
                (cutoff,),
            ).fetchall()
            file_rows = self._execute(
                connection,
                """
                SELECT f.owner_user_id, u.email, u.display_name, f.created_at
                FROM files f
                JOIN users u ON u.id = f.owner_user_id
                WHERE u.deleted_at IS NULL AND f.created_at >= ?
                """,
                (cutoff,),
            ).fetchall()

        users_usage: dict[str, dict] = {}

        def user_usage(owner_user_id: str, email: str = "", display_name: str = "") -> dict:
            owner_user_id = str(owner_user_id or "")
            item = users_usage.setdefault(
                owner_user_id,
                {
                    "owner_user_id": owner_user_id,
                    "email": email or "",
                    "display_name": display_name or "",
                    "history_total": 0,
                    "job_done": 0,
                    "job_error": 0,
                    "file_count": 0,
                    "last_activity_at": "",
                },
            )
            if email and not item.get("email"):
                item["email"] = email
            if display_name and not item.get("display_name"):
                item["display_name"] = display_name
            return item

        for row in history_rows:
            owner_user_id, email, display_name = str(row[0] or ""), str(row[1] or ""), str(row[2] or "")
            history_id, kind, status, title, payload, updated_at = row[3], str(row[4] or ""), str(row[5] or ""), str(row[6] or ""), row[7], str(row[8] or "")
            metrics["usage"]["history_total"] += 1
            if kind in metrics["usage"]:
                metrics["usage"][kind] += 1
            usage = user_usage(owner_user_id, email, display_name)
            usage["history_total"] += 1
            usage["last_activity_at"] = max(str(usage.get("last_activity_at") or ""), updated_at)
            history_payload = decode_json_value(payload, {})
            counts = history_payload.get("counts") if isinstance(history_payload, dict) and isinstance(history_payload.get("counts"), dict) else {}
            metrics["quality"]["candidate_total"] += self._first_int(counts, "candidates", "candidate_count", "raw", "raw_count", "references")
            metrics["quality"]["qualified_total"] += self._first_int(counts, "qualified", "qualified_count")
            metrics["quality"]["needs_review_total"] += self._first_int(counts, "needs_review", "needs_review_count")
            metrics["quality"]["rejected_total"] += self._first_int(counts, "rejected", "rejected_count", "filtered", "filtered_count")
            error = ""
            if status == "error" and isinstance(history_payload, dict):
                error = str(history_payload.get("error") or history_payload.get("stage") or "History failed.")
            if error and len(metrics["recent_errors"]) < 10:
                metrics["recent_errors"].append(
                    {
                        "history_id": history_id,
                        "owner_user_id": owner_user_id,
                        "user": email or display_name or owner_user_id,
                        "kind": kind,
                        "title": title,
                        "error": error,
                        "updated_at": updated_at,
                    }
                )

        for row in job_rows:
            owner_user_id, email, display_name = str(row[0] or ""), str(row[1] or ""), str(row[2] or "")
            status, updated_at = str(row[3] or ""), str(row[4] or "")
            if status in metrics["jobs"]:
                metrics["jobs"][status] += 1
            elif status in {"failed", "cancelled"}:
                metrics["jobs"]["error"] += 1
            usage = user_usage(owner_user_id, email, display_name)
            if status == "done":
                usage["job_done"] += 1
            elif status in {"error", "failed", "cancelled"}:
                usage["job_error"] += 1
            usage["last_activity_at"] = max(str(usage.get("last_activity_at") or ""), updated_at)

        for row in file_rows:
            owner_user_id, email, display_name = str(row[0] or ""), str(row[1] or ""), str(row[2] or "")
            created_at = str(row[3] or "")
            metrics["storage"]["file_count"] += 1
            usage = user_usage(owner_user_id, email, display_name)
            usage["file_count"] += 1
            usage["last_activity_at"] = max(str(usage.get("last_activity_at") or ""), created_at)

        metrics["users_usage"] = sorted(
            users_usage.values(),
            key=lambda item: (str(item.get("last_activity_at") or ""), int(item.get("history_total") or 0)),
            reverse=True,
        )
        return metrics

    @staticmethod
    def _first_int(source: dict, *keys: str) -> int:
        for key in keys:
            try:
                value = int(source.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                return value
        return 0

    def update_history_internal(self, history_id: str, mutate) -> bool:
        with self._connect() as connection:
            row = self._execute(connection, "SELECT owner_user_id, payload FROM history_entries WHERE id=?", (history_id,)).fetchone()
            if not row:
                return False
            owner, payload = row[0], decode_json_value(row[1], {})
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
        payload = decode_json_value(row.get("payload"), {})
        # Structured columns are authoritative, including nullable values.  A retry
        # deliberately clears fields such as error and finished_at; ignoring NULL
        # here would resurrect their stale values from the compatibility payload.
        payload.update({
            key: row.get(key)
            for key in (
                "id", "owner_user_id", "kind", "status", "stage", "progress",
                "error", "idempotency_key", "task_id", "attempt", "created_at",
                "started_at", "finished_at", "expires_at",
            )
            if key in row
        })
        for key, column in (("request", "request_json"), ("result", "result_json")):
            if column in row:
                payload[key] = decode_json_value(row.get(column), {})
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
                "UPDATE jobs SET status=?, stage=?, progress=?, task_id=?, attempt=?, started_at=?, finished_at=NULL, error=NULL, updated_at=? WHERE id=? AND (status='queued' OR (status='running' AND updated_at < ?))",
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
            return [{"status": row[0], "stage": row[1], "progress": row[2], "detail": decode_json_value(row[3], {}), "created_at": row[4]} for row in rows]

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

    def admin_user_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS count FROM users WHERE role='admin' AND deleted_at IS NULL")
        return int((row or {}).get("count") or 0)

    def list_users(self, limit: int = 200) -> list[dict]:
        with self._connect() as connection:
            rows = self._execute(
                connection,
                """
                SELECT
                    u.id, u.email, u.display_name, u.role, u.status, u.created_at, u.last_login_at,
                    COUNT(DISTINCT h.id) AS history_count,
                    COUNT(DISTINCT j.id) AS job_count,
                    COUNT(DISTINCT f.id) AS file_count
                FROM users u
                LEFT JOIN history_entries h ON h.owner_user_id = u.id
                LEFT JOIN jobs j ON j.owner_user_id = u.id
                LEFT JOIN files f ON f.owner_user_id = u.id
                WHERE u.deleted_at IS NULL
                GROUP BY u.id, u.email, u.display_name, u.role, u.status, u.created_at, u.last_login_at
                ORDER BY u.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        users = []
        for row in rows:
            if isinstance(row, sqlite3.Row):
                item = dict(row)
            else:
                columns = ("id", "email", "display_name", "role", "status", "created_at", "last_login_at", "history_count", "job_count", "file_count")
                item = dict(zip(columns, row))
            users.append(item)
        return users

    def update_user_admin(self, actor: str, user_id: str, *, role: str | None = None, status: str | None = None, display_name: str | None = None) -> dict | None:
        existing = self._fetchone("SELECT * FROM users WHERE id=? AND deleted_at IS NULL", (user_id,))
        if not existing:
            return None
        next_role = existing.get("role") or "user"
        next_status = existing.get("status") or "active"
        next_display_name = existing.get("display_name") or ""
        if role is not None:
            next_role = "admin" if role == "admin" else "user"
        if status is not None:
            if status not in {"active", "disabled"}:
                raise ValueError("用户状态只能是 active 或 disabled。")
            next_status = status
        if display_name is not None:
            next_display_name = str(display_name or "").strip()
        if existing.get("role") == "admin" and next_role != "admin" and self.admin_user_count() <= 1:
            raise ValueError("至少需要保留一个管理员。")
        if existing.get("role") == "admin" and next_status != "active" and self.admin_user_count() <= 1:
            raise ValueError("不能禁用最后一个管理员。")
        with self._connect() as connection:
            self._execute(connection, "UPDATE users SET role=?, status=?, display_name=? WHERE id=?", (next_role, next_status, next_display_name, user_id))
            if next_status != "active":
                self._execute(connection, "DELETE FROM sessions WHERE user_id=?", (user_id,))
            connection.commit()
        self.audit(actor, "user.admin_updated", "user", user_id, {"role": next_role, "status": next_status})
        return self.public_user({**existing, "role": next_role, "status": next_status, "display_name": next_display_name})

    def create_evaluation_run(self, owner: str, name: str, kind: str, request: dict, queries: list[str]) -> str:
        run_id, now = uuid.uuid4().hex, utc_now()
        with self._connect() as connection:
            self._execute(
                connection,
                "INSERT INTO evaluation_runs (id, owner_user_id, name, kind, status, request_json, total_count, completed_count, created_at, updated_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)",
                (run_id, owner, name, kind, "queued", json.dumps(request, ensure_ascii=False), len(queries), now, now),
            )
            for query in queries:
                self._execute(
                    connection,
                    "INSERT INTO evaluation_items (id, run_id, query, status, total_duration_ms, result_json, error, created_at, started_at, finished_at) VALUES (?, ?, ?, 'queued', NULL, ?, NULL, ?, NULL, NULL)",
                    (uuid.uuid4().hex, run_id, query, json.dumps({}, ensure_ascii=False), now),
                )
            connection.commit()
        self.audit(owner, "evaluation.created", "evaluation", run_id, {"kind": kind, "count": len(queries)})
        return run_id

    def evaluation_runs(self, owner: str, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            cursor = self._execute(connection, "SELECT * FROM evaluation_runs WHERE owner_user_id=? ORDER BY updated_at DESC LIMIT ?", (owner, limit))
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description]
        return [self._evaluation_run_from_row(dict(row) if isinstance(row, sqlite3.Row) else dict(zip(columns, row))) for row in rows]

    def evaluation_run_detail(self, owner: str, run_id: str) -> dict | None:
        run = self._fetchone("SELECT * FROM evaluation_runs WHERE id=? AND owner_user_id=?", (run_id, owner))
        if not run:
            return None
        detail = self._evaluation_run_from_row(run)
        with self._connect() as connection:
            item_cursor = self._execute(connection, "SELECT * FROM evaluation_items WHERE run_id=? ORDER BY created_at", (run_id,))
            item_rows = item_cursor.fetchall()
            item_columns = [item[0] for item in item_cursor.description]
            items = []
            for row in item_rows:
                item = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(item_columns, row))
                event_cursor = self._execute(connection, "SELECT stage, source, duration_ms, input_count, output_count, detail_json, created_at FROM evaluation_stage_events WHERE item_id=? ORDER BY created_at", (item["id"],))
                event_rows = event_cursor.fetchall()
                events = []
                for event_row in event_rows:
                    if isinstance(event_row, sqlite3.Row):
                        event = dict(event_row)
                    else:
                        event = dict(zip([column[0] for column in event_cursor.description], event_row))
                    event["detail"] = decode_json_value(event.pop("detail_json"), {})
                    events.append(event)
                item["result"] = decode_json_value(item.pop("result_json"), {})
                item["events"] = events
                items.append(item)
        detail["items"] = items
        return detail

    def evaluation_items_for_run(self, run_id: str) -> list[dict]:
        with self._connect() as connection:
            cursor = self._execute(connection, "SELECT * FROM evaluation_items WHERE run_id=? ORDER BY created_at", (run_id,))
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description]
        items = []
        for row in rows:
            item = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(columns, row))
            item["result"] = decode_json_value(item.pop("result_json"), {})
            items.append(item)
        return items

    def update_evaluation_run(self, run_id: str, *, status: str, finished: bool = False) -> None:
        now = utc_now()
        finished_at = now if finished else None
        with self._connect() as connection:
            if finished:
                self._execute(connection, "UPDATE evaluation_runs SET status=?, completed_count=(SELECT COUNT(*) FROM evaluation_items WHERE run_id=? AND status IN ('done','error')), updated_at=?, finished_at=? WHERE id=?", (status, run_id, now, finished_at, run_id))
            else:
                self._execute(connection, "UPDATE evaluation_runs SET status=?, completed_count=(SELECT COUNT(*) FROM evaluation_items WHERE run_id=? AND status IN ('done','error')), updated_at=? WHERE id=?", (status, run_id, now, run_id))
            connection.commit()

    def update_evaluation_item(self, item_id: str, *, status: str, result: dict | None = None, error: str = "", total_duration_ms: int | None = None, started: bool = False, finished: bool = False) -> None:
        now = utc_now()
        with self._connect() as connection:
            self._execute(
                connection,
                "UPDATE evaluation_items SET status=?, result_json=?, error=?, total_duration_ms=?, started_at=COALESCE(started_at, ?), finished_at=?, WHERE id=?".replace(", WHERE", " WHERE"),
                (status, json.dumps(result or {}, ensure_ascii=False), error or None, total_duration_ms, now if started else None, now if finished else None, item_id),
            )
            connection.commit()

    def add_evaluation_stage_event(self, item_id: str, stage: str, *, source: str = "", duration_ms: int | None = None, input_count: int | None = None, output_count: int | None = None, detail: dict | None = None) -> None:
        with self._connect() as connection:
            self._execute(
                connection,
                "INSERT INTO evaluation_stage_events (id, item_id, stage, source, duration_ms, input_count, output_count, detail_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, item_id, stage, source or None, duration_ms, input_count, output_count, json.dumps(detail or {}, ensure_ascii=False), utc_now()),
            )
            connection.commit()

    @staticmethod
    def _evaluation_run_from_row(row: dict) -> dict:
        item = dict(row)
        item["request"] = decode_json_value(item.pop("request_json"), {})
        return item

    def enforce_retention(self, days: int) -> None:
        cutoff = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            self._execute(connection, "DELETE FROM history_entries WHERE updated_at < ?", (cutoff,))
            self._execute(connection, "DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
            self._execute(connection, "DELETE FROM reference_feedback WHERE updated_at < ?", (cutoff,))
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
