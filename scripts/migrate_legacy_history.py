"""One-time import of history_records.json into the configured database."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.research_agent.web_security import UserDataStore


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env" if (ROOT / ".env").exists() else ROOT / ".env.example")


def main() -> None:
    source = Path(os.getenv("LEGACY_HISTORY_PATH", str(ROOT / "history_records.json")))
    store = UserDataStore(ROOT)
    store.migrate_legacy_history(source)
    admin = store.provision_user(os.getenv("LEGACY_ADMIN_OIDC_SUBJECT", "legacy-history-admin"))
    print(f"Imported history entries: {len(store.histories(admin['id'], 1_000_000))}")
    print(f"Backup retained at: {source.with_suffix(source.suffix + '.pre-db-migration.bak')}")


if __name__ == "__main__":
    main()
