from __future__ import annotations

import getpass
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research_agent.web_security import UserDataStore  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env" if (ROOT / ".env").exists() else ROOT / ".env.example")
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/create_admin.py admin@example.com")
    email = sys.argv[1].strip()
    password = getpass.getpass("Admin password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        raise SystemExit("Passwords do not match.")
    store = UserDataStore(ROOT)
    user = store.create_local_user(email, password, email.split("@", 1)[0], role="admin")
    print(f"Created admin user: {user['email']}")


if __name__ == "__main__":
    main()
