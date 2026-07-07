from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time


class ProcessFileLock:
    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._thread_lock = threading.RLock()
        self._local = threading.local()

    def __enter__(self):
        self._thread_lock.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            self._acquire_file_lock()
        self._local.depth = depth + 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        depth = max(0, getattr(self._local, "depth", 1) - 1)
        self._local.depth = depth
        try:
            if depth == 0:
                self._release_file_lock()
        finally:
            self._thread_lock.release()

    def _acquire_file_lock(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.monotonic()
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
                os.close(fd)
                return
            except FileExistsError:
                if time.monotonic() - started_at > self.timeout_seconds:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    started_at = time.monotonic()
                time.sleep(0.05)

    def _release_file_lock(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class TeeStream:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "replace")

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def enable_auto_file_logging(log_dir: Path, *, port: int | str = "") -> tuple[Path, Path] | None:
    raw = os.getenv("WEB_AUTO_LOGS", "1").strip().casefold()
    if raw in {"0", "false", "no", "off"}:
        return None
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    port_part = f"_{port}" if str(port or "").strip() else ""
    stdout_path = log_dir / f"web_backend{port_part}_{stamp}.out.log"
    stderr_path = log_dir / f"web_backend{port_part}_{stamp}.err.log"
    stdout_log = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_log = stderr_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, stdout_log)
    sys.stderr = TeeStream(sys.stderr, stderr_log)
    return stdout_path, stderr_path
