from __future__ import annotations

from http import HTTPStatus
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote

from src.research_agent.web_utils import parse_byte_range


def frontend_index_path(web_dir: Path, web_dist_dir: Path) -> Path:
    dist_index = web_dist_dir / "index.html"
    return dist_index if dist_index.exists() else web_dir / "index.html"


def frontend_dist_file(web_dist_dir: Path, request_path: str) -> Path | None:
    if not web_dist_dir.exists():
        return None
    relative = request_path.lstrip("/")
    if not relative:
        return None
    candidate = (web_dist_dir / relative).resolve()
    try:
        candidate.relative_to(web_dist_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def content_type_for_path(path: Path) -> str:
    if path.suffix == ".js":
        return "application/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    content_type, _ = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def send_json(handler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_binary(handler, body: bytes, content_type: str, filename: str) -> None:
    fallback_filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename).strip("_") or "download.pdf"
    encoded_filename = quote(filename)
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header(
        "Content-Disposition",
        f"attachment; filename=\"{fallback_filename}\"; filename*=UTF-8''{encoded_filename}",
    )
    handler.end_headers()
    handler.wfile.write(body)


def send_file(handler, path: Path, content_type: str) -> None:
    if not path.exists():
        handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
        return

    file_size = path.stat().st_size
    range_header = handler.headers.get("Range", "")
    byte_range = parse_byte_range(range_header, file_size)
    if range_header and byte_range is None:
        handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        handler.send_header("Content-Range", f"bytes */{file_size}")
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        return

    start, end = byte_range if byte_range else (0, max(file_size - 1, 0))
    content_length = max(0, end - start + 1)
    handler.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(content_length))
    if byte_range:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.end_headers()
    if content_length <= 0:
        return
    with path.open("rb") as file:
        file.seek(start)
        remaining = content_length
        while remaining > 0:
            chunk = file.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)
