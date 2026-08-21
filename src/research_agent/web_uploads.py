from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import io
import json
from pathlib import Path
import re
import socket
import struct
import unicodedata
import zipfile
from xml.etree import ElementTree


MULTIPART_FIELD_KEYS = [
    "topic",
    "max_results",
    "category",
    "start_date",
    "end_date",
    "citation_format",
    "source_mode",
    "user_context",
    "output_language",
]


@dataclass(frozen=True)
class UploadSecurityPolicy:
    max_total_bytes: int = 30 * 1024 * 1024
    max_total_mb: int = 30
    max_file_bytes: int = 15 * 1024 * 1024
    max_file_mb: int = 15
    max_file_count: int = 4
    max_docx_uncompressed_bytes: int = 20 * 1024 * 1024
    max_docx_xml_bytes: int = 10 * 1024 * 1024
    max_docx_zip_entries: int = 200
    max_docx_compression_ratio: int = 100
    max_extracted_text_chars: int = 300_000
    virus_scan_mode: str = "off"
    clamav_host: str = "127.0.0.1"
    clamav_port: int = 3310
    clamav_timeout_seconds: float = 10.0


DOCX_REQUIRED_PARTS = {"[Content_Types].xml", "word/document.xml"}
DOCX_TEXT_PART_RE = re.compile(r"^word/(?:document|header\d*|footer\d*)\.xml$", re.IGNORECASE)
SAFE_UPLOAD_FILENAME_RE = re.compile(r"[^0-9A-Za-z._() \-\u4e00-\u9fff]+")


def normalize_extracted_text(value: str) -> str:
    text = repair_mojibake(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = normalize_pdf_text_symbols(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\u00ad", "")
    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line)).strip()


def repair_mojibake(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""

    def badness(candidate: str) -> int:
        private_use = sum(1 for char in candidate if "\ue000" <= char <= "\uf8ff")
        controls = sum(1 for char in candidate if "\x80" <= char <= "\x9f")
        replacement = candidate.count("\ufffd")
        latin_mojibake = len(re.findall(r"[\u00c2\u00c3\u00e2][\u0080-\u00ff]?", candidate))
        return replacement * 8 + private_use * 5 + controls * 3 + latin_mojibake * 2

    best = text
    best_score = badness(text)
    for encoding in ("latin-1", "cp1252", "gb18030"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        score = badness(repaired)
        if score < best_score:
            best = repaired
            best_score = score
    return best


def truncate_extracted_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Text truncated because the upload exceeded the extraction safety budget.]"


def normalize_pdf_text_symbols(value: str) -> str:
    return str(value or "").translate(
        str.maketrans(
            {
                "\ufb00": "ff",
                "\ufb01": "fi",
                "\ufb02": "fl",
                "\ufb03": "ffi",
                "\ufb04": "ffl",
            }
        )
    )


def extract_docx_text(
    content: bytes,
    *,
    policy: UploadSecurityPolicy | None = None,
    max_text_chars: int | None = None,
) -> str:
    policy = policy or UploadSecurityPolicy()
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("DOCX file is invalid or corrupted.") from error
    validate_docx_archive(archive, policy=policy)

    parts = []
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    xml_names = [
        name
        for name in archive.namelist()
        if DOCX_TEXT_PART_RE.match(name)
    ]
    for xml_name in xml_names:
        try:
            root = ElementTree.fromstring(archive.read(xml_name))
        except (ElementTree.ParseError, UnicodeDecodeError):
            continue
        for paragraph in root.findall(".//w:p", namespaces):
            runs = [
                node.text or ""
                for node in paragraph.findall(".//w:t", namespaces)
                if node.text
            ]
            line = "".join(runs).strip()
            if line:
                parts.append(line)
    max_chars = policy.max_extracted_text_chars if max_text_chars is None else max_text_chars
    return truncate_extracted_text(normalize_extracted_text("\n".join(parts)), max_chars)


def validate_docx_archive(archive: zipfile.ZipFile, *, policy: UploadSecurityPolicy) -> None:
    infos = archive.infolist()
    if not infos:
        raise ValueError("DOCX file is empty or corrupted.")
    if len(infos) > policy.max_docx_zip_entries:
        raise ValueError(
            f"DOCX file has too many internal parts. Please upload a smaller document "
            f"with at most {policy.max_docx_zip_entries} parts."
        )

    names = {info.filename.replace("\\", "/") for info in infos}
    if not DOCX_REQUIRED_PARTS.issubset(names):
        raise ValueError("DOCX file structure is invalid or unsafe.")

    total_uncompressed = 0
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if normalized.startswith("/") or ".." in parts:
            raise ValueError("DOCX file contains unsafe internal paths.")
        total_uncompressed += int(info.file_size or 0)
        if total_uncompressed > policy.max_docx_uncompressed_bytes:
            raise ValueError(
                f"DOCX file expands to more than {policy.max_docx_uncompressed_bytes // (1024 * 1024)} MB. "
                "Please upload a smaller document."
            )
        if DOCX_TEXT_PART_RE.match(normalized) and info.file_size > policy.max_docx_xml_bytes:
            raise ValueError(
                f"DOCX text content is too large. Please keep each document XML part under "
                f"{policy.max_docx_xml_bytes // (1024 * 1024)} MB."
            )
        compressed = max(int(info.compress_size or 0), 1)
        if info.file_size > compressed * policy.max_docx_compression_ratio:
            raise ValueError("DOCX file has an unsafe compression ratio and was rejected.")


def read_multipart_uploads(
    *,
    headers,
    rfile,
    max_upload_bytes: int,
    max_upload_mb: int,
    security_policy: UploadSecurityPolicy | None = None,
    allow_empty: bool = False,
) -> tuple[list[tuple[str, bytes]], list[dict], dict[str, str]]:
    policy_config = security_policy or UploadSecurityPolicy(
        max_total_bytes=max_upload_bytes,
        max_total_mb=max_upload_mb,
    )
    max_upload_bytes = policy_config.max_total_bytes
    max_upload_mb = policy_config.max_total_mb
    content_type = headers.get("Content-Type", "")
    if not str(content_type or "").lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

    try:
        content_length = int(headers.get("Content-Length", "0"))
    except (TypeError, ValueError) as error:
        raise ValueError("Upload body length is invalid.") from error
    if content_length <= 0:
        raise ValueError("Upload body cannot be empty.")
    if content_length > max_upload_bytes:
        raise ValueError(
            f"Upload is too large. Please keep it under {max_upload_mb} MB. "
            "Try uploading fewer files at a time."
        )

    raw_body = rfile.read(content_length)
    message = BytesParser(policy=policy.default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        + raw_body
    )
    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

    values: dict[str, list[str]] = {}
    files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        if part.is_multipart() or part.get_content_disposition() != "form-data":
            continue
        name = str(part.get_param("name", header="content-disposition") or "")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            if name != "pdf" or not payload:
                continue
            if len(files) >= policy_config.max_file_count:
                raise ValueError(
                    f"Literature analysis accepts at most {policy_config.max_file_count} PDF/DOCX files per upload. "
                    "Please upload fewer files or split them into batches."
                )
            files.append(_validated_upload_file(filename, payload, policy=policy_config))
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset)
        except (LookupError, UnicodeDecodeError):
            text = payload.decode("utf-8", errors="replace")
        values.setdefault(name, []).append(text)

    references = multipart_references_from_value(_first_value(values, "references", "[]"))
    fields = multipart_fields_from_values(values)

    if not allow_empty and not files and not references:
        raise ValueError("Please upload at least one non-empty PDF/DOCX file or provide references.")
    return files, references, fields


def _validated_upload_file(
    filename: str,
    content: bytes,
    *,
    policy: UploadSecurityPolicy | None = None,
) -> tuple[str, bytes]:
    policy = policy or UploadSecurityPolicy()
    filename = sanitize_upload_filename(filename)
    if len(content) > policy.max_file_bytes:
        raise ValueError(
            f"{filename} is too large. Please keep each file under {policy.max_file_mb} MB."
        )
    suffix = Path(filename).suffix.lower()
    if suffix == ".doc":
        raise ValueError(
            f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
        )
    if suffix not in {".pdf", ".docx"}:
        raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")
    detected_type = detect_upload_file_type(content)
    expected_type = suffix.lstrip(".")
    if detected_type != expected_type:
        raise ValueError(
            f"{filename} does not match its file extension. Please upload a valid PDF or DOCX file."
        )
    if detected_type == "docx":
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as error:
            raise ValueError("DOCX file is invalid or corrupted.") from error
        validate_docx_archive(archive, policy=policy)
    scan_upload_for_viruses(content, filename=filename, policy=policy)
    return filename, content


def sanitize_upload_filename(filename: str) -> str:
    name = unicodedata.normalize("NFKC", Path(filename or "uploaded-document").name)
    name = name.replace("\x00", "")
    name = SAFE_UPLOAD_FILENAME_RE.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    if not name:
        return "uploaded-document"
    if len(name) <= 160:
        return name
    suffix = Path(name).suffix[:20]
    stem = Path(name).stem[: max(1, 160 - len(suffix))]
    return f"{stem}{suffix}" or "uploaded-document"


def detect_upload_file_type(content: bytes) -> str:
    head = bytes(content[:1024])
    if head.startswith(b"%PDF-"):
        return "pdf"
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return "unknown"
    names = {name.replace("\\", "/") for name in archive.namelist()}
    if DOCX_REQUIRED_PARTS.issubset(names):
        return "docx"
    return "unknown"


def scan_upload_for_viruses(content: bytes, *, filename: str, policy: UploadSecurityPolicy) -> None:
    mode = str(policy.virus_scan_mode or "off").strip().casefold()
    if mode in {"", "off", "disabled", "false", "0"}:
        return
    if mode not in {"clamav", "required", "on", "true", "1"}:
        raise ValueError("Upload virus scan mode is invalid.")
    try:
        clamav_instream_scan(
            content,
            host=policy.clamav_host,
            port=policy.clamav_port,
            timeout_seconds=policy.clamav_timeout_seconds,
        )
    except ValueError:
        raise
    except Exception as error:
        if mode == "required":
            raise ValueError("Upload virus scanner is unavailable; refusing file for safety.") from error
        print(
            f"[web] upload virus scan skipped for {filename}: {type(error).__name__}: {error}",
            flush=True,
        )


def clamav_instream_scan(
    content: bytes,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    target = str(host or "").strip()
    if target.startswith("unix:"):
        target = target.removeprefix("unix:")
    if target.startswith("/"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        sock.connect(target)
    else:
        sock = socket.create_connection((target, port), timeout=timeout_seconds)
    with sock:
        sock.settimeout(timeout_seconds)
        sock.sendall(b"zINSTREAM\0")
        chunk_size = 1024 * 1024
        for offset in range(0, len(content), chunk_size):
            chunk = content[offset : offset + chunk_size]
            sock.sendall(struct.pack("!I", len(chunk)))
            sock.sendall(chunk)
        sock.sendall(struct.pack("!I", 0))
        response = sock.recv(4096).decode("utf-8", errors="replace")
    if "FOUND" in response:
        raise ValueError("Upload was rejected by virus scanning.")
    if "OK" not in response:
        raise RuntimeError(f"Unexpected ClamAV response: {response.strip()}")


def _first_value(values: dict[str, list[str]], key: str, default: str = "") -> str:
    items = values.get(key) or []
    return str(items[0] if items else default)


def multipart_references(form) -> list[dict]:
    raw = form.getvalue("references", "[]")
    if isinstance(raw, list):
        raw = raw[0] if raw else "[]"
    return multipart_references_from_value(raw)


def multipart_references_from_value(raw: str) -> list[dict]:
    try:
        data = json.loads(str(raw or "[]"))
    except json.JSONDecodeError as error:
        raise ValueError("References field must be valid JSON.") from error
    if not isinstance(data, list):
        raise ValueError("References field must be a JSON list.")
    return [dict(item) for item in data if isinstance(item, dict) and str(item.get("title", "")).strip()]


def multipart_fields(form) -> dict[str, str]:
    values = {}
    for key in MULTIPART_FIELD_KEYS:
        value = form.getvalue(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        values[key] = [str(value or "")]
    return multipart_fields_from_values(values)


def multipart_fields_from_values(values: dict[str, list[str]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in MULTIPART_FIELD_KEYS:
        fields[key] = _first_value(values, key, "")
    return fields
