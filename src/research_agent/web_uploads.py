from __future__ import annotations

from email import policy
from email.parser import BytesParser
import io
import json
from pathlib import Path
import re
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


def extract_docx_text(content: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("DOCX file is invalid or corrupted.") from error

    parts = []
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    xml_names = [
        name
        for name in archive.namelist()
        if name == "word/document.xml"
        or name.startswith("word/header")
        or name.startswith("word/footer")
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
    return normalize_extracted_text("\n".join(parts))


def read_multipart_uploads(
    *,
    headers,
    rfile,
    max_upload_bytes: int,
    max_upload_mb: int,
    allow_empty: bool = False,
) -> tuple[list[tuple[str, bytes]], list[dict], dict[str, str]]:
    content_type = headers.get("Content-Type", "")
    if not str(content_type or "").lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data with PDF or DOCX files.")

    content_length = int(headers.get("Content-Length", "0"))
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
            files.append(_validated_upload_file(filename, payload))
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


def _validated_upload_file(filename: str, content: bytes) -> tuple[str, bytes]:
    filename = Path(filename or "uploaded-document").name
    suffix = Path(filename).suffix.lower()
    if suffix == ".doc":
        raise ValueError(
            f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
        )
    if suffix not in {".pdf", ".docx"}:
        raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")
    return filename, content


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
