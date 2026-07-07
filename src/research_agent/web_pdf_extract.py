from __future__ import annotations

import importlib
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Callable


def _extract_pdf_content(
    content: bytes,
    *,
    pdf_parser_mode: Callable[[], str],
    extract_pdf_content_basic: Callable[[bytes], dict],
    should_try_opendataloader_pdf: Callable[[dict], bool],
    extract_pdf_content_with_opendataloader: Callable[[bytes], dict | None],
) -> dict:
    mode = pdf_parser_mode()
    if mode == "opendataloader":
        enhanced = extract_pdf_content_with_opendataloader(content)
        if enhanced:
            return enhanced
        print(
            "[web] OpenDataLoader PDF extraction was requested but unavailable; falling back to basic PDF extraction.",
            flush=True,
        )
        return extract_pdf_content_basic(content)

    basic = extract_pdf_content_basic(content)
    if mode == "auto" and should_try_opendataloader_pdf(basic):
        enhanced = extract_pdf_content_with_opendataloader(content)
        if enhanced and enhanced.get("text"):
            enhanced["metadata"] = enhanced.get("metadata") or basic.get("metadata", {})
            enhanced["page_count"] = max(
                int(enhanced.get("page_count") or 0),
                int(basic.get("page_count") or 0),
            )
            if not int(enhanced.get("extracted_pages") or 0):
                enhanced["extracted_pages"] = int(basic.get("extracted_pages") or 0)
            enhanced["note"] = (
                f"{enhanced.get('note', 'Extracted with OpenDataLoader PDF.')} "
                f"Used because basic PDF extraction looked incomplete. Basic note: {basic.get('note', '')}"
            ).strip()
            return enhanced
    return basic


def _extract_pdf_content_basic(
    content: bytes,
    *,
    pdf_extract_page_limit: Callable[[], int | None],
    clean_pdf_metadata: Callable[[object], dict],
    clean_pdf_page_text: Callable[[str], str],
    extract_pdf_content_with_pymupdf: Callable[[bytes], dict | None],
) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF extraction requires the pypdf package. Run: pip install pypdf") from error

    reader = PdfReader(io.BytesIO(content))
    page_count = len(reader.pages)
    metadata = clean_pdf_metadata(reader.metadata)
    parts = []
    extracted_pages = 0
    skipped_page_notes = []
    page_limit = pdf_extract_page_limit()
    pages_to_scan = reader.pages if page_limit is None else reader.pages[:page_limit]
    for page_number, page in enumerate(pages_to_scan, start=1):
        try:
            page_text = clean_pdf_page_text(page.extract_text() or "")
        except Exception as error:
            skipped_page_notes.append(
                f"page {page_number}: {type(error).__name__}: {error}"
            )
            print(
                f"[web] skipped PDF page {page_number} during text extraction: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        if not page_text:
            continue
        extracted_pages += 1
        parts.append(f"[Page {page_number}]\n{page_text}")

    text = "\n\n".join(parts).strip()
    fallback_note = ""
    if skipped_page_notes or not text:
        fallback = extract_pdf_content_with_pymupdf(content)
        if fallback and fallback["extracted_pages"] > extracted_pages:
            text = fallback["text"]
            extracted_pages = fallback["extracted_pages"]
            page_count = max(page_count, fallback["page_count"])
            fallback_note = " Used PyMuPDF fallback extraction."
            print(
                f"[web] used PyMuPDF fallback PDF extraction; extracted "
                f"{extracted_pages}/{page_count} pages.",
                flush=True,
            )
    if text:
        scan_note = "all pages scanned" if page_limit is None else f"first {min(page_count, page_limit)} pages scanned"
        note = (
            f"Extracted text from {extracted_pages}/{page_count} pages "
            f"({scan_note}).{fallback_note}"
        )
    else:
        note = "No readable text extracted; the PDF may be scanned, image-only, or protected."
    if skipped_page_notes:
        note = (
            f"{note} Skipped {len(skipped_page_notes)} page(s) because pypdf "
            "could not extract text from them."
        )
    return {
        "text": text,
        "page_count": page_count,
        "extracted_pages": extracted_pages,
        "metadata": metadata,
        "note": note,
    }


def _should_try_opendataloader_pdf(
    extracted: dict,
    *,
    pdf_opendataloader_min_chars: Callable[[], int],
    pdf_opendataloader_min_page_ratio: Callable[[], float],
) -> bool:
    text = str(extracted.get("text") or "")
    if len(text) < pdf_opendataloader_min_chars():
        return True
    page_count = int(extracted.get("page_count") or 0)
    extracted_pages = int(extracted.get("extracted_pages") or 0)
    if page_count <= 0:
        return not text
    if extracted_pages <= 0:
        return True
    page_ratio = extracted_pages / max(page_count, 1)
    return page_ratio < pdf_opendataloader_min_page_ratio()


def _opendataloader_convert_function():
    for module_name in (
        "opendataloader_pdf",
        "opendataloader_pdf.convert",
        "opendataloader.pdf",
        "opendataloader",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        converter = getattr(module, "convert", None)
        if callable(converter):
            return converter
    return None


def _extract_pdf_content_with_opendataloader(
    content: bytes,
    *,
    opendataloader_convert_function: Callable[[], object],
    run_opendataloader_convert: Callable[..., object],
    opendataloader_result_markdown: Callable[[object], str],
    read_opendataloader_markdown: Callable[[Path], str],
    opendataloader_result_json: Callable[[object], object],
    read_opendataloader_json: Callable[[Path], object],
    text_from_opendataloader_json: Callable[[object], str],
    clean_pdf_page_text: Callable[[str], str],
    page_count_from_opendataloader_json: Callable[[object], int],
    page_count_from_marked_text: Callable[[str], int],
) -> dict | None:
    converter = opendataloader_convert_function()
    if not converter:
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="research_pdf_odl_") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "upload.pdf"
            output_dir = temp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(content)

            result = run_opendataloader_convert(
                converter,
                input_path=input_path,
                output_dir=output_dir,
            )
            markdown = opendataloader_result_markdown(result)
            if not markdown:
                markdown = read_opendataloader_markdown(output_dir)
            json_data = opendataloader_result_json(result)
            if json_data is None:
                json_data = read_opendataloader_json(output_dir)
            if not markdown and json_data is not None:
                markdown = text_from_opendataloader_json(json_data)
            text = clean_pdf_page_text(markdown or "")
            if not text:
                return None
            page_count = page_count_from_opendataloader_json(json_data)
            extracted_pages = page_count or page_count_from_marked_text(text) or 1
            return {
                "text": text,
                "page_count": page_count or extracted_pages,
                "extracted_pages": extracted_pages,
                "metadata": {},
                "note": "Extracted with OpenDataLoader PDF.",
            }
    except Exception as error:
        print(
            f"[web] OpenDataLoader PDF extraction failed: {type(error).__name__}: {error}",
            flush=True,
        )
        return None


def _run_opendataloader_convert(converter, *, input_path: Path, output_dir: Path):
    attempts = [
        {
            "input_path": [str(input_path)],
            "output_dir": str(output_dir),
            "format": "markdown,json",
        },
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "format": "markdown,json",
        },
        {
            "pdf_path": str(input_path),
            "output_dir": str(output_dir),
            "format": "markdown,json",
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return converter(**kwargs)
        except TypeError as error:
            last_error = error
            continue
    try:
        return converter(str(input_path), str(output_dir))
    except TypeError as error:
        raise last_error or error


def _opendataloader_result_markdown(result) -> str:
    for attr in ("markdown", "md", "text", "content"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(result, dict):
        for key in ("markdown", "md", "text", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _opendataloader_result_json(result):
    for attr in ("json", "data", "result"):
        value = getattr(result, attr, None)
        if value is not None:
            return value
    if isinstance(result, dict):
        for key in ("json", "data", "result"):
            if key in result:
                return result[key]
    return None


def _read_opendataloader_markdown(output_dir: Path) -> str:
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".md", ".markdown"}
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return candidates[0].read_text(encoding="utf-8", errors="replace")


def _read_opendataloader_json(output_dir: Path):
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".json"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    with candidates[0].open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _text_from_opendataloader_json(data) -> str:
    parts: list[str] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"markdown", "md", "text", "content"} and isinstance(child, str):
                    if child.strip():
                        parts.append(child.strip())
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return "\n\n".join(parts).strip()


def _page_count_from_opendataloader_json(data) -> int:
    if data is None:
        return 0
    counts: list[int] = []
    pages: list[int] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).casefold()
                if key_lower in {"page_count", "pagecount", "num_pages", "total_pages"}:
                    if isinstance(child, int):
                        counts.append(child)
                elif key_lower in {"page", "page_number", "page_num"}:
                    if isinstance(child, int):
                        pages.append(child)
                elif key_lower == "page_index":
                    if isinstance(child, int):
                        pages.append(child + 1)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return max(counts + pages, default=0)


def _page_count_from_marked_text(text: str) -> int:
    pages = [
        int(match)
        for match in re.findall(r"(?im)^\s*\[?page\s+(\d+)\]?\s*$", text or "")
    ]
    return max(pages, default=0)


def _clean_pdf_page_text(raw_text: str, *, normalize_extracted_text: Callable[[str], str]) -> str:
    page_text = normalize_extracted_text(raw_text or "")
    page_lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in page_text.splitlines()
    ]
    return "\n".join(line for line in page_lines if line).strip()


def _extract_pdf_content_with_pymupdf(
    content: bytes,
    *,
    page_limit: int | None,
    clean_pdf_page_text: Callable[[str], str],
) -> dict | None:
    try:
        import fitz
    except ImportError:
        return None

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as error:
        print(
            f"[web] PyMuPDF fallback PDF extraction failed to open document: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return None

    try:
        page_count = int(document.page_count)
        scan_count = page_count if page_limit is None else min(page_count, page_limit)
        parts = []
        extracted_pages = 0
        for page_index in range(scan_count):
            try:
                page = document.load_page(page_index)
                page_text = clean_pdf_page_text(page.get_text("text") or "")
            except Exception as error:
                print(
                    f"[web] PyMuPDF skipped PDF page {page_index + 1}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            if not page_text:
                continue
            extracted_pages += 1
            parts.append(f"[Page {page_index + 1}]\n{page_text}")
        return {
            "text": "\n\n".join(parts).strip(),
            "page_count": page_count,
            "extracted_pages": extracted_pages,
        }
    finally:
        document.close()


def _clean_pdf_metadata(metadata, *, normalize_extracted_text: Callable[[str], str]) -> dict:
    if not metadata:
        return {}
    cleaned = {}
    for key, value in dict(metadata).items():
        normalized_key = str(key).lstrip("/").lower()
        if normalized_key in {"title", "author", "subject", "creator", "producer"}:
            cleaned[normalized_key] = normalize_extracted_text(str(value or ""))[:500]
    return cleaned
