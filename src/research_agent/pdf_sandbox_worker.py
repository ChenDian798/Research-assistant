from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from src.research_agent.web_uploads import normalize_extracted_text, truncate_extracted_text


def _clean_pdf_page_text(raw_text: str, *, max_chars: int) -> str:
    return truncate_extracted_text(normalize_extracted_text(raw_text or ""), max_chars)


def _clean_pdf_metadata(metadata) -> dict:
    if not metadata:
        return {}
    cleaned = {}
    for key, value in dict(metadata).items():
        normalized_key = str(key).lstrip("/").lower()
        if normalized_key in {"title", "author", "subject", "creator", "producer"}:
            cleaned[normalized_key] = normalize_extracted_text(str(value or ""))[:500]
    return cleaned


def _extract_with_pymupdf(content: bytes, *, page_limit: int | None, max_text_chars: int) -> dict | None:
    try:
        import fitz
    except ImportError:
        return None

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return None

    try:
        page_count = int(document.page_count)
        scan_count = page_count if page_limit is None else min(page_count, page_limit)
        parts = []
        extracted_pages = 0
        remaining = max_text_chars
        for page_index in range(scan_count):
            if remaining <= 0:
                break
            try:
                page = document.load_page(page_index)
                page_text = _clean_pdf_page_text(page.get_text("text") or "", max_chars=remaining)
            except Exception:
                continue
            if not page_text:
                continue
            extracted_pages += 1
            part = f"[Page {page_index + 1}]\n{page_text}"
            parts.append(part)
            remaining -= len(part)
        text = "\n\n".join(parts).strip()
        return {
            "text": truncate_extracted_text(text, max_text_chars),
            "page_count": page_count,
            "extracted_pages": extracted_pages,
        }
    finally:
        document.close()


def extract_pdf(
    content: bytes,
    *,
    page_limit: int | None,
    max_page_count: int,
    max_text_chars: int,
) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF extraction requires the pypdf package. Run: pip install pypdf") from error

    reader = PdfReader(io.BytesIO(content))
    page_count = len(reader.pages)
    if max_page_count > 0 and page_count > max_page_count:
        raise ValueError(
            f"PDF has {page_count} pages, which exceeds the {max_page_count}-page upload safety limit."
        )
    metadata = _clean_pdf_metadata(reader.metadata)
    parts = []
    extracted_pages = 0
    skipped_pages = 0
    scan_count = page_count if page_limit is None else min(page_count, page_limit)
    remaining = max_text_chars
    for page_number, page in enumerate(reader.pages[:scan_count], start=1):
        if remaining <= 0:
            break
        try:
            page_text = _clean_pdf_page_text(page.extract_text() or "", max_chars=remaining)
        except Exception:
            skipped_pages += 1
            continue
        if not page_text:
            continue
        extracted_pages += 1
        part = f"[Page {page_number}]\n{page_text}"
        parts.append(part)
        remaining -= len(part)

    text = truncate_extracted_text("\n\n".join(parts).strip(), max_text_chars)
    fallback_note = ""
    if skipped_pages or not text:
        fallback = _extract_with_pymupdf(content, page_limit=page_limit, max_text_chars=max_text_chars)
        if fallback and int(fallback.get("extracted_pages") or 0) > extracted_pages:
            text = fallback["text"]
            extracted_pages = int(fallback["extracted_pages"])
            page_count = max(page_count, int(fallback["page_count"]))
            fallback_note = " Used PyMuPDF fallback extraction."

    if text:
        scan_note = "all pages scanned" if page_limit is None else f"first {min(page_count, page_limit)} pages scanned"
        note = f"Extracted text from {extracted_pages}/{page_count} pages ({scan_note}).{fallback_note}"
    else:
        note = "No readable text extracted; the PDF may be scanned, image-only, or protected."
    if skipped_pages:
        note = f"{note} Skipped {skipped_pages} page(s) because pypdf could not extract text from them."
    return {
        "text": text,
        "page_count": page_count,
        "extracted_pages": extracted_pages,
        "metadata": metadata,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--page-limit", default="120")
    parser.add_argument("--max-page-count", type=int, default=300)
    parser.add_argument("--max-text-chars", type=int, default=300_000)
    args = parser.parse_args()

    page_limit = None if str(args.page_limit).casefold() in {"all", "none", "unlimited", "0"} else max(1, int(args.page_limit))
    try:
        result = extract_pdf(
            Path(args.input).read_bytes(),
            page_limit=page_limit,
            max_page_count=max(0, int(args.max_page_count)),
            max_text_chars=max(1000, int(args.max_text_chars)),
        )
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
