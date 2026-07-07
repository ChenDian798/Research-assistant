from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

from src.research_agent.doi import extract_arxiv_id, extract_doi, extract_pmid
from src.research_agent.web_uploads import normalize_pdf_text_symbols


def _uploaded_file_to_reference(
    filename: str,
    content: bytes,
    *,
    expected_context: str = "",
    extract_pdf_content: Callable[[bytes], dict],
    extract_docx_text: Callable[[bytes], str],
    build_high_information_package: Callable[[str], str],
    extract_doi_func: Callable[[dict], str | None] = extract_doi,
    extract_arxiv_id_func: Callable[[dict], str | None] = extract_arxiv_id,
    extract_pmid_func: Callable[[dict], str | None] = extract_pmid,
) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        extracted = extract_pdf_content(content)
        text = extracted["text"]
        has_text = bool(text)
        if not has_text:
            text = "未能从 PDF 中提取到可读文本，可能是扫描件或受保护文档。"
        excerpt = build_high_information_package(text)
        metadata = extracted["metadata"]
        bibliographic = _infer_pdf_bibliographic_metadata(
            filename=filename,
            text=text,
            metadata=metadata,
        )
        bibliographic_identity = _build_pdf_bibliographic_identity(
            filename=filename,
            text=text if has_text else "",
            metadata=metadata,
        )
        identity_gate = _pdf_identity_review_gate(
            filename=filename,
            title=bibliographic.get("title") or Path(filename).stem or filename,
            text=text if has_text else "",
            metadata=metadata,
            expected_context=expected_context,
        )
        reference = {
            "title": bibliographic.get("title") or Path(filename).stem or filename,
            "source": filename,
            "uploaded_filename": filename,
            "relevance": (
                "用户上传 PDF，系统已提取正文片段用于分析。"
                if has_text
                else "用户上传 PDF，但系统未提取到可读正文；需基于元数据谨慎分析。"
            ),
            "branch_name": "文件上传",
            "abstract": excerpt,
            "content_excerpt": excerpt,
            "evidence_source_text": text if has_text else "",
            "full_text_for_evidence": text if has_text else "",
            "pdf_text_available": has_text,
            "pdf_page_count": extracted["page_count"],
            "pdf_extracted_pages": extracted["extracted_pages"],
            "pdf_extraction_note": extracted["note"],
            "pdf_content_sha256": hashlib.sha256(content).hexdigest(),
            "authors": bibliographic.get("authors", []),
            "year": bibliographic.get("year", ""),
            "journal": bibliographic.get("journal", ""),
            "bibliographic_identity": bibliographic_identity,
            "pdf_metadata": metadata,
            "document_type": "PDF",
            "source_origin": "user_upload",
            "source_label": "Uploaded PDF",
        }
        if identity_gate:
            reference.update(identity_gate)
            reference["relevance"] = identity_gate["review_note"]
            reference["branch_name"] = "Uploaded PDF needs review"
            reference["abstract"] = (
                f"{identity_gate['review_note']}\n\nExtracted text preview:\n"
                f"{_debug_preview(text, 1200)}"
            ).strip()
            reference["content_excerpt"] = reference["abstract"]
            reference["evidence_source_text"] = ""
            reference["full_text_for_evidence"] = ""
        _log_uploaded_pdf_reference_debug(
            filename=filename,
            content=content,
            extracted=extracted,
            bibliographic=bibliographic,
            bibliographic_identity=bibliographic_identity,
            text=text if has_text else "",
            reference=reference,
        )
        doi = extract_doi_func(reference)
        arxiv_id = extract_arxiv_id_func(reference)
        pmid = extract_pmid_func(reference)
        if doi:
            reference["doi"] = doi
        if arxiv_id:
            reference["arxiv_id"] = arxiv_id
        if pmid:
            reference["pmid"] = pmid
        return reference

    if suffix == ".docx":
        text = extract_docx_text(content)
        has_text = bool(text)
        if not has_text:
            text = "未能从 DOCX 中提取到可读文本。"
        excerpt = build_high_information_package(text)
        return {
            "title": Path(filename).stem or filename,
            "source": filename,
            "uploaded_filename": filename,
            "relevance": (
                "用户上传 DOCX，系统已提取正文片段用于分析。"
                if has_text
                else "用户上传 DOCX，但系统未提取到可读正文；需谨慎分析。"
            ),
            "branch_name": "文件上传",
            "abstract": excerpt,
            "content_excerpt": excerpt,
            "evidence_source_text": text if has_text else "",
            "full_text_for_evidence": text if has_text else "",
            "pdf_text_available": has_text,
            "pdf_page_count": 0,
            "pdf_extracted_pages": 0,
            "pdf_extraction_note": "Extracted text from DOCX." if has_text else "No readable text extracted from DOCX.",
            "authors": "",
            "year": "",
            "journal": "",
            "pdf_metadata": {},
            "document_type": "DOCX",
            "source_origin": "user_upload",
            "source_label": "Uploaded DOCX",
        }

    if suffix == ".doc":
        raise ValueError(
            f"{filename} is a legacy .doc file. Please save/export it as .docx or PDF, then upload again."
        )
    raise ValueError(f"{filename} is not supported. Please upload PDF or DOCX files.")


def _pdf_identity_review_gate(
    *,
    filename: str,
    title: str,
    text: str,
    metadata: dict | None = None,
    expected_context: str = "",
) -> dict:
    metadata = metadata or {}
    identity_text = "\n".join(
        [
            filename,
            title,
            str(metadata.get("title", "") or ""),
            str(metadata.get("subject", "") or ""),
        ]
    ).casefold()
    body = str(text or "")[:12000].casefold()
    compact_body = re.sub(r"\s+", " ", body)
    expected = str(expected_context or "").casefold()

    production_title = bool(
        re.search(r"\b(?:indd|v\d{2,}|proof|galley|layout|typeset)\b", identity_text)
        or re.search(r"\b\d{3,}\s*-\s*v\d+\b", identity_text)
    )
    metadata_title = str(metadata.get("title", "") or "").strip().casefold()
    selected_title = str(title or "").strip()
    selected_title_is_stable = bool(
        selected_title
        and _clean_bibliographic_title(selected_title)
        and not _looks_like_pdf_body_sentence(selected_title)
        and selected_title.casefold() != metadata_title
        and selected_title.casefold() in compact_body
    )
    fossil_topic_hits = sum(
        1
        for term in [
            "ams",
            "14c",
            "amino acid",
            "enantiomer",
            "fossil",
            "indigeneity",
            "racemization",
            "diagenesis",
            "stable isotope",
            "letters to nature",
        ]
        if term in body
    )
    ml_expected_hits = sum(
        1
        for term in [
            "machine learning",
            "deep learning",
            "neural",
            "random forest",
            "support vector",
            "classification",
            "classifier",
            "regression",
            "dataset",
            "training",
        ]
        if term in expected
    )
    ml_body_hits = sum(
        1
        for term in [
            "machine learning",
            "deep learning",
            "neural network",
            "random forest",
            "support-vector",
            "support vector",
            "classifier",
            "training set",
            "test set",
        ]
        if term in body
    )

    reasons = []
    context_conflict = fossil_topic_hits >= 3 and ml_expected_hits >= 1
    mixed_body_topics = fossil_topic_hits >= 3 and ml_body_hits >= 1 and not selected_title_is_stable

    if production_title:
        reasons.append("PDF title/metadata looks like a production or layout filename, not a stable paper title")
    if mixed_body_topics:
        reasons.append("extracted text mixes fossil/AMS/amino-acid content with machine-learning text inside the same PDF")
    if production_title and context_conflict and not selected_title_is_stable:
        reasons.append("first-page evidence suggests an unrelated fossil/AMS article under an unstable title")
    if selected_title_is_stable and reasons == ["PDF title/metadata looks like a production or layout filename, not a stable paper title"]:
        return {}
    if context_conflict and not (production_title or mixed_body_topics):
        return {}
    if not reasons:
        return {}

    note = (
        "待复核材料：PDF 文本与用户预期主题或文献身份不一致，请确认上传文件、拼接页、隐藏文本/OCR 层或 PDF 元数据。"
    )
    return {
        "document_role": "review_needed",
        "is_literature_source": False,
        "pdf_identity_status": "needs_review",
        "review_note": note,
        "review_reasons": reasons,
    }


def _debug_preview(value: str, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _log_uploaded_pdf_reference_debug(
    *,
    filename: str,
    content: bytes,
    extracted: dict,
    bibliographic: dict,
    bibliographic_identity: str,
    text: str,
    reference: dict,
) -> None:
    debug_payload = {
        "filename": filename,
        "content_length": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "pdf_page_count": extracted.get("page_count"),
        "pdf_extracted_pages": extracted.get("extracted_pages"),
        "pdf_extraction_note": extracted.get("note"),
        "metadata_title": (extracted.get("metadata") or {}).get("title", ""),
        "metadata_author": (extracted.get("metadata") or {}).get("author", ""),
        "inferred_title": bibliographic.get("title", ""),
        "document_role": reference.get("document_role", "literature"),
        "pdf_identity_status": reference.get("pdf_identity_status", "ok"),
        "review_reasons": reference.get("review_reasons", []),
        "first_800_chars": _debug_preview(text, 800),
        "bibliographic_identity_first_800_chars": _debug_preview(
            bibliographic_identity,
            800,
        ),
    }
    print(
        "[web] uploaded PDF reference debug "
        + json.dumps(debug_payload, ensure_ascii=False),
        flush=True,
    )


def _infer_pdf_bibliographic_metadata(filename: str, text: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    metadata_title = _clean_bibliographic_title(str(metadata.get("title", "") or ""))
    metadata_author = _clean_pdf_author_string(str(metadata.get("author", "") or ""))
    first_page = _first_pdf_page_text(text)
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in first_page.splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ][:80]

    title = metadata_title
    if not title:
        title = _infer_title_from_pdf_lines(lines)

    authors = _split_pdf_authors(metadata_author)
    if not authors:
        authors = _infer_authors_from_pdf_lines(lines, title)

    year = ""
    for source in [first_page, str(metadata.get("subject", "") or ""), filename]:
        match = re.search(r"\b(19|20)\d{2}\b", source)
        if match:
            year = match.group(0)
            break

    return {
        "title": title or Path(filename).stem or filename,
        "authors": authors,
        "year": year,
        "journal": "",
    }


def _first_pdf_page_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\[Page\s+1\]\s*(.*?)(?=\n\n\[Page\s+2\]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text[:5000]


def _build_pdf_bibliographic_identity(filename: str, text: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    first_page = _first_pdf_page_text(text)
    first_page = re.split(r"\b(?:references|bibliography|works cited)\b", first_page, maxsplit=1, flags=re.IGNORECASE)[0]
    parts = [
        filename,
        str(metadata.get("title", "") or ""),
        str(metadata.get("author", "") or ""),
        str(metadata.get("subject", "") or ""),
        first_page[:4000],
    ]
    return "\n".join(part for part in parts if str(part).strip())


def _clean_bibliographic_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    if not title:
        return ""
    if title.lower() in {"untitled", "document", "paper"}:
        return ""
    if re.fullmatch(r"\[?\s*page\s+\d+\s*\]?", title, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"[\w\s.-]+\.(?:pdf|docx?|indd|qxd)", title, flags=re.IGNORECASE):
        return ""
    if re.search(r"\b(?:indd|proof|galley|layout|typeset)\b", title, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"(?:[a-f0-9]{8,}|\d{3,}|[\d\s_-]*v\d+[\d\s_-]*)", title, flags=re.IGNORECASE):
        return ""
    if re.search(r"\b(?:vol\.?|volume)\s*\d+|\b(?:letters\s+to\s+nature|nature\s+vol)\b", title, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"\d{3,}\s*-\s*v\d+", title, flags=re.IGNORECASE):
        return ""
    if re.search(r"^microsoft word|^latex|^overleaf", title, flags=re.IGNORECASE):
        return ""
    return title[:300]


def _clean_pdf_author_string(author: str) -> str:
    author = re.sub(r"\s+", " ", author or "").strip()
    if re.search(r"^(unknown|anonymous|uploaded|user|pdf)$", author, flags=re.IGNORECASE):
        return ""
    return author


def _infer_title_from_pdf_lines(lines: list[str]) -> str:
    best = _best_scored_title_from_pdf_lines(lines)
    if best:
        return best

    skip = re.compile(
        r"^(?:arxiv|abstract|keywords?|introduction|preprint|submitted|accepted|published|doi\b|"
        r"journal|conference|proceedings|\[?\s*page\s+\d+\s*\]?|\d+|"
        r"letters\s+to\s+nature|nature\s+vol|medical imaging with deep learning|midl\b|short paper\b)",
        flags=re.IGNORECASE,
    )
    candidates = []
    for line in lines[:16]:
        clean = normalize_pdf_text_symbols(line).strip(" -")
        if not clean or skip.search(clean):
            continue
        if not _clean_bibliographic_title(clean):
            continue
        if _looks_like_pdf_body_sentence(clean):
            continue
        if re.search(r"@|www\.|https?://", clean, flags=re.IGNORECASE):
            break
        if re.search(r"\b(?:abstract|university|department|institute|hospital|school|center|centre|corresponding author|word count|short title)\b", clean, flags=re.IGNORECASE):
            break
        if _looks_like_pdf_author_line(clean):
            break
        if len(clean) < 3 and not candidates:
            continue
        if len(clean) > 180 and not candidates:
            return _trim_pdf_title_noise(clean)
        candidates.append(clean)
        if len(candidates) >= 5:
            break
    if not candidates:
        return ""
    title = " ".join(candidates)
    title = re.sub(r"\s+", " ", title).strip()
    return _trim_pdf_title_noise(title)[:300]


def _best_scored_title_from_pdf_lines(lines: list[str]) -> str:
    scored: list[tuple[int, int, str]] = []
    for index, raw_line in enumerate(lines[:80]):
        first = normalize_pdf_text_symbols(raw_line).strip(" -")
        if not _is_plausible_pdf_title_line(first):
            continue
        candidate_lines = [first]
        for next_line in lines[index + 1 : min(len(lines), index + 5)]:
            clean_next = normalize_pdf_text_symbols(next_line).strip(" -")
            if not clean_next:
                continue
            if _looks_like_pdf_author_line(clean_next):
                break
            if _looks_like_pdf_affiliation_line(clean_next):
                break
            if not _is_plausible_pdf_title_continuation(clean_next):
                break
            candidate_lines.append(clean_next)
        candidate = _trim_pdf_title_noise(" ".join(candidate_lines))
        if not candidate or len(candidate) < 8:
            continue
        score = _score_pdf_title_candidate(candidate, lines, index, len(candidate_lines))
        if score >= 18:
            scored.append((score, -index, candidate[:300]))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def _is_plausible_pdf_title_line(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text or len(text) < 4 or len(text) > 180:
        return False
    if _clean_bibliographic_title(text) == "":
        return False
    if _looks_like_pdf_author_line(text):
        return False
    if _looks_like_pdf_affiliation_line(text):
        return False
    if re.search(r"@|https?://|www\.|^\[?page\s+\d+\]?$", text, flags=re.IGNORECASE):
        return False
    if re.search(
        r"^(?:abstract|keywords?|introduction|references|bibliography|received|accepted|editor:|"
        r"journal|conference|proceedings|figure|table)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:nature publishing group|kluwer academic publishers|manufactured in)\b", text, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[\d\s.,;:()/-]+", text):
        return False
    return True


def _is_plausible_pdf_title_continuation(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not _is_plausible_pdf_title_line(text):
        return False
    if len(text.split()) > 12:
        return False
    if text.endswith(".") and not re.search(r"\b(?:vs\.|etc\.)$", text, flags=re.IGNORECASE):
        return False
    if re.match(r"^(?:we|this|the|as|in|for|there|older|finally)\b", text, flags=re.IGNORECASE):
        return False
    return True


def _looks_like_pdf_body_sentence(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return False
    if re.match(r"^(?:as|older|finally|also|there|if|to|this|the|we describe|we report)\b", text, flags=re.IGNORECASE):
        return True
    if text.endswith(".") and len(text.split()) > 10:
        return True
    return False


def _score_pdf_title_candidate(candidate: str, lines: list[str], start_index: int, line_count: int) -> int:
    text = re.sub(r"\s+", " ", candidate or "").strip()
    score = 0
    words = re.findall(r"[A-Za-z][A-Za-z-]+", text)
    if 2 <= len(words) <= 18:
        score += 10
    if text and text[0].isupper():
        score += 5
    if not text.endswith("."):
        score += 4
    if line_count >= 2:
        score += 5
    if re.search(r"\b(?:learning|network|neural|representation|classification|regression|segmentation|model|algorithm)\b", text, flags=re.IGNORECASE):
        score += 4

    following = [
        normalize_pdf_text_symbols(line).strip()
        for line in lines[start_index + line_count : min(len(lines), start_index + line_count + 8)]
    ]
    if any(_looks_like_pdf_author_line(line) for line in following[:4]):
        score += 35
    if any(_looks_like_pdf_affiliation_line(line) for line in following[:6]):
        score += 8
    if any(re.match(r"^(?:abstract\b|we\s+describe\b|this\s+paper\b)", line, flags=re.IGNORECASE) for line in following):
        score += 6

    if text and text[0].islower():
        score -= 18
    if text.endswith("."):
        score -= 4
    if re.match(r"^(?:as|the|this|we|older|finally|also|there|if|to)\b", text, flags=re.IGNORECASE):
        score -= 10
    if re.search(r"\b(?:grant|acknowledge|support|received|accepted|references?)\b", text, flags=re.IGNORECASE):
        score -= 12
    return score


def _looks_like_pdf_author_line(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return False
    if re.search(r"\b[A-Z][a-zA-Z.-]+\s+[A-Z]\.\s+[A-Z][a-zA-Z.-]+\b", text):
        return True
    if re.search(r"\s*&\s*[A-Z][a-zA-Z.-]+", text) and len(re.findall(r"\b[A-Z][a-zA-Z.-]+\b", text)) >= 3:
        return True
    if re.search(r"\b(?:MD|PhD|MSc|MS|Dr\.|Prof\.)\b", text):
        return True
    if len(re.findall(r"\b[A-Z]\.", text)) >= 2:
        return True
    if "," in text and len(re.findall(r"\b[A-Z][a-zA-Z.-]{2,}\b", text)) >= 3:
        return True
    return False


def _looks_like_pdf_affiliation_line(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:university|department|institute|laborator(?:y|ies)|hospital|school|center|centre|college|"
            r"faculty|academy|at&t|carnegie|mellon|california|pittsburgh)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _trim_pdf_title_noise(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^arXiv:\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w+\s+\d{4}\s+", "", text, flags=re.IGNORECASE)
    stop_patterns = [
        r"\s+\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:MD|PhD|MSc|MS)\b",
        r"\s+\b[A-Z]\.[A-Z]\.\s*[A-Z][a-zA-Z-]+\d*",
        r"\s+\b[A-Z][a-zA-Z-]+\s+[A-Z][a-zA-Z-]+\s+\d",
        r"\s+\bAbstract\b",
        r"\s+\bShort Title\b",
        r"\s+\bWord Count\b",
        r"\s+\b\d[A-Z]?[A-Za-z ]*(?:University|Department|Institute|Hospital|School|Center|Centre)\b",
    ]
    cut = len(text)
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            cut = min(cut, match.start())
    return text[:cut].strip(" ,;:-")


def _infer_authors_from_pdf_lines(lines: list[str], title: str) -> list[str]:
    if not lines or not title:
        return []
    title_words = set(re.findall(r"[A-Za-z]{4,}", title.casefold()))
    title_key = re.sub(r"\s+", " ", title.casefold()).strip()
    start = 0
    for index in range(min(len(lines), 80)):
        parts = []
        for end_index in range(index, min(len(lines), index + 5)):
            parts.append(re.sub(r"\s+", " ", lines[end_index]).strip())
            if re.sub(r"\s+", " ", " ".join(parts).casefold()).strip() == title_key:
                start = end_index + 1
                break
        if start:
            break

    scan_lines = lines[start : min(len(lines), start + 10)] if start else lines[:16]
    for line in scan_lines:
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean or clean == title:
            continue
        if not _clean_bibliographic_title(clean) and re.search(
            r"\b(?:nature|vol|letters)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"abstract|keywords?|doi|arxiv|university|department|institute|hospital|@|www\.|https?://", clean, flags=re.IGNORECASE):
            continue
        words = re.findall(r"[A-Za-z]{2,}", clean)
        if len(words) < 2 or len(words) > 24:
            continue
        overlap = sum(1 for word in words if word.casefold() in title_words)
        if overlap >= max(2, len(words) // 3):
            continue
        authors = _split_pdf_authors(clean)
        if authors:
            return authors[:20]
    return []


def _split_pdf_authors(value: str) -> list[str]:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return []
    text = re.sub(r"\b(corresponding author|affiliations?|authors?)\b:?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\d+|\*|†|‡|§", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    if not text:
        return []
    if ";" in text:
        parts = text.split(";")
    elif re.search(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE):
        parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
    elif "," in text and not re.search(r"[A-Z][a-z]+,\s*[A-Z]\.", text):
        parts = text.split(",")
    else:
        parts = [text]
    authors = []
    for part in parts:
        author = re.sub(r"\([^)]*\)", "", part).strip(" ,")
        if not author or len(author) > 80:
            continue
        if re.search(r"\b(university|department|institute|hospital|center|centre|school|college|lab)\b", author, flags=re.IGNORECASE):
            continue
        if len(re.findall(r"[A-Za-z]", author)) < 3:
            continue
        authors.append(author)
    return authors


def _infer_uploaded_document_role(
    *,
    filename: str,
    text: str,
    metadata: dict | None = None,
    document_type: str = "",
) -> str:
    metadata = metadata or {}
    sample = f"{filename}\n{metadata.get('title', '')}\n{text[:6000]}".casefold()
    requirement_terms = [
        "assignment",
        "rubric",
        "grading",
        "marking criteria",
        "assessment",
        "writing requirement",
        "format requirement",
        "style guide",
        "instructions",
        "brief",
        "word count",
        "submission",
        "deadline",
        "课程要求",
        "作业要求",
        "写作要求",
        "评分标准",
        "评分细则",
        "格式要求",
        "字数",
        "提交",
        "老师要求",
        "导师要求",
    ]
    scholarly_terms = [
        "abstract",
        "introduction",
        "methodology",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "doi:",
        "关键词",
        "摘要",
        "引言",
        "研究方法",
        "实验",
        "结果",
        "讨论",
        "参考文献",
    ]
    requirement_score = sum(1 for term in requirement_terms if term in sample)
    scholarly_score = sum(1 for term in scholarly_terms if term in sample)
    requirement_score += sum(
        1
        for term in [
            "\u8bfe\u7a0b\u8981\u6c42",
            "\u4f5c\u4e1a\u8981\u6c42",
            "\u5199\u4f5c\u8981\u6c42",
            "\u8bc4\u5206\u6807\u51c6",
            "\u8bc4\u5206\u7ec6\u5219",
            "\u683c\u5f0f\u8981\u6c42",
            "\u5b57\u6570",
            "\u63d0\u4ea4",
            "\u8001\u5e08\u8981\u6c42",
            "\u5bfc\u5e08\u8981\u6c42",
        ]
        if term in sample
    )
    scholarly_score += sum(
        1
        for term in [
            "\u5173\u952e\u8bcd",
            "\u6458\u8981",
            "\u5f15\u8a00",
            "\u7814\u7a76\u65b9\u6cd5",
            "\u5b9e\u9a8c",
            "\u7ed3\u679c",
            "\u8ba8\u8bba",
            "\u53c2\u8003\u6587\u732e",
        ]
        if term in sample
    )
    has_identifier = bool(re.search(r"\bdoi\b|10\.\d{4,9}/|arxiv", sample, re.IGNORECASE))
    has_pdf_title_metadata = document_type == "PDF" and bool(str(metadata.get("title", "")).strip())

    if requirement_score >= 2 and scholarly_score < 4:
        return "instructions"
    if document_type == "DOCX" and requirement_score >= 1 and not has_identifier:
        return "instructions"
    if has_identifier or has_pdf_title_metadata or scholarly_score >= 3:
        return "literature"
    if requirement_score >= 1:
        return "instructions"
    return "literature"


def _split_reference_roles(references: list[dict]) -> tuple[list[dict], list[dict]]:
    literature = []
    context_documents = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        item = dict(reference)
        role = str(item.get("document_role") or "").strip().lower()
        if not role and item.get("document_type"):
            role = _infer_uploaded_document_role(
                filename=str(item.get("source") or item.get("title") or ""),
                text=str(item.get("content_excerpt") or item.get("abstract") or ""),
                metadata=item.get("pdf_metadata") if isinstance(item.get("pdf_metadata"), dict) else {},
                document_type=str(item.get("document_type") or ""),
            )
            item["document_role"] = role
            item["is_literature_source"] = role == "literature"
        if role and role != "literature":
            context_documents.append(item)
        else:
            literature.append(item)
    return literature, context_documents


def _build_uploaded_context(
    documents: list[dict],
    *,
    build_context_document_package: Callable[[str], str],
) -> str:
    sections = [
        "Uploaded auxiliary documents. These may be writing requirements, rubrics, assignment prompts, style constraints, background notes, or source evidence.",
        "Use them as source evidence only when they clearly contain substantive research content.",
    ]
    for index, document in enumerate(documents, start=1):
        title = document.get("title") or document.get("source") or f"Uploaded document {index}"
        source = document.get("source", "")
        role = document.get("document_role", "unknown")
        excerpt = build_context_document_package(
            document.get("content_excerpt") or document.get("abstract") or ""
        )
        sections.append(
            f"## Auxiliary document {index}: {title}\n\n"
            f"Source: {source}\n\n"
            f"Detected role: {role}\n\n"
            "Use note: classify this document before using it; if it is a rubric or writing requirement, apply it as task constraints rather than literature evidence.\n\n"
            f"Extracted excerpt:\n{excerpt}"
        )
    return "\n\n".join(section.strip() for section in sections if section).strip()
