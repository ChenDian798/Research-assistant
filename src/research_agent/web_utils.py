from __future__ import annotations

import re


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on"}


def normalize_output_language(value) -> str:
    return "en" if str(value or "").strip().casefold() == "en" else "zh"


def bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    header = str(range_header or "").strip()
    if not header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or file_size < 0:
        return None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return None
        start = max(file_size - suffix_length, 0)
        end = max(file_size - 1, 0)
    else:
        start = int(start_text)
        end = int(end_text) if end_text else max(file_size - 1, 0)
    if start >= file_size or start < 0 or end < start:
        return None
    return start, min(end, max(file_size - 1, 0))


def markdown_has_wide_table(markdown: str) -> bool:
    lines = markdown.splitlines()
    for index, _line in enumerate(lines):
        if not is_markdown_table_start(lines, index):
            continue
        columns = len(split_markdown_table_row(lines[index]))
        if columns >= 5:
            return True
    return False


def pdf_table_column_widths(total_width: float, column_count: int) -> list[float]:
    if column_count == 6:
        weights = [1.35, 1.15, 1.35, 1.5, 1.2, 1.2]
    elif column_count == 5:
        weights = [1.35, 1.15, 1.4, 1.45, 1.2]
    else:
        weights = [1.0] * column_count
    total_weight = sum(weights)
    return [total_width * weight / total_weight for weight in weights]


def is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index]
    separator = lines[index + 1]
    return bool(
        re.match(r"^\s*\|.*\|\s*$", current)
        and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", separator)
    )


def split_markdown_table_row(line: str) -> list[str]:
    text = str(line or "").strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    cells: list[str] = []
    cell = []
    for index, char in enumerate(text):
        if char == "|" and (index == 0 or text[index - 1] != "\\"):
            cells.append("".join(cell).replace("\\|", "|").strip())
            cell = []
        else:
            cell.append(char)
    cells.append("".join(cell).replace("\\|", "|").strip())
    return cells
