from __future__ import annotations

import html
import io
from pathlib import Path
import re

from src.research_agent.web_utils import (
    is_markdown_table_start,
    markdown_has_wide_table,
    pdf_table_column_widths,
    split_markdown_table_row,
)


def markdown_to_pdf_bytes(title: str, markdown: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as error:
        raise RuntimeError("PDF export requires reportlab. Run: pip install -r requirements.txt") from error

    font_name = "Helvetica"
    for font_path in [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ResearchFont", str(font_path)))
            font_name = "ResearchFont"
            break
        except Exception:
            continue

    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "ResearchNormal",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )
    heading1 = ParagraphStyle(
        "ResearchHeading1",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=18,
        leading=23,
        spaceBefore=8,
        spaceAfter=8,
    )
    heading2 = ParagraphStyle(
        "ResearchHeading2",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=19,
        spaceBefore=8,
        spaceAfter=6,
    )
    heading3 = ParagraphStyle(
        "ResearchHeading3",
        parent=styles["Heading3"],
        fontName=font_name,
        fontSize=12,
        leading=17,
        spaceBefore=6,
        spaceAfter=4,
    )

    pagesize = landscape(A4) if markdown_has_wide_table(markdown) else A4
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )
    table_normal = ParagraphStyle(
        "ResearchTableNormal",
        parent=normal,
        fontSize=8.5,
        leading=12,
        spaceAfter=0,
    )
    story = [Paragraph(pdf_inline_text(title), heading1), Spacer(1, 4)]
    pending_list = []

    def flush_list() -> None:
        nonlocal pending_list
        if not pending_list:
            return
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, normal)) for item in pending_list],
                bulletType="bullet",
                leftIndent=14,
            )
        )
        pending_list = []

    def append_table(table_lines: list[str]) -> None:
        rows = [split_markdown_table_row(line) for line in table_lines]
        if len(rows) < 2:
            return
        data = [
            [Paragraph(pdf_inline_text(cell), table_normal) for cell in row]
            for row in [rows[0], *rows[2:]]
            if row
        ]
        if not data:
            return
        column_count = max(len(row) for row in data)
        for row in data:
            while len(row) < column_count:
                row.append(Paragraph("", normal))
        table = Table(
            data,
            colWidths=pdf_table_column_widths(doc.width, column_count),
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7dee8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 6))

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush_list()
            story.append(Spacer(1, 4))
            index += 1
            continue
        if is_markdown_table_start(lines, index):
            flush_list()
            table_lines = []
            while index < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[index]):
                table_lines.append(lines[index])
                index += 1
            append_table(table_lines)
            continue
        if line.startswith("### "):
            flush_list()
            story.append(Paragraph(pdf_inline_text(line[4:]), heading3))
        elif line.startswith("## "):
            flush_list()
            story.append(Paragraph(pdf_inline_text(line[3:]), heading2))
        elif line.startswith("# "):
            flush_list()
            story.append(Paragraph(pdf_inline_text(line[2:]), heading1))
        elif re.match(r"^[-*]\s+", line):
            pending_list.append(pdf_inline_text(re.sub(r"^[-*]\s+", "", line)))
        else:
            flush_list()
            story.append(Paragraph(pdf_inline_text(line), normal))
        index += 1
    flush_list()
    doc.build(story)
    return buffer.getvalue()


def pdf_inline_text(value: str) -> str:
    text = html.escape(normalize_pdf_symbols(str(value or "")))
    text = re.sub(r"&lt;br\s*/?&gt;", "<br/>", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
    return text


def normalize_pdf_symbols(value: str) -> str:
    superscripts = str.maketrans(
        {
            "\u2070": "0",
            "\u00b9": "1",
            "\u00b2": "2",
            "\u00b3": "3",
            "\u2074": "4",
            "\u2075": "5",
            "\u2076": "6",
            "\u2077": "7",
            "\u2078": "8",
            "\u2079": "9",
            "\u207b": "-",
            "\u207a": "+",
        }
    )
    text = str(value or "").translate(superscripts)
    text = re.sub(r"(?<=\d)-(?=\d)", "^-", text)
    text = text.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u00d7", "x")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\ufffd", "")
    return text
