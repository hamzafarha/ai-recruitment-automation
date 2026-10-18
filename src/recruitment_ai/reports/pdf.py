"""Markdown -> styled PDF (ReportLab), for analysis reports and project reports.

Supports: #/##/### headings, paragraphs, '---' rules, '-' and '1.' lists (one nesting level), pipe
tables, ``` code blocks, **bold**, *italic*, `code`. Uses Arial/Consolas on Windows (full Unicode); elsewhere the
built-in Helvetica/Courier (a few symbols such as arrows are replaced by ASCII).
"""

import io
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
HEAD_BG = colors.HexColor("#e4ecf5")
ZEBRA = colors.HexColor("#f5f7fa")
_FONTS: dict[str, str] = {}
ASCII_FALLBACK = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "✓": "v", "✗": "x", "…": "...", "∼": "~",
                  "│": "|", "─": "-", "┌": "+", "┐": "+", "└": "+", "┘": "+", "├": "+", "┤": "+", "┬": "+",
                  "┴": "+", "▼": "v", "▶": ">"}


def _fonts() -> dict[str, str]:
    """Register Arial/Consolas when available (Windows), else use the built-in fonts."""
    if _FONTS:
        return _FONTS
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    windows = Path("C:/Windows/Fonts")
    try:
        pdfmetrics.registerFont(TTFont("Body", str(windows / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Bold", str(windows / "arialbd.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Italic", str(windows / "ariali.ttf")))
        pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(windows / "arialbi.ttf")))
        pdfmetrics.registerFont(TTFont("Mono", str(windows / "consola.ttf")))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                      boldItalic="Body-BoldItalic")
        _FONTS.update(body="Body", bold="Body-Bold", mono="Mono", unicode="yes")
    except Exception:
        _FONTS.update(body="Helvetica", bold="Helvetica-Bold", mono="Courier", unicode="")
    return _FONTS


def _styles() -> dict[str, ParagraphStyle]:
    f = _fonts()
    return {
        "title": ParagraphStyle("title", fontName=f["bold"], fontSize=19, leading=24, textColor=ACCENT, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=f["bold"], fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12,
                             spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=f["bold"], fontSize=11.5, leading=15, textColor=INK, spaceBefore=10,
                             spaceAfter=4),
        "body": ParagraphStyle("body", fontName=f["body"], fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=5,
                               alignment=TA_LEFT),
        "meta": ParagraphStyle("meta", fontName=f["body"], fontSize=9.6, leading=13.4, textColor=MUTED, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName=f["body"], fontSize=8.6, leading=11, textColor=INK),
        "cellhead": ParagraphStyle("cellhead", fontName=f["bold"], fontSize=8.6, leading=11, textColor=INK),
        "code": ParagraphStyle("code", fontName=f["mono"], fontSize=7.8, leading=9.8, textColor=INK),
    }


def inline(text: str) -> str:
    """Markdown inline syntax -> ReportLab paragraph markup (escape first)."""
    if not _fonts()["unicode"]:
        for symbol, ascii_text in ASCII_FALLBACK.items():
            text = text.replace(symbol, ascii_text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes: list[str] = []  # code spans first: their content may contain '*'

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    mono = _fonts()["mono"]
    return re.sub(r"\x00(\d+)\x00", lambda m: f'<font name="{mono}" color="#7b2d26">{codes[int(m.group(1))]}</font>',
                  text)


def _column_widths(rows: list[list[str]], width: float) -> list[float]:
    """Proportional to the text length, but every column at least as wide as its longest word."""
    ncols = len(rows[0])
    lengths = [min(max(max(len(r[c]) for r in rows), 8), 90) for c in range(ncols)]
    widths = [width * n / sum(lengths) for n in lengths]
    longest = [max((len(w) for r in rows for w in re.sub(r"[`*]", "", r[c]).split()), default=4)
               for c in range(ncols)]
    need = [min(n * 5.0 + 12, width / ncols) for n in longest]  # ~5 pt per character at the table's font size
    missing = sum(max(n - w, 0) for w, n in zip(widths, need, strict=True))
    spare = sum(max(w - n, 0) for w, n in zip(widths, need, strict=True))
    if missing > 0 and spare > 0:  # wide columns give some room to the narrow ones
        share = min(missing, spare)
        widths = [w + share * (n - w) / missing if w < n else w - share * (w - n) / spare
                  for w, n in zip(widths, need, strict=True)]
    return widths


def _table(rows: list[list[str]], width: float, s: dict[str, ParagraphStyle]) -> Table:
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = _column_widths(rows, width)
    data = [[Paragraph(inline(cell), s["cellhead"] if i == 0 else s["cell"]) for cell in row]
            for i, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    style += [("BACKGROUND", (0, r), (-1, r), ZEBRA) for r in range(2, len(rows), 2)]
    table.setStyle(TableStyle(style))
    return table


def _code(text: str, width: float, s: dict[str, ParagraphStyle]) -> Table:
    """A code block: monospace lines as written, on a light background."""
    if not _fonts()["unicode"]:
        for symbol, ascii_text in ASCII_FALLBACK.items():
            text = text.replace(symbol, ascii_text)
    table = Table([[Preformatted(text, s["code"])]], colWidths=[width], hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ZEBRA), ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return table


def _list(items: list[tuple[int, str]], ordered: bool, s: dict[str, ParagraphStyle],
          start: int = 1) -> ListFlowable:
    """items: (indent_level, text); level-1 items attach to the previous level-0 item."""
    body = _fonts()["body"]
    flow, i = [], 0
    while i < len(items):
        _, text = items[i]
        children = []
        i += 1
        while i < len(items) and items[i][0] == 1:
            children.append(items[i][1])
            i += 1
        parts = [Paragraph(inline(text), s["body"])]
        if children:
            parts.append(ListFlowable([ListItem(Paragraph(inline(c), s["body"]), leftIndent=12) for c in children],
                                      bulletType="bullet", start="•", leftIndent=12, bulletFontName=body,
                                      bulletFontSize=7))
        flow.append(ListItem(parts, leftIndent=14))
    return ListFlowable(flow, bulletType="1" if ordered else "bullet", start=str(start) if ordered else "•",
                        leftIndent=14, bulletFontName=body, bulletFontSize=9.6,
                        bulletFormat="%s." if ordered else None)


def parse(md: str, width: float) -> list:
    s = _styles()
    story: list = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            story += [_code("\n".join(code), width, s), Spacer(1, 6)]
            i += 1  # the closing fence
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), s["title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), s["h2"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), s["h3"]))
        elif stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=8))
        elif stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [Spacer(1, 3), _table(rows, width, s), Spacer(1, 7)]
            continue
        elif re.match(r"^(\s*)([-*]|\d+\.)\s", line):
            ordered = bool(re.match(r"^\s*\d+\.\s", line))
            items: list[tuple[int, str]] = []
            while i < len(lines) and lines[i].strip():
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if m:
                    items.append((1 if len(m.group(1)) >= 2 else 0, m.group(3)))
                elif items:  # wrapped continuation line
                    level, text = items[-1]
                    items[-1] = (level, text + " " + lines[i].strip())
                i += 1
            first = re.match(r"^\s*(\d+)\.", line)  # "2." after a code block continues at 2
            story += [_list(items, ordered, s, int(first.group(1)) if first else 1), Spacer(1, 4)]
            continue
        else:
            block = []
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(\s*)([-*]|\d+\.)\s|^#|^\||^---$|^```", lines[i].strip()):
                block.append(lines[i].strip())
                i += 1
            if all(b.startswith("**") for b in block) and len(block) > 1:  # "**Key:** value" header lines
                story.append(Paragraph("<br/>".join(inline(b) for b in block), s["meta"]))
            else:
                story.append(Paragraph(inline(" ".join(block)), s["body"]))
            continue
        i += 1
    return story


def markdown_to_pdf(markdown: str, title: str = "Report", footer_text: str = "AI Recruitment Automation") -> bytes:
    """PDF bytes of a Markdown document."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=title, author="AI Recruitment Automation")
    story = parse(markdown, doc.width)
    merged, k = [], 0
    while k < len(story):  # keep each heading with the element that follows it
        if isinstance(story[k], Paragraph) and story[k].style.name in ("h2", "h3") and k + 1 < len(story):
            merged.append(KeepTogether([story[k], story[k + 1]]))
            k += 2
        else:
            merged.append(story[k])
            k += 1
    body = _fonts()["body"]

    def footer(canvas, doc_) -> None:
        canvas.saveState()
        canvas.setFont(body, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc_.leftMargin, 10 * mm, footer_text)
        canvas.drawRightString(A4[0] - doc_.rightMargin, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(merged, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
