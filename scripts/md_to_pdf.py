"""Convert the data-inspection Markdown report to a styled PDF (ReportLab).

Supports what the report uses: #/##/### headings, paragraphs, '---' rules,
'-' and '1.' lists (one nesting level), pipe tables, **bold**, *italic*, `code`.
Usage: python scripts/md_to_pdf.py input.md output.pdf   (needs: pip install reportlab; Windows Arial fonts)
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(FONTS / "arialbi.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic"
)

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
HEAD_BG = colors.HexColor("#e4ecf5")
ZEBRA = colors.HexColor("#f5f7fa")

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=19, leading=24, textColor=ACCENT, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "meta": ParagraphStyle("meta", fontName="Body", fontSize=9.6, leading=13.4, textColor=MUTED, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.6, leading=11, textColor=INK),
    "cellhead": ParagraphStyle("cellhead", fontName="Body-Bold", fontSize=8.6, leading=11, textColor=INK),
}


def inline(text: str) -> str:
    """Markdown inline syntax -> ReportLab paragraph markup (escape first)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Protect code spans first: their content may contain '*' (e.g. raw/**/sample_*.json)
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return re.sub(
        r"\x00(\d+)\x00",
        lambda m: f'<font name="Mono" color="#7b2d26">{codes[int(m.group(1))]}</font>',
        text,
    )


def make_table(rows: list[list[str]], width: float) -> Table:
    ncols = len(rows[0])
    # Column widths proportional to content length, with a floor so short columns stay readable
    lengths = [max(len(r[c]) if c < len(r) else 0 for r in rows) for c in range(ncols)]
    lengths = [min(max(n, 8), 90) for n in lengths]
    widths = [width * n / sum(lengths) for n in lengths]
    data = [
        [Paragraph(inline(cell), S["cellhead"] if i == 0 else S["cell"]) for cell in row]
        for i, row in enumerate(rows)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    style += [("BACKGROUND", (0, r), (-1, r), ZEBRA) for r in range(2, len(rows), 2)]
    table.setStyle(TableStyle(style))
    return table


def make_list(items: list[tuple[int, str]], ordered: bool) -> ListFlowable:
    """items: (indent_level, text); nested level-1 items attach to the previous level-0 item."""
    flow, i = [], 0
    while i < len(items):
        _, text = items[i]
        children = []
        i += 1
        while i < len(items) and items[i][0] == 1:
            children.append(items[i][1])
            i += 1
        parts = [Paragraph(inline(text), S["body"])]
        if children:
            parts.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(c), S["body"]), leftIndent=12) for c in children],
                    bulletType="bullet", start="•", leftIndent=12, bulletFontName="Body", bulletFontSize=7,
                )
            )
        flow.append(ListItem(parts, leftIndent=14))
    return ListFlowable(
        flow,
        bulletType="1" if ordered else "bullet",
        start="1" if ordered else "•",
        leftIndent=14,
        bulletFontName="Body",
        bulletFontSize=9.6,
        bulletFormat="%s." if ordered else None,
    )


def parse(md: str, width: float) -> list:
    story: list = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), S["title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), S["h2"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), S["h3"]))
        elif stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=8))
        elif stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story.append(Spacer(1, 3))
            story.append(make_table(rows, width))
            story.append(Spacer(1, 7))
            continue
        elif re.match(r"^(\s*)([-*]|\d+\.)\s", line):
            ordered = bool(re.match(r"^\s*\d+\.\s", line))
            items: list[tuple[int, str]] = []
            while i < len(lines) and lines[i].strip():
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if m:
                    items.append((1 if len(m.group(1)) >= 2 else 0, m.group(3)))
                elif items:  # wrapped continuation line
                    lvl, txt = items[-1]
                    items[-1] = (lvl, txt + " " + lines[i].strip())
                i += 1
            story.append(make_list(items, ordered))
            story.append(Spacer(1, 4))
            continue
        else:
            block = []
            while i < len(lines) and lines[i].strip() and not re.match(
                r"^(\s*)([-*]|\d+\.)\s|^#|^\||^---$", lines[i].strip()
            ):
                block.append(lines[i].strip())
                i += 1
            # A block of "**Key:** value" lines (the header) keeps its line breaks
            if all(b.startswith("**") for b in block) and len(block) > 1:
                story.append(Paragraph("<br/>".join(inline(b) for b in block), S["meta"]))
            else:
                story.append(Paragraph(inline(" ".join(block)), S["body"]))
            continue
        i += 1
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 10 * mm, "AI Recruitment Automation — Data Inspection Report")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
        title="Data Inspection Report — AI Recruitment Automation", author="AI Recruitment Automation team",
    )
    story = parse(src.read_text(encoding="utf-8"), doc.width)
    # Keep each h3 heading with the element that follows it
    merged, k = [], 0
    while k < len(story):
        if isinstance(story[k], Paragraph) and story[k].style.name in ("h2", "h3") and k + 1 < len(story):
            merged.append(KeepTogether([story[k], story[k + 1]]))
            k += 2
        else:
            merged.append(story[k])
            k += 1
    doc.build(merged, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
