"""Convert a Markdown report to a styled PDF (ReportLab).

Usage: python scripts/md_to_pdf.py input.md output.pdf ["Footer text"]
The conversion itself lives in recruitment_ai.reports.pdf (also used by the API for analysis reports).
"""

import sys
from pathlib import Path

from recruitment_ai.reports.pdf import markdown_to_pdf


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    footer = sys.argv[3] if len(sys.argv) > 3 else "AI Recruitment Automation"
    markdown = src.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), src.stem)
    out.write_bytes(markdown_to_pdf(markdown, title=title, footer_text=footer))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
