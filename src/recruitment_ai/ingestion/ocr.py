"""OCR for scanned CVs: PaddleOCR PP-OCRv5 models run through RapidOCR (ONNX Runtime).

Same PaddleOCR models as in the architecture, without installing the PaddlePaddle
framework. The Latin recognizer reads English and French (accents) correctly; the
default Chinese recognizer drops the spaces between English words.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    line_count: int
    mean_confidence: float


def merge_lines(boxes: list[tuple[list[list[float]], str, float]]) -> list[str]:
    """Rebuild visual lines from OCR boxes.

    Boxes whose vertical centres are closer than half a line height are on the same
    line (e.g. a job title on the left and its dates on the right); they are joined left
    to right, with two spaces marking the gap between columns.
    """
    items = []
    for points, text, _ in boxes:
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        items.append({"y": (min(ys) + max(ys)) / 2, "h": max(max(ys) - min(ys), 1.0), "x": min(xs), "text": text})
    items.sort(key=lambda b: (b["y"], b["x"]))

    lines: list[list[dict[str, Any]]] = []
    for item in items:
        if lines and abs(item["y"] - lines[-1][0]["y"]) < 0.5 * min(item["h"], lines[-1][0]["h"]):
            lines[-1].append(item)
        else:
            lines.append([item])
    return ["  ".join(b["text"] for b in sorted(line, key=lambda b: b["x"])) for line in lines]


class OCREngine:
    """Lazy-loaded OCR engine (models load once, on first use)."""

    def __init__(self) -> None:
        self._engine: Any = None

    @property
    def engine(self) -> Any:
        if self._engine is None:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            self._engine = RapidOCR(params={
                "Rec.lang_type": LangRec.LATIN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
            })
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        return self._engine

    def image_to_text(self, image: np.ndarray) -> OCRResult:
        """OCR an RGB image array (H x W x 3)."""
        out = self.engine(image)
        if not out.txts:
            return OCRResult(text="", line_count=0, mean_confidence=0.0)
        boxes = list(zip(out.boxes.tolist(), out.txts, out.scores, strict=True))
        lines = merge_lines(boxes)
        return OCRResult(text="\n".join(lines), line_count=len(lines), mean_confidence=float(np.mean(out.scores)))


def page_image(page: Any, dpi: int = 200) -> np.ndarray:
    """RGB image of a PDF page.

    A scanned page is usually one embedded picture: it is used at its native resolution
    (sharper and faster than rendering, which would only upscale it). Other pages are
    rendered at ``dpi``.
    """
    import pymupdf

    images = page.get_images(full=True)
    pix = None
    if len(images) == 1 and not page.get_text().strip():
        try:
            pix = pymupdf.Pixmap(page.parent, images[0][0])
            if pix.alpha or pix.n not in (1, 3):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        except Exception:  # unusual image encodings: fall back to rendering
            pix = None
    if pix is None:
        pix = page.get_pixmap(dpi=dpi)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        array = np.repeat(array, 3, axis=2)
    return array[:, :, :3]


_DEFAULT_ENGINE: OCREngine | None = None


def default_engine() -> OCREngine:
    """Process-wide engine, so models are loaded only once."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = OCREngine()
    return _DEFAULT_ENGINE
