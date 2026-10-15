"""EN/FR skill lexicon built from ESCO v1.2.1 + a project-specific extension.

``build_skill_lexicon`` turns ESCO's skills_en.csv / skills_fr.csv and
data/reference/custom_skills.json into one JSON file mapping every label and
synonym (both languages) to a skill concept. ``SkillNormalizer`` uses it to turn
raw strings ("ReactJS", "Python 3", "apprentissage automatique") into canonical skills.

Precedence when two concepts share a label: custom entries > ESCO preferred labels
> ESCO alternative labels > ESCO hidden labels > ESCO labels without their
"(qualifier)". Custom entries fix ESCO gaps
(Docker, AWS, PyTorch...) and coarse mappings (ESCO maps "TensorFlow" to "computer vision").
"""

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ESCO_VERSION = "1.2.1"


def skill_key(text: str) -> str:
    """Lookup key: lowercase, unified quotes, single spaces, no trailing punctuation."""
    text = text.replace("’", "'").replace("‘", "'").lower()
    return re.sub(r"\s+", " ", text).strip(" .,;:-")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _strip_accents(text.lower())).strip("-")


def _split_labels(cell: str | None) -> list[str]:
    return [label.strip() for label in (cell or "").split("\n") if label.strip()]


class _IndexBuilder:
    def __init__(self) -> None:
        self.concepts: dict[str, dict[str, str]] = {}
        self.index: dict[str, str] = {}

    def register(self, label: str, concept_id: str) -> None:
        """First registration wins (that is how precedence is implemented)."""
        key = skill_key(label)
        if key:
            self.index.setdefault(key, concept_id)
            self.index.setdefault(_strip_accents(key), concept_id)


def _read_esco(path: Path) -> dict[str, dict[str, str]]:
    """conceptUri -> row (first occurrence: ESCO's CSV repeats 21 concept ids)."""
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["conceptUri"], row)
    return rows


def build_skill_lexicon(
    esco_dir: Path = Path("data/raw/ontologies/esco/hf"),
    custom_path: Path = Path("data/reference/custom_skills.json"),
    out_path: Path = Path("data/interim/skills/skill_lexicon.json"),
) -> dict[str, int]:
    """Build and write the lexicon; returns counts (concepts, index entries, sources)."""
    builder = _IndexBuilder()

    custom = json.loads(custom_path.read_text(encoding="utf-8"))["skills"]
    for skill in custom:
        cid = f"custom:{_slug(skill['label_en'])}"
        builder.concepts[cid] = {
            "label_en": skill["label_en"], "label_fr": skill["label_fr"],
            "type": skill.get("type", "tool"), "source": "custom",
        }
        for label in [skill["label_en"], skill["label_fr"], *skill.get("aliases", [])]:
            builder.register(label, cid)

    en = _read_esco(esco_dir / "en" / "skills_en.csv")
    fr = _read_esco(esco_dir / "fr" / "skills_fr.csv")
    for uri, row in en.items():
        builder.concepts[uri] = {
            "label_en": row["preferredLabel"],
            "label_fr": fr.get(uri, {}).get("preferredLabel", row["preferredLabel"]),
            "type": row.get("skillType") or "skill/competence",
            "source": f"esco-{ESCO_VERSION}",
        }
    # Three passes so a preferred label always beats another concept's synonym
    for column in ("preferredLabel", "altLabels", "hiddenLabels"):
        for uri, row in en.items():
            labels = _split_labels(row.get(column))
            labels += _split_labels(fr.get(uri, {}).get(column))
            for label in labels:
                builder.register(label, uri)
    # Lowest priority: labels without their qualifier, "C (computer programming)" -> "C"
    for uri, row in en.items():
        for label in [row["preferredLabel"], fr.get(uri, {}).get("preferredLabel", "")]:
            if "(" in label:
                builder.register(re.sub(r"\s*\(.*?\)", "", label), uri)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"version": f"esco-{ESCO_VERSION}+custom", "concepts": builder.concepts, "index": builder.index},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "concepts": len(builder.concepts),
        "custom_concepts": len(custom),
        "esco_concepts": len(en),
        "index_entries": len(builder.index),
    }


@dataclass(frozen=True)
class SkillConcept:
    """A canonical skill, in English and French."""

    id: str
    label_en: str
    label_fr: str
    type: str
    source: str


class SkillNormalizer:
    """Map raw skill strings to canonical ``SkillConcept``s using the built lexicon."""

    def __init__(self, lexicon: dict[str, Any]) -> None:
        self._concepts = lexicon["concepts"]
        self._index = lexicon["index"]

    @classmethod
    def from_file(cls, path: Path = Path("data/interim/skills/skill_lexicon.json")) -> "SkillNormalizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _candidates(self, raw: str) -> list[str]:
        key = skill_key(raw)
        variants = [
            key,
            re.sub(r"\s*\(.*?\)\s*", " ", key).strip(),   # "python (programming)" -> "python"
            re.sub(r"\s*v?\d+(\.\d+)*(\.x)?\+?$", "", key),  # "python 3", "angular 5.x" -> name
            re.sub(r"\.?\s?js$", "", key),                   # "react.js" -> "react"
        ]
        variants += [_strip_accents(v) for v in variants]
        return list(dict.fromkeys(v for v in variants if v))

    def normalize(self, raw: str) -> SkillConcept | None:
        """Canonical concept for a raw skill string, or None if unknown."""
        for candidate in self._candidates(raw):
            if (cid := self._index.get(candidate)) is not None:
                return SkillConcept(id=cid, **self._concepts[cid])
        return None

    def coverage(self, raw_skills: list[str]) -> float:
        """Share of raw skills the lexicon recognises (a quick quality indicator)."""
        if not raw_skills:
            return 0.0
        return sum(self.normalize(s) is not None for s in raw_skills) / len(raw_skills)
