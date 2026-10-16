"""Labeling app for the evaluation set (Step 1c).

Run from the project root:
    python -m streamlit run frontend/labeling_app.py

Task 1 - grade CV-job pairs 0-3 with the rubric (KB-RR-002)   -> labels/matches.csv
Task 2 - check real CV answer keys against the CV image      -> labels/extraction_checks.csv

Design rules: no grade is pre-selected and no model output is shown (no anchoring);
whether a CV is real or synthetic, and what it was designed to test, are hidden.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recruitment_ai.evaluation.labeling import (  # noqa: E402
    CHECK_COLUMNS,
    CHECK_SECTIONS,
    CHECK_VALUES,
    GRADES,
    MATCH_COLUMNS,
    REQUIREMENT_STATUSES,
    LabelingData,
    match_row,
    now,
    upsert,
    validate_match,
)

st.set_page_config(page_title="Evaluation labeling", page_icon="🏷️", layout="wide")
# Keep the navigation row clear of Streamlit's top toolbar
st.markdown(
    "<style>.block-container, [data-testid='stMainBlockContainer'] {padding-top: 4rem;}</style>",
    unsafe_allow_html=True,
)

RUBRIC = """
| Grade | Label | When |
|---|---|---|
| **3** | Strong fit | All must-haves demonstrated or equivalent; experience meets the requirement |
| **2** | Partial fit | ≥ 70 % weighted must-have coverage, or experience up to 2 years below; gaps trainable |
| **1** | Weak fit | Some relevant skills but < 70 % coverage, or an adjacent role |
| **0** | No fit | Different field, or a hard requirement missing |

**Evidence**: demonstrated in a job/project = 1.0 · equivalent skill = 0.75 · only *listed* in a skills section = 0.5 ·
not in the CV = 0 (a gap to check in interview, not proof of weakness).
**Importance weights**: high = 3, medium = 2, low = 1. Overqualification is not penalised.
"""


@st.cache_resource
def get_data() -> LabelingData:
    return LabelingData()


@st.cache_data(show_spinner=False)
def page_images(pdf_path: str) -> list[bytes]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_pixmap(dpi=110).tobytes("png") for page in doc]


def show_cv(pdf: Path) -> None:
    if not pdf.exists():
        st.error(f"CV file not found: {pdf}. Run `python scripts/build_eval_set.py pool` / `synthetic`.")
        return
    for image in page_images(str(pdf)):
        st.image(image, width="stretch")


def navigation(key: str, total: int) -> int:
    """Previous / next buttons over a list; returns the current index."""
    idx = min(st.session_state.get(key, 0), max(total - 1, 0))
    left, middle, right = st.columns([1, 3, 1])
    if left.button("◀ Previous", disabled=idx == 0, key=f"{key}_prev"):
        idx -= 1
    if right.button("Next ▶", disabled=idx >= total - 1, key=f"{key}_next"):
        idx += 1
    middle.markdown(f"<div style='text-align:center'>Item {idx + 1} of {total}</div>", unsafe_allow_html=True)
    st.session_state[key] = idx
    return idx


# --------------------------------------------------------------------------- sidebar

data = get_data()
with st.sidebar:
    st.title("🏷️ Labeling")
    annotator = st.text_input("Your initials (annotator)", help="Use initials, not your full name: labels are committed to git.")
    task = st.radio("Task", ["Grade CV–job pairs", "Check extraction labels"])
    show_done = st.checkbox("Show items I already labeled", help="To review or change a label.")
    st.caption("Labels are saved to data/processed/eval_v1/labels/ each time you press Save.")

if not annotator.strip():
    st.info("Enter your initials in the sidebar to start.")
    st.stop()


# --------------------------------------------------------------------------- task 1: match grading


def grade_pairs() -> None:
    pairs = data.pairs()
    if not pairs:
        st.warning("No pairs yet. Run `python scripts/build_eval_set.py pairs` first.")
        return
    done = data.labels(data.matches_path, annotator, "pair_id")
    st.progress(len(done) / len(pairs), text=f"{len(done)} / {len(pairs)} pairs graded")
    items = pairs if show_done else [p for p in pairs if p["pair_id"] not in done]
    if not items:
        st.success("All pairs are graded. Thank you! Tick 'Show items I already labeled' to review them.")
        return

    pair = items[navigation("pair_idx", len(items))]
    job = data.job(pair["job_role"])
    existing = done.get(pair["pair_id"])
    st.subheader(f"{pair['pair_id']} · {job['title']} ({job['source']['company']})")

    cv_col, job_col = st.columns(2, gap="large")
    with cv_col:
        st.markdown("#### Candidate CV")
        show_cv(data.cv_pdf(pair["cv_id"], pair["cv_source"]))
    with job_col:
        st.markdown("#### Job requirements")
        st.table([
            {"ID": r["id"], "Requirement": r["requirement_text"], "Importance": r["importance"],
             "Min years": r["min_years_experience"] or ""}
            for r in job["requirements"]
        ])
        if job.get("not_scored"):
            st.caption("Not scored (checked by the recruiter): " + "; ".join(job["not_scored"]))
        with st.expander("Grading rubric (KB-RR-002)"):
            st.markdown(RUBRIC)
        with st.expander("Full job posting"):
            st.text(job["description"])

        with st.form(f"form_{pair['pair_id']}"):
            # Requirements first, overall grade second: the grade should follow from the
            # evidence, not from a first impression.
            st.markdown("**1 · Each requirement** — met = used in a job/project · partial = only listed, "
                        "or an equivalent skill · missing = not in the CV")
            statuses: dict[str, str] = {}
            previous_reqs = json.loads(existing["requirements"]) if existing else {}
            choices = ["—", *REQUIREMENT_STATUSES]
            for r in job["requirements"]:
                statuses[r["id"]] = st.radio(
                    f"{r['id']} ({r['importance']}) · {r['requirement_text']}", choices, horizontal=True,
                    index=choices.index(previous_reqs.get(r["id"], "—")), key=f"{pair['pair_id']}_{r['id']}",
                )
            st.markdown("**2 · Overall fit** (weights: high = 3, medium = 2, low = 1; see the rubric above)")
            options = list(GRADES)
            previous = int(existing["grade"]) if existing else None
            grade = st.radio(
                "Overall fit", options, index=options.index(previous) if previous is not None else None,
                format_func=lambda g: f"{g} — {GRADES[g]}", horizontal=True, label_visibility="collapsed",
            )
            reason = st.text_area(
                "3 · Reason (which requirements decided the grade; no personal information)",
                value=existing["reason"] if existing else "",
            )
            if st.form_submit_button("💾 Save", type="primary"):
                errors = validate_match(grade, reason)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    row = match_row(pair["pair_id"], annotator, grade, reason, statuses)
                    upsert(data.matches_path, row, ("pair_id", "annotator"), MATCH_COLUMNS)
                    st.toast(f"{pair['pair_id']} saved")
                    if show_done:
                        st.session_state["pair_idx"] = st.session_state.get("pair_idx", 0) + 1
                    st.rerun()


# --------------------------------------------------------------------------- task 2: extraction check


def show_answer_key(key: dict) -> None:
    st.markdown("**Identity**")
    st.write({k: key.get(k) for k in ("full_name", "email", "phone", "location")})
    st.markdown("**Experience**")
    for x in key.get("experience", []):
        dates = " – ".join(v for v in (x.get("start_date"), x.get("end_date")) if v) or "no dates"
        st.markdown(f"- **{x.get('title') or '—'}** @ {x.get('company') or '—'} · {dates}")
    st.markdown("**Education**")
    for ed in key.get("education", []):
        st.markdown(f"- {ed.get('degree') or '—'} · {ed.get('field_of_study') or ''} · "
                    f"{ed.get('institution') or '—'} · {ed.get('end_date') or ''}")
    st.markdown("**Skills**")
    st.write(", ".join(s["name"] for s in key.get("skills", [])) or "—")
    if key.get("certifications") or key.get("languages"):
        st.markdown("**Certifications / languages** (not graded)")
        st.write([c["name"] for c in key.get("certifications", [])], [x["language"] for x in key.get("languages", [])])


def check_extraction() -> None:
    sample = data.extraction_sample()
    if not sample:
        st.warning("No sample yet. Run `python scripts/build_eval_set.py pairs` first.")
        return
    done = data.labels(data.checks_path, annotator, "resume_id")
    st.progress(len(done) / len(sample), text=f"{len(done)} / {len(sample)} CVs checked")
    items = sample if show_done else [s for s in sample if s["resume_id"] not in done]
    if not items:
        st.success("All CVs are checked. Thank you!")
        return

    item = items[navigation("check_idx", len(items))]
    existing = done.get(item["resume_id"])
    st.subheader(f"Does the answer key describe this CV correctly? ({item['resume_id']})")
    cv_col, key_col = st.columns(2, gap="large")
    with cv_col:
        st.markdown("#### CV image")
        show_cv(data.cv_pdf(item["resume_id"], "real"))
    with key_col:
        st.markdown("#### Answer key (what extraction will be compared with)")
        show_answer_key(data.answer_key(item["resume_id"], "real", item["eval_split"]))
        with st.form(f"check_{item['resume_id']}"):
            st.caption("minor_errors = a typo or one small item wrong/missing · major_errors = wrong or missing entries")
            values = {
                section: st.radio(
                    section.capitalize(), CHECK_VALUES, horizontal=True, key=f"{item['resume_id']}_{section}",
                    index=CHECK_VALUES.index(existing[section]) if existing else None,
                )
                for section in CHECK_SECTIONS
            }
            notes = st.text_area("Notes (what is wrong?)", value=existing["notes"] if existing else "")
            if st.form_submit_button("💾 Save", type="primary"):
                if any(v is None for v in values.values()):
                    st.error("Rate every section.")
                else:
                    row = {"resume_id": item["resume_id"], "annotator": annotator.strip(), **values,
                           "notes": notes.strip(), "labeled_at": now()}
                    upsert(data.checks_path, row, ("resume_id", "annotator"), CHECK_COLUMNS)
                    st.toast(f"{item['resume_id']} saved")
                    if show_done:
                        st.session_state["check_idx"] = st.session_state.get("check_idx", 0) + 1
                    st.rerun()


if task == "Grade CV–job pairs":
    grade_pairs()
else:
    check_extraction()
