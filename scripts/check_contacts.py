"""Check the e-mail and phone of the stored candidates against their CV text.

The extraction now does it for every new CV (``check_contacts`` in
recruitment_ai/extraction/extractor.py): the model sometimes scrambles an address while copying it.
This script applies the same check to the candidates extracted before. No model runs: only the
stored CV text is read. Candidates without CV text (demo profiles) are skipped.

The contact details are stored in three places: the candidate's columns (used to send e-mails), the
profile JSON (shown by the pipeline page) and the e-mails not sent yet (outbox, failed). The columns
are checked and the two other places follow them, so a column already corrected by hand is carried
over too. E-mails already sent are the record of what happened: they are left as they are.

    python scripts/check_contacts.py            # shows what would change
    python scripts/check_contacts.py --apply    # saves it
"""

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

UNSENT = ("outbox", "failed")  # the e-mails that "Send now" can still send


def contact_changes(db: Any, apply: bool = False) -> list[tuple[Any, list[tuple[str, Any, Any]]]]:
    """(candidate, [(place, stored value, value of the CV)]) for every candidate whose stored contact
    details differ from its CV; ``apply`` also writes the CV's values (the caller commits)."""
    from recruitment_ai.database.models import Candidate
    from recruitment_ai.extraction.extractor import check_contacts

    out = []
    for candidate in db.query(Candidate).filter(Candidate.cv_text.isnot(None)).order_by(Candidate.created_at):
        email, phone = check_contacts(candidate.email, candidate.phone, candidate.cv_text)
        checked = {"email": email, "phone": phone}
        columns = {"email": candidate.email, "phone": candidate.phone}
        profile = candidate.profile or {}
        unsent = [e for e in candidate.emails if e.status in UNSENT and e.candidate_address != email]
        changes = [(field, columns[field], new) for field, new in checked.items() if columns[field] != new]
        changes += [(f"profile {field}", profile.get(field), new) for field, new in checked.items()
                    if profile.get(field) != new]
        changes += [(f"{e.kind} e-mail ({e.status})", e.candidate_address, email) for e in unsent]
        if not changes:
            continue
        out.append((candidate, changes))
        if not apply:
            continue
        # the profile keeps what the model wrote; a column may already hold a hand correction
        fixes = {}
        for field, new in checked.items():
            wrong = [old for old in (profile.get(field), columns[field]) if old != new]
            if wrong:
                fixes[field] = {"model": wrong[0], "cv": new}
        candidate.email, candidate.phone = email, phone
        candidate.profile = {**profile, **checked}
        if fixes:
            extraction = candidate.extraction or {}
            candidate.extraction = {**extraction,
                                    "contact_fixes": {**(extraction.get("contact_fixes") or {}), **fixes}}
        for e in unsent:
            if e.to_address == e.candidate_address:  # not redirected to EMAIL_REDIRECT_TO
                e.to_address = email
            e.candidate_address = email
    return out


def main() -> None:
    from recruitment_ai.database.connection import get_session_factory

    parser = argparse.ArgumentParser(description="Check stored candidates' e-mail and phone against their CV.")
    parser.add_argument("--apply", action="store_true", help="save the corrections (default: only show them)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db = get_session_factory()()
    try:
        changed = contact_changes(db, apply=args.apply)
        for candidate, changes in changed:
            print(f"{candidate.full_name or candidate.id} ({candidate.id})")
            for place, old, new in changes:
                print(f"  {place}: {old!r} -> {new!r}")
        if args.apply:
            db.commit()
    finally:
        db.close()
    if not changed:
        print("Every stored e-mail and phone matches its CV.")
    elif args.apply:
        print(f"{len(changed)} candidate(s) corrected.")
    else:
        print(f"{len(changed)} candidate(s) would change: run again with --apply to save.")


if __name__ == "__main__":
    main()
