---
doc_id: KB-RR-001
title: Screening Process and Role of the AI Assistant
category: recruitment_rules
language: en
roles: [all]
version: 1.0
last_reviewed: 2026-10-15
---

# Screening Process and Role of the AI Assistant

## Process overview

1. **Reception**: the CV arrives (upload, email or careers page) and is linked to one open position.
2. **Extraction**: the CV is converted into a structured profile (identity, experience, education, skills,
   certifications, languages). Scanned CVs go through OCR or a vision-language model.
3. **Contextual analysis**: the profile is compared with the job description and the relevant knowledge-base
   documents (job profile, scoring rubric, experience rules).
4. **Report**: a structured, evidence-backed analysis is produced using the fit scoring rubric (KB-RR-002).
5. **Human review**: a recruiter validates, corrects or rejects the analysis and decides the next step.
6. **Communication**: the candidate is informed of the outcome by a recruiter.

## What the AI assistant may do

- Extract and structure information from CVs.
- Retrieve requirements, policies and definitions from the knowledge base.
- Compare a profile with a job description and propose a grade with supporting evidence.
- Suggest interview topics based on identified gaps.
- Prepare draft messages for recruiters.

## What the AI assistant must never do

- Make a final hiring or rejection decision.
- Send a rejection or offer to a candidate without recruiter approval.
- Use protected personal attributes in its analysis (see the fairness policy, KB-HR-003).
- Invent information absent from the CV, or present an assumption as a fact.
- Score a candidate without citing the CV excerpts that justify the score.

## Eligibility checks

- Candidates must be legally authorised to work in the country of the position, or the position must offer
  sponsorship. This is checked by the recruiter, not inferred from the candidate's name or nationality.
- Applications for a position are only compared with that position's job description. A candidate who seems better
  suited to another open role can be flagged to the recruiter, who decides whether to re-route the application.

## Service levels

- A first analysis should be available within one working day of reception.
- Every candidate receives an answer within ten working days.
- Shortlisted candidates are contacted for an interview within five working days of the shortlist decision.

## Traceability

Each analysis records the CV version, the job description version, the knowledge-base documents used (by
`doc_id`), the model versions and the reviewer's decision. This makes every recommendation auditable.
