---
doc_id: KB-RR-002
title: Candidate–Job Fit Scoring Rubric
category: recruitment_rules
language: en
roles: [all]
version: 1.0
last_reviewed: 2026-10-15
---

# Candidate–Job Fit Scoring Rubric

This rubric defines how every application is graded against a job description. It is used in the same way by
recruiters, hiring managers and the AI screening assistant, so that analyses are consistent and comparable.

## Requirement types

Each job description lists requirements of three kinds:

- **Must-have (M)**: required to do the job from day one. Example: "3+ years of Python in production".
- **Nice-to-have (N)**: adds value but can be learned on the job. Example: "experience with Kubernetes".
- **Hard requirement**: a must-have that is legally or contractually mandatory (work authorisation, a regulated
  certification). Only requirements explicitly marked as mandatory in the job description are hard requirements.

## Evidence levels

A requirement is assessed from what the CV actually shows, never from assumptions:

| Evidence level | Meaning | Weight |
|---|---|---|
| Demonstrated | The skill appears in a job, project or achievement with concrete context | 1.0 |
| Listed | The skill only appears in a skills list, with no usage context | 0.5 |
| Equivalent | An equivalent or transferable skill is demonstrated (see the job profile) | 0.75 |
| Not evidenced | Nothing in the CV supports the requirement | 0 |

"Not evidenced" is not the same as "lacks the skill": the CV may simply not mention it. It is reported as a gap to
check in interview, never as a proven weakness.

## Fit grades

| Grade | Label | Definition | Score band |
|---|---|---|---|
| 3 | Strong fit | All must-haves demonstrated or equivalent; experience meets the requirement; most nice-to-haves present | 75–100 |
| 2 | Partial fit | At least 70 % of weighted must-have coverage, or all must-haves with experience up to 2 years below the requirement; gaps are trainable | 50–74 |
| 1 | Weak fit | Some relevant skills but less than 70 % must-have coverage, or an adjacent role with a different core skill set | 25–49 |
| 0 | No fit | Different field, or a hard requirement is missing | 0–24 |

Must-have coverage is the weighted sum of evidence over the must-haves, divided by the number of must-haves.

## Decision guidance

- **Grade 3**: recommended for the shortlist.
- **Grade 2**: requires recruiter review; often worth an interview when gaps are trainable.
- **Grades 0 and 1**: not recommended, but never rejected automatically. A recruiter confirms every rejection.
- Experience tolerance: a candidate up to 1 year below the required experience, with strong project or
  open-source evidence, is treated as meeting the requirement.

## What every analysis must contain

1. The grade (0–3) and the score (0–100).
2. For each requirement: its status (met, partial, missing) and the **CV excerpt** that supports it.
3. Gaps to explore in interview.
4. A short summary and a proposed action (shortlist, review, not recommended).

An analysis without evidence excerpts is incomplete and must not be used for a decision.

## Common pitfalls

- **Keyword stuffing**: long skill lists with no matching experience. Count such skills as "Listed" (0.5), not
  "Demonstrated".
- **Synonyms**: "ML", "machine learning" and "apprentissage automatique" are the same skill. Use the skill glossary
  and the equivalences in each job profile before marking a requirement as missing.
- **Seniority by title only**: a "Senior" title does not prove senior experience. Count years as defined in the
  experience and seniority rules (KB-RR-003).
