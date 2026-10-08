# Data Management & Privacy Policy

This project enforces strict data governance, GDPR compliance guidelines, and privacy-preserving practices.

---

## 1. Directory Structure

```text
data/
├── raw/
│   ├── cvs/       # Uploaded raw resume files (PDF, DOCX) - GITIGNORED
│   └── jobs/      # Raw job description text/documents - GITIGNORED
├── interim/       # Cleaned, normalized intermediate text - GITIGNORED
├── processed/     # Chunked and serialized vectors ready for indexing
└── evaluation/    # Synthetic benchmark datasets
```

---

## 2. Privacy Guidelines

1. **No Real PII in Source Control:**
   - Real candidate CVs, emails, phone numbers, and identifying documents MUST NEVER be committed to Git.
   - `.gitignore` explicitly excludes all files within `data/raw/` and `data/interim/`.

2. **Synthetic Data Policy:**
   - All automated unit tests, integration tests, and benchmarks must use synthetic or anonymized resumes.
   - Run `python scripts/seed_data.py` to generate synthetic sample profiles.

3. **Untrusted Input Handling:**
   - Resumes uploaded through the API or UI are treated as untrusted data.
   - Filenames are sanitized, paths are locked against traversal, and binary files are parsed using sandboxed libraries.
