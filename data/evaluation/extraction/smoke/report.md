# Extraction evaluation — smoke

Engine `ocr_llm` · model `qwen2.5:3b` · split `dev` · 3 documents · 1 errors · 43.2 s per document

| Group | Docs | Experience F1 | Dates exact | Education F1 | Skills P | Skills R | Skills F1 | Identity acc. | Identity halluc. |
|---|---|---|---|---|---|---|---|---|---|
| **All** | 3 | 36% | 100% | 75% | 92% | 51% | 65% | 75% | 0 |
| source = real | 3 | 36% | 100% | 75% | 92% | 51% | 65% | 75% | 0 |
| input_type = scanned_pdf | 3 | 36% | 100% | 75% | 92% | 51% | 65% | 75% | 0 |
| language = en | 3 | 36% | 100% | 75% | 92% | 51% | 65% | 75% | 0 |

Overall F1 (mean of experience, education, skills): 59%

Errors: resume_0b3d3635f3471352
