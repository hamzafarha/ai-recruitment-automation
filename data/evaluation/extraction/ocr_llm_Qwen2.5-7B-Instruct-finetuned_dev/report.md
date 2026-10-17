# Extraction evaluation — ocr_llm_Qwen2.5-7B-Instruct-finetuned_dev

Engine `ocr_llm` · model `Qwen/Qwen2.5-7B-Instruct` · split `dev` · 71 documents · 1 errors · 72.1 s per document

| Group | Docs | Experience F1 | Dates exact | Education F1 | Skills P | Skills R | Skills F1 | Identity acc. | Identity halluc. |
|---|---|---|---|---|---|---|---|---|---|
| **All** | 71 | 93% | 98% | 89% | 72% | 54% | 62% | 95% | 5 |
| source = real | 60 | 92% | 97% | 88% | 72% | 52% | 60% | 92% | 5 |
| source = synthetic | 11 | 100% | 100% | 92% | 84% | 99% | 91% | 100% | 0 |
| input_type = docx | 5 | 100% | 100% | 91% | 82% | 100% | 90% | 100% | 0 |
| input_type = scanned_pdf | 60 | 92% | 97% | 88% | 72% | 52% | 60% | 92% | 5 |
| input_type = text_pdf | 6 | 100% | 100% | 92% | 85% | 99% | 91% | 100% | 0 |
| language = en | 66 | 92% | 97% | 89% | 72% | 53% | 61% | 94% | 5 |
| language = fr | 5 | 100% | 100% | 83% | 82% | 98% | 89% | 100% | 0 |

Overall F1 (mean of experience, education, skills): 81%

Errors: resume_2caa3439e12019c7
