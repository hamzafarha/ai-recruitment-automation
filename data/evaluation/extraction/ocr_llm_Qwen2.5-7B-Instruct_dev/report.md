# Extraction evaluation — ocr_llm_Qwen2.5-7B-Instruct_dev

Engine `ocr_llm` · model `Qwen/Qwen2.5-7B-Instruct` · split `dev` · 71 documents · 3 errors · 70.4 s per document

| Group | Docs | Experience F1 | Dates exact | Education F1 | Skills P | Skills R | Skills F1 | Identity acc. | Identity halluc. |
|---|---|---|---|---|---|---|---|---|---|
| **All** | 71 | 89% | 96% | 88% | 69% | 49% | 57% | 93% | 13 |
| source = real | 60 | 88% | 96% | 86% | 67% | 47% | 55% | 89% | 13 |
| source = synthetic | 11 | 100% | 94% | 100% | 91% | 96% | 94% | 100% | 0 |
| input_type = docx | 5 | 100% | 94% | 100% | 90% | 97% | 93% | 100% | 0 |
| input_type = scanned_pdf | 60 | 88% | 96% | 86% | 67% | 47% | 55% | 89% | 13 |
| input_type = text_pdf | 6 | 100% | 95% | 100% | 92% | 96% | 94% | 100% | 0 |
| language = en | 66 | 88% | 96% | 87% | 68% | 48% | 56% | 91% | 13 |
| language = fr | 5 | 100% | 90% | 100% | 80% | 91% | 85% | 100% | 0 |

Overall F1 (mean of experience, education, skills): 78%

Errors: resume_2caa3439e12019c7, resume_4a66e8e614f42cd2, resume_8ebd19823195f86e
