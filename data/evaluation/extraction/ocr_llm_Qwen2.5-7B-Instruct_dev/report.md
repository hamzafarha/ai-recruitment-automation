# Extraction evaluation — ocr_llm_Qwen2.5-7B-Instruct_dev

Engine `ocr_llm` · model `Qwen/Qwen2.5-7B-Instruct` · split `dev` · 71 documents · 1 errors · 90.4 s per document

| Group | Docs | Experience F1 | Dates exact | Education F1 | Skills P | Skills R | Skills F1 | Identity acc. | Identity halluc. |
|---|---|---|---|---|---|---|---|---|---|
| **All** | 71 | 88% | 96% | 92% | 68% | 48% | 56% | 93% | 12 |
| source = real | 60 | 87% | 96% | 91% | 66% | 45% | 54% | 90% | 12 |
| source = synthetic | 11 | 100% | 94% | 100% | 88% | 96% | 92% | 100% | 0 |
| input_type = docx | 5 | 100% | 94% | 100% | 88% | 97% | 92% | 100% | 0 |
| input_type = scanned_pdf | 60 | 87% | 96% | 91% | 66% | 45% | 54% | 90% | 12 |
| input_type = text_pdf | 6 | 100% | 95% | 100% | 89% | 96% | 92% | 100% | 0 |
| language = en | 66 | 88% | 96% | 92% | 67% | 47% | 55% | 92% | 12 |
| language = fr | 5 | 100% | 90% | 100% | 79% | 91% | 84% | 100% | 0 |

Overall F1 (mean of experience, education, skills): 79%

Errors: resume_0a01a2f51fc21218
