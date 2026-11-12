# API specification

Base URL: `http://localhost:8000/api/v1`. Interactive docs: `/docs`. Errors use `{"error": {"code": "...", "message": "..."}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and Tesseract version |
| POST | `/documents` | Upload one or more files (multipart `files[]`, optional `recipe`); processing starts in the background |
| GET | `/documents` | List with search, type, status, date filters and paging |
| GET | `/documents/{id}` | Full detail: pages, classification, fields, validation |
| DELETE | `/documents/{id}` | Delete the document and its files |
| GET | `/documents/{id}/status` | Lightweight progress (`uploaded / processing / processed / failed`) |
| POST | `/documents/{id}/process` | Re-run the whole pipeline with a chosen recipe |
| POST | `/documents/{id}/preprocess` | Run only the image processing stage |
| GET | `/documents/{id}/pages/{n}/stages` | Stage images and metrics for the Studio |
| GET / POST | `/documents/{id}/ocr` | Read or re-run OCR (PSM, confidence threshold) |
| POST | `/documents/{id}/classify` | Classify only |
| POST | `/documents/{id}/extract` | Extract fields only |
| POST | `/documents/{id}/validate` | Re-run validation |
| PATCH | `/documents/{id}/fields` | Manual corrections; validation re-runs and the status updates |
| GET | `/documents/{id}/report` | Download the report (`format=json` or `pdf`) |
| GET | `/stats/summary` | Dashboard figures |
| GET | `/pipeline/recipes` | Recipes and per-step explanations |
| GET | `/classification/info`, `/extraction/schema` | Categories, keywords and field sets |
| GET | `/evaluation/summary`, `/evaluation/charts/{name}` | Evaluation results for the dashboard |
| GET | `/files/{path}` | Stored images (originals, stages, thumbnails) |

Upload limits: 20 MB per file, 10 pages per document, file type checked by extension and magic bytes.
