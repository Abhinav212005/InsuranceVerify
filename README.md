# Insurance Document Intelligence for Automated Policy and Claim Intake

A locally run DIP (Digital Image Processing) project. A user uploads insurance documents (PDF/PNG/JPG). The backend cleans each page with **OpenCV**, reads it with **Tesseract OCR**, **classifies** it (policy, claim form, identity, medical bill, repair invoice, other), **extracts** key fields, **validates** them and stores everything in **SQLite**. A **Next.js** dashboard shows the results and lets a person review and correct them.

> This system assists data entry. It never approves or rejects insurance claims. Statuses are only `Valid (data complete)`, `Needs Review` and `Incomplete`.

Planning document: [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md). Evaluation: [`evaluation_results/RESULTS.md`](evaluation_results/RESULTS.md). Report, slides and demo notes: [`documentation/`](documentation).

## Prerequisites
- Python 3.10+ and Node 20+
- Tesseract 5 (UB-Mannheim Windows build). If it is not at `C:\Program Files\Tesseract-OCR\tesseract.exe`, set `TESSERACT_CMD` in `backend/.env`.

## Run the backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```
Check http://localhost:8000/api/v1/health (reports the Tesseract version) and http://localhost:8000/docs.

## Run the frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:3000. The API URL is set in `frontend/.env.local` (`NEXT_PUBLIC_API_URL`).

## Tests
```bash
cd backend
python -m pytest -q
```

## Reproduce the evaluation
```bash
cd backend
python -m scripts.dataset.build_dataset          # synthetic documents + ground truth
python -m scripts.train_classifier               # TF-IDF + logistic regression
python -m scripts.evaluation.run_experiments     # 11 experiments -> evaluation_results/*.csv
python -m scripts.evaluation.make_charts         # charts, summary.json, RESULTS.md
```
No real personal data is used anywhere; all documents and identifiers are synthetic.

## Pipeline
`Upload -> PDF/image acquisition -> OpenCV preprocessing -> Tesseract OCR -> classification -> field extraction -> validation -> SQLite -> dashboard`

Preprocessing recipes (selectable per document): R0 baseline, R1 basic, R2 enhanced grayscale, **R3 standard (default)**, R4 standard + CLAHE, R5 full (page boundary + perspective for phone photos).

## Status
All twelve phases are complete, except that the final report PDF, the slide deck and the demo recording are written from the drafts in `documentation/` by the author.
