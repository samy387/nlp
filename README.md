# NLP Expense Classifier

Streamlit app + training code for classifying short expense sentences into categories (Food, Health, Clothing, Entertainment, Transport, etc.) using DistilBERT.

Contents
- `app.py` — Streamlit app and model loading/classification logic.
- `expense_dataset_10000.csv` — training dataset (10k rows).
- `NLP_project.ipynb` — notebook with model experiments.
- `requirements.txt` — Python dependencies.

Quick start
1. Create and activate a virtual environment (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Run the app:
   ```powershell
   streamlit run app.py
   ```

Notes
- The first run may download a DistilBERT model and tokenizer and/or retrain a local classifier; this can take time and requires internet access.
- The app automatically extracts amounts from sentences like "I bought lunch for $12.50".
