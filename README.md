# NLP Expense Classifier

A small Streamlit application to classify short expense sentences (e.g. "Bought lunch for $12.50") into categories such as Food, Health, Clothing, Entertainment, Transport, etc., and to track spending per date and per category.

This repository contains the Streamlit app, training code (uses DistilBERT), and a companion notebook with experiments.

Features
- Classify a one-line sentence into a predefined expense category using a DistilBERT-based classifier.
- Automatically extract the amount from the sentence (e.g. `$12.50`).
- Store classified records (sentence, predicted class, date, amount) in the Streamlit session and display full history.
- Search records by date and see the total spent on that date.
- Visualize total spending per category (bar chart + totals).
- Includes training pipeline in `app.py` (first run will train if no pre-trained model is present) and an experiment notebook `NLP_project.ipynb`.

Repository structure
- `app.py` — Streamlit application, classification, training and UI logic.
- `expense_dataset_10000.csv` — Training dataset used to train the classifier (10k rows).
- `NLP_project.ipynb` — Notebook with experiments, alternative training runs and notes.
- `requirements.txt` — Python dependencies to install.
- `.gitignore` — Ignored files for git (virtualenvs, model artifacts, etc.).

Prerequisites
- Python 3.8+ (tested on 3.9/3.10+).
- At least 8 GB of RAM for training locally (transformer training is memory-intensive).
- (Optional) GPU + CUDA if you want faster training/inference.

Installation (Windows PowerShell)
1. Create and activate a virtual environment in the project root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```

Running the app
1. Start the Streamlit app:
```powershell
streamlit run app.py
```
2. Open the URL printed in the console (usually http://localhost:8501) and use the UI to:
   - Enter a sentence containing an expense (e.g. "Bought lunch for $12.50") and a date.
   - Click **Classify Sentence** to classify and store the record.
   - Use **Search Records by Date** to show records and total spent on a specific date.
   - Use **Spending by Category** to view category totals and a bar chart.

Training notes
- On first run, if `expense_classifier_model/` is not present, `app.py` will run a training routine using `expense_dataset_10000.csv` and save:
  - `expense_classifier_model/` (transformer weights)
  - `expense_classifier_tokenizer/` (tokenizer files)
  - `label_encoder.pkl` (mapping of labels)
- Training uses Hugging Face `Trainer` with a small number of epochs by default. Training can take a long time on CPU and will download DistilBERT weights on the first run (internet required).

Improving accuracy
- If classification accuracy is insufficient:
  - Increase `num_train_epochs` or batch size (requires more RAM/GPU).
  - Use a larger transformer or fine-tune hyperparameters.
  - Add the `Amount` column as an explicit feature (currently the app extracts the amount from the sentence automatically).
  - Consider feature engineering or using an external pre-trained classifier (BERT, RoBERTa) depending on label complexity.

Model loading and device notes
- The app attempts to use GPU if available, but some environments (models saved with meta tensors or using special device maps) can raise errors when switching devices. The app contains robust fallback logic to load and run on CPU if GPU moves fail.

Security & safety
- The project saves model artifacts and a small `label_encoder.pkl`. Do NOT load `model.pkl` or other pickled objects from untrusted sources — pickles can execute arbitrary code.

Troubleshooting
- If you see errors about missing packages, ensure your virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- If the app reports model-loading errors, check the `expense_classifier_model/` folder and `label_encoder.pkl` are present. You can delete those and re-run the app to retrain.

Contributing
- Contributions, bug reports and improvement suggestions are welcome. Open an issue or create a pull request describing changes.

License
- This repository does not include an explicit license. Add a license file (e.g., MIT) if you intend to make the project open-source.

Contact
- For questions about the project or help with training/pushing to GitHub, open an issue or contact the repository owner.
