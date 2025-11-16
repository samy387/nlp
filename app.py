import streamlit as st
import pandas as pd
import datetime
import re
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# -------------------------------------------------------------------
# DATASET CLASS
# -------------------------------------------------------------------
class ExpenseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -------------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------------
def train_model():
    print("Loading dataset...")
    df = pd.read_csv('expense_dataset_10000.csv')
    
    # Clean text
    df['Text'] = df['Text'].str.lower()
    df['Text'] = df['Text'].str.replace(r'[^a-zA-Z0-9\s$\'-]', '', regex=True)
    
    # Encode labels
    le = LabelEncoder()
    df['Category_encoded'] = le.fit_transform(df['Category'])
    
    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['Text'], df['Category_encoded'], test_size=0.2, random_state=42, stratify=df['Category_encoded']
    )
    
    # Load tokenizer
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Create datasets
    train_dataset = ExpenseDataset(train_texts, train_labels, tokenizer)
    test_dataset = ExpenseDataset(test_texts, test_labels, tokenizer)
    
    # Load model
    print("Loading DistilBERT model...")
    num_labels = len(le.classes_)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    
    # Trainer
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Save model and metadata
    print("Saving model...")
    model.save_pretrained("expense_classifier_model")
    tokenizer.save_pretrained("expense_classifier_tokenizer")
    
    # Save label encoder metadata
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le.classes_, f)
    
    print("Training complete!")

if not os.path.exists('expense_classifier_model'):
    train_model()

# -------------------------------------------------------------------
# ML-BASED CLASSIFIER
# -------------------------------------------------------------------
def classify_sentence(sentence):
    try:
        # Extract amount from sentence
        import re
        amount_match = re.search(r'\$?([\d,]+\.?\d*)', sentence)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0.0
        
        # Load model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained("expense_classifier_model")
        tokenizer = DistilBertTokenizerFast.from_pretrained("expense_classifier_tokenizer")
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            classes = pickle.load(f)
        
        # Clean text
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s$\'-]', '', sentence)
        
        # Tokenize
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        category = classes[pred_class]
        
        return category, amount
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 0.0

# -------------------------------------------------------------------
# TEST CLASSIFICATION
# -------------------------------------------------------------------
def test_classification():
    test_sentences = [
        ("I bought lunch and paid $111 at work", "Food"),
        ("I purchased supplements and paid $141 for health", "Health"),
        ("I spent $50 on a new shirt", "Clothing"),
        ("I paid $30 for a movie ticket", "Entertainment"),
        ("I took a taxi and paid $20", "Transport")
    ]
    st.subheader("Test Classification Results")
    for sentence, expected in test_sentences:
        result = classify_sentence(sentence)
        if result[0]:
            predicted, amount = result
            result_text = "✓ Correct" if predicted == expected else f"✗ Incorrect (Expected: {expected})"
            st.write(f"**Sentence:** {sentence}")
            st.write(f"**Predicted:** {predicted}, {result_text} | **Amount Extracted:** ${amount:.2f}")
            st.write("---")

# -------------------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------------------
st.title("expense tracker")
st.write("Enter a sentence and it will be classified into one of the categories using DistilBERT.")

# Storage for classified results
if "records" not in st.session_state:
    st.session_state["records"] = pd.DataFrame(columns=["sentence", "class", "date", "amount"])

# Test classification button
if st.button("Test Model"):
    test_classification()

# -------------------------
# USER INPUT
# -------------------------
sentence_input = st.text_input("Enter a sentence (e.g., 'I bought lunch for $15.50'):")
date_input = st.date_input("Select a date", datetime.date.today())

if st.button("Classify Sentence"):
    if sentence_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        result = classify_sentence(sentence_input)
        if result[0]:
            label, amount = result
            new_row = {"sentence": sentence_input, "class": label, "date": date_input, "amount": amount}
            st.session_state["records"] = pd.concat(
                [st.session_state["records"], pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success(f"Sentence classified as: **{label}** | Amount extracted: **${amount:.2f}**")

# -------------------------
# SEARCH BY DATE
# -------------------------
st.subheader("Search Records by Date")

search_date = st.date_input("Choose a date to search", key="search_date")

if st.button("Search"):
    df = st.session_state["records"]
    if len(df) == 0:
        st.info("No records found. Start classifying expenses!")
    else:
        # Convert both to date objects for proper comparison
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date
        result = df_copy[df_copy['date'] == search_date]
        
        if result.empty:
            st.info(f"No records found for {search_date}.")
        else:
            st.write(f"### Records for {search_date}:")
            st.dataframe(result)
            total_spent = result["amount"].sum()
            st.metric("Total Spent on This Date", f"${total_spent:.2f}")

# -------------------------
# CATEGORY SPENDING SUMMARY
# -------------------------
st.subheader("Spending by Category")
if len(st.session_state["records"]) > 0:
    df = st.session_state["records"]
    category_summary = df.groupby("class")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(category_summary)
    st.write("### Total Spending per Category:")
    for category, total in category_summary.items():
        st.write(f"**{category}**: ${total:.2f}")
else:
    st.info("No records yet. Start classifying expenses!")

# Show full history
st.subheader("All Stored Records")
st.dataframe(st.session_state["records"])
