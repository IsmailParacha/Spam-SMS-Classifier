"""Spam Message Classification Project.

This script:
1. Downloads the UCI SMS Spam Collection dataset
2. Cleans the text data
3. Splits the data into train and test sets
4. Builds TF-IDF + model pipelines
5. Trains and evaluates multiple classifiers
6. Saves metrics and confusion matrices
7. Allows a user to test a custom message
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Data loading
# -----------------------------
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"]
    )
    return df

    """Fetch the SMS Spam Collection dataset from UCI and return a clean DataFrame."""
   #  dataset = fetch_ucirepo(id=228)
   


    # UCI returns targets/features separately. This project only needs label + text.
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Normalize column names safely.
    X.columns = [str(col).strip().lower() for col in X.columns]
    y.columns = [str(col).strip().lower() for col in y.columns]

    # Try common text/label names defensively.
    text_col = next((c for c in X.columns if c in {"message", "text", "sms"}), X.columns[0])
    label_col = next((c for c in y.columns if c in {"class", "label", "target"}), y.columns[0])

    df = pd.DataFrame({
        "message": X[text_col].astype(str),
        "label": y[label_col].astype(str).str.lower(),
    })

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_csv(DATA_DIR / "sms_spam_dataset.csv", index=False)
    return df


# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    """Basic text cleaning for beginner-friendly spam classification."""
    text = text.lower()
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned text and simple exploratory features."""
    processed = df.copy()
    processed["clean_message"] = processed["message"].apply(clean_text)
    processed["message_length"] = processed["message"].apply(len)
    processed["word_count"] = processed["message"].apply(lambda x: len(str(x).split()))
    processed["label_num"] = processed["label"].map({"ham": 0, "spam": 1})
    return processed


# -----------------------------
# Modeling
# -----------------------------
def build_models() -> Dict[str, Pipeline]:
    """Return a dictionary of beginner-friendly text classification models."""
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

    models: Dict[str, Pipeline] = {
        "MultinomialNB": Pipeline([
            ("tfidf", vectorizer),
            ("model", MultinomialNB()),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "DecisionTree": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("model", DecisionTreeClassifier(max_depth=25, random_state=42)),
        ]),
    }
    return models


def evaluate_model(model_name: str, model: Pipeline, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """Train and evaluate one model, then save outputs."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0)
    (RESULTS_DIR / f"classification_report_{model_name}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()

    return metrics


# -----------------------------
# Analysis helpers
# -----------------------------
def save_basic_analysis(df: pd.DataFrame) -> None:
    """Save simple statistics that can be used in the report."""
    summary_lines = []
    summary_lines.append("Dataset Summary")
    summary_lines.append("=" * 40)
    summary_lines.append(f"Number of messages: {len(df)}")
    summary_lines.append(f"Number of ham messages: {(df['label'] == 'ham').sum()}")
    summary_lines.append(f"Number of spam messages: {(df['label'] == 'spam').sum()}")
    summary_lines.append(f"Average message length: {df['message_length'].mean():.2f}")
    summary_lines.append(f"Average word count: {df['word_count'].mean():.2f}")

    ham_avg_len = df.loc[df["label"] == "ham", "message_length"].mean()
    spam_avg_len = df.loc[df["label"] == "spam", "message_length"].mean()
    summary_lines.append(f"Average ham length: {ham_avg_len:.2f}")
    summary_lines.append(f"Average spam length: {spam_avg_len:.2f}")

    (RESULTS_DIR / "dataset_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Simple bar plot for class distribution
    class_counts = df["label"].value_counts()
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    plt.close()

    # Histogram of message lengths
    df["message_length"].plot(kind="hist", bins=30)
    plt.title("Message Length Distribution")
    plt.xlabel("Number of characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "message_length_distribution.png")
    plt.close()


# -----------------------------
# Custom prediction
# -----------------------------
def predict_custom_message(best_model: Pipeline) -> None:
    """Allow user to classify a custom message from terminal input."""
    print("\nTry your own message.")
    custom_text = input("Enter a message (or press Enter to skip): ").strip()
    if not custom_text:
        print("Skipped custom prediction.")
        return

    cleaned = clean_text(custom_text)
    prediction = best_model.predict([cleaned])[0]
    label = "spam" if prediction == 1 else "ham"
    print(f"Prediction: {label}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print(df.head())

    print("Preprocessing data...")
    df = preprocess_dataframe(df)
    save_basic_analysis(df)

    X = df["clean_message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training models...")
    models = build_models()
    results = []
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        fitted_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model based on F1-score: {best_model_name}")

    predict_custom_message(best_model)

    print("\nProject finished.")
    print(f"Dataset saved to: {DATA_DIR / 'sms_spam_dataset.csv'}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
