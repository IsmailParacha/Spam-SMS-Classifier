import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR


def save_basic_analysis(df: pd.DataFrame) -> None:
    """
    Save dataset summary and basic graphs.
    """

    summary = []

    summary.append("Dataset Summary")
    summary.append("=" * 40)
    summary.append(f"Number of messages: {len(df)}")
    summary.append(f"Number of ham messages: {(df['label'] == 'ham').sum()}")
    summary.append(f"Number of spam messages: {(df['label'] == 'spam').sum()}")
    summary.append(f"Average message length: {df['message_length'].mean():.2f}")
    summary.append(f"Average word count: {df['word_count'].mean():.2f}")

    ham_avg = df.loc[df["label"] == "ham", "message_length"].mean()
    spam_avg = df.loc[df["label"] == "spam", "message_length"].mean()

    summary.append(f"Average ham length: {ham_avg:.2f}")
    summary.append(f"Average spam length: {spam_avg:.2f}")

    summary_path = RESULTS_DIR / "dataset_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    # Class distribution graph
    df["label"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    plt.close()

    # Message length graph
    df["message_length"].plot(kind="hist", bins=30)
    plt.title("Message Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "message_length_distribution.png")
    plt.close()


def save_top_words(best_model: Pipeline, model_name: str) -> None:
    """
    Save top spam and ham words.
    This works for Logistic Regression and Linear SVM.
    """

    vectorizer = best_model.named_steps["tfidf"]
    model = best_model.named_steps["model"]

    if not hasattr(model, "coef_"):
        return

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_spam_indices = coefficients.argsort()[-20:][::-1]
    top_ham_indices = coefficients.argsort()[:20]

    lines = []
    lines.append(f"Top words for model: {model_name}")
    lines.append("=" * 40)

    lines.append("\nTop spam words:")
    for index in top_spam_indices:
        lines.append(f"{feature_names[index]}: {coefficients[index]:.4f}")

    lines.append("\nTop ham words:")
    for index in top_ham_indices:
        lines.append(f"{feature_names[index]}: {coefficients[index]:.4f}")

    output_path = RESULTS_DIR / f"top_words_{model_name}.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")