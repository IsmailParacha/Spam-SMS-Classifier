from typing import Dict

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test
) -> Dict[str, float]:
    """
    Train one model and save its report and confusion matrix.
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"],
        zero_division=0
    )

    report_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["ham", "spam"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()

    return metrics