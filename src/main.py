from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.analysis import save_basic_analysis, save_top_words
from src.data_loader import load_dataset
from src.evaluation import evaluate_model
from src.models import build_models
from src.prediction import predict_custom_message, save_model


def main() -> None:
    """
    Main function to run the full SMS classifier project.
    """

    print("Loading dataset...")
    df = load_dataset()
    print(df.head())

    print("\nPreprocessing dataset...")
    from src.preprocessing import preprocess_dataframe
    df = preprocess_dataframe(df)

    print("\nSaving dataset analysis...")
    save_basic_analysis(df)

    X = df["clean_message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nBuilding models...")
    models = build_models()

    results = []
    fitted_models: Dict[str, Pipeline] = {}

    print("\nTraining and evaluating models...")

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        metrics = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(metrics)
        fitted_models[model_name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv("results/metrics_summary.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print(f"\nBest model based on F1-score: {best_model_name}")

    save_model(best_model)
    save_top_words(best_model, best_model_name)

    predict_custom_message(best_model)

    print("\nProject finished.")
    print("Results saved in results folder.")
    print("Best model saved in models folder.")