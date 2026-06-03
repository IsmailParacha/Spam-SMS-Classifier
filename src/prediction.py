import joblib
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_PATH
from src.preprocessing import clean_text


def save_model(model: Pipeline) -> None:
    """
    Save trained best model.
    """

    joblib.dump(model, BEST_MODEL_PATH)


def predict_message(model: Pipeline, message: str) -> str:
    """
    Predict one SMS message.
    """

    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]

    if prediction == 1:
        return "spam"

    return "ham"


def predict_custom_message(model: Pipeline) -> None:
    """
    Take SMS from user input and predict spam or ham.
    """

    print("\nTry your own SMS message.")
    message = input("Enter message or press Enter to skip: ").strip()

    if not message:
        print("Skipped custom prediction.")
        return

    prediction = predict_message(model, message)

    print(f"Prediction: {prediction}")

    classifier = model.named_steps["model"]

    if hasattr(classifier, "predict_proba"):
        cleaned_message = clean_text(message)
        probabilities = model.predict_proba([cleaned_message])[0]
        predicted_index = model.predict([cleaned_message])[0]
        confidence = probabilities[predicted_index] * 100

        print(f"Confidence: {confidence:.2f}%")