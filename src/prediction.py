import joblib
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_PATH
from src.preprocessing import clean_text


def save_model(model: Pipeline) -> None:
    """
    Save the trained best model to models folder.
    """
    joblib.dump(model, BEST_MODEL_PATH)


def load_model() -> Pipeline:
    """
    Load the saved model from models folder.
    """

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Please run python run_project.py first to train and save the model."
        )

    model = joblib.load(BEST_MODEL_PATH)
    return model


def predict_message(model: Pipeline, message: str) -> str:
    """
    Predict one SMS message as spam or ham.
    """

    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]

    if prediction == 1:
        return "spam"

    return "ham"


def predict_custom_message(model: Pipeline) -> None:
    """
    Take SMS from terminal input and predict spam or ham.
    """

    print("\nTry your own SMS message.")
    message = input("Enter message or press Enter to skip: ").strip()

    if not message:
        print("Skipped custom prediction.")
        return

    prediction = predict_message(model, message)
    print(f"Prediction: {prediction}")