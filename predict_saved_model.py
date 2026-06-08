from src.prediction import load_model, predict_message


def main():
    """
    Load saved model and predict SMS without training again.
    """

    print("Loading saved model...")
    model = load_model()

    print("Model loaded successfully.")

    while True:
        message = input("\nEnter SMS message or type 'exit' to stop: ").strip()

        if message.lower() == "exit":
            print("Program stopped.")
            break

        if not message:
            print("Please enter a message.")
            continue

        prediction = predict_message(model, message)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()