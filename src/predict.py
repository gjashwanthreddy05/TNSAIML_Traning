
from pathlib import Path

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "best_model.joblib"
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"


def load_model():
    """Load the trained classification model."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src/train_model.py first."
        )

    return joblib.load(MODEL_FILE)


def predict_transaction(transaction_data):
    """
    Predict whether a transaction is normal or fraudulent.

    transaction_data must contain the same 30 features
    used during model training.
    """
    model = load_model()

    # Convert transaction data into a DataFrame
    transaction_df = pd.DataFrame([transaction_data])

    # Match the feature order used during training
    if hasattr(model, "feature_names_in_"):
        transaction_df = transaction_df[
            list(model.feature_names_in_)
        ]

    prediction = model.predict(transaction_df)[0]

    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Normal Transaction"

    return result


if __name__ == "__main__":

    print("===== CREDIT CARD FRAUD DETECTION =====")

    # Load one transaction from the dataset for testing
    sample_data = pd.read_csv(DATA_FILE, nrows=1)

    # Remove target column
    sample_transaction = sample_data.drop(
        columns=["Class"]
    ).iloc[0].to_dict()

    result = predict_transaction(sample_transaction)

    print("\nPrediction Result:", result)
