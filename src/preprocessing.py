
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"


def load_dataset():
    return pd.read_csv(DATA_FILE)


def prepare_data():

    df = load_dataset()

    # Remove duplicate records
    print("Original Dataset Shape:", df.shape)

    df = df.drop_duplicates()

    print("After Removing Duplicates:", df.shape)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n===== DATA SPLITTING =====")
    print("Training Features:", X_train.shape)
    print("Testing Features:", X_test.shape)

    print("\n===== TRAINING CLASS DISTRIBUTION =====")
    print(y_train.value_counts())

    print("\n===== TESTING CLASS DISTRIBUTION =====")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
