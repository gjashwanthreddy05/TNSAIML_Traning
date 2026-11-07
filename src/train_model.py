
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import prepare_data


def train_models():

    print("\n===== LOADING DATA =====")

    X_train, X_test, y_train, y_test = prepare_data()

    # Classification models
    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            max_depth=15,
            n_jobs=-1,
            random_state=42
        ),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ))
        ])
    }

    results = []
    trained_models = {}

    print("\n===== MODEL TRAINING STARTED =====")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Prediction scores for PR-AUC
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        pr_auc = average_precision_score(y_test, scores)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "PR-AUC": pr_auc
        })

        trained_models[name] = model

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    # Compare all models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="PR-AUC",
        ascending=False
    )

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    # Save comparison results
    output_dir = BASE_DIR / "outputs"
    model_dir = BASE_DIR / "models"

    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # Select best model based on PR-AUC
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Save best model
    joblib.dump(
        best_model,
        model_dir / "best_model.joblib"
    )

    with open(model_dir / "best_model_name.txt", "w") as file:
        file.write(best_model_name)

    print("\n===== BEST MODEL =====")
    print("Best Model:", best_model_name)

    print("\nBest Model Classification Report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        zero_division=0
    ))

    print("\nModel saved successfully!")
    print("Comparison saved in outputs/model_comparison.csv")


if __name__ == "__main__":
    train_models()
