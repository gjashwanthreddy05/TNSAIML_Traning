
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ================= PROJECT PATHS =================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "best_model.joblib"
MODEL_NAME_FILE = BASE_DIR / "models" / "best_model_name.txt"
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"
RESULTS_FILE = BASE_DIR / "outputs" / "model_comparison.csv"


# ================= PAGE CONFIGURATION =================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ================= LOAD MODEL AND DATA =================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_FILE)


# Check required files

if not MODEL_FILE.exists():
    st.error("Trained model not found. Run python src/train_model.py first.")
    st.stop()

if not DATA_FILE.exists():
    st.error("Dataset not found in the data folder.")
    st.stop()

if not RESULTS_FILE.exists():
    st.error("Model comparison file not found.")
    st.stop()


# Load saved files

model = load_model()
df = load_dataset()
results = load_results()

if MODEL_NAME_FILE.exists():
    best_model_name = MODEL_NAME_FILE.read_text().strip()
else:
    best_model_name = "Best trained model"


# ================= MAIN DASHBOARD =================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Machine Learning project using classification techniques "
    "to identify fraudulent credit card transactions."
)

st.success(f"Selected Best Model: {best_model_name}")


# ================= DASHBOARD TABS =================

tab1, tab2, tab3 = st.tabs([
    "Dataset Overview",
    "Fraud Prediction",
    "Model Comparison"
])


# ==================================================
# TAB 1: DATASET OVERVIEW
# ==================================================

with tab1:

    st.header("Dataset Overview")

    total_transactions = len(df)

    normal_transactions = int(
        (df["Class"] == 0).sum()
    )

    fraud_transactions = int(
        (df["Class"] == 1).sum()
    )

    # Display metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )

    col3.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

    # Transaction distribution chart

    st.subheader("Transaction Class Distribution")

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Normal Transactions",
            "Fraud Transactions"
        ],
        "Count": [
            normal_transactions,
            fraud_transactions
        ]
    })

    fig = px.pie(
        chart_data,
        names="Transaction Type",
        values="Count",
        title="Normal vs Fraud Transactions",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Dataset sample

    st.subheader("Dataset Sample")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# TAB 2: FRAUD PREDICTION
# ==================================================

with tab2:

    st.header("Fraud Prediction")

    st.write(
        "Upload a CSV file containing credit card transaction features "
        "to predict whether transactions are normal or fraudulent."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Transaction Data")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        # Get model feature columns

        if hasattr(model, "feature_names_in_"):

            feature_columns = list(
                model.feature_names_in_
            )

        else:

            feature_columns = [
                column for column in df.columns
                if column != "Class"
            ]

        # Check missing columns

        missing_columns = [
            column for column in feature_columns
            if column not in uploaded_df.columns
        ]

        if missing_columns:

            st.error(
                "Your CSV file is missing required columns."
            )

            st.write("Missing columns:", missing_columns)

        else:

            X_upload = uploaded_df[
                feature_columns
            ].copy()

            # Validate missing values

            if X_upload.isnull().any().any():

                st.error(
                    "Your uploaded file contains missing values. "
                    "Please clean the data before prediction."
                )

            elif st.button("Predict Transactions"):

                # Perform prediction

                predictions = model.predict(X_upload)

                prediction_results = uploaded_df.copy()

                prediction_results["Prediction"] = [
                    "Fraud" if value == 1 else "Normal"
                    for value in predictions
                ]

                # Display results

                st.subheader("Prediction Results")

                st.dataframe(
                    prediction_results,
                    use_container_width=True
                )

                # Prediction counts

                fraud_count = int(
                    (predictions == 1).sum()
                )

                normal_count = int(
                    (predictions == 0).sum()
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Fraud Detected",
                    fraud_count
                )

                col2.metric(
                    "Normal Transactions",
                    normal_count
                )

                # Prediction chart

                st.subheader("Prediction Distribution")

                result_chart = pd.DataFrame({
                    "Transaction Type": [
                        "Fraud",
                        "Normal"
                    ],
                    "Count": [
                        fraud_count,
                        normal_count
                    ]
                })

                fig = px.bar(
                    result_chart,
                    x="Transaction Type",
                    y="Count",
                    title="Prediction Summary",
                    text="Count"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Download prediction results

                csv_data = prediction_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )


# ==================================================
# TAB 3: MODEL COMPARISON
# ==================================================

with tab3:

    st.header("Classification Model Comparison")

    st.write(
        "Compare the performance of the four trained "
        "machine learning classification models."
    )

    # Display comparison table

    st.subheader("Model Performance Metrics")

    st.dataframe(
        results,
        use_container_width=True
    )

    # PR-AUC comparison

    if "PR-AUC" in results.columns:

        st.subheader("PR-AUC Comparison")

        fig = px.bar(
            results,
            x="Model",
            y="PR-AUC",
            title="Model Comparison - PR-AUC",
            text="PR-AUC"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # F1 Score comparison

    if "F1 Score" in results.columns:

        st.subheader("F1 Score Comparison")

        fig = px.bar(
            results,
            x="Model",
            y="F1 Score",
            title="Model Comparison - F1 Score",
            text="F1 Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ================= FOOTER =================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Classification Project"
)
