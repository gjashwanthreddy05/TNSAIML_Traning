<<<<<<< HEAD
# Credit Card Fraud Detection Using Classification Techniques

## Project Overview
This project uses Machine Learning classification algorithms to detect fraudulent credit card transactions. It analyzes transaction data and predicts whether a transaction is Normal or Fraudulent.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Streamlit
- Plotly
- Joblib

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Linear Support Vector Classifier (LinearSVC)

## Project Structure
```text
Credit_Card_Fraud_Detection/
├── data/
├── models/
├── outputs/
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── dashboard/
│   └── app.py
├── requirements.txt
└── README.md
```

## Dataset
Dataset: Kaggle Credit Card Fraud Detection

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it inside the `data/` folder.

The dataset contains 284,807 transactions and 30 input features, with a highly imbalanced fraud class.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Project

Train the classification models:

```powershell
python src/train_model.py
```

Run the prediction script:

```powershell
python src/predict.py
```

Start the Streamlit dashboard:

```powershell
streamlit run dashboard/app.py
```

Open the dashboard at:

http://localhost:8501

## Dashboard Features
- Dataset overview
- Transaction distribution chart
- Upload transaction CSV files
- Predict normal and fraudulent transactions
- Compare classification models
- Download prediction results

## Project Objective
To build a machine learning-based system that helps identify potentially fraudulent credit card transactions using classification techniques.

## Author
Jashwanth Reddy
=======
# TNSAIML_Traning
>>>>>>> fafb797cb95d96b709f272ce6f2f9c56dd153721
