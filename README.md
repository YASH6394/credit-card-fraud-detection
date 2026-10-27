# 🛡️ FinGuard AI – Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions. It trains and compares two classifiers on the well-known Kaggle credit card fraud dataset and serves the best model through an interactive **Streamlit** dashboard.

> ⚠️ **Educational project.** Predictions are model estimates, not confirmed fraud. This is not a production banking security system.

---

## ✨ Features

- **Model training pipeline** (`train_model.py`)
  - Data validation (missing values, duplicates, `Class` column checks)
  - Exploratory charts (class balance, amount distribution)
  - Trains **Logistic Regression** (with scaling) and **Random Forest**, both using `class_weight="balanced"` to handle the heavy class imbalance
  - Stratified 80/20 train–test split (`random_state=42`)
  - Evaluates with Precision, Recall, F1-score and Average Precision (PR-AUC)
  - Saves confusion matrices, precision-recall curves and a model comparison table
  - Automatically saves the model with the highest Average Precision
- **Streamlit dashboard** (`app.py`)
  - **Dashboard** – upload a labelled CSV and explore fraud vs. legitimate distribution and transaction amounts
  - **Fraud Detection** – upload transactions, get a prediction and fraud probability (%) for each one, filter results, and download them as CSV
  - **Model Information** – view the model comparison table and chart

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py               # Streamlit dashboard
├── train_model.py       # Training & evaluation script
├── requirements.txt     # Python dependencies
├── .gitignore
├── data/                # Put creditcard.csv here (not tracked by git)
├── models/              # Generated: fraud_model.pkl, feature_names.pkl
└── reports/             # Generated: charts and model_comparison.csv
```

The `data/`, `models/` and `reports/` folders are created / populated as you follow the steps below.

---

## 📊 Dataset

The project expects the **Credit Card Fraud Detection** dataset (European cardholders, September 2013) from Kaggle:
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1` – `V28` | Anonymised features obtained via PCA |
| `Amount` | Transaction amount |
| `Class` | Target: `0` = legitimate, `1` = fraud |

The dataset is highly imbalanced (fraud is a tiny fraction of all transactions), which is why Average Precision, Recall and Precision are used instead of plain accuracy.

Download `creditcard.csv` and place it at `data/creditcard.csv`.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or newer

### 2. Clone and set up the environment

```bash
git clone <your-repo-url>
cd credit-card-fraud-detection

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the dataset

```bash
mkdir data
# copy creditcard.csv into the data/ folder
```

### 4. Train the model

```bash
python train_model.py
```

This prints the metrics for both models, saves the evaluation charts in `reports/`, and writes the best model to `models/fraud_model.pkl` (plus `models/feature_names.pkl` with the exact feature order).

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually <http://localhost:8501>).

---

## 🧪 Using the App

1. Go to **Fraud Detection** in the sidebar.
2. Upload a CSV that contains the same feature columns as training (`Time`, `V1`–`V28`, `Amount`). A `Class` column, if present, is ignored for prediction.
3. Click **Analyse Transactions**.
4. Review flagged transactions, filter the table, and download the results.

Input rows with missing, non-numeric or infinite feature values are rejected with a clear error message.

---

## 🧠 Models

| Model | Notes |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=1000)` |
| Random Forest | `RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)` |

Both models are evaluated on the same held-out test set. The one with the higher **Average Precision** is saved for use in the dashboard.

### Generated reports (`reports/`)
- `class_distribution.png`
- `amount_distribution.png`
- `<model>_confusion_matrix.png`
- `<model>_precision_recall.png`
- `model_comparison.csv`

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Plotly · Streamlit · joblib

---

## 🔮 Possible Improvements

- Try resampling (SMOTE) or gradient boosting models (XGBoost / LightGBM)
- Tune the decision threshold to balance precision and recall
- Add SHAP-based explanations for flagged transactions
- Add automated tests and a Dockerfile

---

## 📄 License

Add a license of your choice (e.g. MIT) before publishing.