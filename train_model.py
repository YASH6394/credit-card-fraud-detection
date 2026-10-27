import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading credit card transaction dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Place creditcard.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

if "Class" not in df.columns:
    raise ValueError("Dataset must contain a 'Class' target column.")

if df["Class"].isnull().any():
    raise ValueError("Target column contains missing values.")

if not df["Class"].isin([0, 1]).all():
    raise ValueError("Class column must contain only 0 and 1.")

# --------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

class_counts = df["Class"].value_counts().sort_index()

print("\nTransaction class distribution:")
print(class_counts)

print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "class_distribution.png"),
    dpi=150
)

plt.close()

if "Amount" in df.columns:
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=50,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Transaction Amount Distribution")
    plt.tight_layout()

    plt.savefig(
        os.path.join(REPORT_DIR, "amount_distribution.png"),
        dpi=150
    )

    plt.close()

print("\nEDA charts saved in the reports folder.")

# --------------------------------------------------
# 4. PREPARE DATA
# --------------------------------------------------

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Preserve the same test set for both models.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    average_precision = average_precision_score(
        y_test, probabilities
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Average Precision": average_precision,
        "Training Time (seconds)": training_time
    })

    trained_models[model_name] = model

    # Save confusion matrix.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Legitimate", "Fraud"],
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=150
    )

    plt.close()

    # Save precision-recall curve.
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        name=model_name
    )

    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            f"{safe_name}_precision_recall.png"
        ),
        dpi=150
    )

    plt.close()

# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    os.path.join(REPORT_DIR, "model_comparison.csv"),
    index=False
)

# Select the model with the highest average precision.
best_model_name = results_df.sort_values(
    "Average Precision",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

# --------------------------------------------------
# 8. SAVE BEST MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "fraud_model.pkl"
)

joblib.dump(best_model, model_path)

# Save the exact feature order required for predictions.
joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

print("\nBest model:", best_model_name)
print("Saved model:", model_path)

print("\nTraining completed successfully!")
print("Check the reports folder for evaluation charts.")

