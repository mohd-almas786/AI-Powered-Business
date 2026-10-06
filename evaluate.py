# CUSTOMER CHURN - MODEL EVALUATION

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customers.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"


# LOAD DATA

print("=" * 70)
print("CUSTOMER CHURN MODEL EVALUATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# PREPARE FEATURES

target_column = "churn"

y = df[target_column]

X = df.drop(columns=[target_column, "customer_id"])

# Convert categorical variables into numerical variables
X = pd.get_dummies(X, drop_first=True)

# Convert boolean columns to integers
bool_columns = X.select_dtypes(include=["bool"]).columns
X[bool_columns] = X[bool_columns].astype(int)


# LOAD SAVED FEATURE COLUMNS

feature_columns = joblib.load(FEATURE_PATH)

# Make sure evaluation data has exactly the same columns
X = X.reindex(columns=feature_columns, fill_value=0)


print("\nNumber of features:", X.shape[1])
print("Number of samples :", X.shape[0])


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# LOAD SAVED SCALER

scaler = joblib.load(SCALER_PATH)

X_test_scaled = scaler.transform(X_test)


# LOAD BEST MODEL

model = joblib.load(MODEL_PATH)

print("\nBest model loaded successfully!")
print("Model:", type(model).__name__)


# MAKE PREDICTIONS

y_pred = model.predict(X_test_scaled)


# EVALUATION METRICS

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)


# ROC-AUC

roc_auc = None

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_test_scaled)

    # Find probability column corresponding to "Yes"
    class_names = list(model.classes_)

    if "Yes" in class_names:

        yes_index = class_names.index("Yes")

        y_probability = probabilities[:, yes_index]

        y_test_binary = (y_test == "Yes").astype(int)

        roc_auc = roc_auc_score(
            y_test_binary,
            y_probability
        )


# DISPLAY RESULTS

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC  : {roc_auc:.4f}")


# CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n                Predicted")
print("              No      Yes")
print(f"Actual No    {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"Actual Yes   {cm[1][0]:5d}   {cm[1][1]:5d}")


# CLASSIFICATION REPORT

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=["No", "Yes"],
        zero_division=0
    )
)


# FINAL INTERPRETATION

print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nImportant:")
print("- Accuracy alone should not be used for churn prediction.")
print("- Churn data is imbalanced.")
print("- Recall tells us how many actual churn customers we detected.")
print("- Precision tells us how many predicted churn customers were actually churners.")
print("- F1 Score balances precision and recall.")
print("- ROC-AUC measures the model's ability to distinguish churn vs non-churn.")
