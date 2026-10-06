import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customers.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the customer dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Prepare features and target for machine learning.
    """

    df = df.copy()

    # Target variable
    y = df["churn"]

    # Remove target and customer ID
    columns_to_drop = ["churn"]

    if "customer_id" in df.columns:
        columns_to_drop.append("customer_id")

    X = df.drop(
        columns=columns_to_drop
    )

    # Convert categorical columns to numerical
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # Convert boolean columns to integers
    boolean_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    for column in boolean_columns:
        X[column] = X[column].astype(int)

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    """
    Split dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_data(X_train, X_test):
    """
    Scale numerical features.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():
    """
    Create multiple machine learning models.
    """

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=6,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),

        "AdaBoost":
            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "SVM":
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
    }

    return models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate classification metrics.

    Positive class:
    Yes = Customer churned
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Yes",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Yes",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Yes",
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate all models.
    """

    results = {}

    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[name] = metrics

        trained_models[name] = model

        print(
            f"Accuracy : {metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['Recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['F1 Score']:.4f}"
        )

    return results, trained_models


# ============================================================
# FIND BEST MODEL
# ============================================================

def find_best_model(results, trained_models):
    """
    Select the model with the highest F1 score.
    """

    best_model_name = max(
        results,
        key=lambda model_name:
        results[model_name]["F1 Score"]
    )

    best_model = trained_models[
        best_model_name
    ]

    return (
        best_model_name,
        best_model
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_model(
    model,
    scaler,
    feature_columns,
    model_name
):
    """
    Save the trained model, scaler,
    and feature columns.
    """

    model_path = MODEL_DIR / "best_model.pkl"

    scaler_path = MODEL_DIR / "scaler.pkl"

    features_path = MODEL_DIR / "feature_columns.pkl"

    metadata_path = MODEL_DIR / "model_metadata.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    joblib.dump(
        feature_columns,
        features_path
    )

    metadata = {
        "model_name": model_name
    }

    joblib.dump(
        metadata,
        metadata_path
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\nBest model: {model_name}"
    )

    print(
        f"Model path: {model_path}"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CHURN MACHINE LEARNING")
    print("=" * 70)

    # 1. Load data

    df = load_data()

    # 2. Prepare features

    X, y = prepare_features(
        df
    )

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    print(
        f"Number of samples : {X.shape[0]}"
    )

    # 3. Train/test split

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # 4. Scaling

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_data(
        X_train,
        X_test
    )

    # 5. Create models

    models = create_models()

    # 6. Train models

    (
        results,
        trained_models
    ) = train_models(
        models,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    # 7. Display comparison

    results_df = pd.DataFrame(
        results
    ).T

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.round(4)
    )

    # 8. Select best model

    (
        best_model_name,
        best_model
    ) = find_best_model(
        results,
        trained_models
    )

    # 9. Save model

    save_model(
        best_model,
        scaler,
        X.columns.tolist(),
        best_model_name
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()