import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# PATHS


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"



# LOAD DATA


def load_data():
    """
    Load the customer dataset from the data folder.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")

    return df



# BASIC DATA CLEANING


def clean_data(df):
    """
    Perform basic data cleaning.
    """

    df = df.copy()

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows...")
        df = df.drop_duplicates()

    # Handle missing numerical values
    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(
                df[column].median()
            )

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    print("\nData cleaning completed.")
    print(f"Final shape: {df.shape}")

    return df



# PREPARE FEATURES AND TARGET


def prepare_features(df):
    """
    Separate features (X) and target (y).

    Target:
        churn

    Returns:
        X, y
    """

    df = df.copy()

    if "churn" not in df.columns:
        raise ValueError(
            "'churn' column is missing from the dataset."
        )

    # Remove customer_id because it is only an identifier
    columns_to_drop = ["churn"]

    if "customer_id" in df.columns:
        columns_to_drop.append("customer_id")

    X = df.drop(columns=columns_to_drop)
    y = df["churn"]

    # Convert categorical columns into numerical columns
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

    print("\nFeatures prepared successfully.")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    return X, y



# TRAIN / TEST SPLIT


def split_data(X, y):
    """
    Split data into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nData split completed.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test



# FEATURE SCALING


def scale_features(X_train, X_test):
    """
    Standardize numerical features.

    IMPORTANT:
    The scaler is fitted ONLY on training data.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling completed.")

    return X_train_scaled, X_test_scaled, scaler



# COMPLETE PREPROCESSING PIPELINE


def preprocess_data():
    """
    Run the complete preprocessing pipeline.

    Steps:
        1. Load data
        2. Clean data
        3. Prepare features and target
        4. Train/test split
        5. Feature scaling

    Returns:
        X_train
        X_test
        y_train
        y_test
        scaler
    """

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Features and target
    X, y = prepare_features(df)

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Step 5: Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )



# TEST THE FILE DIRECTLY


if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOMER DATA PREPROCESSING")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTraining samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")