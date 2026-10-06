import pandas as pd
from pathlib import Path


# PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"


# LOAD DATA

def load_data():
    """
    Load customer dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# CREATE ENGAGEMENT FEATURES

def create_engagement_features(df):
    """
    Create customer engagement-related features.
    """

    df = df.copy()

    # Total digital engagement
    df["total_engagement"] = (
        df["website_visits"]
        + df["app_usage_hours"]
    )

    # Website visits per unit of app usage
    df["website_app_ratio"] = (
        df["website_visits"]
        / (df["app_usage_hours"] + 1)
    )

    return df


# CREATE CUSTOMER VALUE FEATURES

def create_customer_value_features(df):
    """
    Create customer value-related features.
    """

    df = df.copy()

    # Income per year normalized by age
    df["income_age_ratio"] = (
        df["income"]
        / (df["age"] + 1)
    )

    # Approximate customer value score
    df["customer_value_score"] = (
        df["income"] * 0.5
        + df["satisfaction_score"] * 1000 * 0.3
        + df["website_visits"] * 100 * 0.2
    )

    return df


# Create age group


def create_age_groups(df):
    """
    Convert continuous age into meaningful age groups.
    """

    df = df.copy()

    bins = [
        0,
        25,
        35,
        45,
        55,
        100
    ]

    labels = [
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


# create income group

def create_income_groups(df):
    """
    Divide customers into income categories.
    """

    df = df.copy()

    df["income_group"] = pd.qcut(
        df["income"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        duplicates="drop"
    )

    return df


# create satisfaction category

def create_satisfaction_category(df):
    """
    Convert satisfaction score into categories.
    """

    df = df.copy()

    def satisfaction_label(score):

        if score < 0.4:
            return "Low"

        elif score < 0.7:
            return "Medium"

        else:
            return "High"

    df["satisfaction_category"] = (
        df["satisfaction_score"]
        .apply(satisfaction_label)
    )

    return df


# create engagement category

def create_engagement_category(df):
    """
    Categorize customers based on digital engagement.
    """

    df = df.copy()

    def engagement_label(value):

        if value < 30:
            return "Low"

        elif value < 60:
            return "Medium"

        else:
            return "High"

    df["engagement_category"] = (
        df["total_engagement"]
        .apply(engagement_label)
    )

    return df


# create risk features  

def create_risk_features(df):
    """
    Create simple business-oriented churn risk indicators.
    """

    df = df.copy()

    # Low satisfaction indicator
    df["low_satisfaction_flag"] = (
        df["satisfaction_score"] < 0.4
    ).astype(int)

    # Low engagement indicator
    df["low_engagement_flag"] = (
        df["total_engagement"] < 30
    ).astype(int)

    # Combined risk score
    df["risk_score"] = (
        df["low_satisfaction_flag"]
        + df["low_engagement_flag"]
    )

    return df

# Run complete feature engineering pipeline

def engineer_features(df):
    """
    Run the complete feature engineering pipeline.
    """

    df = create_engagement_features(df)

    df = create_customer_value_features(df)

    df = create_age_groups(df)

    df = create_income_groups(df)

    df = create_satisfaction_category(df)

    df = create_engagement_category(df)

    df = create_risk_features(df)

    return df


# main execution block

if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOMER CHURN - FEATURE ENGINEERING")
    print("=" * 60)

    # Load data
    df = load_data()

    print("\nOriginal dataset shape:")
    print(df.shape)

    # Create features
    df_engineered = engineer_features(df)

    print("\nFeature engineering completed!")

    print("\nNew dataset shape:")
    print(df_engineered.shape)

    print("\nNew features created:")

    original_columns = df.columns.tolist()

    new_columns = [
        column
        for column in df_engineered.columns
        if column not in original_columns
    ]

    for column in new_columns:
        print(f" - {column}")

    print("\nSample of engineered data:")
    print(
        df_engineered[
            new_columns
        ].head()
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 60)