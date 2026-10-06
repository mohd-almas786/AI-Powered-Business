import pandas as pd
import matplotlib.pyplot as plt
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

    df = pd.read_csv(DATA_PATH)

    return df



# BASIC DATA INFORMATION


def basic_analysis(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("BASIC DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())



# CHURN DISTRIBUTION


def churn_distribution(df):
    """
    Analyze customer churn distribution.
    """

    print("\n" + "=" * 60)
    print("CHURN DISTRIBUTION")
    print("=" * 60)

    churn_counts = df["churn"].value_counts()

    print("\nChurn Counts:")
    print(churn_counts)

    print("\nChurn Percentage:")
    print(
        (df["churn"].value_counts(normalize=True) * 100)
        .round(2)
    )

    plt.figure(figsize=(7, 5))

    churn_counts.plot(
        kind="bar"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



# AGE DISTRIBUTION


def age_distribution(df):
    """
    Analyze customer age distribution.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["age"],
        bins=20
    )

    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()


# INCOME DISTRIBUTION

def income_distribution(df):
    """
    Analyze customer income distribution.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["income"],
        bins=20
    )

    plt.title("Customer Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()


# CHURN BY GENDER

def churn_by_gender(df):
    """
    Analyze churn across genders.
    """

    churn_gender = pd.crosstab(
        df["gender"],
        df["churn"]
    )

    print("\n" + "=" * 60)
    print("CHURN BY GENDER")
    print("=" * 60)

    print(churn_gender)

    churn_gender.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Customer Churn by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# CHURN VS INCOME

def churn_vs_income(df):
    """
    Compare income between churned and non-churned customers.
    """

    plt.figure(figsize=(8, 5))

    df.boxplot(
        column="income",
        by="churn"
    )

    plt.title("Income Distribution by Churn")
    plt.suptitle("")

    plt.xlabel("Churn")
    plt.ylabel("Income")

    plt.tight_layout()
    plt.show()


# CHURN VS AGE

def churn_vs_age(df):
    """
    Compare age between churned and non-churned customers.
    """

    plt.figure(figsize=(8, 5))

    df.boxplot(
        column="age",
        by="churn"
    )

    plt.title("Age Distribution by Churn")
    plt.suptitle("")

    plt.xlabel("Churn")
    plt.ylabel("Age")

    plt.tight_layout()
    plt.show()


# WEBSITE VISITS VS CHURN

def website_visits_vs_churn(df):
    """
    Analyze relationship between website visits and churn.
    """

    churn_visits = df.groupby("churn")[
        "website_visits"
    ].mean()

    print("\n" + "=" * 60)
    print("AVERAGE WEBSITE VISITS BY CHURN")
    print("=" * 60)

    print(churn_visits)

    churn_visits.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Average Website Visits by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Average Website Visits")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# APP USAGE VS CHURN

def app_usage_vs_churn(df):
    """
    Analyze relationship between app usage and churn.
    """

    app_usage = df.groupby("churn")[
        "app_usage_hours"
    ].mean()

    print("\n" + "=" * 60)
    print("AVERAGE APP USAGE BY CHURN")
    print("=" * 60)

    print(app_usage)

    app_usage.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Average App Usage by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Average App Usage Hours")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# SATISFACTION VS CHURN

def satisfaction_vs_churn(df):
    """
    Analyze relationship between satisfaction score and churn.
    """

    satisfaction = df.groupby("churn")[
        "satisfaction_score"
    ].mean()

    print("\n" + "=" * 60)
    print("AVERAGE SATISFACTION SCORE BY CHURN")
    print("=" * 60)

    print(satisfaction)

    satisfaction.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Average Satisfaction Score by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Average Satisfaction Score")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# CORRELATION ANALYSIS

def correlation_analysis(df):
    """
    Analyze numerical feature correlations.
    """

    numerical_df = df.select_dtypes(
        include=["number"]
    )

    correlation = numerical_df.corr()

    print("\n" + "=" * 60)
    print("CORRELATION WITH CHURN")
    print("=" * 60)

    if "churn" in correlation.columns:

        churn_correlation = (
            correlation["churn"]
            .sort_values(
                ascending=False
            )
        )

        print(churn_correlation)

    plt.figure(figsize=(10, 7))

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.show()


# COMPLETE EDA

def run_eda():
    """
    Execute the complete EDA process.
    """

    print("=" * 60)
    print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()

    # Basic analysis
    basic_analysis(df)

    # Churn analysis
    churn_distribution(df)

    # Numerical distributions
    age_distribution(df)
    income_distribution(df)

    # Churn relationships
    churn_by_gender(df)
    churn_vs_income(df)
    churn_vs_age(df)

    # Behavioral analysis
    website_visits_vs_churn(df)
    app_usage_vs_churn(df)

    # Customer satisfaction
    satisfaction_vs_churn(df)

    # Correlation
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)


# RUN EDA

if __name__ == "__main__":
    run_eda()