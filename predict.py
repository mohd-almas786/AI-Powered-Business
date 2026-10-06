# CUSTOMER CHURN - CUSTOMER PREDICTION

import pandas as pd
import joblib

from pathlib import Path


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"


# LOAD SAVED MODEL COMPONENTS

print("=" * 70)
print("CUSTOMER CHURN PREDICTION")
print("=" * 70)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

print("\nModel loaded successfully!")
print("Model:", type(model).__name__)


# GET CUSTOMER INFORMATION

print("\n" + "=" * 70)
print("ENTER CUSTOMER INFORMATION")
print("=" * 70)

age = int(input("\nAge: "))

gender = input("Gender (Male/Female): ")

income = float(input("Annual Income: "))

tenure_months = int(input("Tenure in months: "))

support_tickets = int(input("Number of support tickets: "))

website_visits = int(input("Website visits per month: "))

app_usage_hours = float(input("App usage hours per month: "))

satisfaction_score = float(
    input("Satisfaction score (1-5): ")
)


# CREATE CUSTOMER DATAFRAME

customer = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "income": [income],
    "tenure_months": [tenure_months],
    "support_tickets": [support_tickets],
    "website_visits": [website_visits],
    "app_usage_hours": [app_usage_hours],
    "satisfaction_score": [satisfaction_score]
})


# ONE-HOT ENCODING

customer = pd.get_dummies(
    customer,
    columns=["gender"],
    drop_first=True
)


# MATCH TRAINING FEATURES

customer = customer.reindex(
    columns=feature_columns,
    fill_value=0
)


# SCALE CUSTOMER DATA

customer_scaled = scaler.transform(customer)


# MAKE PREDICTION

prediction = model.predict(customer_scaled)[0]


# CHURN PROBABILITY

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(customer_scaled)

    class_names = list(model.classes_)

    if "Yes" in class_names:

        yes_index = class_names.index("Yes")

        churn_probability = probabilities[
            0,
            yes_index
        ]

    else:
        churn_probability = 0.0

else:

    churn_probability = 0.0


# DISPLAY RESULT

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

if prediction == "Yes":

    print("\n⚠️  CHURN RISK: HIGH")

else:

    print("\n✅ CHURN RISK: LOW")

print(f"Prediction: {prediction}")

print(
    f"Churn Probability: "
    f"{churn_probability * 100:.2f}%"
)


# BUSINESS RECOMMENDATION

print("\n" + "=" * 70)
print("BUSINESS RECOMMENDATION")
print("=" * 70)

if prediction == "Yes":

    print("""
Recommended actions:

1. Contact the customer.
2. Offer a retention discount.
3. Check recent support issues.
4. Improve customer engagement.
5. Investigate satisfaction problems.
""")

else:

    print("""
Recommended actions:

1. Continue regular engagement.
2. Monitor customer satisfaction.
3. Encourage product usage.
4. Offer loyalty benefits.
5. Continue monitoring churn probability.
""")


print("=" * 70)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)
