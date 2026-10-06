import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible results
np.random.seed(42)

# Number of customers
n = 1000

# Generate customer data
data = {
    "customer_id": range(1, n + 1),
    "age": np.random.randint(18, 70, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "income": np.random.randint(20000, 150000, n),
    "tenure_months": np.random.randint(1, 120, n),
    "monthly_spend": np.round(np.random.uniform(20, 1000, n), 2),
    "support_tickets": np.random.randint(0, 15, n),
    "website_visits": np.random.randint(1, 50, n),
    "app_usage_hours": np.round(np.random.uniform(0.5, 20, n), 2),
    "satisfaction_score": np.random.randint(1, 11, n),
}

df = pd.DataFrame(data)

# Create a realistic churn target
churn_probability = (
    0.10
    + 0.025 * df["support_tickets"]
    - 0.015 * df["satisfaction_score"]
    - 0.002 * df["tenure_months"]
    + 0.001 * (df["monthly_spend"] / 10)
)

churn_probability = np.clip(churn_probability, 0.02, 0.90)

df["churn"] = np.random.binomial(1, churn_probability)

# Convert churn to readable labels
df["churn"] = df["churn"].map({
    0: "No",
    1: "Yes"
})

# Save CSV
output_path = Path(__file__).parent / "customers.csv"
df.to_csv(output_path, index=False)

print(f"Dataset created successfully: {output_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(df.head())