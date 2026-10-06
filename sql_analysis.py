# CUSTOMER CHURN - SQL ANALYSIS

import pandas as pd
import sqlite3

from pathlib import Path


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customers.csv"


# LOAD DATA

print("=" * 70)
print("CUSTOMER CHURN - SQL BUSINESS ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# CREATE SQLITE DATABASE IN MEMORY

connection = sqlite3.connect(":memory:")

df.to_sql(
    "customers",
    connection,
    index=False,
    if_exists="replace"
)

print("\nCustomer data loaded into SQL table successfully.")


# 1. TOTAL CUSTOMERS

query = """
SELECT COUNT(*) AS total_customers
FROM customers;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("1. TOTAL CUSTOMERS")
print("=" * 70)

print(result.to_string(index=False))


# 2. CHURNED CUSTOMERS

query = """
SELECT COUNT(*) AS churned_customers
FROM customers
WHERE churn = 'Yes';
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("2. CHURNED CUSTOMERS")
print("=" * 70)

print(result.to_string(index=False))


# 3. CHURN RATE

query = """
SELECT
    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customers;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("3. OVERALL CHURN RATE")
print("=" * 70)

print(result.to_string(index=False))


# 4. CUSTOMERS BY GENDER

query = """
SELECT
    gender,
    COUNT(*) AS customer_count,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY gender
ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("4. CHURN BY GENDER")
print("=" * 70)

print(result.to_string(index=False))


# 5. CHURN BY AGE GROUP

query = """
SELECT
    CASE
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END AS age_group,

    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY age_group

ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("5. CHURN BY AGE GROUP")
print("=" * 70)

print(result.to_string(index=False))


# 6. CHURN BY SATISFACTION SCORE

query = """
SELECT
    satisfaction_score,
    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY satisfaction_score

ORDER BY satisfaction_score;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("6. CHURN BY SATISFACTION SCORE")
print("=" * 70)

print(result.to_string(index=False))


# 7. CHURN BY TENURE

query = """
SELECT
    CASE
        WHEN tenure_months < 6 THEN '0-5 months'
        WHEN tenure_months < 12 THEN '6-11 months'
        WHEN tenure_months < 24 THEN '12-23 months'
        ELSE '24+ months'
    END AS tenure_group,

    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY tenure_group

ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("7. CHURN BY TENURE")
print("=" * 70)

print(result.to_string(index=False))


# 8. SUPPORT TICKETS VS CHURN

query = """
SELECT
    support_tickets,
    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY support_tickets

ORDER BY support_tickets;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("8. SUPPORT TICKETS VS CHURN")
print("=" * 70)

print(result.to_string(index=False))


# 9. WEBSITE VISITS VS CHURN

query = """
SELECT
    CASE
        WHEN website_visits < 10 THEN 'Low'
        WHEN website_visits < 30 THEN 'Medium'
        ELSE 'High'
    END AS website_activity,

    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY website_activity

ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("9. WEBSITE ACTIVITY VS CHURN")
print("=" * 70)

print(result.to_string(index=False))


# 10. APP USAGE VS CHURN

query = """
SELECT
    CASE
        WHEN app_usage_hours < 5 THEN 'Low Usage'
        WHEN app_usage_hours < 15 THEN 'Medium Usage'
        ELSE 'High Usage'
    END AS app_usage_group,

    COUNT(*) AS customer_count,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY app_usage_group

ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("10. APP USAGE VS CHURN")
print("=" * 70)

print(result.to_string(index=False))


# 11. HIGH-RISK CUSTOMERS

query = """
SELECT
    customer_id,
    age,
    gender,
    income,
    tenure_months,
    support_tickets,
    website_visits,
    app_usage_hours,
    satisfaction_score,
    churn

FROM customers

WHERE
    satisfaction_score <= 2
    AND support_tickets >= 3

ORDER BY support_tickets DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("11. HIGH-RISK CUSTOMER SEGMENT")
print("=" * 70)

print(result.head(20).to_string(index=False))


# 12. AVERAGE CUSTOMER PROFILE

query = """
SELECT
    ROUND(AVG(age), 2) AS average_age,
    ROUND(AVG(income), 2) AS average_income,
    ROUND(AVG(tenure_months), 2) AS average_tenure,
    ROUND(AVG(support_tickets), 2) AS average_support_tickets,
    ROUND(AVG(website_visits), 2) AS average_website_visits,
    ROUND(AVG(app_usage_hours), 2) AS average_app_usage,
    ROUND(AVG(satisfaction_score), 2) AS average_satisfaction

FROM customers;
"""

result = pd.read_sql_query(query, connection)

print("\n" + "=" * 70)
print("12. AVERAGE CUSTOMER PROFILE")
print("=" * 70)

print(result.to_string(index=False))


# CLOSE DATABASE

connection.close()

print("\n" + "=" * 70)
print("SQL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
