import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("churnguard_data.csv")

# DATA CLEANING
# Drop customerID column
df.drop(columns=["customerID"], inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Strip whitespace
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()

# Standardize casing
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()

# Fix Contract column
df["Contract"] = df["Contract"].str.strip().str.lower()

contract_mapping = {
    "month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "monthly": "Month-to-month",
    "monthtomonth": "Month-to-month",

    "one year": "One year",
    "1 year": "One year",
    "oneyear": "One year",

    "two year": "Two year",
    "2 year": "Two year",
    "twoyear": "Two year"
}

df["Contract"] = df["Contract"].replace(contract_mapping)

# Convert Contract to numeric values
contract_encode = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

df["Contract"] = df["Contract"].map(contract_encode)

# Fix InternetService column
df["InternetService"] = df["InternetService"].str.strip().str.lower()

internet_mapping = {
    "dsl": "DSL",
    "fiber optic": "Fiber optic",
    "fiber": "Fiber optic",
    "fiberoptic": "Fiber optic",
    "no": "No",
    "none": "No",
    "no internet": "No"
}

df["InternetService"] = df["InternetService"].replace(internet_mapping)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove invalid rows
df = df[df["tenure"] > 0]

df = df[
    (df["MonthlyCharges"] >= 10) &
    (df["MonthlyCharges"] <= 200)
]

# Fill missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].mean()
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)

df["tenure"] = df["tenure"].fillna(
    round(df["tenure"].median())
)

# Encode target column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# SELECT FEATURES

X = df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "Contract"
    ]
]

y = df["Churn"]

# TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X, y)


# USER INPUT
tenure = int(
    input("Enter tenure (months): ")
)

monthly_charges = float(
    input("Enter Monthly Charges: ")
)

total_charges = float(
    input("Enter Total Charges: ")
)

senior_citizen = int(
    input("Senior Citizen? (1 = Yes, 0 = No): ")
)

contract = int(
    input(
        "Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "
    )
)

# Create new customer data
new_customer = pd.DataFrame(
    [[
        tenure,
        monthly_charges,
        total_charges,
        senior_citizen,
        contract
    ]],
    columns=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "Contract"
    ]
)

# Predict churn
prediction = model.predict(new_customer)

# Print result
if prediction[0] == 1:
    print(
        "Prediction: This customer is likely to CHURN."
    )
else:
    print(
        "Prediction: This customer is likely to STAY."
    )