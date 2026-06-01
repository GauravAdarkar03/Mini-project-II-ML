#Load and clean churnguard_data.csv (apply all Task 2 cleaning steps)
#Encode the target column Churn: Yes → 1, No → 0
#Encode categorical columns using pd.get_dummies() with drop_first=True:
#gender, PhoneService, InternetService, Contract, PaperlessBilling, PaymentMethod
#Separate the data into:
#X — all columns except Churn
#y — the Churn column
#Split into train and test sets — 80% train, 20% test, random_state=42
#Train a LogisticRegression model with max_iter=1000
#Print the accuracy score on the test set
#Print the classification report using classification_report with target_names=['Stay', 'Churn']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("churnguard_data.csv")

# -----------------------------
# DATA CLEANING (Task 2 Steps)
# -----------------------------

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
df["Contract"] = df["Contract"].fillna("Month-to-month")

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
df["InternetService"] = df["InternetService"].fillna("No")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove invalid rows
df = df[df["tenure"] > 0]
df = df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200)]

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

# -----------------------------
# ENCODING
# -----------------------------

# Encode target column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=[
        "gender",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ],
    drop_first=True
)

# -----------------------------
# SPLIT FEATURES AND TARGET
# -----------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stay", "Churn"]
    )
)