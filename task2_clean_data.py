#Task Instructions
#Write a Python script task2_clean_data.py that applies all of the following cleaning steps in order:

#Load churnguard_data.csv
#Drop the customerID column
#Remove duplicate rows
#Strip whitespace from gender and PaymentMethod using .str.strip()
#Standardise casing — convert Churn, PhoneService, and PaperlessBilling to title case using .str.strip().str.title()
#Fix Contract — map all variations to one of three valid values:
#Month-to-month, One year, Two year
#Fix InternetService — map all variations to one of three valid values:
#DSL, Fiber optic, No
#Fix TotalCharges — convert to numeric using pd.to_numeric(..., errors='coerce') so junk becomes NaN
#Remove rows where tenure is zero or negative
#Remove rows where MonthlyCharges is less than 10 or greater than 200
#Fill missing values:
#MonthlyCharges → column mean
#TotalCharges → column mean
#tenure → column median (use integer rounding)
#Print the shape of the cleaned DataFrame
#Print missing value counts to confirm all issues are resolved

import pandas as pd

# Load the dataset
df = pd.read_csv("churnguard_data.csv")

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

# Fill remaining unknown Contract values
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

# Fill remaining unknown InternetService values
df["InternetService"] = df["InternetService"].fillna("No")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove invalid tenure rows
df = df[df["tenure"] > 0]

# Remove invalid MonthlyCharges rows
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

# Print shape
print("Shape of the cleaned dataset:", df.shape)

# Print missing values
print("\nCount of missing values in each column:")
print(df.isnull().sum())