#Task Instructions
#Write a Python script task1_load_explore.py that does all of the following:

#Import pandas and load churnguard_data.csv into a DataFrame
#Print the shape of the dataset (rows, columns)
#Print the first 5 rows
#Print column names and data types using .info()
#Print the count of missing values in each column
#Print the number of duplicate rows
#Print the value counts of the Churn column — you will notice inconsistent entries
#Print the unique values in the Contract column — you will notice typos

import pandas as pd


# Load the dataset
df = pd.read_csv("churnguard_data.csv")

# Print the shape of the dataset
print("Shape of the dataset:", df.shape)

# Print the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Print column names and data types
print("\nColumn names and data types:")
print(df.info())

# Print the count of missing values in each column
print("\nCount of missing values in each column:")
print(df.isnull().sum())

# Print the number of duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Print the value counts of the Churn column
print("\nValue counts of the Churn column:")

print(df["Churn"].value_counts())

# Print the unique values in the Contract column
print("\nUnique values in the Contract column:")
print(df["Contract"].unique())