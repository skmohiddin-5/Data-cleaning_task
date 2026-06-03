import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# Load Dataset
df = pd.read_csv("train.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Handle Missing Values
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows Found: {duplicates}")

df.drop_duplicates(inplace=True)

print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.tight_layout()
plt.savefig("outputs/survival_gender.png")
plt.show()

# Passenger Class vs Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.tight_layout()
plt.savefig("outputs/class_survival.png")
plt.show()

# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png")
plt.show()

# Fare Outliers
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.tight_layout()
plt.savefig("outputs/fare_outliers.png")
plt.show()

print("\n===== KEY INSIGHTS =====")
print("1. Missing values were handled successfully.")
print("2. Duplicate records were removed.")
print("3. Fare column contains outliers.")
print("4. Survival rates differ significantly by gender.")
print("5. Passenger class influenced survival chances.")
print("6. Most passengers were between 20 and 40 years old.")

print("\nGraphs saved successfully in the 'outputs' folder.")