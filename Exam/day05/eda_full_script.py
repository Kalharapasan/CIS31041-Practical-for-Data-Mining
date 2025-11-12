
# Explanatory Data Analysis - Combined Python Script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 01. Pandas Basics
# -----------------------------
print("=== 01. Pandas Basics ===")
series = pd.Series([10, 20, 30, 40, 50])
print("Series Data Type:", series.dtype)
print("Series Index:", series.index)
print("Series * 2:\n", series * 2)
print("Series + 10:\n", series + 10)

date_series = pd.date_range(start="2023-01-01", periods=5)
print("Date Series:\n", date_series)

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print("DataFrame:\n", df)
print("DataFrame Types:\n", df.dtypes)
print("First Rows:\n", df.head())
print("Last Rows:\n", df.tail())
print("Index Values:", df.index)
print("Columns:", df.columns)
print("To Numpy:\n", df.to_numpy())
print("Summary:\n", df.describe(include='all'))
print("Transpose:\n", df.T)
print("Sorted by Age:\n", df.sort_values(by='Age'))
print("Value at row 1, column 'Name':", df.loc[1, 'Name'])

# -----------------------------
# 02. heart.csv Analysis
# -----------------------------
print("\n=== 02. Heart Dataset ===")
heart_df = pd.read_csv('/mnt/data/heart.csv')
print("First 5 rows:\n", heart_df.head())
print("Data Types:\n", heart_df.dtypes)
print("Dataset Info:")
heart_df.info()
print("Summary:\n", heart_df.describe())

if 'cp' in heart_df.columns:
    filtered_cp = heart_df[heart_df['cp'] > 2]
    print("Records where cp > 2:\n", filtered_cp)

# -----------------------------
# 03. bank-data.csv Visualizations
# -----------------------------
print("\n=== 03. Data Visualization ===")
bank_df = pd.read_csv('/mnt/data/bank-data.csv')
print("First 5 rows:\n", bank_df.head())
print("Summary:\n", bank_df.describe(include='all'))

sns.pairplot(bank_df.select_dtypes(include=['float64', 'int64']))
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

if 'income' in bank_df.columns and 'children' in bank_df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='income', y='children', data=bank_df)
    plt.title("Income vs Children")
    plt.grid(True)
    plt.show()

if 'region' in bank_df.columns:
    plt.figure(figsize=(8,5))
    sns.countplot(x='region', data=bank_df)
    plt.title("Count by Region")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(bank_df['children'], bins=10, kde=True)
    plt.title("Children Histogram")
    plt.xlabel("Children")
    plt.show()

# -----------------------------
# 04. EDA on bank-data.csv
# -----------------------------
print("\n=== 04. EDA ===")
data = bank_df.copy()

print("Data Types:\n", data.dtypes)
print("Shape:", data.shape)
print("Columns:\n", data.columns.tolist())
print("Null Values:\n", data.isnull().sum())

if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

if 'income' in data.columns:
    data.rename(columns={'income': 'Income_USD'}, inplace=True)

if 'pep' in data.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='pep', data=data)
    plt.title("Distribution of PEP")
    plt.show()

if 'pep' in data.columns and 'sex' in data.columns:
    grouped = data.groupby(['sex', 'pep']).size().reset_index(name='count')
    print("Grouped by sex and pep:\n", grouped)

    plt.figure(figsize=(6,4))
    sns.countplot(x='sex', hue='pep', data=data)
    plt.title("PEP vs Sex")
    plt.show()

if 'children' in data.columns:
    print("Children Value Counts:\n", data['children'].value_counts())

    plt.figure(figsize=(6,4))
    sns.countplot(x='children', data=data)
    plt.title("Children Distribution")
    plt.show()

if {'pep', 'children', 'sex'}.issubset(data.columns):
    pivot = data.groupby(['pep', 'sex', 'children']).size().reset_index(name='count')
    sns.catplot(x='children', y='count', hue='sex', col='pep', kind='bar', data=pivot)
    plt.subplots_adjust(top=0.85)
    plt.suptitle("PEP vs Children vs Sex")
    plt.show()
