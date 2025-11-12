# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Pandas Basics ---
s = pd.Series([10, 20, 30])
print(s.dtype, s.index, s * 2, s + 10)

df = pd.DataFrame({'Name': ['A', 'B'], 'Age': [25, 30]})
print(df.dtypes, df.head(), df.tail(), df.columns, df.to_numpy())
print(df.describe(), df.T, df.sort_values('Age'), df.loc[0, 'Name'])

# --- 2. heart.csv ---
heart = pd.read_csv('/mnt/data/heart.csv')
print(heart.head(), heart.dtypes, heart.describe())
print(heart[heart['cp'] > 2])  # filter cp > 2

# --- 3. bank-data.csv ---
bank = pd.read_csv('/mnt/data/bank-data.csv')
print(bank.head(), bank.describe())

sns.scatterplot(x='income', y='children', data=bank)
plt.title("Income vs Children")
plt.show()

sns.countplot(x='region', data=bank)
plt.title("Region Count")
plt.show()

sns.histplot(bank['children'], kde=True)
plt.title("Children Distribution")
plt.show()

# --- 4. EDA on bank-data.csv ---
data = bank.copy()
data.rename(columns={'income': 'Income_USD'}, inplace=True)

sns.countplot(x='pep', data=data)
plt.title("PEP Distribution")
plt.show()

sns.countplot(x='sex', hue='pep', data=data)
plt.title("PEP vs Sex")
plt.show()

sns.countplot(x='children', data=data)
plt.title("Children Count")
plt.show()

pivot = data.groupby(['pep', 'sex', 'children']).size().reset_index(name='count')
sns.catplot(x='children', y='count', hue='sex', col='pep', kind='bar', data=pivot)
plt.suptitle("PEP vs Children vs Sex")
plt.show()
