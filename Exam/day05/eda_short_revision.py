
# --- EDA Revision Summary ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
heart = pd.read_csv('/mnt/data/heart.csv')
bank = pd.read_csv('/mnt/data/bank-data.csv')

# --- Pandas Basics ---
s = pd.Series([10, 20])
df = pd.DataFrame({'Name': ['A', 'B'], 'Age': [25, 30]})

# --- Quick Checks ---
print(df.head(), df.dtypes, df.describe())
print(heart[heart['cp'] > 2])  # Filter

# --- Simple Plots ---
sns.countplot(x='pep', data=bank); plt.show()
sns.scatterplot(x='income', y='children', data=bank); plt.show()
sns.histplot(bank['children']); plt.show()
sns.countplot(x='sex', hue='pep', data=bank); plt.show()

# --- Grouped Plot ---
g = bank.groupby(['pep', 'sex', 'children']).size().reset_index(name='count')
sns.catplot(x='children', y='count', hue='sex', col='pep', kind='bar', data=g)
plt.show()
