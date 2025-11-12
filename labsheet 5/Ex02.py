# Import pandas
import pandas as pd

# 02. Import the data set
df = pd.read_csv("heart.csv")
  # Make sure heart.csv is in the same directory

# a. Filter the first five records of the data set
print("First 5 Records:\n", df.head())

# b. Find the data type of attributes
print("\nData Types:\n", df.dtypes)

# c. Find the information of data set
print("\nDataFrame Info:")
df.info()

# d. Find the summary statistics of the data set
print("\nSummary Statistics:\n", df.describe())

# e. Retrieve the information where cp > 2
print("\nRecords where 'cp' > 2:\n", df[df['cp'] > 2])
