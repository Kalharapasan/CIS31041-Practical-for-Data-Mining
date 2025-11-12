#01. Pandas
# a. Import the pandas as pd
import pandas as pd

# b. Create the series
series = pd.Series([10, 20, 30, 40, 50])

# c. Find the data type of the series
print("Data type of series:", series.dtype)

# d. Print the index of the element
print("Index of series:", series.index)

# e. Multiply and add the element in the series
print("Series multiplied by 2:\n", series * 2)
print("Series plus 10:\n", series + 10)

# f. Print the date
date_series = pd.date_range(start="2023-01-01", periods=5)
print("Date series:\n", date_series)

# g. Create the data frame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# h. Insert the given data frame and explain the output
print("DataFrame:\n", df)

# i. Find the datatype of data frame
print("Data types:\n", df.dtypes)

# j. Find the first few values of the data frame
print("First few records:\n", df.head())

# k. Find the last records of the data frame
print("Last few records:\n", df.tail())

# l. Index values of data frame
print("Index of DataFrame:", df.index)

# m. Find the column of the data frame
print("Columns of DataFrame:", df.columns)

# n. Convert the data frame into numpy array
print("DataFrame as numpy array:\n", df.to_numpy())

# o. Find the statistics summary of data frame
print("Summary:\n", df.describe(include='all'))

# p. Transpose the data frame
print("Transposed:\n", df.T)

# q. Sort the data frame (by Age)
print("Sorted by Age:\n", df.sort_values(by='Age'))

# r. Retrieve the value from the data frame
print("Retrieve value in row 1, column 'Name':", df.loc[1, 'Name'])


#02. Import the Data Set (heart.csv)
import pandas as pd

# a. Filter the first five records of the data set heart.csv
heart_df = pd.read_csv('/mnt/data/heart.csv')
print("First 5 records:\n", heart_df.head())

# b. Find the data type of attributes
print("\nData types of each attribute:\n", heart_df.dtypes)

# c. Find the information of data set
print("\nDataset Information:")
heart_df.info()

# d. Find the summary statistics of the data set
print("\nSummary Statistics:\n", heart_df.describe())

# e. Retrieve the information cp greater than 2
filtered_cp = heart_df[heart_df['cp'] > 2]
print("\nRecords where cp > 2:\n", filtered_cp)

#03. Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# a. Import the bank-data.csv file
bank_df = pd.read_csv('/mnt/data/bank-data.csv')

# b. Retrieve the first five records in the data set
print("First 5 records:\n", bank_df.head())

# c. Plot the data set (basic pairplot to visualize relationships)
sns.pairplot(bank_df.select_dtypes(include=['float64', 'int64']))
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# d. Describe the data set
print("\nDescriptive statistics:\n", bank_df.describe(include='all'))

# e. Plot the data set income Vs children
plt.figure(figsize=(8, 5))
sns.scatterplot(x='income', y='children', data=bank_df)
plt.title("Income vs Children")
plt.xlabel("Income")
plt.ylabel("Children")
plt.grid(True)
plt.show()
