# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# a. Import the bank-data.csv file
df = pd.read_csv('bank-data.csv')  # Ensure the file is located correctly or provide the full path

# b. Retrieve the first five records in the data set
print("First 5 Records:\n", df.head())

# c. Plot the data set (For example, a pairplot for the numerical columns)
print("\nPlotting Pairplot for the dataset:")
sns.pairplot(df)
plt.show()

# d. Describe the data set
print("\nDataset Description:\n", df.describe())

# e. Plot the data set income vs children
plt.figure(figsize=(8, 6))
sns.scatterplot(x='income', y='children', data=df)
plt.title('Income vs Children')
plt.xlabel('Income')
plt.ylabel('Children')
plt.show()

# f. Construct the bar chart for region
plt.figure(figsize=(8, 6))
sns.countplot(x='region', data=df)
plt.title('Bar chart for Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.show()

# g. Construct the scatter plot for income and children with title
plt.figure(figsize=(8, 6))
sns.scatterplot(x='income', y='children', data=df)
plt.title('Income vs Children')
plt.xlabel('Income')
plt.ylabel('Children')
plt.show()

# h. Construct the histogram for children
plt.figure(figsize=(8, 6))
sns.histplot(df['children'], bins=10, kde=True)
plt.title('Histogram for Children')
plt.xlabel('Number of Children')
plt.ylabel('Frequency')
plt.show()
