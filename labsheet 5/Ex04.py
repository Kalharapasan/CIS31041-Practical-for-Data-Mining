# a. Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# b. Import the dataset bank-data.csv as data
data = pd.read_csv('bank-data.csv')  # Ensure the file is in the correct path

# c. Check the data type
print("\nData Types:\n", data.dtypes)

# d. Check the data shape
print("\nData Shape:", data.shape)

# e. Check the attribute name of the dataset
print("\nAttribute Names:\n", data.columns)

# f. Check for the null values
print("\nNull Values:\n", data.isnull().sum())

# g. Remove unwanted attributes (example: 'unwanted_column' can be replaced with actual unwanted columns)
# Example: data = data.drop(columns=['unwanted_column'])
# data = data.drop(columns=['unwanted_column'])

# h. Rename the attribute name (example: renaming 'old_name' to 'new_name')
# Example: data.rename(columns={'old_name': 'new_name'}, inplace=True)

# i. Construct the plot for pep (assuming pep is a column in the dataset)
# Let's plot the distribution of the 'pep' column
plt.figure(figsize=(8, 6))
sns.countplot(x='pep', data=data)
plt.title('Distribution of pep')
plt.xlabel('PEP')
plt.ylabel('Count')
plt.show()

# j. Group the data by 'sex' and 'pep' according to the count
grouped_data = data.groupby(['sex', 'pep']).size().reset_index(name='count')
print("\nGrouped Data (sex and pep count):\n", grouped_data)

# k. Construct the graph for pep vs sex
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='pep', data=data)
plt.title('PEP vs Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# l. Find the count of the attribute children according to the category
children_count = data['children'].value_counts().reset_index()
children_count.columns = ['Children', 'Count']
print("\nChildren Count:\n", children_count)

# m. Construct the graph for the attribute children according to the category
plt.figure(figsize=(8, 6))
sns.countplot(x='children', data=data)
plt.title('Distribution of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

# n. Construct the chart to illustrate pep, children, and sex of the customer based on the count
plt.figure(figsize=(8, 6))
sns.countplot(x='children', hue='pep', data=data)
plt.title('PEP, Children, and Sex Distribution')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()
