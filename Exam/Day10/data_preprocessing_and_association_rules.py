
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Import numpy and pandas - already done
# 2. Load the dataset ls-01
df = pd.read_csv('ls-01.csv')

# 3. Create two copies of the dataset
df_copy1 = df.copy()
df_copy2 = df.copy()

# 4. Check for null values
print("Null values in each column:")
print(df.isnull().sum())

# 5. Missing data handling

# 6. Remove missing values
df_removed = df.dropna()
print("\nData after removing missing values:")
print(df_removed)

# 7. Fill missing values
df_filled = df.fillna("Missing")
print("\nData after filling missing values with 'Missing':")
print(df_filled)

# 8. Group the data by a sample column (update as per actual data)
# Example assuming a 'Department' column exists
# grouped = df.groupby('Department')

# 9. Label missing values in a specific column (example: 'Gender')
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].fillna('Unknown')

# 10. Group by gender (if exists)
if 'Gender' in df.columns:
    gender_grouped = df.groupby('Gender')
    print("\nGrouped by Gender:")
    for name, group in gender_grouped:
        print(f"\n{name}:")
        print(group)

# 11–12. Define dataset for association rule mining
dataset = [
    ['milk','onion','nugmeg','kidney beans','eggs','yogurt'],
    ['oil','onion','nutmeg','kidney beans','eggs','yogurt'],
    ['milk','apple','kidney beans','eggs'],
    ['milk','unicorn','corn','kidney beans','yogurt'],
    ['corn','onion','onion','kidney beans','ice cream','eggs']
]

# 13. Import libraries - already done above

# 14. Create the dataset as a DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# 15. Generate frequent itemsets with min_support=0.6
frequent_itemsets = apriori(df_trans, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# 16. Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules)

# 17. Create subset with key metrics
rules_subset = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print("\nRules Subset:")
print(rules_subset)

# 18. Retrieve rules with confidence >= 1
high_confidence = rules[rules['confidence'] >= 1]
print("\nHigh Confidence Rules (confidence >= 1):")
print(high_confidence)
