# CIS31041 - Practical for Data Mining

## 📋 Overview
This repository contains practical exercises, lab sheets, and exam materials for the CIS31041 Data Mining course. It includes implementations of various data mining techniques including data preprocessing, exploratory data analysis (EDA), classification, clustering, and association rule mining.

## 📁 Workspace Structure

### Lab Sheets (Labsheet 1-12)
- **Labsheet 1-3**: Introduction to Python and data structures
- **Labsheet 4**: Basic Python programming with arrays and functions
- **Labsheet 5**: Data exploration and visualization using pandas
- **Labsheet 6**: Data preprocessing and ARFF file handling
- **Labsheet 7**: Advanced data preprocessing techniques
- **Labsheet 8**: Classification algorithms (Decision Trees, etc.)
- **Labsheet 9**: Model evaluation and prediction
- **Labsheet 10**: Association rule mining and data preprocessing
- **Labsheet 11**: Clustering and classification with banking data
- **Labsheet 12**: Time series analysis with rainfall dataset

### Exam Materials
- **Exam/2019/**: Past exam questions and datasets
  - Q01: Employee dataset analysis
  - Q02-Q04: Additional exam questions
- **CA4, CA4,5/**: Continuous assessment materials

### Practice Code Directory (code/)
Contains working examples and practice notebooks:
- Various Jupyter notebooks (Lb04, Lb05, Lb10, Lb11)
- Practice questions (Q01, Q02)
- Multiple datasets for experimentation

### Day-wise Practice (Day 1-11)
Daily practice sessions covering:
- Data loading and exploration
- Preprocessing techniques
- Classification algorithms
- Clustering methods
- Association rules
- Model evaluation

## 🔧 Technologies & Libraries

### Core Libraries
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **scipy**: Scientific computing

### Data Mining Tools
- **Weka** (ARFF file format support)
- **mlxtend**: Association rule mining (Apriori algorithm)

## 📊 Datasets

### Banking & Finance
- `bank-data.csv` / `Bank1.csv`: Customer banking information
- `bank.arff`: Banking dataset in ARFF format

### Health & Medical
- `heart.csv`: Heart disease dataset

### Business & HR
- `Company_ABC_HumanResource.csv`: Human resource management data
- `Employee.arff`: Employee information

### Other Datasets
- `iris.arff`: Classic iris flower dataset
- `glass.arff`: Glass identification dataset
- `telco_*.csv/arff`: Telecom customer data
- `Components_of_Fertilizer.csv`: Agricultural data
- `RainfallDataset.csv`: Weather/climate data
- `ls-01.csv`: General purpose dataset

## 🚀 Getting Started

### Prerequisites

#### Install Python
Ensure Python 3.8+ is installed on your system:
```bash
python --version
```

#### Install Required Libraries
```bash
# Essential packages
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# Association rule mining
pip install mlxtend

# Additional useful packages
pip install jupyter notebook openpyxl liac-arff

# For data visualization
pip install plotly

# Complete installation in one command
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend scipy jupyter liac-arff plotly openpyxl
```

### Environment Setup

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat

# Install packages
pip install -r requirements.txt
```

### Running Jupyter Notebooks
1. Navigate to the desired labsheet directory
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the `.ipynb` file and run cells sequentially
4. Use `Shift + Enter` to execute cells

### Running Python Scripts
```bash
# Basic execution
python <script_name>.py

# With arguments
python script.py --input data.csv --output results.csv

# Using Python's interactive mode
python -i script.py
```

## 📝 Key Topics Covered

### 1. Data Preprocessing
- **Handling Missing Values**
  - Detection: `df.isnull()`, `df.isna()`, `df.info()`
  - Removal: `df.dropna()`, `df.drop()`
  - Imputation: Mean, median, mode, forward/backward fill
  ```python
  df['column'].fillna(df['column'].mean(), inplace=True)
  ```

- **Data Normalization and Standardization**
  - Min-Max Scaling: Scale to [0,1]
  - Z-score Standardization: Mean=0, StdDev=1
  ```python
  from sklearn.preprocessing import MinMaxScaler, StandardScaler
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)
  ```

- **Feature Encoding**
  - Label Encoding: Convert categories to numbers
  - One-Hot Encoding: Create binary columns
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df['encoded'] = le.fit_transform(df['category'])
  ```

- **Data Transformation**
  - Log transformation
  - Box-Cox transformation
  - Power transformation

- **Outlier Detection and Treatment**
  - IQR method
  - Z-score method
  - Domain-specific rules

### 2. Exploratory Data Analysis (EDA)
- **Statistical Summaries**
  ```python
  df.describe()  # Summary statistics
  df.info()      # Data types and non-null counts
  df.shape       # Dimensions
  df.columns     # Column names
  ```

- **Data Visualization**
  - Histograms: Distribution of numerical data
  - Box plots: Identify outliers
  - Scatter plots: Relationships between variables
  - Correlation heatmaps: Feature relationships
  - Bar charts: Categorical data
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Correlation heatmap
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  plt.show()
  ```

- **Correlation Analysis**
  ```python
  df.corr()  # Pearson correlation
  df.corr(method='spearman')  # Spearman correlation
  ```

- **Distribution Analysis**
  - Normal distribution tests
  - Skewness and kurtosis
  - Q-Q plots

### 3. Classification
- **Decision Trees**
  ```python
  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier(max_depth=5)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)
  ```

- **k-Nearest Neighbors (k-NN)**
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train, y_train)
  ```

- **Naive Bayes**
  ```python
  from sklearn.naive_bayes import GaussianNB
  nb = GaussianNB()
  nb.fit(X_train, y_train)
  ```

- **Model Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curve
  - Classification report

### 4. Clustering
- **K-Means Clustering**
  ```python
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=3, random_state=42)
  clusters = kmeans.fit_predict(X)
  ```

- **Hierarchical Clustering**
  ```python
  from sklearn.cluster import AgglomerativeClustering
  hc = AgglomerativeClustering(n_clusters=3)
  labels = hc.fit_predict(X)
  ```

- **Cluster Evaluation**
  - Silhouette score
  - Elbow method
  - Davies-Bouldin index
  ```python
  from sklearn.metrics import silhouette_score
  score = silhouette_score(X, labels)
  ```

### 5. Association Rule Mining
- **Apriori Algorithm**
  ```python
  from mlxtend.frequent_patterns import apriori, association_rules
  
  # Find frequent itemsets
  frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
  
  # Generate association rules
  rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
  ```

- **Key Metrics**
  - **Support**: Frequency of itemset
    - `support(A) = count(A) / total_transactions`
  - **Confidence**: Probability of B given A
    - `confidence(A→B) = support(A∪B) / support(A)`
  - **Lift**: Strength of association
    - `lift(A→B) = confidence(A→B) / support(B)`

- **Market Basket Analysis**
  - Identifying product associations
  - Cross-selling opportunities
  - Store layout optimization

### 6. Model Evaluation
- **Train-Test Split**
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42
  )
  ```

- **Cross-Validation**
  ```python
  from sklearn.model_selection import cross_val_score
  scores = cross_val_score(model, X, y, cv=5)
  print(f"Average accuracy: {scores.mean():.2f}")
  ```

- **Confusion Matrix**
  ```python
  from sklearn.metrics import confusion_matrix, classification_report
  cm = confusion_matrix(y_test, y_pred)
  print(classification_report(y_test, y_pred))
  ```

- **Performance Metrics**
  - **Accuracy**: (TP + TN) / Total
  - **Precision**: TP / (TP + FP)
  - **Recall (Sensitivity)**: TP / (TP + FN)
  - **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

## 📖 File Formats

### ARFF Files
ARFF (Attribute-Relation File Format) is used by Weka. Structure:
```
@relation dataset_name
@attribute attribute_name {value1, value2, ...}
@data
value1, value2, ...
```

### CSV Files
Standard comma-separated values format for tabular data.

## 🔍 Important Scripts

### EDA Scripts
- **`eda_full_script.py`**: Comprehensive exploratory data analysis
  - Complete statistical analysis
  - Multiple visualization types
  - Correlation analysis
  - Missing value detection
  
- **`eda_short_revision.py`**: Quick EDA revision notes
  - Condensed analysis workflow
  - Key visualization examples
  - Quick reference guide

### Specialized Scripts
- **`data_preprocessing_and_association_rules.py`**: 
  - Data cleaning pipeline
  - Feature engineering
  - Association rule mining implementation
  - Frequent itemset generation
  
- **`bank_clustering_classification.py`**: 
  - Banking data analysis
  - Customer segmentation using clustering
  - Predictive classification models
  - Model comparison and evaluation

### Lab-Specific Scripts
- **`code.py`**: General purpose data mining examples
- **`04.py`, `05.py`**: Lab-specific implementations
- **`LB04.py`, `LB06.py`**: Labsheet code solutions

## 💡 Common Code Patterns

### Loading Data
```python
import pandas as pd

# CSV files
df = pd.read_csv('data.csv')

# ARFF files (using liac-arff)
import arff
with open('data.arff', 'r') as f:
    dataset = arff.load(f)
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# Excel files
df = pd.read_excel('data.xlsx')
```

### Basic Data Exploration
```python
# Quick overview
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Unique values in categorical columns
print(df['column'].value_counts())
```

### Data Cleaning Pipeline
```python
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean())  # For numerical columns
df = df.fillna(df.mode().iloc[0])  # For categorical columns

# Remove outliers using IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['column'] < (Q1 - 1.5 * IQR)) | (df['column'] > (Q3 + 1.5 * IQR)))]
```

### Building a Complete Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier())
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 🎓 Study Tips & Best Practices

### 1. Workflow Organization
- Keep each labsheet in its dedicated folder
- Use descriptive file names
- Comment your code extensively
- Save multiple versions during development

### 2. Data Handling
- Always create a copy before modifying: `df_clean = df.copy()`
- Understand your data before preprocessing
- Document all transformations
- Keep raw data unchanged

### 3. Model Development
- Start simple, then increase complexity
- Always split data before any preprocessing
- Use cross-validation for robust evaluation
- Compare multiple algorithms

### 4. Jupyter Notebook Tips
- Use markdown cells for documentation
- Restart kernel and run all cells before submission
- Include visualizations to support findings
- Clear outputs before version control

### 5. Debugging Strategies
- Print intermediate results
- Use `df.head()` frequently
- Check data types with `df.dtypes`
- Verify shapes after transformations
- Use small data samples for testing

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue: Module Not Found
```bash
# Solution: Install missing package
pip install package_name

# or upgrade pip first
python -m pip install --upgrade pip
```

#### Issue: Jupyter Kernel Not Found
```bash
# Solution: Install ipykernel
pip install ipykernel
python -m ipykernel install --user
```

#### Issue: ARFF File Reading Error
```python
# Solution: Use scipy or liac-arff
pip install liac-arff

# Alternative using scipy
from scipy.io import arff
data, meta = arff.loadarff('file.arff')
df = pd.DataFrame(data)
```

#### Issue: Memory Error with Large Datasets
```python
# Solution: Read data in chunks
chunk_iter = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunk_iter:
    process(chunk)
```

#### Issue: Categorical Data in Scikit-learn
```python
# Solution: Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['category_column'])
```

## 📊 Dataset Details

### Banking Datasets
**Files**: `bank-data.csv`, `Bank1.csv`, `bank.arff`

**Features**:
- Customer demographics (age, job, marital status, education)
- Financial information (balance, loan status)
- Marketing campaign data (contact, duration, campaign)
- Target variable: deposit subscription (yes/no)

**Use Cases**:
- Customer segmentation
- Churn prediction
- Marketing campaign optimization

### Heart Disease Dataset
**File**: `heart.csv`

**Features**:
- Age, sex, chest pain type
- Resting blood pressure, cholesterol
- Fasting blood sugar, ECG results
- Maximum heart rate, exercise-induced angina
- Target: Heart disease presence (0/1)

**Use Cases**:
- Disease prediction
- Risk factor analysis
- Healthcare decision support

### HR Dataset
**File**: `Company_ABC_HumanResource.csv`

**Features**:
- Employee demographics
- Job satisfaction, performance rating
- Work-life balance, years at company
- Attrition status

**Use Cases**:
- Employee attrition prediction
- HR analytics
- Workforce planning

### Telecom Dataset
**Files**: `telco_*.csv`, `telco_*.arff`

**Features**:
- Customer information
- Service subscriptions
- Billing information
- Churn status

**Use Cases**:
- Customer retention
- Service optimization
- Revenue forecasting

## 📚 Resources

### Documentation Files
- `Notes.txt`, `text.txt`: Various course notes and summaries
- `Function Explanation.txt`: Python function explanations and examples
- `ReadMe.txt`: Specific instructions for assignments and exercises
- `Q.txt`: Question files with problem statements

### Official Documentation Links
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [MLxtend Documentation](http://rasbt.github.io/mlxtend/)

### Recommended Reading
- Python for Data Analysis by Wes McKinney
- Hands-On Machine Learning with Scikit-Learn by Aurélien Géron
- Introduction to Data Mining by Tan, Steinbach & Kumar

### Useful Websites
- [Kaggle](https://www.kaggle.com/) - Datasets and competitions
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Towards Data Science](https://towardsdatascience.com/) - Tutorials and articles

## 🔑 Quick Reference Guide

### Pandas Cheat Sheet
```python
# Data Selection
df.loc[row_label, column_label]  # By label
df.iloc[row_index, column_index]  # By position
df['column']                       # Single column
df[['col1', 'col2']]              # Multiple columns

# Filtering
df[df['age'] > 30]                # Boolean indexing
df.query('age > 30 and city == "NYC"')  # Query syntax

# Grouping
df.groupby('category').mean()     # Group by and aggregate
df.pivot_table(values='sales', index='month', columns='year', aggfunc='sum')

# Sorting
df.sort_values('column', ascending=False)
df.sort_index()
```

### Scikit-learn Model Selection
```python
# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```

### Common Evaluation Metrics
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

# Classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Regression
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
```

## 🎯 Exam Preparation Tips

### Past Exam Analysis (2019)
- **Q01**: Employee dataset analysis
  - Data preprocessing
  - Handling categorical variables
  - Classification tasks
  
- **Q02-Q04**: Various data mining scenarios
  - Association rule mining
  - Clustering analysis
  - Model evaluation

### Key Skills to Master
1. **Data Preprocessing** (30% of exam)
   - Missing value handling
   - Encoding techniques
   - Feature scaling
   
2. **Model Implementation** (40% of exam)
   - Classification algorithms
   - Clustering methods
   - Association rules
   
3. **Evaluation & Interpretation** (30% of exam)
   - Performance metrics
   - Result visualization
   - Model comparison

### Time Management
- Read all questions first (5 minutes)
- Allocate time based on marks
- Leave time for review (10-15 minutes)
- Complete easier questions first

### Common Exam Patterns
- Load and explore dataset
- Handle missing values
- Encode categorical features
- Build and train model
- Evaluate performance
- Interpret results

## 🚦 Project Workflow

### 1. Problem Understanding
- Define business objective
- Identify target variable
- Understand constraints

### 2. Data Collection & Exploration
```python
# Load data
df = pd.read_csv('dataset.csv')

# Initial exploration
print(df.head())
print(df.info())
print(df.describe())
```

### 3. Data Preprocessing
```python
# Handle missing values
df = df.fillna(df.mean())

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Feature Engineering
- Create new features
- Select important features
- Reduce dimensionality if needed

### 5. Model Selection & Training
```python
# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### 6. Model Evaluation
```python
# Make predictions
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

### 7. Model Optimization
- Hyperparameter tuning
- Cross-validation
- Ensemble methods

### 8. Deployment & Documentation
- Save trained model
- Document findings
- Create visualizations

## 🎯 Learning Objectives

By completing this coursework, you will:
- ✅ Master data preprocessing techniques for real-world datasets
- ✅ Understand and implement classification algorithms (Decision Trees, k-NN, Naive Bayes)
- ✅ Perform clustering analysis for customer segmentation
- ✅ Extract association rules from transactional data
- ✅ Evaluate machine learning models effectively using appropriate metrics
- ✅ Visualize and communicate data insights professionally
- ✅ Work with multiple data formats (CSV, ARFF, Excel)
- ✅ Build complete data mining pipelines from raw data to insights
- ✅ Apply cross-validation and other model validation techniques
- ✅ Interpret model results and make data-driven recommendations

## 📈 Performance Metrics Reference

### Classification Metrics

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| **Accuracy** | (TP + TN) / Total | Balanced datasets | 0 to 1 |
| **Precision** | TP / (TP + FP) | Minimize false positives | 0 to 1 |
| **Recall** | TP / (TP + FN) | Minimize false negatives | 0 to 1 |
| **F1-Score** | 2 × (P × R) / (P + R) | Balance precision & recall | 0 to 1 |
| **ROC-AUC** | Area under ROC curve | Overall model performance | 0 to 1 |

### Clustering Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Silhouette Score** | Cohesion vs separation | Higher is better (-1 to 1) |
| **Davies-Bouldin Index** | Average similarity ratio | Lower is better (≥0) |
| **Inertia** | Sum of squared distances | Lower is better |
| **Calinski-Harabasz** | Variance ratio | Higher is better |

### Regression Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | mean(\|y_true - y_pred\|) | Mean absolute error |
| **MSE** | mean((y_true - y_pred)²) | Mean squared error |
| **RMSE** | √MSE | Root mean squared error |
| **R²** | 1 - (SS_res / SS_tot) | Coefficient of determination |

## 🔧 Advanced Techniques

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2f}")
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

# Chi-square test
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Feature importance from tree-based model
rf = RandomForestClassifier()
rf.fit(X, y)
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### Handling Imbalanced Datasets
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Under-sampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Manual over-sampling
df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]
df_minority_upsampled = resample(df_minority, 
                                  replace=True,
                                  n_samples=len(df_majority),
                                  random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
```

### Ensemble Methods
```python
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression()
)
```

## 📊 Visualization Gallery

### Essential Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['column'], kde=True)
plt.title('Distribution of Column')
plt.show()

# Box plot for outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Box Plot for Outlier Detection')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(df, hue='target')
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Feature Importances')
plt.show()
```

## 🔐 Data Privacy & Ethics

### Best Practices
- Anonymize personal identifiable information (PII)
- Respect data usage agreements
- Consider bias in datasets and models
- Document data sources and transformations
- Ensure reproducibility

### Ethical Considerations
- Avoid discriminatory features (race, gender, etc.)
- Validate model fairness across groups
- Consider societal impact of predictions
- Maintain transparency in model decisions
- Regular bias audits

## 💾 Saving & Loading Models

### Model Persistence
```python
import pickle
import joblib

# Using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Using joblib (recommended for large models)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
```

### Saving Preprocessors
```python
# Save scaler
joblib.dump(scaler, 'scaler.joblib')

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.joblib')

# Complete pipeline
joblib.dump(pipeline, 'complete_pipeline.joblib')
```

## 🗂️ Directory Organization Tips

```
project/
│
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Third-party data
│
├── notebooks/            # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/                  # Source code
│   ├── data/            # Data loading and processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   └── visualization/   # Plotting functions
│
├── models/              # Trained models
├── reports/             # Analysis reports
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
```

## 📧 Course Information
**Course Code**: CIS31041  
**Course Title**: Practical for Data Mining  
**Focus**: Hands-on implementation of data mining techniques using Python

## 📄 License
Educational use only - Course materials for CIS31041

---

**Last Updated**: January 9, 2026

*This workspace is organized for learning data mining concepts through practical implementation. Each labsheet builds upon previous concepts, providing a structured learning path.*
