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
- `eda_full_script.py`: Comprehensive exploratory data analysis
- `eda_short_revision.py`: Quick EDA revision notes

### Specialized Scripts
- `data_preprocessing_and_association_rules.py`: Combined preprocessing and association rules
- `bank_clustering_classification.py`: Banking data analysis with clustering and classification

## 📚 Resources

### Documentation Files
- `Notes.txt`, `text.txt`: Various course notes
- `Function Explanation.txt`: Python function explanations
- `ReadMe.txt`: Specific instructions for assignments
- `Q.txt`: Question files

## 🎯 Learning Objectives

By completing this coursework, you will:
- Master data preprocessing techniques
- Understand and implement classification algorithms
- Perform clustering analysis
- Extract association rules from transactional data
- Evaluate machine learning models effectively
- Visualize and communicate data insights

## 📧 Course Information
**Course Code**: CIS31041  
**Course Title**: Practical for Data Mining  
**Focus**: Hands-on implementation of data mining techniques using Python

## 📄 License
Educational use only - Course materials for CIS31041

---

**Last Updated**: January 9, 2026

*This workspace is organized for learning data mining concepts through practical implementation. Each labsheet builds upon previous concepts, providing a structured learning path.*
