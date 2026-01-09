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
```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend scipy
```

### Running Jupyter Notebooks
1. Navigate to the desired labsheet directory
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the `.ipynb` file and run cells sequentially

### Running Python Scripts
```bash
python <script_name>.py
```

## 📝 Key Topics Covered

### 1. Data Preprocessing
- Handling missing values
- Data normalization and standardization
- Feature encoding (Label Encoding, One-Hot Encoding)
- Data transformation
- Outlier detection and treatment

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries
- Data visualization
- Correlation analysis
- Distribution analysis

### 3. Classification
- Decision Trees
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- Model evaluation metrics

### 4. Clustering
- K-Means clustering
- Hierarchical clustering
- Cluster evaluation

### 5. Association Rule Mining
- Apriori algorithm
- Support, Confidence, and Lift metrics
- Market basket analysis

### 6. Model Evaluation
- Train-test split
- Cross-validation
- Confusion matrix
- Accuracy, Precision, Recall, F1-Score

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
