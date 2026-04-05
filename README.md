# Heart-Failure-Prediction

<div align="center">

# 🫀 Heart Disease Dataset — Supervised Learning

*A complete Python pipeline covering Decision Tree, Linear Regression, Logistic Regression, and Naïve Bayes — with model comparison*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-11557C?style=flat)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/Dataset-Public%20Domain-brightgreen?style=flat)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

</div>

---

## 📋 Abstract

This project applies four supervised learning algorithms to the Heart Disease dataset to predict the presence or absence of heart disease in a patient. The pipeline begins with dataset loading and exploration, moves through categorical encoding and a deliberate choice not to scale features for tree-based models, and then trains and evaluates a Decision Tree, a Linear Regression baseline, a Logistic Regression classifier, and a Naïve Bayes classifier. Each model is assessed using accuracy, confusion matrix, precision, recall, and F1-score. A final comparison summarises which model best suits this binary classification task and why.

---

## 🗂️ Dataset

The dataset contains **918 patient records** with 11 clinical features and a binary target variable indicating the presence of heart disease.

| Feature | Type | Description |
|:---|:---|:---|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | M = Male, F = Female |
| `ChestPainType` | Categorical | ATA / NAP / ASY / TA |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mm/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl (1 = true) |
| `RestingECG` | Categorical | Normal / ST / LVH |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y / N) |
| `Oldpeak` | Numeric | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment (Up / Flat / Down) |
| `HeartDisease` | Binary | Target — 0 = no disease, 1 = disease |

> **Class balance:** 508 positive (55.3%) · 410 negative (44.7%) — mildly imbalanced but workable without resampling.

---

## 📁 Project Structure

```
heart-supervised/
│
├── 📄 heart.csv                    # Raw dataset
├── 📓 heartfailure.ipynb       # Main notebook
└── 📝 README.md
```

---

## ⚙️ Setup

```bash
pip install pandas numpy matplotlib scikit-learn
```

> **🪐 JupyterLite users:**
> ```python
> import micropip
> await micropip.install(['scikit-learn'])
> ```

---

## 🔍 Pipeline Steps

### `01` · Dataset Overview

Load the dataset and perform a quick structural inspection before any modelling begins.

```python
df = pd.read_csv('heart.csv')
df.info()
df.describe()
df['HeartDisease'].value_counts()
```

Covers shape, data types, missing value check, and target class distribution. The dataset is clean with no missing values across all 918 rows and 12 columns.

---

### `02` · Encoding Categorical Variables

Five categorical columns are present and must be encoded before any model can be trained. Label Encoding is applied to binary columns and One-Hot Encoding to multi-class columns.

| Column | Unique Values | Strategy |
|:---|:---|:---|
| `Sex` | M, F | Label Encoding → 0 / 1 |
| `ExerciseAngina` | Y, N | Label Encoding → 0 / 1 |
| `ChestPainType` | ATA, NAP, ASY, TA | One-Hot Encoding |
| `RestingECG` | Normal, ST, LVH | One-Hot Encoding |
| `ST_Slope` | Up, Flat, Down | One-Hot Encoding |

After encoding the dataset expands from 12 to 19 columns, all numeric.

---

### `03` · Decision Tree

A Decision Tree classifier is trained on the raw (unscaled) encoded features. Decision trees are invariant to feature scale — bringing all values to a common scale is unnecessary and would not change the tree structure or its predictions.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
```

The full tree is plotted using `sklearn.tree.plot_tree` with feature names, class labels, and colour-coded nodes. Feature importances are extracted and ranked to show which clinical measurements drive the splits most strongly.

> **Why no scaling?** Decision trees split on thresholds — the absolute scale of a feature has no effect on where the best split falls. Scaling is required for distance-based and gradient-based models but not for tree ensembles.

---

### `04` · Linear Regression

Linear Regression is applied as a numeric baseline. Although the target is binary (0 / 1), Linear Regression can still produce a continuous prediction that, when thresholded at 0.5, yields a binary classification. This step illustrates both the capability and the limitations of using a regression model for a classification task.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = (reg.predict(X_test) >= 0.5).astype(int)
```

> **Note:** Feature scaling is applied here since Linear Regression is sensitive to feature magnitude when interpreting coefficients.

---

### `05` · Logistic Regression

Logistic Regression is the natural model for binary classification. It outputs calibrated probabilities through the sigmoid function and is directly interpretable through its coefficients.

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
```

Coefficients are extracted and ranked to show which features push predictions toward heart disease and which push away from it. A confusion matrix and full classification report are produced.

---

### `06` · Naïve Bayes

Gaussian Naïve Bayes assumes each feature is conditionally independent given the class label and follows a normal distribution. It is a fast, probabilistic classifier well-suited for medical datasets where features often carry independent signal.

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
```

Despite its strong independence assumption — which is unlikely to hold perfectly for clinical measurements — Naïve Bayes often performs competitively in practice, especially when training data is limited.

---

### `07` · Notebook to GitHub

The completed notebook is exported and pushed to GitHub for version control and sharing.

```bash
# Export notebook to clean format
jupyter nbconvert --to notebook --ClearOutputs.enabled=True heart_supervised.ipynb

# Push to GitHub
git init
git add .
git commit -m "Heart disease supervised learning pipeline"
git remote add origin https://github.com/your-username/heart-supervised.git
git push -u origin main
```

---

### `08` · Summary & Model Comparison

All four models are evaluated on the same test set using the same metrics, then compared in a summary table.

| Metric | Description |
|:---|:---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of all predicted positives, how many are truly positive |
| **Recall** | Of all actual positives, how many were correctly identified |
| **F1-score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Full breakdown of TP, FP, TN, FN |

For a medical diagnosis task, **recall** is the most critical metric — a false negative (predicting no disease when disease is present) is more dangerous than a false positive.

---

## 🛠️ Libraries Used

| Library | Purpose |
|:---|:---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Tree visualisation and plots |
| `scikit-learn` | All models, metrics, and train/test split |

---

