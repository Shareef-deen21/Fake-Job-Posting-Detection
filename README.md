# Fake Job Posting Detection

> Computational Intelligence Module Assignment - Cardiff Metropolitan University
> Binary classification system to detect fraudulent job postings using NLP and Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/fake-job-posting)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Online recruitment fraud is a growing global threat. This project builds a **binary classification system** that detects whether a job posting is **fraudulent (1)** or **legitimate (0)** using Natural Language Processing (NLP) and Machine Learning.

The system was developed as a submission for the Kaggle competition [Fake Job Posting Detection](https://www.kaggle.com/competitions/fake-job-posting) and deployed as a Streamlit web application for real-time predictions.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | [Kaggle: Fake Job Posting](https://www.kaggle.com/competitions/fake-job-posting) |
| Training rows | 12,001 |
| Columns | 18 (text + binary + categorical) |
| Target | `fraudulent` - 0 (Real) or 1 (Fake) |
| Class imbalance | 95.1% Real vs 4.9% Fake (19.4:1 ratio) |

**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| title | Text | Job title |
| description | Text | Full job description |
| requirements | Text | Required qualifications |
| company_profile | Text | Brief company description |
| has_company_logo | Binary | 1 if company logo present |
| has_questions | Binary | 1 if screening questions included |
| telecommuting | Binary | 1 if remote position |
| fraudulent | Binary | **Target — 1 = Fake, 0 = Real** |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| IDE | PyCharm |
| Notebook | Jupyter Notebook |
| NLP | NLTK, scikit-learn TfidfVectorizer |
| ML Models | scikit-learn, XGBoost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Model Saving | joblib |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |

---

## Models Trained

| # | Model | Technique | F1 Score |
|---|-------|-----------|----------|
| 1 | Logistic Regression | Baseline | 0.5667 |
| 2 | Random Forest | Ensemble (Bagging) | 0.7870 |
| 3 | XGBoost | Ensemble (Boosting) | 0.7707 |
| 4 | SVM (LinearSVC) | Support Vector Machine | 0.7981 |
| 5 | Neural Network (MLP) | Neural Network | 0.8017 |
| 6 | **SVM (GridSearch Tuned)** | **Best Model** | **0.8058** |

> **Primary metric: F1-Score** — Accuracy is misleading due to the 19.4:1 class imbalance.

---

## System Architecture

```
DATA LAYER
train.csv + test.csv (12,001 rows × 18 columns)
        ↓
NLP PREPROCESSING LAYER
Text cleaning → HTML removal → Lemmatisation → Stopword removal
TF-IDF Vectorisation (5,000 features, 1+2 ngrams)
Feature Engineering (desc_len, has_company_logo, has_questions...)
        ↓
CLASS BALANCING LAYER
SMOTE — balances 4.9% → 50% fraud in training set only
        ↓
MODEL LAYER
LR | Random Forest | XGBoost | SVM | MLP
Best model selected by F1-Score
        ↓
APPLICATION LAYER
Kaggle submission.csv + Streamlit Web App
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shareefdeen/Fake-Job-Posting-Detection.git
cd fake-job-posting-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Go to the [Kaggle competition page](https://www.kaggle.com/competitions/fake-job-posting/data), download `train.csv` and `test.csv`, and place them in the `Data/` folder.

### 4. Run the Jupyter notebook

```bash
jupyter notebook Fake_Job_Detection_CI.ipynb
```

Run all cells from top to bottom. This will:
- Perform EDA (9 analysis steps)
- Preprocess and engineer features
- Train and evaluate all 5 models
- Apply hyperparameter tuning
- Generate 9 Kaggle submission files
- Save the best model to `Models/`

### 5. Launch the web application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Web Application

The Streamlit app allows real-time fraud detection on new job postings.

**Input fields:**
- Job Title, Company Profile, Location, Industry
- Employment Type, Required Experience
- Job Description, Requirements, Benefits
- Remote position, Company logo, Screening questions

**Output:**
- 🚨 FAKE JOB POSTING DETECTED or ✅ LEGITIMATE JOB POSTING
- Fraud probability score (0.0 – 1.0)
- Confidence breakdown with progress bar

---

## Results

**Best model: SVM (GridSearch Tuned)**

| Metric | Score |
|--------|-------|
| Accuracy | 98.3% |
| AUC-ROC | 0.9825 |
| Precision | 0.9432 |
| Recall | 0.7034 |
| F1-Score | **0.8058** |

**5-Fold Cross-Validation:** Mean F1 = 0.8233 ± 0.0191 ✅ Stable

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
nltk
joblib
scipy
streamlit
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## References

- Vidros, S. et al. (2017). Automatic Detection of Online Recruitment Frauds. *Future Internet*, 9(1), 6. https://doi.org/10.3390/fi9010006
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*. https://doi.org/10.1145/2939672.2939785
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, pp.321–357. https://doi.org/10.1613/jair.953
- He, H. and Garcia, E.A. (2009). Learning from Imbalanced Data. *IEEE TKDE*, 21(9). https://doi.org/10.1109/TKDE.2008.239

---

## Author

**Shareefdeen**
Final-year BSc (Hons) Software Engineering Student
Cardiff Metropolitan University via ICBT Campus, Colombo, Sri Lanka

[![Portfolio](https://img.shields.io/badge/Portfolio-shareefdeen--portfolio.netlify.app-blue)](https://shareefdeen-portfolio.netlify.app)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed for the Computational Intelligence (CI) Module - Cardiff Metropolitan University, 2026*
