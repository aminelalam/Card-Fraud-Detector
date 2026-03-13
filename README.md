# FraudGuard AI - Credit Card Fraud Detection System

A complete machine learning pipeline + interactive dashboard for detecting fraudulent credit card transactions.

Built with **XGBoost**, **scikit-learn**, **imbalanced-learn**, and **Streamlit**.

---

## Features

### ML Pipeline (`train_model.py`)
- **RobustScaler** preprocessing for Time & Amount features
- **Temporal train/test split** (80/20 chronological - no data leakage)
- **5 model comparison**: Logistic Regression, Random Forest, XGBoost (balanced/unbalanced), XGBoost + SMOTE
- **StratifiedKFold** 5-fold cross-validation
- **RandomizedSearchCV** hyperparameter tuning (30 iterations)
- Business-oriented evaluation: **Recall**, **PR-AUC**, **Confusion Matrix**

### Streamlit Dashboard (`app.py`)
- **Dashboard EDA** - Interactive charts: V1 vs V2 scatter, fraud-by-hour, amount distribution
- **Fraud Simulator** - Real-time prediction with confidence score and latency
- **Model Performance** - Confusion matrix, PR-AUC curve, benchmark table, expert review notes

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates models/*.joblib + charts)
python train_model.py

# 3. Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
archive/
├── app.py                 # Streamlit dashboard (3 pages)
├── train_model.py         # Full ML training pipeline
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── creditcard.csv          # Dataset (284,807 transactions)
├── models/
│   ├── best_model.joblib  # Tuned XGBoost model
│   ├── scaler.joblib      # Fitted RobustScaler
│   ├── metrics.json       # All model metrics & comparison data
│   ├── eda_charts.png     # EDA visualizations
│   ├── confusion_pr_curve.png
│   └── model_comparison.png
└── notebooks/             # Jupyter notebooks (EDA research)
```

---

## Model Results

| Model | Recall | Precision | PR-AUC | ROC-AUC |
|-------|--------|-----------|--------|---------|
| Logistic Regression | 0.893 | 0.071 | 0.760 | 0.986 |
| Random Forest | 0.653 | 0.980 | 0.802 | 0.938 |
| **XGBoost (Tuned)** | **0.760** | **0.851** | **0.794** | **0.982** |
| XGBoost + SMOTE | 0.760 | 0.851 | 0.788 | 0.977 |
| XGBoost (unbalanced) | 0.707 | 0.914 | 0.728 | 0.967 |

> **Why not Accuracy?** With 99.83% legitimate transactions, a model that predicts "legit" for everything scores 99.83% accuracy. That's useless. We optimize for **Recall** (catching fraudsters) and **PR-AUC** (the industry standard for imbalanced data).

---

## Key Design Decisions

1. **RobustScaler > StandardScaler**: Time and Amount have extreme outliers from fraudulent transactions. RobustScaler uses the IQR, making it resilient to these spikes.

2. **Temporal Split > Random Split**: In production, you predict future transactions based on past data. A random split would leak future information into training.

3. **Balanced vs Unbalanced**: Using `scale_pos_weight` improved XGBoost recall by **+5.3%** — a critical gain when each missed fraud costs real money.

4. **SMOTE applied only to training data**: Synthetic oversampling on test data would give misleadingly optimistic results.

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions (Sep 2013, European cardholders)
- 492 frauds (0.17%)
- V1-V28: PCA-transformed features (anonymized)
- Time: seconds elapsed from first transaction
- Amount: transaction amount

---

## Tech Stack

- **Python 3.13** | **scikit-learn** | **XGBoost** | **imbalanced-learn**
- **Streamlit** | **Plotly** | **Seaborn** | **Pandas**

---

*Built as a portfolio project demonstrating end-to-end ML engineering for fraud detection.*
