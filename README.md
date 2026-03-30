# Telecom Customer Churn Intelligence System

> An end-to-end Data Science project combining ML and NLP built to predict and explain customer churn for a telecom operator.

---

## Project Overview

| Component | Tech Used | Output |
|-----------|-----------|--------|
| ML Churn Prediction | XGBoost, Random Forest, Logistic Regression, SHAP | ~0.86 AUC + feature explainability |
| NLP Complaint Analysis | TF-IDF, LDA Topic Modelling, NLTK | Sentiment + complaint topic per customer |
| LLM + RAG (OpenAI/Groq) | Natural language risk reports  coming soon |


---

## Business Problem

Telecom companies lose 20–40% of customers annually to churn, costing millions in lost revenue. This project builds an intelligent system that:

- **Predicts** which customers are likely to churn (ML)
- **Understands** why they are unhappy (NLP on complaint text)
- **Explains** which features drive each prediction (SHAP)

---

## Dataset

Synthetic dataset of 1,000 telecom customers with:

- 20 features: demographics, usage, billing, service quality
- Complaint text per customer (NLP input)
- Churn label (ML target)
- Realistic correlations: contract type and complaints are top churn drivers

**Churn rate: ~44%** handled with class weighting in tree models

---

## Repository Structure
```
telecom-churn-intelligence_report/
│
├── data/
│   └── telecom_churn_data.csv          # Raw dataset (1000 rows, 20 features)
│
├── outputs/
│   ├── predictions_for_powerbi.csv     # Model predictions + risk segments
│   ├── enriched_with_nlp.csv           # Dataset enriched with sentiment + topics
│   ├── feature_importance.csv          # XGBoost feature importance ranking
│   └── *.png                           # EDA, correlation, SHAP, NLP charts
│
├── telecom-churn-intelligence.ipynb    # Main notebook — all modules
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Results

| Model | CV AUC | Test AUC |
|-------|--------|----------|
| Logistic Regression | ~0.74 | ~0.73 |
| Random Forest | ~0.83 | ~0.82 |
| **XGBoost** | **~0.87** | **~0.86** |

**Top churn drivers (SHAP):**
1. Contract type month-to-month customers are highest risk
2. Complaint rate complaints per month of tenure
3. Network issues count
4. Tenure month newer customers churn more
5. is_month_to_month engineered binary flag

---

## How to Run
```bash
# 1. Clone repo
git clone https://github.com/Sonalmaral/telecom-churn-intelligence_report.git
cd telecom-churn-intelligence_report

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open notebook
jupyter notebook telecom_churn_fixed.ipynb

# 4. Run all cells top to bottom
```

---

## Tech Stack
```
Python 3.10+       pandas, numpy, matplotlib, seaborn
Machine Learning   scikit-learn, xgboost, shap
NLP                nltk, scikit-learn TF-IDF, LDA
```

---

## Author

**Sonal Maral**
4.8 years Software Test Engineer at Amdocs | Transitioning to Data Science

*"Built this project to demonstrate end-to-end DS skills from raw data to business ready insights. My QA background shaped the feature engineering: thinking in rates and ratios rather than raw counts is exactly how defect density analysis works in software testing."*

LinkedIn: https://www.linkedin.com/in/sonal-maral-1904641ab

---

