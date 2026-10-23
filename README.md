# LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable AI

## An End-to-End Machine Learning Solution

### Project Overview

This project implements a complete data science pipeline for predicting startup funding outcomes using public business information, financial metrics, and LLM-derived text features. It covers data acquisition, validation, EDA, feature engineering, model training, evaluation, explainability, deployment, and responsible AI assessment.

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd startup-risk-profiling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Go to: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
# Place startup_data.csv in data/

# 4. Run the pipeline (notebooks or scripts)
# Option A: Run notebooks in order
jupyter notebook notebooks/

# Option B: Run training script
python -m src.train

# 5. Launch the application
# Streamlit dashboard
streamlit run app/streamlit_app.py

# FastAPI service
uvicorn api.main:app --reload --port 8000

# 6. Run tests
pytest tests/ -v
```

### Project Structure

```
startup-risk-profiling/
├── config/
│   └── config.yaml              # Project configuration
├── data/
│   ├── README.md                # Dataset documentation
│   └── startup_data.csv         # Dataset (download separately)
├── notebooks/
│   ├── 01_data_acquisition_and_validation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_model_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and target encoding
│   ├── validation.py            # Data validation checks
│   ├── preprocessing.py         # Feature engineering, LLM embeddings, pipelines
│   ├── train.py                 # Model training and optimization
│   ├── evaluate.py              # Evaluation metrics and plots
│   └── predict.py              # Inference and SHAP explanations
├── models/
│   └── README.md                # Model artifacts info
├── app/
│   └── streamlit_app.py         # Interactive dashboard
├── api/
│   └── main.py                  # FastAPI REST API
├── tests/
│   ├── test_preprocessing.py    # 12 preprocessing tests
│   └── test_prediction.py       # 13 prediction tests
├── reports/
│   ├── figures/                 # Generated plots
│   ├── model_comparison.md      # Model comparison report
│   ├── technical_report.md      # Complete technical report
│   └── responsible_ai_assessment.md
├── presentation/
│   └── capstone_presentation.md # 15 slides + 15 defence Q&A
├── docs/
│   ├── stage1_documentation.md  # Introduction, Lit Review, Methodology
│   └── stage2_full_dissertation.md
├── README.md
├── requirements.txt
└── .gitignore
```

### Dataset

**Source**: Kaggle - Startup Success Prediction
**URL**: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

- Company-level public business information
- Funding rounds, total funding, founding dates
- Startup descriptions, categories, markets
- Status: acquired, IPO, closed, operating
- Binary target: acquired/IPO = success (1), closed = failure (0)

### Models

| Model | Type | Key Features |
|-------|------|--------------|
| Logistic Regression | Linear (baseline) | Interpretable, class_weight=balanced |
| Decision Tree | Single tree | max_depth=5 |
| Random Forest | Ensemble | 200 trees, class_weight=balanced |
| XGBoost | Gradient boosting | Optuna-optimized hyperparameters |
| LightGBM | Gradient boosting | Leaf-wise growth, balanced |

### Feature Engineering

**Financial**: funding_velocity, rounds_per_year, avg_round_size
**Temporal**: company_age, time_to_first_funding, funding_span
**Text**: description_length, description_readability
**Risk Keywords**: high_risk_score, medium_risk_score, low_risk_score
**LLM**: Optional sentence-transformer embeddings (MiniLM-L6-v2)

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC-AUC, PR-AUC, Brier Score
- Cost-sensitive threshold optimization
- Calibration curves

### Responsible AI

- PDPA 2010 (Malaysia) compliance assessment
- Fairness: subgroup analysis by country, market
- Explainability: SHAP global + local explanations
- Human oversight: predictions are decision-support, not automated decisions
- Data drift monitoring strategy

### Key Design Decisions

- **Prediction ≠ Intervention**: Model identifies risk, doesn't prove retention actions work
- **LLM for feature extraction**: Startup descriptions → risk keywords + embeddings
- **Leakage-free**: Pipelines fit on train only, no target-derived features
- **Cost-aware threshold**: Minimizes FP×$50K + FN×$5M

### Author

**ADS** | October 2026
