# LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable AI

## An End-to-End Machine Learning Solution

### Project Overview

This project implements a complete data science pipeline for predicting startup funding outcomes using public business information, financial metrics, and LLM-derived text features. It covers data acquisition, validation, EDA, feature engineering, model training, evaluation, explainability, deployment, and responsible AI assessment.

### Deliverables

| File | Description |
|------|-------------|
| `README.md` | Project overview and quick start guide |
| `docs/stage1_documentation.md` | Stage 1: Introduction, Literature Review, Methodology |
| `docs/stage2_full_dissertation.md` | Stage 2: Full 14-chapter dissertation + appendices |
| `presentation/capstone_presentation.md` | 15-slide presentation + 15 defence Q&A |

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
