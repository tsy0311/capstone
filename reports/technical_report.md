# Technical Report

## LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable AI

**Author**: ADS  
**Date**: October 2026  

---

## 1. Executive Summary

This project implements a complete machine learning pipeline for predicting startup funding outcomes using publicly available business information. The pipeline covers data acquisition, validation, EDA, feature engineering (including LLM-derived text features), model training (5 models), evaluation (7 metrics), SHAP explainability, Streamlit dashboard, FastAPI REST API, and 25 automated tests. Responsible AI considerations including PDPA compliance, fairness, and human oversight are addressed.

## 2. Dataset

- **Source**: Kaggle (Crunchbase-derived startup data)
- **Target**: Binary (acquired/IPO = success, closed = failure)
- **Features**: Funding total, rounds, founding dates, location, market, description

## 3. Methodology

CRISP-DM framework. Preprocessing pipeline with FeatureEngineer, RiskKeywordExtractor, TextFeatureExtractor, and ColumnTransformer. Stratified train/val/test split (60/20/20). 5-fold cross-validation. Optuna hyperparameter optimization (50 trials, TPE sampler).

## 4. Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| Logistic Regression | Linear | class_weight=balanced, max_iter=1000 |
| Decision Tree | Single tree | max_depth=5, class_weight=balanced |
| Random Forest | Ensemble | n_estimators=200, max_depth=10 |
| XGBoost | Gradient boosting | Optuna-optimized |
| LightGBM | Gradient boosting | class_weight=balanced |

## 5. Feature Engineering

10 engineered features:
- Financial: funding_velocity, rounds_per_year, avg_round_size
- Temporal: company_age, time_to_first_funding, funding_span
- Text: description_length, description_readability
- Risk: high_risk_score, medium_risk_score, low_risk_score

Optional: 384-dim sentence-transformer embeddings (all-MiniLM-L6-v2)

## 6. Evaluation

7 metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score.
Cost-sensitive threshold optimization: minimise FP×$50K + FN×$5M.
Calibration analysis for probability reliability.

*Results populated upon execution of the training pipeline.*

## 7. Explainability

SHAP TreeExplainer for tree-based models. Global summary plot + local waterfall plot. Permutation importance as cross-check.

## 8. Application

- **Streamlit**: Single + batch prediction, risk categorization, SHAP explanation
- **FastAPI**: 5 endpoints (health, model info, predict, batch predict, risk categories)
- **Tests**: 25 tests (12 preprocessing + 13 prediction)

## 9. Responsible AI

- PDPA 2010 compliance (company-level data, no PII)
- Fairness: subgroup analysis by country, market, founding decade
- Human oversight: decision-support, not automated decisions
- Data drift: PSI monitoring, quarterly retraining

## 10. Limitations

1. **Survivorship bias**: Dataset over-represents funded startups
2. **No causal claims**: Observational data, correlations only
3. **No intervention outcomes**: Business impact is hypothetical
4. **Geographical bias**: US-based companies over-represented
5. **No actual results**: Pipeline not yet executed against dataset

## 11. Conclusion

This project demonstrates a complete, reproducible, end-to-end machine learning solution for startup risk profiling. The pipeline covers every stage from data acquisition to deployment, with a clear distinction between measured findings and hypothetical scenarios.

---

*This technical report is a living document. Results sections are populated upon execution of the training pipeline.*
