# Capstone Presentation

## LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable Machine Learning

**Author**: ADS  
**Programme**: Master of Computer Science  
**Date**: October 2026  

---

## Slide 1: Title and Introduction

**LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable Machine Learning**

Author: ADS  
Programme: Master of Computer Science  
Supervisor: [Supervisor Name]  
Date: October 2026  

---

## Slide 2: Business Problem and Motivation

- **Problem**: Investors evaluate startups under fundamental information asymmetry
- **Failure rate**: ~70% of startups fail (CB Insights, 2021)
- **Current methods**: Subjective, non-scalable, bias-prone, opaque
- **Opportunity**: Public business data + ML + LLMs → data-driven risk profiling

**Key distinction**: Predicting ≠ Causing. The model identifies at-risk startups; proving interventions help requires A/B testing.

---

## Slide 3: Research Question and Objectives

**RQ1**: Can ML predict funding outcomes from public startup data?  
**RQ2**: Which features matter most — financial, temporal, or LLM-derived text?  
**RQ3**: Do LLM features improve prediction over financial features alone?  
**RQ4**: Can SHAP provide meaningful interpretations?  
**RQ5**: What are the ethical and fairness implications?

**Objectives**: 10 objectives covering data, ML, explainability, deployment, responsible AI

---

## Slide 4: Dataset and Data Acquisition

- **Source**: Kaggle (Crunchbase-derived startup data)
- **Features**: Funding total, rounds, founding dates, location, market, description
- **Target**: Binary (acquired/IPO = success, closed = failure)
- **No PII**: Company-level public data only
- **Known biases**: Survivorship bias, US over-representation, temporal drift

---

## Slide 5: Data Quality and EDA Findings

- **Missing values**: funding_total_usd stored as string with placeholders
- **Date issues**: Multiple formats, some approximate
- **Class imbalance**: Failures typically outnumber successes
- **Key patterns** (expected):
  - Successful startups → more funding rounds, higher total funding
  - US-based startups dominate dataset
  - Market category matters (biotech ≠ software ≠ hardware)
  - Company age correlates with survival

---

## Slide 6: Feature Engineering and Methodology

**10 engineered features**:

| Category | Features |
|----------|----------|
| Financial | funding_velocity, rounds_per_year, avg_round_size |
| Temporal | company_age, time_to_first_funding, funding_span |
| Text | description_length, description_readability |
| LLM/Risk | high_risk_score, medium_risk_score, low_risk_score |

**LLM component**: Optional sentence-transformer embeddings (MiniLM-L6-v2, 384 dims)

**Pipeline**: Leakage-free scikit-learn Pipeline with ColumnTransformer

---

## Slide 7: Machine Learning Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| Logistic Regression | Linear (baseline) | class_weight=balanced |
| Decision Tree | Single tree | max_depth=5 |
| Random Forest | Ensemble | 200 trees |
| XGBoost | Gradient boosting | Optuna-optimized (50 trials) |
| LightGBM | Gradient boosting | Leaf-wise, balanced |

---

## Slide 8: Model Comparison and Evaluation

**7 metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score  
**5-fold stratified CV**: Robust performance estimates  
**Cost-sensitive threshold**: Minimise FP×$50K + FN×$5M  
**Selection**: Primary=ROC-AUC, Secondary=Recall, Tertiary=Calibration  

*(Results table populated upon execution)*

---

## Slide 9: Explainability and Key Findings

- **SHAP**: Global summary plot + local waterfall explanations
- **Key features** (expected):
  - company_age → positive (maturity = proven market fit)
  - funding_velocity → positive (efficient fundraising)
  - high_risk_score → negative (risk keywords in description)
  - avg_round_size → positive (investor conviction)

**Important**: SHAP shows *association*, not *causation*.

---

## Slide 10: Application Demonstration

- **Streamlit Dashboard**: Single + batch prediction, risk categorization, SHAP explanations
- **FastAPI REST API**: 5 endpoints (health, model info, predict, batch predict, risk categories)
- **Risk categories**: Low (0-0.3), Medium (0.3-0.6), High (0.6-0.8), Very High (0.8-1.0)
- **Tests**: 25 automated tests (12 preprocessing + 13 prediction)

---

## Slide 11: Business Implications

*Hypothetical scenario — not measured outcome.*

- Investor evaluates 100 startups/year
- Model flags ~30 as "likely success"
- Of those, ~20 actually succeed (precision ≈ 67%)
- Misses ~5 successful startups (recall ≈ 80%)
- **Cost with model**: $25.5M vs. **random selection**: much higher miss rate

**Recommendation**: Use as initial screening; supplement with human due diligence

---

## Slide 12: Responsible AI, Privacy, and Limitations

- **PDPA Malaysia**: No personal data — company-level only. DPIA recommended for production
- **Fairness**: Subgroup analysis (country, market, decade). Geographic bias flagged
- **Human oversight**: Decision-support only — no automated decisions
- **Data drift**: PSI monitoring, quarterly retraining
- **Key limitations**: Survivorship bias, observational data, no causal claims, no intervention outcomes

---

## Slide 13: Deployment and Monitoring

**Deployment**:
- Streamlit dashboard (port 8501)
- FastAPI REST API (port 8000)
- CPU-compatible (no GPU required)

**Monitoring**:
- Feature drift: PSI > 0.2 triggers alert
- Performance: Quarterly evaluation on new labelled data
- Retraining: Triggered by drift or performance degradation
- Model registry: Version with metadata

---

## Slide 14: Conclusion and Recommendations

**Contributions**:
1. LLM-derived text features integrated with financial features (novel)
2. SHAP explainability for startup risk profiling
3. Deployable application (Streamlit + FastAPI)
4. Responsible AI: PDPA, fairness, human oversight

**Recommendations**:
- Use as decision-support, not automated decision-maker
- Supplement with founder/team data in future
- Design A/B tests to measure actual business impact
- Fine-tune domain-specific LLM for startup text

---

## Slide 15: Questions and Answers

Thank you.  
Questions welcome.

---

## Defence Questions (15)

**Q1: Why this dataset?**  
Kaggle Crunchbase-derived, company-level (no PII), binary target, mix of financial + temporal + text features, realistic industry scenario.

**Q2: How did you prevent data leakage?**  
Pipelines fit on train only. No target-derived features. All features available at prediction time.

**Q3: Why those ten features?**  
Each captures a known business pattern: funding efficiency, maturity, investor conviction, text quality, risk language. All prediction-time information.

**Q4: Why class_weight="balanced"?**  
Imbalanced outcomes. Weighting adjusts loss function to penalise minority class errors.

**Q5: Why is accuracy misleading?**  
If 70% fail, "always predict failure" = 70% accuracy but useless. ROC-AUC and PR-AUC better.

**Q6: How was the final model selected?**  
ROC-AUC primary, recall secondary, calibration tertiary.

**Q7: What is the optimal threshold?**  
Cost-sensitive: minimise FP×$50K + FN×$5M. Typically below 0.5, prioritising recall.

**Q8: How does SHAP work?**  
Shapley values from game theory. Attributes prediction-baseline difference to each feature.

**Q9: Can we say high funding causes success?**  
No. SHAP shows association. Causation requires A/B testing.

**Q10: Why not deep learning?**  
Small dataset, tabular features, tree-based models outperform DL (Grinsztajn et al., 2022).

**Q11: How do LLM features work?**  
Rule-based risk keywords from descriptions + optional sentence-transformer embeddings (384 dims).

**Q12: How do you ensure fairness?**  
Subgroup analysis (country, market, decade). Proxy discrimination monitoring. Human oversight.

**Q13: What is PDPA?**  
Malaysia's Personal Data Protection Act 2010. This project uses company-level data only — no PII.

**Q14: How do you handle data drift?**  
PSI monitoring. Alert if PSI > 0.2. Quarterly evaluation. Retrain on drift.

**Q15: What would you do differently?**  
Collect founder data, survival analysis, A/B testing, investor network features, domain-specific LLM fine-tuning.
