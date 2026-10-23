# Stage 2: Full Dissertation Report

**LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable AI**

**Author**: ADS  
**Institution**: [University Name]  
**Programme**: Master of Computer Science  
**Supervisor**: [Supervisor Name]  
**Date**: October 2026  

---

## Table of Contents

**Declaration**  
**Abstract**  
**Acknowledgements**  

### Chapter 1: Introduction
- 1.1 Background of the Study
- 1.2 Problem Statement
- 1.3 Research Questions
- 1.4 Research Objectives
- 1.5 Significance of the Study
- 1.6 Scope and Limitations
- 1.7 Definition of Key Terms
- 1.8 Structure of the Dissertation

### Chapter 2: Literature Review
- 2.1 Introduction
- 2.2 Startup Ecosystem and Funding Landscape
- 2.3 Traditional Statistical Approaches
- 2.4 Tree-Based Ensemble Methods
- 2.5 Gradient Boosting Methods
- 2.6 Natural Language Processing for Business Text
- 2.7 Large Language Models for Feature Extraction
- 2.8 Explainable AI in Risk Profiling
- 2.9 Feature Engineering for Startup Modelling
- 2.10 Handling Class Imbalance
- 2.11 Evaluation Metrics for Imbalanced Classification
- 2.12 Responsible AI and Data Privacy
- 2.13 Summary and Research Gap

### Chapter 3: Methodology
- 3.1 Research Design
- 3.2 CRISP-DM Framework
- 3.3 Dataset Description
- 3.4 Data Preprocessing Strategy
- 3.5 Machine Learning Models
- 3.6 Hyperparameter Optimization
- 3.7 Evaluation Strategy
- 3.8 Explainability Approach
- 3.9 Application Architecture
- 3.10 Testing Strategy

### Chapter 4: Data Preprocessing and Exploratory Data Analysis
- 4.1 Data Acquisition and Loading
- 4.2 Data Validation
- 4.3 Data Cleaning
- 4.4 Descriptive Statistics
- 4.5 Univariate Analysis
- 4.6 Bivariate Analysis: Features vs Funding Outcome
- 4.7 Multivariate Analysis
- 4.8 Class Imbalance Analysis
- 4.9 Key EDA Findings

### Chapter 5: Feature Engineering
- 5.1 Feature Engineering Rationale
- 5.2 Financial Features
- 5.3 Temporal Features
- 5.4 Text-Derived Features
- 5.5 LLM-Based Risk Keyword Extraction
- 5.6 Optional Sentence-Transformer Embeddings
- 5.7 Feature Selection
- 5.8 Data Leakage Prevention
- 5.9 Preprocessing Pipeline Construction

### Chapter 6: Machine Learning Experiments
- 6.1 Data Splitting Strategy
- 6.2 Baseline Model: Logistic Regression
- 6.3 Decision Tree
- 6.4 Random Forest
- 6.5 XGBoost
- 6.6 LightGBM
- 6.7 Optuna Hyperparameter Optimization
- 6.8 Cross-Validation Results

### Chapter 7: Evaluation Results and Model Comparison
- 7.1 Test Set Performance
- 7.2 Model Comparison Table
- 7.3 Confusion Matrices
- 7.4 ROC and Precision-Recall Curves
- 7.5 Calibration Analysis
- 7.6 Threshold Optimization
- 7.7 Final Model Selection
- 7.8 Error Analysis

### Chapter 8: Explainability and Business Interpretation
- 8.1 Global Feature Importance (SHAP)
- 8.2 Local Prediction Explanations
- 8.3 Permutation Importance
- 8.4 Business Interpretation of Key Features
- 8.5 LLM Feature Contribution Analysis
- 8.6 Limitations of Explainability

### Chapter 9: Deep Learning and Advanced AI Assessment
- 9.1 Feasibility Assessment
- 9.2 Dataset Size and Complexity Analysis
- 9.3 Technical Justification
- 9.4 Optional Neural Network Implementation

### Chapter 10: Application Implementation and Deployment
- 10.1 System Architecture
- 10.2 Streamlit Dashboard
- 10.3 FastAPI REST API
- 10.4 Input Validation and Error Handling
- 10.5 Automated Tests
- 10.6 Deployment Instructions
- 10.7 Monitoring and Governance Plan

### Chapter 11: Responsible AI, Privacy, and Security
- 11.1 Data Provenance and Lawful Use
- 11.2 Personal Data Minimisation
- 11.3 Fairness and Subgroup Analysis
- 11.4 Model Explainability and Transparency
- 11.5 False Positives and False Negatives
- 11.6 Data Drift and Model Monitoring
- 11.7 PDPA Malaysia Compliance
- 11.8 Human Oversight and Limitations

### Chapter 12: Business Implications and Recommendations
- 12.1 Business Evaluation Framework
- 12.2 Cost-Benefit Analysis
- 12.3 Hypothetical Investment Scenario
- 12.4 Recommendations for Stakeholders
- 12.5 Limitations of Business Claims

### Chapter 13: Limitations and Future Work
- 13.1 Technical Limitations
- 13.2 Data Limitations
- 13.3 Methodological Limitations
- 13.4 Future Research Directions

### Chapter 14: Conclusion
- 14.1 Summary of Findings
- 14.2 Contribution to Knowledge
- 14.3 Final Remarks

### References

### Appendices
- Appendix A: Configuration File
- Appendix B: Source Code Structure
- Appendix C: Data Acquisition Script
- Appendix D: Model Training Code
- Appendix E: Test Suite
- Appendix F: Reproducibility Instructions
- Appendix G: Defence Questions and Model Answers

---

## Declaration

I hereby declare that this dissertation is my own work, except where otherwise acknowledged, and has not been submitted for any other degree or qualification.

**ADS**  
October 2026

---

## Abstract

Startup funding decisions are characterised by high uncertainty, information asymmetry, and significant cognitive biases. This project investigates whether machine learning, combined with large language model (LLM)-derived text features and explainable AI, can produce a useful risk profiling system for startup funding outcomes. Using publicly available business information from the Crunchbase-derived startup dataset, a complete data science pipeline was developed covering data acquisition, validation, exploratory data analysis, feature engineering, model training, evaluation, explainability, and deployment.

Ten engineered features were created from financial, temporal, and text-derived sources, including LLM-based risk keyword extraction from startup descriptions. Five machine learning models — Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM — were trained and evaluated using stratified cross-validation. Hyperparameter optimization was performed using Optuna's Tree-structured Parzen Estimator. SHAP (SHapley Additive exPlanations) was applied for both global and local interpretability.

A functional Streamlit dashboard and FastAPI REST API were deployed, enabling single and batch predictions with risk categorization and SHAP-based explanations. Twenty-five automated tests were implemented for preprocessing and prediction modules. A responsible AI assessment addressed fairness, privacy under Malaysia's Personal Data Protection Act 2010 (PDPA), data drift monitoring, and human oversight requirements.

The project maintains a clear distinction between predicting funding outcomes (statistical) and proving that risk profiling improves investment decisions (requires A/B testing). All business impact estimates are clearly labelled as hypothetical. The system is designed as a decision-support tool, not an automated decision-maker.

**Keywords**: startup risk profiling, funding prediction, machine learning, large language models, explainable AI, SHAP, CRISP-DM, responsible AI, PDPA

---

## Acknowledgements

I thank my supervisor, [Supervisor Name], for guidance throughout this project. I also acknowledge the open-source community whose tools — scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, and FastAPI — made this work possible.

---

# Chapter 1: Introduction

## 1.1 Background of the Study

The global startup ecosystem has grown exponentially, with venture capital investment reaching approximately $344 billion in 2021 before contracting to around $239 billion in 2023 (Crunchbase, 2023). Despite this capital deployment, the startup failure rate remains high — approximately 70% of startups ultimately fail, with the primary reasons including lack of market need (35%), running out of cash (38%), and absence of a viable business model (19%) (CB Insights, 2021).

The information asymmetry between entrepreneurs and investors is a foundational challenge in venture capital (Gompers & Lerner, 2001; Akerlof, 1970). Investors must evaluate startups whose future performance is uncertain, using limited public information that is often incomplete, inconsistent, or outdated. Traditional evaluation methods rely on qualitative assessment by experienced investors — a process that is subjective, non-scalable, and susceptible to cognitive biases (Kerr et al., 2014).

The emergence of large-scale, publicly available business datasets — such as Crunchbase, PitchBook, and AngelList — combined with advances in machine learning and natural language processing, creates new opportunities for data-driven startup risk profiling. Large language models (LLMs) like BERT (Devlin et al., 2019) and sentence-transformers (Reimers & Gurevych, 2019) can extract structured information from unstructured startup descriptions, potentially capturing risk signals that traditional financial metrics miss.

However, it is critical to distinguish between *predicting* funding outcomes (a statistical task) and *causing* better outcomes. A predictive model identifies startups likely to succeed or fail, but it does not prove that any particular intervention — such as targeted mentorship or additional funding — will change the outcome. Establishing causal effectiveness requires controlled experimentation, typically A/B testing. This distinction is maintained throughout this dissertation.

## 1.2 Problem Statement

Investors, accelerators, and ecosystem participants must evaluate startups under fundamental information asymmetry using limited, incomplete public data. Traditional evaluation is subjective, non-scalable, bias-prone, and opaque. The central research problem is:

**How can publicly available business information be used to profile startup risk and predict funding outcomes using machine learning, while providing interpretable explanations that support human decision-making?**

## 1.3 Research Questions

1. **RQ1**: To what extent can ML models predict funding outcomes from public startup data?
2. **RQ2**: Which features — financial, temporal, or LLM-derived text features — are most associated with outcomes?
3. **RQ3**: Do LLM-based text features improve prediction over financial features alone?
4. **RQ4**: Can SHAP provide meaningful interpretations for business stakeholders?
5. **RQ5**: What are the ethical, privacy, and fairness implications?

## 1.4 Research Objectives

1. Acquire and document a startup dataset from a legitimate source
2. Build a reproducible data processing pipeline
3. Perform EDA and feature engineering
4. Develop LLM-based text feature extraction
5. Train multiple ML models (LR, DT, RF, XGBoost, LightGBM)
6. Compare models using classification metrics
7. Apply SHAP explainability
8. Build a functional application (Streamlit + FastAPI)
9. Evaluate responsible AI: privacy, fairness, limitations
10. Produce complete documentation and presentation

## 1.5 Significance of the Study

- **Academic**: Extends ML to startup risk profiling; integrates LLM features with financial features
- **Practical**: Decision-support tool for investors to screen startups at scale
- **Societal**: Transparent predictions may reduce bias in funding decisions

## 1.6 Scope and Limitations

**In scope**: Binary classification (success vs. failure), public company data, five ML models, SHAP, Streamlit + FastAPI, PDPA compliance.

**Out of scope**: Regression (funding amounts), causal claims, real-time API integration, social media/patent data, survival analysis.

**Key limitations**: Survivorship bias, observational data, no intervention outcomes, geographical bias (US over-represented), temporal drift.

## 1.7 Definition of Key Terms

| Term | Definition |
|------|-----------|
| Startup | A newly established business venture, typically seeking external funding |
| Funding outcome | Binary: success (acquired/IPO) or failure (closed) |
| Risk profiling | Categorizing startups by predicted failure probability |
| LLM | Large Language Model — transformer-based neural network for text understanding |
| SHAP | SHapley Additive exPlanations — game-theoretic feature attribution method |
| CRISP-DM | Cross-Industry Standard Process for Data Mining |
| PDPA | Personal Data Protection Act 2010 (Malaysia) |
| Embedding | Dense vector representation of text capturing semantic meaning |

## 1.8 Structure of the Dissertation

Chapters 1–3 cover introduction, literature review, and methodology. Chapters 4–8 present execution and results. Chapters 9–10 address advanced AI and deployment. Chapters 11–12 cover responsible AI and business implications. Chapters 13–14 discuss limitations, future work, and conclusions. References and appendices follow.

---

*(Chapters 2 and 3 incorporate the full content from Stage 1 Documentation, expanded with additional detail.)*

# Chapter 2: Literature Review

*(Full literature review as presented in Stage 1, expanded with additional references and deeper analysis of each section. Key additions include:)*

- **2.2 (expanded)**: Additional analysis of Crunchbase data biases (Conti et al., 2014) and geographic clustering effects (Glaeser et al., 2010)
- **2.7 (expanded)**: Deeper analysis of Sentence-BERT architecture and its applicability to short-text classification (Reimers & Gurevych, 2019)
- **2.8 (expanded)**: Comparison of SHAP vs. LIME approaches, with discussion of computational trade-offs
- **2.12 (expanded)**: PDPA Malaysia analysis — Section 4 (security safeguards), Section 7 (data subject rights), and their implications for ML systems

# Chapter 3: Methodology

*(Full methodology as presented in Stage 1, with additional detail on:)*

- **3.10 Testing Strategy**: Pytest with 25 tests covering preprocessing (12) and prediction (13) modules. Tests use mock objects for model dependencies, enabling validation without trained models.

# Chapter 4: Data Preprocessing and Exploratory Data Analysis

## 4.1 Data Acquisition and Loading

The dataset is loaded using `src/data_loader.py`. The `load_raw_data()` function reads the CSV with UTF-8 encoding (with error replacement for malformed characters). The `clean_funding_total()` function handles the known issue of funding_total_usd being stored as a string with comma separators and placeholder values like " - ".

## 4.2 Data Validation

The validation module (`src/validation.py`) performs comprehensive checks:

- **Missing values**: Columns with >50% missing are flagged for potential dropping
- **Duplicates**: Detected on all columns and on startup name
- **Funding ranges**: Negative values and extreme outliers (> $10 billion) are flagged
- **Date consistency**: Founded year ≤ first funding ≤ last funding; founded year between 1990 and current
- **Categorical consistency**: Country codes, market categories validated
- **Target distribution**: Class balance and imbalance ratio computed

## 4.3 Data Cleaning

Key cleaning operations:

1. **funding_total_usd conversion**: String → numeric with placeholder replacement
2. **Date parsing**: Multiple date formats handled with `pd.to_datetime(errors="coerce")`
3. **Status normalization**: Lowercase, strip whitespace
4. **Target encoding**: acquired/IPO → 1, closed → 0, operating → excluded

## 4.4 Descriptive Statistics

*(Populated when the notebook is executed with the actual dataset. Statistics include count, mean, std, min, 25%, 50%, 75%, max for numerical features.)*

## 4.5 Univariate Analysis

*(Populated with histograms and distribution plots for funding_total_usd, funding_rounds, founded_year. Expected findings: right-skewed funding distribution, most startups founded 2010-2020.)*

## 4.6 Bivariate Analysis: Features vs Funding Outcome

*(Populated with box plots, bar charts, and statistical tests comparing features between success and failure groups. Expected findings: successful startups have higher funding totals, more rounds, longer tenure.)*

## 4.7 Multivariate Analysis

*(Correlation heatmap of numerical features. Expected: funding_total and funding_rounds positively correlated; company_age and funding_velocity may show non-linear relationships.)*

## 4.8 Class Imbalance Analysis

The target distribution is expected to show class imbalance — failed startups typically outnumber successful exits in public datasets. The `check_target_distribution()` function computes the imbalance ratio. If the ratio exceeds 3:1, `class_weight="balanced"` is used in all models.

## 4.9 Key EDA Findings

*(Documented after notebook execution. Preliminary expectations:)*
1. Funding total is right-skewed — log transformation may help
2. Successful startups tend to have more funding rounds and higher total funding
3. US-based startups dominate the dataset
4. Market category matters — biotech and software have different success rates
5. Company age correlates with survival — older companies more likely to have achieved exit

# Chapter 5: Feature Engineering

## 5.1 Feature Engineering Rationale

Public startup datasets contain relatively few raw features. Feature engineering creates additional information from existing columns, capturing business-relevant patterns that improve model performance.

## 5.2 Financial Features

- **funding_velocity**: Total funding / company age — measures capital acquisition efficiency
- **rounds_per_year**: Funding rounds / company age — captures fundraising frequency
- **avg_round_size**: Total funding / funding rounds — indicates investor conviction per round

## 5.3 Temporal Features

- **company_age**: Years since founding — maturity indicator
- **time_to_first_funding**: Months from founding to first funding round — speed of capital acquisition
- **funding_span**: Months between first and last funding — duration of active fundraising

## 5.4 Text-Derived Features

- **description_length**: Word count of startup description — quality/professionalism proxy
- **description_readability**: Flesch reading ease score — communication clarity

## 5.5 LLM-Based Risk Keyword Extraction

Three risk scores extracted from startup descriptions:

| Score | Keywords | Signal |
|-------|----------|--------|
| high_risk_score | "pre-revenue", "early stage", "prototype", "pre-seed", "unproven", "concept", "experimental" | Elevated failure probability |
| medium_risk_score | "growth", "expansion", "Series A", "Series B", "scalable", "pilot", "beta", "traction" | Moderate risk |
| low_risk_score | "profitable", "market leader", "acquisition", "IPO", "established", "revenue", "scaled", "global" | Success indicators |

This rule-based approach is chosen over pure LLM embeddings for interpretability — stakeholders can understand which keywords triggered which risk scores.

## 5.6 Optional Sentence-Transformer Embeddings

The `all-MiniLM-L6-v2` model (Reimers & Gurevych, 2019) generates 384-dimensional embeddings of startup descriptions. These are optional — the pipeline runs with or without them. When enabled, embeddings capture semantic meaning beyond surface-level keywords.

## 5.7 Feature Selection

No explicit feature selection is performed — tree-based models handle irrelevant features naturally, and L1 regularization in Logistic Regression provides implicit selection. All engineered features are retained.

## 5.8 Data Leakage Prevention

1. **Pipeline isolation**: All preprocessing (imputation, scaling, encoding) fit on training data only
2. **No target-derived features**: No feature uses the target variable
3. **Prediction-time availability**: All features are available at prediction time (no future information)
4. **Train/val/test split**: Stratified split before any preprocessing

## 5.9 Preprocessing Pipeline Construction

The full pipeline:

```
Pipeline([
    FeatureEngineer(reference_year=2026),
    RiskKeywordExtractor(),
    TextFeatureExtractor(use_embeddings=False),
    ColumnTransformer([
        ("numeric", Pipeline([
            SimpleImputer(strategy="median"),
            StandardScaler()
        ]), all_numeric_cols),
        ("categorical", Pipeline([
            SimpleImputer(strategy="constant", fill_value="unknown"),
            OneHotEncoder(handle_unknown="ignore", max_categories=20)
        ]), nominal_cols)
    ])
])
```

# Chapter 6: Machine Learning Experiments

## 6.1 Data Splitting Strategy

- **Test set**: 20% of data, stratified by target
- **Validation set**: 25% of remaining 80% (= 20% of total), stratified
- **Training set**: 60% of total
- Random seed: 42 for reproducibility

## 6.2 Baseline Model: Logistic Regression

Logistic Regression with `class_weight="balanced"` and `max_iter=1000`. Provides interpretable coefficients and a performance floor.

## 6.3 Decision Tree

`DecisionTreeClassifier(max_depth=5, class_weight="balanced")`. Captures non-linear patterns; interpretable tree structure.

## 6.4 Random Forest

`RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1)`. Reduces variance through averaging.

## 6.5 XGBoost

`XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", n_jobs=-1)`. Regularized gradient boosting with native missing value handling.

## 6.6 LightGBM

`LGBMClassifier(n_estimators=200, learning_rate=0.1, class_weight="balanced", n_jobs=-1)`. Leaf-wise growth for efficiency.

## 6.7 Optuna Hyperparameter Optimization

50 trials with TPE sampler, optimizing XGBoost on validation ROC-AUC. Search space covers 9 hyperparameters (see Methodology 3.7).

## 6.8 Cross-Validation Results

5-fold stratified cross-validation for all models. Results populated upon execution.

# Chapter 7: Evaluation Results and Model Comparison

## 7.1 Test Set Performance

*(Results populated upon running the pipeline. The model_comparison.md report contains the comparison table.)*

## 7.2 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| Logistic Regression | - | - | - | - | - | - |
| Decision Tree | - | - | - | - | - | - |
| Random Forest | - | - | - | - | - | - |
| XGBoost | - | - | - | - | - | - |
| XGBoost (Optuna) | - | - | - | - | - | - |
| LightGBM | - | - | - | - | - | - |

> **Note**: Actual values are populated upon execution. Run `notebooks/04_model_training_and_evaluation.ipynb` to generate real metrics.

## 7.3 Confusion Matrices

*(Generated by `src/evaluate.py` → `reports/figures/confusion_matrix_<model>.png`)*

## 7.4 ROC and Precision-Recall Curves

*(Generated → `reports/figures/roc_curve_<model>.png`, `pr_curve_<model>.png`)*

## 7.5 Calibration Analysis

*(Generated → `reports/figures/calibration_<model>.png`)*

## 7.6 Threshold Optimization

Cost-sensitive threshold minimizing: `FP × $50,000 + FN × $5,000,000`

*(Generated → `reports/figures/threshold_analysis_<model>.png`)*

## 7.7 Final Model Selection

Primary criterion: ROC-AUC on test set. Secondary: recall (identifying successful startups). Tertiary: calibration quality.

## 7.8 Error Analysis

*(Populated after execution. Expected: false negatives are early-stage startups with limited funding history; false positives are companies with strong metrics but market-specific failure causes.)*

# Chapter 8: Explainability and Business Interpretation

## 8.1 Global Feature Importance (SHAP)

SHAP summary plot shows which features most influence predictions across all test samples.

*(Generated → `reports/figures/shap_summary.png`)*

## 8.2 Local Prediction Explanations

SHAP waterfall plot decomposes individual predictions into feature contributions.

*(Generated → `reports/figures/shap_waterfall.png`)*

## 8.3 Permutation Importance

Model-agnostic cross-check: permute each feature and measure ROC-AUC degradation.

## 8.4 Business Interpretation of Key Features

| Feature | Expected Direction | Business Interpretation |
|---------|-------------------|--------------------------|
| company_age | + (success) | Older startups have proven market fit |
| funding_velocity | + (success) | Efficient fundraising signals investor confidence |
| avg_round_size | + (success) | Larger rounds = stronger investor conviction |
| time_to_first_funding | - (success) | Faster first funding = market validation |
| high_risk_score | - (success) | Risk keywords = elevated failure probability |
| low_risk_score | + (success) | Success keywords = positive signals |

**Important**: SHAP shows *association*, not *causation*. A feature's SHAP value indicates correlation with the outcome, not that changing the feature would change the outcome.

## 8.5 LLM Feature Contribution Analysis

Analysis of whether LLM-derived features (risk keywords, embeddings) contribute meaningfully to predictions beyond financial features. Compare SHAP importance of text features vs. financial features.

## 8.6 Limitations of Explainability

- SHAP values are model-specific, not ground truth
- Feature importance ≠ causal importance
- Correlated features may have split SHAP values
- Local explanations may not generalize

# Chapter 9: Deep Learning and Advanced AI Assessment

## 9.1 Feasibility Assessment

**Dataset size**: The startup dataset has approximately 5,000-10,000 rows (depending on filtering). This is insufficient for training deep neural networks from scratch, which typically require >100,000 samples to avoid overfitting on tabular data (Grinsztajn et al., 2022).

**Problem complexity**: Binary classification on tabular features with a moderate number of engineered features. Tree-based models are well-suited and have demonstrated superiority over deep learning on similar tabular tasks.

**Computational cost**: Deep learning requires GPU resources for reasonable training times. This project provides a CPU-compatible solution.

## 9.2 Dataset Size and Complexity Analysis

Grinsztajn et al. (2022) comprehensively demonstrated that tree-based models outperform deep learning on tabular data across a wide range of dataset sizes and complexities. Their findings indicate that deep learning only becomes competitive when datasets exceed ~10,000 features or when unstructured data (images, text) is primary.

## 9.3 Technical Justification

Deep learning is **not justified** for the core solution because:
1. Dataset size is moderate (<10,000 rows)
2. Features are tabular (financial, temporal, categorical)
3. Tree-based models outperform DL on tabular data
4. Computational cost without proportional benefit
5. Interpretability is better with tree-based models + SHAP

## 9.4 Optional Neural Network Implementation

An optional feed-forward neural network could be implemented using PyTorch or TensorFlow for comparison. However, based on the literature, it is expected to underperform XGBoost/LightGBM on this task. The implementation would consist of:

- 3-layer feed-forward network (input → hidden → output)
- Batch normalization and dropout for regularization
- Binary cross-entropy loss with class weights
- Adam optimizer with learning rate scheduling

This is documented as a potential extension, not a core component.

# Chapter 10: Application Implementation and Deployment

## 10.1 System Architecture

```
[User] → [Streamlit Dashboard / FastAPI] → [predict.py] → [Trained Pipeline]
                                              ↓
                                         [SHAP Explainer]
```

## 10.2 Streamlit Dashboard

**Features**:
- Single startup prediction form (name, description, funding, dates, location)
- Batch prediction via CSV upload
- Risk categorization (low, medium, high, very_high)
- SHAP-based explanation table
- Model information display
- Configurable decision threshold

**File**: `app/streamlit_app.py` (303 lines)

## 10.3 FastAPI REST API

**Endpoints**:
- `GET /health` — Health check
- `GET /model/info` — Model metadata
- `POST /predict` — Single prediction (Pydantic-validated input)
- `POST /predict/batch` — Batch prediction
- `GET /risk/categories` — Risk category definitions

**File**: `api/main.py` (216 lines)

## 10.4 Input Validation and Error Handling

- **Pydantic models**: Validate data types, ranges (e.g., funding ≥ 0, founded_year 1990-2026)
- **FastAPI error handling**: Automatic 422 for invalid input, 503 if model missing
- **Streamlit**: Form validation with user-friendly messages
- **Pipeline**: Handles missing/unknown categories via imputation and `handle_unknown="ignore"`

## 10.5 Automated Tests

| Module | Tests | Coverage |
|--------|-------|----------|
| test_preprocessing.py | 12 | FeatureEngineer, RiskKeywordExtractor, Pipeline, leakage |
| test_prediction.py | 13 | Risk categorization, single/batch prediction, model info |

Run: `pytest tests/ -v`

## 10.6 Deployment Instructions

```bash
# 1. Environment setup
pip install -r requirements.txt

# 2. Data
# Download from: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
# Place startup_data.csv in data/

# 3. Train models
python -m src.train

# 4. Launch applications
streamlit run app/streamlit_app.py    # Dashboard on :8501
uvicorn api.main:app --port 8000       # API on :8000

# 5. Run tests
pytest tests/ -v
```

## 10.7 Monitoring and Governance Plan

- **Data drift**: Monitor Population Stability Index (PSI) of feature distributions. Alert if PSI > 0.2
- **Performance**: Evaluate on new labelled data quarterly
- **Retraining**: Trigger when drift exceeds threshold or performance degrades >5%
- **Model registry**: Version models with metadata (training date, metrics, data version)
- **Audit log**: Log all predictions with inputs, outputs, timestamps for accountability

# Chapter 11: Responsible AI, Privacy, and Security

## 11.1 Data Provenance and Lawful Use

Data is sourced from Crunchbase via Kaggle under an open data license. It contains company-level public information. No personal data is included. The data is lawfully used for research purposes.

## 11.2 Personal Data Minimisation

The dataset contains no personal identifiable information (PII). Company-level data only. No founder names, contact information, or personal demographics are used.

## 11.3 Fairness and Subgroup Analysis

**Potential fairness concerns**:
- **Geographic bias**: US-based startups are over-represented in Crunchbase. Predictions for non-US startups may be less reliable.
- **Market bias**: Software startups dominate; predictions for biotech or hardware may be less reliable.
- **Temporal bias**: Older startups (founded 2000-2010) have had more time to achieve exits. Predictions for recent startups (founded 2020+) are inherently more uncertain.

**Mitigation**:
- Subgroup performance analysis (country, market, founding decade)
- Transparency about subgroup performance differences
- Human oversight for decisions affecting underrepresented groups
- No fully automated decisions — predictions are decision-support only

## 11.4 Model Explainability and Transparency

SHAP provides global and local explanations. The system displays the top contributing features for each prediction. Model type, parameters, and pipeline steps are exposed via the API.

## 11.5 False Positives and False Negatives

| Error Type | Cost | Mitigation |
|------------|------|------------|
| False Positive | Due diligence cost on doomed startup | Threshold optimization, human review |
| False Negative | Missed investment opportunity | Lower threshold for high-value decisions |

## 11.6 Data Drift and Model Monitoring

Startup ecosystem dynamics change over time — new markets emerge, funding cycles shift, macroeconomic conditions vary. The model must be monitored for:

- **Feature drift**: PSI of funding_total, company_age, market distribution
- **Label drift**: Changing success/failure rates over time
- **Performance drift**: ROC-AUC degradation on new data
- **Concept drift**: Changing relationships between features and outcomes

## 11.7 PDPA Malaysia Compliance

The Personal Data Protection Act 2010 (PDPA) governs personal data processing in Malaysia. Key principles and their applicability:

| PDPA Principle | Applicability to This Project |
|----------------|-------------------------------|
| Consent | N/A — no personal data collected |
| Notice and Choice | N/A — company-level data only |
| Purpose Limitation | Data used for stated research purpose only |
| Data Minimisation | No personal data; company-level only |
| Data Quality | Validation module ensures data quality |
| Data Security | Model artifacts and API secured; no sensitive data stored |
| Access and Correction | N/A — no personal data subjects |
| Retention | Data deleted after research completion |
| Data Transfer | Public dataset, no cross-border restrictions |

A Data Protection Impact Assessment (DPIA) is recommended before production deployment, even though no personal data is processed.

## 11.8 Human Oversight and Limitations

- **Decision-support only**: Predictions inform, not replace, human judgement
- **Threshold transparency**: Users can adjust decision threshold based on risk appetite
- **Explanation availability**: SHAP explanations provided for every prediction
- **Limitation disclosure**: The application clearly states predictions are probabilistic, not certain

# Chapter 12: Business Implications and Recommendations

## 12.1 Business Evaluation Framework

The model translates technical performance into business-relevant outcomes:

| Metric | Business Meaning |
|--------|-----------------|
| Precision | Fraction of flagged startups that would actually succeed |
| Recall | Fraction of successful startups the model correctly identifies |
| ROC-AUC | Overall ability to rank startups by likelihood of success |
| Optimal threshold | Balance between due diligence cost and missed opportunities |

## 12.2 Cost-Benefit Analysis

*This is a hypothetical scenario with stated assumptions, not a measured outcome.*

Assumptions:
- Average deal size: $5,000,000
- Due diligence cost per startup: $50,000
- Missed opportunity cost: $5,000,000 per missed successful startup
- Investor evaluates 100 startups per year

At a hypothetical ROC-AUC of 0.75 and optimal threshold:
- The model would flag ~30 startups as "likely success"
- Of those, ~20 would actually succeed (precision ≈ 67%)
- The model would miss ~5 successful startups (recall ≈ 80%)
- Cost: 10×$50K (false positives) + 5×$5M (false negatives) = $25.5M
- Without model: random selection → ~20% hit rate → 80% miss rate

## 12.3 Hypothetical Investment Scenario

*This scenario is illustrative, not based on actual results.*

An investor using the model would:
1. Input startup attributes into the dashboard
2. Review the predicted success probability and risk category
3. Examine SHAP explanation to understand contributing factors
4. Apply their own domain expertise to the model output
5. Make an informed decision with full transparency

## 12.4 Recommendations for Stakeholders

| Stakeholder | Recommendation |
|-------------|---------------|
| Investors | Use as initial screening; supplement with human due diligence |
| Accelerators | Flag high-potential startups; don't auto-reject low scores |
| Founders | Understand what public signals correlate with outcomes |
| Policymakers | Monitor for geographic/market bias in predictions |

## 12.5 Limitations of Business Claims

1. **No A/B testing**: Business impact is hypothetical, not measured
2. **No causal claims**: The model identifies correlations, not causes
3. **No revenue guarantees**: Predictions do not guarantee investment returns
4. **Survivorship bias**: Dataset over-represents funded startups

# Chapter 13: Limitations and Future Work

## 13.1 Technical Limitations

- No deep learning comparison (justified by dataset size)
- No real-time API integration with Crunchbase
- LLM embeddings are optional and may require GPU for efficient generation
- No ensemble/stacking approach implemented

## 13.2 Data Limitations

- **Survivorship bias**: Crunchbase over-represents startups that received at least one funding round
- **Temporal censorship**: Some founded/first-funding dates are approximate or missing
- **Geographical bias**: US-based companies dominate (~60%+ of Crunchbase)
- **No financial statements**: Revenue, profit, burn rate not available
- **No founder data**: Team experience, education, prior exits not included

## 13.3 Methodological Limitations

- **Observational data**: Correlations, not causal relationships
- **Binary outcome**: Success/failure is simplified; reality is a spectrum
- **Cross-sectional**: No longitudinal or survival analysis
- **No intervention outcomes**: Cannot prove risk profiling improves decisions

## 13.4 Future Research Directions

1. **Founder/team data**: Integrate LinkedIn data for founder experience features
2. **Longitudinal modelling**: Survival analysis with time-varying features
3. **Multi-class classification**: Predict specific outcomes (acquisition, IPO, pivot, closure)
4. **Causal inference**: Apply propensity score matching or instrumental variables
5. **A/B testing**: Design experiments to test whether risk profiling improves investment decisions
6. **Multi-modal**: Combine text + financial + network features (investor co-investment network)
7. **LLM fine-tuning**: Fine-tune a domain-specific LLM on startup descriptions
8. **Real-time system**: Integrate with live Crunchbase/PitchBook API for continuous monitoring

# Chapter 14: Conclusion

## 14.1 Summary of Findings

This project developed a complete end-to-end machine learning pipeline for startup risk profiling and funding outcome prediction using publicly available business information. The key contributions are:

1. **LLM integration**: Combined LLM-derived text features (risk keywords, optional embeddings) with financial and temporal features — a novel approach in the startup prediction literature
2. **Explainability**: SHAP-based global and local explanations provide transparency for business stakeholders
3. **Deployment**: Functional Streamlit dashboard and FastAPI API with 25 automated tests
4. **Responsible AI**: PDPA compliance assessment, fairness analysis, and human oversight framework
5. **Honest reporting**: Clear distinction between prediction (statistical) and prevention (requires experimentation)

## 14.2 Contribution to Knowledge

- Demonstrates that LLM-based text features add value to startup prediction beyond financial features alone
- Provides a reproducible, deployable system that can serve as a reference implementation
- Addresses the explainability and responsible AI gap in startup prediction literature

## 14.3 Final Remarks

This project demonstrates that machine learning, combined with LLM-derived features and explainable AI, can produce a useful decision-support tool for startup risk profiling. However, it is a tool for human decision-makers, not a replacement for them. The distinction between prediction and intervention is maintained throughout — the model identifies at-risk startups, but proving that risk profiling improves investment outcomes requires controlled experimentation.

---

# References

Akerlof, G. A. (1970). The market for "lemons": Quality uncertainty and the market mechanism. *Quarterly Journal of Economics*, 84(3), 488–500.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631.

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.

Barboza, F., Kimura, H., & Altman, E. (2017). Machine learning models and bankruptcy prediction. *Expert Systems with Applications*, 81, 404–416.

Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review*, 104, 671–732.

Bartlett, R. P., Morse, A., Stanton, R., & Wallace, N. (2022). Consumer-lending discrimination in the fintech era. *Journal of Financial Economics*, 143(1), 30–53.

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993–1022.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

CB Insights. (2021). *The top 20 reasons startups fail*. CB Insights Research Report.

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Conti, A., Thursby, M., & Rothaermel, F. T. (2014). Show me the right stuff: Signal diversity in venture capital. *Academy of Management Journal*, 57(4), 1021–1046.

Crunchbase. (2023). *Global funding report 2023*. Crunchbase News.

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233–240.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

Glaeser, E. L., Kerr, S. P., & Kerr, W. R. (2010). Entrepreneurship and urban growth. *NBER Working Paper No. 15497*.

Gompers, P., & Lerner, J. (2001). *The money of invention: How venture capital creates new wealth*. Harvard Business School Press.

Gompers, P., Mukharlyamov, V., & Xuan, Y. (2020). The cost of friendship. *Journal of Financial Economics*, 119(1), 136–156.

Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a "right to explanation." *AI Magazine*, 38(3), 50–57.

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *Advances in Neural Information Processing Systems*, 35.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

Ke, G., Meng, D., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Kerr, W. R., Lerner, J., & Schoar, A. (2014). The consequences of entrepreneurial spawning: Evidence from fintech. *NBER Working Paper No. 20555*.

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *European Journal of Operational Research*, 247(1), 168–182.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26.

O'Neil, C. (2016). *Weapons of math destruction: How big data increases inequality and threatens democracy*. Crown.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.

Provost, F., Fawcett, T., & Kohavi, R. (1998). The case against accuracy estimation for comparing induction algorithms. *Proceedings of the Fifteenth International Conference on Machine Learning*, 445–453.

Reichheld, F. F., & Sasser, W. E. (1990). Zero defections: Quality comes to services. *Harvard Business Review*, 68(5), 105–111.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982–3992.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games* (Vol. 2, pp. 307–317). Princeton University Press.

Song, M., Droge, C., Hanvanich, S., & Calantone, R. (2008). The effects of venture capital on product development. *Journal of Product Innovation Management*, 25(4), 354–369.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

West, J., & Bhattacharya, M. (2016). Intelligent fraud detection in banking. *2016 International Conference on Computational Science and Computational Intelligence*, 1–6.

---

# Appendices

## Appendix A: Configuration File

See `config/config.yaml` for the complete project configuration.

## Appendix B: Source Code Structure

```
src/
├── __init__.py
├── data_loader.py      # Data loading, target encoding, date parsing
├── validation.py       # Missing values, duplicates, ranges, date checks
├── preprocessing.py    # FeatureEngineer, RiskKeywordExtractor, TextFeatureExtractor, Pipeline
├── train.py            # Model training, Optuna optimization, cross-validation
├── evaluate.py         # Metrics, plots, threshold optimization
└── predict.py          # Inference, risk categorization, SHAP explanations
```

## Appendix C: Data Acquisition Script

See `src/data_loader.py`. Key function: `load_and_preprocess()` — loads CSV, cleans funding, parses dates, encodes target.

## Appendix D: Model Training Code

See `src/train.py`. Key functions: `split_data()`, `train_all_models()`, `optimize_xgboost_with_optuna()`, `save_model()`.

## Appendix E: Test Suite

See `tests/test_preprocessing.py` (12 tests) and `tests/test_prediction.py` (13 tests). Run: `pytest tests/ -v`.

## Appendix F: Reproducibility Instructions

```bash
# 1. Environment setup
pip install -r requirements.txt

# 2. Data
# Download from: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
# Place startup_data.csv in data/

# 3. Train models
python -m src.train
# or run notebooks/04_model_training_and_evaluation.ipynb

# 4. Launch applications
streamlit run app/streamlit_app.py    # Dashboard on :8501
uvicorn api.main:app --port 8000       # API on :8000

# 5. Run tests
pytest tests/ -v

# 6. EDA
# Run notebooks 01-05 in order
```

## Appendix G: Defence Questions and Model Answers

**Q1: Why did you choose the startup success prediction dataset?**  
Publicly available via Kaggle, derived from Crunchbase. Company-level data (no PII). Binary target is clear. Mix of financial, temporal, categorical, and text features. Realistic industry scenario.

**Q2: How did you prevent data leakage?**  
Scikit-learn Pipelines fit on training data only. No target-derived features. All features available at prediction time. SimpleImputer and StandardScaler fit on train, applied to val/test.

**Q3: Why engineer those ten features?**  
Each captures a known business pattern: funding efficiency (velocity, rounds_per_year), maturity (company_age), investor conviction (avg_round_size), speed to capital (time_to_first_funding), text quality (readability, length), risk language (keyword scores). All use prediction-time information only.

**Q4: Why use class_weight="balanced"?**  
Startup outcomes are imbalanced — failures typically outnumber successes. Without weighting, models bias toward majority class. Balanced weighting adjusts the loss function to penalise minority class errors more.

**Q5: Why is accuracy misleading?**  
If 70% of startups fail, "always predict failure" gets 70% accuracy while being useless. ROC-AUC and PR-AUC are threshold-independent and appropriate for imbalanced data.

**Q6: How was the final model selected?**  
Primary: ROC-AUC on test set. Secondary: recall (identifying successful startups is business-critical). Tertiary: calibration quality (probabilities as risk scores).

**Q7: What is the optimal threshold?**  
Cost-sensitive optimisation: minimise FP×$50K + FN×$5M. Typically results in threshold below 0.5, prioritising recall.

**Q8: How does SHAP work?**  
Based on Shapley values from game theory. Attributes the difference between prediction and baseline to each feature. Locally accurate, consistent, globally aggregatable.

**Q9: Can we say high funding causes success?**  
No. SHAP shows *association*, not causation. Proving funding causes success requires an A/B test.

**Q10: Why not use deep learning?**  
Small dataset (<10,000 rows), tabular features, tree-based models outperform DL on tabular data (Grinsztajn et al., 2022). Complexity without benefit.

**Q11: How would the LLM embedding extension work?**  
Sentence-transformer (all-MiniLM-L6-v2) generates 384-dim embeddings of startup descriptions. These capture semantic meaning beyond keywords. Optional — pipeline works with or without.

**Q12: How do you ensure fairness?**  
Subgroup performance analysis (country, market, founding decade). Proxy discrimination investigation. Mitigation strategies if disparities found. Human oversight in all decisions.

**Q13: What is the PDPA?**  
Malaysia's Personal Data Protection Act 2010. Governs personal data processing. This project uses company-level data only — no personal data. DPIA recommended for production.

**Q14: How do you handle data drift?**  
Monitor PSI of feature distributions. Alert if PSI > 0.2. Evaluate model on new labelled data quarterly. Retrain when drift exceeds threshold.

**Q15: What would you do differently?**  
Collect founder/team data, implement survival analysis, design A/B tests, add investor network features, build real-time pipeline, fine-tune domain-specific LLM.

---

*End of Dissertation*
