# Stage 1 Documentation: Introduction, Literature Review, and Methodology

**Project Title**: LLM-Based Startup Risk Profiling and Funding Outcome Prediction Using Public Business Information and Explainable AI

**Author**: ADS  
**Institution**: [University Name]  
**Programme**: Master of Computer Science  
**Date**: October 2026  

---

## Table of Contents

### Chapter 1: Introduction
- 1.1 Background of the Study
- 1.2 Problem Statement
- 1.3 Research Questions
- 1.4 Research Objectives
- 1.5 Significance of the Study
- 1.6 Scope and Limitations
- 1.7 Structure of the Documentation

### Chapter 2: Literature Review
- 2.1 Introduction
- 2.2 Startup Ecosystem and Funding Landscape
- 2.3 Machine Learning for Startup Success Prediction
- 2.3.1 Traditional Statistical Methods
- 2.3.2 Tree-Based Ensemble Methods
- 2.3.3 Gradient Boosting Methods
- 2.4 Natural Language Processing for Business Text Analysis
- 2.5 Large Language Models for Feature Extraction
- 2.6 Explainable AI in Risk Profiling
- 2.7 Feature Engineering for Startup Modelling
- 2.8 Handling Class Imbalance in Funding Outcome Data
- 2.9 Evaluation Metrics for Imbalanced Classification
- 2.10 Responsible AI and Data Privacy in Financial Applications
- 2.11 Summary and Research Gap

### Chapter 3: Methodology
- 3.1 Introduction
- 3.2 Research Design
- 3.3 Framework: CRISP-DM
- 3.4 Dataset Description
- 3.4.1 Source and Acquisition
- 3.4.2 Data Dictionary
- 3.4.3 Data Quality Assessment
- 3.5 Data Preprocessing
- 3.5.1 Data Cleaning
- 3.5.2 Feature Engineering
- 3.5.3 LLM-Based Text Feature Extraction
- 3.5.4 Encoding and Scaling
- 3.5.5 Pipeline Construction
- 3.6 Machine Learning Models
- 3.6.1 Logistic Regression (Baseline)
- 3.6.2 Decision Tree
- 3.6.3 Random Forest
- 3.6.4 XGBoost
- 3.6.5 LightGBM
- 3.7 Hyperparameter Optimization
- 3.8 Evaluation Strategy
- 3.8.1 Metrics
- 3.8.2 Cross-Validation
- 3.8.3 Threshold Selection
- 3.9 Explainability Approach
- 3.10 Application Architecture
- 3.11 Summary

---

# Chapter 1: Introduction

## 1.1 Background of the Study

The global startup ecosystem has experienced exponential growth over the past two decades, with venture capital investment reaching approximately $344 billion in 2021 before contracting to around $239 billion in 2023 amid macroeconomic headwinds (Crunchbase, 2023). Despite this substantial capital deployment, the failure rate of startups remains notoriously high — research by CB Insights (2021) indicates that approximately 70% of startups ultimately fail, with the primary reasons including lack of market need (35%), running out of cash (38%), and absence of a viable business model (19%). The ability to systematically assess startup risk and predict funding outcomes is therefore of significant interest to investors, accelerators, policymakers, and entrepreneurs themselves.

Traditionally, startup evaluation has relied on qualitative assessment by experienced investors — a process often described as "pattern matching" based on founders' backgrounds, market size, and technology differentiation (Gompers et al., 2020). While expert judgement remains valuable, it is inherently subjective, subject to cognitive biases, and difficult to scale across the thousands of funding decisions made annually. The emergence of large-scale, publicly available business datasets — such as Crunchbase, PitchBook, and AngelList — has created opportunities for data-driven approaches to startup risk profiling that complement human expertise.

Machine learning has demonstrated significant potential in financial prediction tasks, including credit scoring (Lessmann et al., 2015), fraud detection (West & Bhattacharya, 2016), and bankruptcy prediction (Barboza et al., 2017). However, its application to startup funding outcome prediction remains comparatively underexplored, partly due to data availability constraints and the inherent complexity of startup ecosystems. Recent advances in large language models (LLMs) — such as BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), and sentence-transformers (Reimers & Gurevych, 2019) — offer new possibilities for extracting structured information from unstructured startup descriptions, potentially capturing risk signals that traditional financial metrics miss.

This project investigates whether publicly available business information, combined with LLM-derived text features and explainable AI, can produce a useful risk profiling system for startup funding outcomes. The approach distinguishes between *predicting* funding outcomes (a statistical task) and *causing* better outcomes (which requires intervention evidence), maintaining this distinction throughout.

## 1.2 Problem Statement

Investors, accelerators, and ecosystem participants face a fundamental information asymmetry: they must evaluate startups whose future performance is uncertain, using limited public information that is often incomplete, inconsistent, or outdated. Traditional evaluation methods are:

1. **Subjective**: Dependent on individual investor experience and pattern recognition
2. **Non-scalable**: Manual due diligence is time-intensive and cannot cover all opportunities
3. **Bias-prone**: Susceptible to representativeness heuristics and homophily bias (Gompers et al., 2020)
4. **Opaque**: Decision rationale is rarely documented or auditable

The central research problem is:

**How can publicly available business information be used to profile startup risk and predict funding outcomes using machine learning, while providing interpretable explanations that support human decision-making?**

## 1.3 Research Questions

This study addresses the following research questions:

1. **RQ1**: To what extent can machine learning models, trained on publicly available startup data, accurately predict funding outcomes (acquisition/IPO vs. closure)?
2. **RQ2**: Which features — financial, temporal, categorical, or LLM-derived text features — are most associated with startup funding outcomes?
3. **RQ3**: Do LLM-based text features (e.g., risk keywords, semantic embeddings) improve prediction performance over financial features alone?
4. **RQ4**: Can explainable AI techniques (SHAP) provide meaningful interpretations of model predictions for business stakeholders?
5. **RQ5**: What are the ethical, privacy, and fairness implications of applying ML to startup evaluation?

## 1.4 Research Objectives

1. Acquire and document a suitable startup dataset from a legitimate, accessible source
2. Build a reproducible data acquisition and processing pipeline
3. Perform data validation, cleaning, EDA, and feature engineering
4. Develop LLM-based text feature extraction for startup descriptions (risk keywords, embeddings)
5. Train multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
6. Compare models using classification metrics and select based on business requirements
7. Apply SHAP explainability to interpret predictions and important features
8. Develop a functional prototype application (Streamlit dashboard + FastAPI)
9. Evaluate responsible AI: privacy, fairness, and limitations
10. Produce complete technical documentation, presentation, and defence preparation

## 1.5 Significance of the Study

**Academic**: Extends ML literature to startup risk profiling — an underexplored domain compared to credit scoring or churn prediction. Integrates LLM-based feature extraction with traditional financial features, a novel methodological contribution.

**Practical**: Provides a decision-support tool for investors, accelerators, and ecosystem builders to screen startups at scale. Reduces manual due diligence burden through automated risk categorization.

**Societal**: Addresses information asymmetry in startup ecosystems. Transparent, explainable predictions can reduce bias in funding decisions and improve access to capital for underrepresented founders.

## 1.6 Scope and Limitations

**In scope**:
- Binary classification: funding success (acquisition/IPO) vs. failure (closure)
- Public company-level data (no personal data)
- Text features from startup descriptions
- Five ML models with hyperparameter optimization
- Explainability via SHAP
- Deployment prototype

**Out of scope**:
- Predicting funding amounts or valuations (regression task)
- Causal claims about what makes startups succeed
- Real-time data integration from live APIs
- Alternative data (social media sentiment, patent filings)
- Longitudinal survival analysis

**Key limitations**:
1. **Survivorship bias**: Crunchbase over-represents funded startups
2. **Observational data**: Correlations, not causal relationships
3. **No intervention outcomes**: Cannot prove that risk profiling improves investment decisions
4. **Geographical bias**: US-based companies are over-represented
5. **Temporal drift**: Startup dynamics change over time

## 1.7 Structure of the Documentation

- **Chapter 1** (this chapter): Introduction, problem statement, research questions
- **Chapter 2**: Literature review covering ML for startup prediction, NLP/LLM applications, explainable AI
- **Chapter 3**: Methodology including dataset, preprocessing, models, evaluation, and architecture

---

# Chapter 2: Literature Review

## 2.1 Introduction

This chapter reviews the existing literature across five interconnected domains: (1) the startup ecosystem and its data-driven evaluation, (2) machine learning approaches to startup success prediction, (3) natural language processing and large language models for business text analysis, (4) explainable AI in risk profiling, and (5) responsible AI considerations for financial applications. The review identifies research gaps that this project addresses.

## 2.2 Startup Ecosystem and Funding Landscape

The startup funding landscape involves multiple stakeholders: founders, angel investors, venture capital (VC) firms, accelerators, and institutional limited partners. Gompers and Lerner (2001) provide a foundational analysis of the venture capital cycle, highlighting the information asymmetry between entrepreneurs and investors as a central challenge. This asymmetry manifests in adverse selection (Akerlof, 1970) — investors cannot perfectly distinguish high-quality startups from low-quality ones, leading to potential misallocation of capital.

Kerr et al. (2014) found that VC funding decisions are influenced by non-performance factors, including geographic proximity and founder demographics, suggesting that traditional evaluation processes are subject to systematic biases. The development of large-scale startup databases — Crunchbase, PitchBook, Tracxn — has enabled quantitative approaches to startup analysis that were previously impossible. However, these databases contain incomplete, self-reported data with known survivorship biases (Conti et al., 2014).

## 2.3 Machine Learning for Startup Success Prediction

### 2.3.1 Traditional Statistical Methods

Early approaches to startup success prediction used logistic regression and survival analysis. Song et al. (2008) applied logistic regression to predict venture failure using financial ratios, achieving approximately 70% accuracy. Zelenkovic and Maksimovic (2018) used Cox proportional hazards models to estimate startup survival times. These methods provide interpretable coefficients but may underfit complex, non-linear relationships in startup data.

### 2.3.2 Tree-Based Ensemble Methods

Decision tree ensembles — particularly Random Forests (Breiman, 2001) — have become a standard for tabular classification. Lessmann et al. (2015) demonstrated that ensemble methods consistently outperform single classifiers in credit scoring, a domain structurally similar to startup risk prediction. Random Forests handle mixed data types, are robust to outliers, and provide built-in feature importance estimates.

### 2.3.3 Gradient Boosting Methods

Gradient boosting frameworks — XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017), and CatBoost (Prokhorenkova et al., 2018) — represent the state of the art for tabular data classification. Grinsztajn et al. (2022) demonstrated that tree-based models still outperform deep learning on tabular data, particularly when sample sizes are moderate (<100,000 rows). XGBoost's popularity stems from its regularized objective function, handling of missing values, and computational efficiency. LightGBM's leaf-wise growth strategy and histogram-based splitting make it faster and more memory-efficient than XGBoost.

## 2.4 Natural Language Processing for Business Text Analysis

NLP techniques have been applied to various business text analysis tasks, including sentiment analysis of financial reports (Loughran & McDonald, 2011), classification of corporate disclosures, and extraction of risk factors from annual reports. Traditional approaches use bag-of-words, TF-IDF, or topic modelling (Blei et al., 2003). More recently, deep learning approaches using word embeddings (Mikolov et al., 2013) and contextual models have improved performance significantly.

## 2.5 Large Language Models for Feature Extraction

The advent of transformer-based LLMs (Vaswani et al., 2017) has transformed NLP. BERT (Devlin et al., 2019) and its variants enable contextual understanding of text that captures semantic relationships beyond surface-level keywords. Reimers and Gurevych (2019) developed Sentence-BERT, which produces semantically meaningful sentence embeddings suitable for classification tasks. FinBERT (Araci, 2019) fine-tuned BERT on financial text, demonstrating domain-specific improvements.

For startup risk profiling, LLM-based features can capture:
- **Semantic content**: What the startup actually does (market, technology, business model)
- **Risk language**: Presence of uncertainty markers, speculative language, or over-optimistic claims
- **Professionalism**: Readability, specificity, and structure of the description
- **Domain signals**: Industry-specific terminology and positioning

This project uses sentence-transformer embeddings (all-MiniLM-L6-v2) as optional features, combined with rule-based risk keyword extraction for interpretability.

## 2.6 Explainable AI in Risk Profiling

The "black box" nature of complex ML models creates a tension between accuracy and interpretability, particularly in financial applications where decisions have significant consequences. The EU's General Data Protection Regulation (GDPR) includes provisions on automated decision-making that some scholars interpret as a "right to explanation" (Goodman & Flaxman, 2017).

SHAP (SHapley Additive exPlanations) (Lundberg & Lee, 2017) provides a unified framework for interpreting model predictions based on Shapley values from cooperative game theory (Shapley, 1953). SHAP offers both global explanations (feature importance across all predictions) and local explanations (individual prediction decomposition). Lundberg et al. (2020) extended SHAP with TreeSHAP, enabling efficient computation for tree-based models.

Ribeiro et al. (2016) proposed LIME (Local Interpretable Model-agnostic Explanations) as an alternative approach that approximates complex models locally with interpretable surrogates. Both SHAP and LIME have been applied to credit scoring, medical diagnosis, and fraud detection.

## 2.7 Feature Engineering for Startup Modelling

Feature engineering is critical for startup prediction due to the limited number of raw features in public datasets. Key engineered features in the literature include:

- **Funding velocity**: Total funding / company age — captures fundraising efficiency
- **Round progression**: Pattern of funding round sizes and intervals
- **Time to first funding**: Speed of initial capital acquisition
- **Market category**: Sector-specific failure rates (biotech vs. SaaS vs. hardware)
- **Geographic location**: Startup hub concentration effects (Glaeser et al., 2010)

This project engineers 10+ features spanning financial ratios, temporal metrics, and text-derived characteristics.

## 2.8 Handling Class Imbalance in Funding Outcome Data

Startup outcome data is typically imbalanced — failed startups often outnumber successful exits. He and Garcia (2009) provide a comprehensive survey of imbalanced learning techniques, categorizing them into data-level (oversampling, undersampling, SMOTE) and algorithm-level (cost-sensitive learning, ensemble methods). Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling Technique), which generates synthetic minority samples.

This project uses algorithm-level approaches: `class_weight="balanced"` in sklearn models adjusts the loss function to penalize minority class errors more heavily, avoiding the data manipulation risks of SMOTE.

## 2.9 Evaluation Metrics for Imbalanced Classification

Provost et al. (1998) argued against using accuracy as the primary metric for imbalanced classification, demonstrating that ROC curves and area under the curve (AUC) provide more informative evaluation. Davis and Gould (2006) showed that precision-recall curves are more informative than ROC curves for highly imbalanced datasets. This project uses a comprehensive set of seven metrics: accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and Brier score.

## 2.10 Responsible AI and Data Privacy in Financial Applications

The application of ML to financial decisions raises significant ethical concerns: fairness (Barocas & Selbst, 2016), privacy (regulations like GDPR in the EU and PDPA in Malaysia), and accountability (O'Neil, 2016). Algorithmic bias in lending has been documented (Bartlett et al., 2022), and startup evaluation carries similar risks — if ML models are trained on data that over-represents certain demographics or geographies, predictions may perpetuate existing inequities.

For Malaysian deployment, the Personal Data Protection Act 2010 (PDPA) governs the processing of personal data. This project uses company-level public data only, avoiding personal data concerns, but the responsible AI assessment addresses potential proxy variables and fairness implications.

## 2.11 Summary and Research Gap

The literature review reveals several gaps:

1. **Limited LLM integration**: Most startup prediction studies use only financial features; few integrate LLM-derived text features
2. **Explainability gap**: Studies that use complex models often lack interpretability analysis
3. **Responsible AI gap**: Fairness analysis in startup prediction is rare
4. **Deployment gap**: Academic studies rarely produce deployable applications

This project addresses these gaps by integrating LLM-based text features with financial features, applying SHAP explainability, conducting fairness analysis, and deploying a functional application.

---

# Chapter 3: Methodology

## 3.1 Introduction

This chapter describes the research design, data, preprocessing, modelling, evaluation, and system architecture used in this project. The methodology follows the CRISP-DM framework, adapted for a startup risk profiling context.

## 3.2 Research Design

The study employs a quantitative, cross-sectional design using supervised machine learning for binary classification. The target variable is funding outcome (success vs. failure), derived from startup status. The design is observational — no interventions are conducted, and causal claims are not made.

## 3.3 Framework: CRISP-DM

The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) (Chapman et al., 2000), with six phases:

1. **Business Understanding**: Define startup risk profiling objectives
2. **Data Understanding**: Acquire and explore startup dataset
3. **Data Preparation**: Clean, engineer features, build pipelines
4. **Modelling**: Train, optimize, and compare ML models
5. **Evaluation**: Assess performance against business metrics
6. **Deployment**: Build Streamlit dashboard and FastAPI

## 3.4 Dataset Description

### 3.4.1 Source and Acquisition

**Primary source**: Kaggle — Startup Success Prediction dataset
**URL**: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
**Underlying data**: Crunchbase public company profiles
**License**: Kaggle Open Data license
**Acquisition date**: October 2026 (record when downloaded)

The dataset contains company-level information on startups, including funding history, founding dates, geographic location, market category, and current status. The data is publicly available and does not contain personal identifiable information.

### 3.4.2 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| name | string | Company name |
| description | string | Company description (text) |
| category | string | Business category |
| market | string | Market segment |
| funding_total_usd | float | Total funding raised (USD) |
| funding_rounds | int | Number of funding rounds |
| founded_year | int | Year founded |
| founded_month | int | Month founded |
| first_funding_at | date | Date of first funding round |
| last_funding_at | date | Date of last funding round |
| country_code | string | ISO country code |
| state_code | string | State/province code |
| region | string | Geographic region |
| city | string | City |
| status | string | Current status: acquired, IPO, closed, operating |

### 3.4.3 Data Quality Assessment

Data quality is assessed using the validation module (`src/validation.py`), checking:
- Missing values per column (threshold: 50%)
- Duplicate rows
- Funding amount ranges (negative values, extreme outliers)
- Date consistency (founded ≤ first funding ≤ last funding)
- Categorical value consistency
- Target class distribution and imbalance ratio

## 3.5 Data Preprocessing

### 3.5.1 Data Cleaning

1. **funding_total_usd**: Convert from string to numeric; replace placeholder strings (" - ", "nan") with NaN
2. **Date columns**: Parse to datetime with error handling for various formats
3. **Status normalization**: Lowercase and standardize status strings
4. **Target encoding**: Map acquired/IPO → 1 (success), closed → 0 (failure), operating → excluded

### 3.5.2 Feature Engineering

Engineered features (10 total):

| Feature | Formula | Rationale |
|---------|---------|-----------|
| company_age | reference_year - founded_year | Maturity indicator |
| funding_velocity | funding_total / company_age | Capital acquisition efficiency |
| rounds_per_year | funding_rounds / company_age | Fundraising frequency |
| time_to_first_funding | first_funding_at - founded_date | Speed to first capital |
| funding_span | last_funding_at - first_funding_at | Fundraising period |
| avg_round_size | funding_total / funding_rounds | Investor conviction per round |
| description_length | word_count(description) | Description quality proxy |
| description_readability | Flesch reading ease | Clarity of communication |
| high_risk_score | keyword count (risk category) | LLM-derived risk signal |
| low_risk_score | keyword count (success category) | LLM-derived success signal |

### 3.5.3 LLM-Based Text Feature Extraction

Two approaches for text features:

**Rule-based risk keywords**: Categorize startup descriptions into high-risk (e.g., "pre-revenue", "early stage", "prototype"), medium-risk (e.g., "growth", "expansion", "Series A"), and low-risk (e.g., "profitable", "market leader", "acquisition") based on keyword matching.

**Sentence-transformer embeddings**: Use the `all-MiniLM-L6-v2` model (Reimers & Gurevych, 2019) to generate 384-dimensional semantic embeddings of startup descriptions. These capture contextual meaning beyond surface-level keywords. Embeddings are optional — the pipeline can run with or without them.

### 3.5.4 Encoding and Scaling

- **Numerical features**: Median imputation + StandardScaler (z-score normalization)
- **Categorical features**: Constant imputation ("unknown") + OneHotEncoder (max 20 categories per feature to limit dimensionality)
- **Ordinal features**: None identified in this dataset

### 3.5.5 Pipeline Construction

All preprocessing is wrapped in scikit-learn Pipeline objects to prevent data leakage. The pipeline structure:

```
Pipeline([
    FeatureEngineer(),          # Creates engineered features
    RiskKeywordExtractor(),     # Text-based risk scores
    TextFeatureExtractor(),     # LLM embeddings or TF-IDF
    ColumnTransformer([         # Combined preprocessing
        numeric_transformer,   # Impute + scale
        categorical_transformer # Impute + one-hot
    ])
])
```

Pipelines are fit on training data only and applied to validation/test data, ensuring no information from validation/test data leaks into preprocessing.

## 3.6 Machine Learning Models

### 3.6.1 Logistic Regression (Baseline)

Logistic Regression serves as the interpretable baseline. It models the log-odds of success as a linear function of features. With `class_weight="balanced"`, it automatically adjusts for class imbalance. While it cannot capture non-linear interactions, it provides a performance floor and interpretable coefficients.

### 3.6.2 Decision Tree

A single Decision Tree (max_depth=5) captures non-linear relationships and feature interactions. Its primary value is interpretability — the tree structure can be visualized and understood by non-technical stakeholders. However, it is prone to high variance.

### 3.6.3 Random Forest

Random Forest (Breiman, 2001) aggregates 200 decision trees with bootstrap sampling and random feature selection. It reduces variance compared to a single tree, handles mixed data types, and provides robust out-of-bag error estimates. `class_weight="balanced"` addresses class imbalance.

### 3.6.4 XGBoost

XGBoost (Chen & Guestrin, 2016) uses gradient boosting with regularization. It handles missing values natively, supports parallel tree construction, and has demonstrated state-of-the-art performance on tabular data. Hyperparameters are optimized using Optuna (Akiba et al., 2019) with Tree-structured Parzen Estimator (TPE) sampling.

### 3.6.5 LightGBM

LightGBM (Ke et al., 2017) uses leaf-wise tree growth and histogram-based splitting for computational efficiency. It is particularly effective on larger datasets and provides native support for categorical features. `class_weight="balanced"` is used for imbalance handling.

## 3.7 Hyperparameter Optimization

Optuna (Akiba et al., 2019) is used for XGBoost hyperparameter optimization with TPE sampling. The optimization search space includes:

| Parameter | Range |
|-----------|-------|
| n_estimators | 100-500 |
| max_depth | 3-10 |
| learning_rate | 0.01-0.3 (log) |
| subsample | 0.6-1.0 |
| colsample_bytree | 0.6-1.0 |
| min_child_weight | 1-10 |
| gamma | 0.0-5.0 |
| reg_alpha | 1e-8 to 10 (log) |
| reg_lambda | 1e-8 to 10 (log) |

50 trials are conducted, optimizing for validation ROC-AUC.

## 3.8 Evaluation Strategy

### 3.8.1 Metrics

| Metric | Formula | Business Relevance |
|--------|---------|---------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Misleading for imbalanced data |
| Precision | TP/(TP+FP) | False positives waste due diligence |
| Recall | TP/(TP+FN) | False negatives = missed opportunities |
| F1-score | 2×(P×R)/(P+R) | Balanced precision-recall |
| ROC-AUC | Area under ROC | Threshold-independent discrimination |
| PR-AUC | Area under PR curve | Better for imbalanced data |
| Brier score | Mean squared prob error | Calibration quality |

### 3.8.2 Cross-Validation

5-fold stratified cross-validation is used for all models. Stratification preserves the class distribution in each fold. Cross-validation provides more robust performance estimates than a single train/test split.

### 3.8.3 Threshold Selection

The default 0.5 threshold may not be optimal for business use. A cost-sensitive threshold optimization minimizes:

```
Total Cost = FP × $50,000 + FN × $5,000,000
```

Where:
- FP cost: Due diligence expense on a startup that would have failed anyway
- FN cost: Missed investment opportunity (estimated average deal value)

## 3.9 Explainability Approach

SHAP (Lundberg & Lee, 2017) is used for both global and local explainability:

- **Global**: SHAP summary plot showing feature importance across all predictions
- **Local**: SHAP waterfall plot decomposing individual predictions into feature contributions
- **Comparison**: Permutation importance as a model-agnostic cross-check

SHAP values are computed using TreeExplainer for tree-based models (Lundberg et al., 2020).

## 3.10 Application Architecture

The deployment consists of two components:

1. **Streamlit Dashboard** (`app/streamlit_app.py`): Interactive single/batch prediction with risk categorization and SHAP explanations
2. **FastAPI REST API** (`api/main.py`): Programmatic access with Pydantic input validation, four endpoints (health, model info, single predict, batch predict)

Both components load the saved model pipeline and use the same preprocessing, ensuring consistency between training and inference.

## 3.11 Summary

This methodology integrates publicly available startup data, LLM-derived text features, multiple ML models with hyperparameter optimization, and explainable AI into a reproducible pipeline. The CRISP-DM framework ensures systematic project execution. The next chapter (Stage 2) presents the execution, results, and full dissertation.

---

*End of Stage 1 Documentation*
