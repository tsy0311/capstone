"""
Preprocessing and feature engineering module.

Builds scikit-learn Pipelines for reproducible, leakage-free preprocessing.
Handles:
- Missing value imputation
- Categorical encoding (OneHot)
- Numerical scaling
- Feature engineering (financial, temporal, text-derived)
- Risk keyword extraction from startup descriptions
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Dict, Optional
import warnings
from datetime import datetime

# Column groups — adapted to the actual Kaggle startup dataset
NUMERIC_COLS = [
    "funding_total_usd", "funding_rounds", "founded_year",
    "relationships", "milestones", "avg_participants",
    "age_first_funding_year", "age_last_funding_year",
    "age_first_milestone_year", "age_last_milestone_year",
]

NOMINAL_COLS = [
    "state_code", "city", "category_code",
]

BINARY_COLS = [
    "is_CA", "is_NY", "is_MA", "is_TX", "is_otherstate",
    "is_software", "is_web", "is_mobile", "is_enterprise",
    "is_advertising", "is_gamesvideo", "is_ecommerce",
    "is_biotech", "is_consulting", "is_othercategory",
    "has_VC", "has_angel",
    "has_roundA", "has_roundB", "has_roundC", "has_roundD",
    "is_top500",
]

TEXT_COLS = ["description"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.

    Creates:
    - company_age: Years since founding
    - funding_velocity: Total funding / company age
    - rounds_per_year: Funding rounds / company age
    - time_to_first_funding: Months from founding to first funding
    - funding_span: Months between first and last funding
    - avg_round_size: Total funding / number of rounds
    - description_length: Word count of description
    - description_readability: Simplified readability score
    """

    def __init__(self, reference_year: int = None):
        self.reference_year = reference_year or datetime.now().year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Company age
        if "founded_year" in X.columns:
            X["company_age"] = self.reference_year - X["founded_year"]
            X["company_age"] = X["company_age"].clip(lower=0)
        else:
            X["company_age"] = 0

        # Funding velocity (USD per year)
        if "funding_total_usd" in X.columns:
            X["funding_velocity"] = X["funding_total_usd"] / X["company_age"].clip(lower=1)
        else:
            X["funding_velocity"] = 0

        # Rounds per year
        if "funding_rounds" in X.columns:
            X["rounds_per_year"] = X["funding_rounds"] / X["company_age"].clip(lower=1)
        else:
            X["rounds_per_year"] = 0

        # Time to first funding (months)
        if "founded_year" in X.columns and "first_funding_at" in X.columns:
            founded = pd.to_datetime(
                X["founded_year"].astype(str) + "-01-01", errors="coerce"
            )
            first_fund = pd.to_datetime(X["first_funding_at"], errors="coerce")
            X["time_to_first_funding"] = (
                (first_fund - founded).dt.days / 30.44
            )
            X["time_to_first_funding"] = X["time_to_first_funding"].fillna(0).clip(lower=0)
        else:
            X["time_to_first_funding"] = 0

        # Funding span (months between first and last funding)
        if "first_funding_at" in X.columns and "last_funding_at" in X.columns:
            first_fund = pd.to_datetime(X["first_funding_at"], errors="coerce")
            last_fund = pd.to_datetime(X["last_funding_at"], errors="coerce")
            X["funding_span"] = (
                (last_fund - first_fund).dt.days / 30.44
            )
            X["funding_span"] = X["funding_span"].fillna(0).clip(lower=0)
        else:
            X["funding_span"] = 0

        # Average round size
        if "funding_total_usd" in X.columns and "funding_rounds" in X.columns:
            X["avg_round_size"] = (
                X["funding_total_usd"] / X["funding_rounds"].clip(lower=1)
            )
        else:
            X["avg_round_size"] = 0

        # Description length (word count)
        if "description" in X.columns:
            X["description_length"] = (
                X["description"].astype(str).str.split().str.len()
            )
            X["description_length"] = X["description_length"].fillna(0)
        else:
            X["description_length"] = 0

        # Description readability (simplified)
        if "description" in X.columns:
            X["description_readability"] = self._compute_readability(X["description"])
        else:
            X["description_readability"] = 0

        return X

    def _compute_readability(self, descriptions: pd.Series) -> pd.Series:
        """Simplified readability score: based on avg word length."""
        def avg_word_len(text):
            words = str(text).split()
            if not words:
                return 0
            return np.mean([len(w) for w in words])
        return descriptions.astype(str).apply(avg_word_len)

    def get_feature_names_out(self, input_features=None):
        return [
            "company_age", "funding_velocity", "rounds_per_year",
            "time_to_first_funding", "funding_span", "avg_round_size",
            "description_length", "description_readability"
        ]


class RiskKeywordExtractor(BaseEstimator, TransformerMixin):
    """
    Extract risk-related keyword features from startup descriptions.

    Flags high-risk, medium-risk, and low-risk keywords.
    """

    HIGH_RISK_KW = [
        "pre-revenue", "early stage", "prototype", "pre-seed", "unproven",
        "concept", "idea stage", "research", "experimental"
    ]
    MEDIUM_RISK_KW = [
        "growth", "expansion", "series a", "series b", "scalable",
        "pilot", "launch", "beta", "traction"
    ]
    LOW_RISK_KW = [
        "profitable", "acquisition", "ipo", "market leader", "established",
        "revenue", "scaled", "enterprise", "global"
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "description" not in X.columns:
            X["high_risk_score"] = 0
            X["medium_risk_score"] = 0
            X["low_risk_score"] = 0
            return X

        desc = X["description"].astype(str).str.lower()

        X["high_risk_score"] = desc.apply(
            lambda t: sum(1 for kw in self.HIGH_RISK_KW if kw in t)
        )
        X["medium_risk_score"] = desc.apply(
            lambda t: sum(1 for kw in self.MEDIUM_RISK_KW if kw in t)
        )
        X["low_risk_score"] = desc.apply(
            lambda t: sum(1 for kw in self.LOW_RISK_KW if kw in t)
        )

        return X

    def get_feature_names_out(self, input_features=None):
        return ["high_risk_score", "medium_risk_score", "low_risk_score"]


def build_preprocessing_pipeline(
    use_llm_embeddings: bool = False,
    reference_year: int = None
) -> Pipeline:
    """
    Build the complete preprocessing pipeline.

    Combines:
    - Feature engineering (financial + temporal + text)
    - Risk keyword extraction
    - Numerical imputation + scaling
    - Categorical imputation + one-hot encoding
    - Binary feature passthrough

    Parameters
    ----------
    use_llm_embeddings : bool
        If True, would include sentence-transformer embeddings (not used in default).
    reference_year : int
        Reference year for company_age calculation.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline with all preprocessing steps.
    """
    # Feature engineering transformer
    feature_engineer = FeatureEngineer(reference_year=reference_year)

    # Define engineered numeric columns
    engineered_numeric = [
        "company_age", "funding_velocity", "rounds_per_year",
        "time_to_first_funding", "funding_span", "avg_round_size",
        "description_length", "description_readability"
    ]

    # Original numeric columns that exist in the dataset
    original_numeric = [c for c in NUMERIC_COLS]

    # Risk keyword features
    risk_numeric = ["high_risk_score", "medium_risk_score", "low_risk_score"]

    all_numeric = original_numeric + engineered_numeric + risk_numeric

    # Numeric transformer: impute + scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical transformer: impute + one-hot
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
    ])

    # Binary features: just pass through (they are already 0/1)
    binary_passthrough = "passthrough"

    # Build ColumnTransformer
    # Note: FeatureEngineer and RiskKeywordExtractor run first in the pipeline,
    # creating engineered columns, then ColumnTransformer processes them.
    # Binary columns pass through as-is.
    available_nominal = NOMINAL_COLS
    available_binary = BINARY_COLS

    preprocessor = Pipeline([
        ("feature_engineer", feature_engineer),
        ("risk_keywords", RiskKeywordExtractor()),
        ("column_transform", ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, all_numeric),
                ("categorical", categorical_transformer, available_nominal),
                ("binary", binary_passthrough, available_binary),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor


def get_feature_names() -> List[str]:
    """Return the list of expected feature names after preprocessing."""
    return NUMERIC_COLS + [
        "company_age", "funding_velocity", "rounds_per_year",
        "time_to_first_funding", "funding_span", "avg_round_size",
        "description_length", "description_readability",
        "high_risk_score", "medium_risk_score", "low_risk_score"
    ] + BINARY_COLS


if __name__ == "__main__":
    pipe = build_preprocessing_pipeline(use_llm_embeddings=False)
    print("Pipeline structure:")
    print(pipe)
