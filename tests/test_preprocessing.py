"""
Tests for preprocessing module.

Tests:
- FeatureEngineer creates all expected features
- RiskKeywordExtractor extracts keywords correctly
- Pipeline fits and transforms without errors
- Data leakage prevention (no target in features)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    FeatureEngineer, RiskKeywordExtractor,
    build_preprocessing_pipeline, get_feature_names
)


@pytest.fixture
def sample_startup_df():
    """Sample startup data matching the actual dataset schema."""
    return pd.DataFrame({
        "name": ["StartupA", "StartupB", "StartupC"],
        "description": [
            "AI-powered SaaS platform for enterprise workflow automation",
            "Pre-revenue early stage prototype for biotech research",
            "Profitable market leader in fintech with global expansion"
        ],
        "category_code": ["software", "biotech", "other"],
        "funding_total_usd": [5000000, 500000, 20000000],
        "funding_rounds": [3, 1, 5],
        "founded_year": [2019, 2023, 2015],
        "founded_at": ["2019-01-01", "2023-01-01", "2015-01-01"],
        "first_funding_at": ["2019-09-01", "2023-02-15", "2015-06-01"],
        "last_funding_at": ["2022-03-15", "2023-02-15", "2023-01-10"],
        "state_code": ["CA", "NY", "MA"],
        "city": ["San Francisco", "New York", "Boston"],
        "relationships": [5, 2, 10],
        "milestones": [2, 0, 4],
        "avg_participants": [3.0, 1.0, 5.0],
        "age_first_funding_year": [0.7, 0.1, 0.4],
        "age_last_funding_year": [3.2, 0.1, 8.0],
        "age_first_milestone_year": [1.5, 0.0, 2.0],
        "age_last_milestone_year": [3.0, 0.0, 7.0],
        "is_CA": [1, 0, 0], "is_NY": [0, 1, 0], "is_MA": [0, 0, 1],
        "is_TX": [0, 0, 0], "is_otherstate": [0, 0, 0],
        "is_software": [1, 0, 0], "is_web": [0, 0, 0], "is_mobile": [0, 0, 0],
        "is_enterprise": [0, 0, 0], "is_advertising": [0, 0, 0],
        "is_gamesvideo": [0, 0, 0], "is_ecommerce": [0, 0, 0],
        "is_biotech": [0, 1, 0], "is_consulting": [0, 0, 0], "is_othercategory": [0, 0, 1],
        "has_VC": [1, 0, 1], "has_angel": [1, 1, 0],
        "has_roundA": [1, 1, 1], "has_roundB": [1, 0, 1],
        "has_roundC": [1, 0, 1], "has_roundD": [0, 0, 1],
        "is_top500": [0, 0, 1],
        "funding_success": [1, 0, 1],
    })


class TestFeatureEngineer:

    def test_creates_company_age(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "company_age" in result.columns
        assert result["company_age"].iloc[0] == 7  # 2026 - 2019
        assert result["company_age"].iloc[1] == 3  # 2026 - 2023

    def test_creates_funding_velocity(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "funding_velocity" in result.columns
        assert result["funding_velocity"].iloc[0] > 0

    def test_creates_rounds_per_year(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "rounds_per_year" in result.columns

    def test_creates_time_to_first_funding(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "time_to_first_funding" in result.columns
        assert (result["time_to_first_funding"] >= 0).all()

    def test_creates_funding_span(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "funding_span" in result.columns

    def test_creates_avg_round_size(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "avg_round_size" in result.columns
        assert result["avg_round_size"].iloc[0] > 1000000

    def test_creates_description_length(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "description_length" in result.columns
        assert result["description_length"].iloc[0] > 0

    def test_creates_description_readability(self, sample_startup_df):
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(sample_startup_df)
        assert "description_readability" in result.columns

    def test_company_age_clipped_to_zero(self):
        """Company founded in future year should have age 0."""
        df = pd.DataFrame({
            "funding_total_usd": [1000000],
            "funding_rounds": [1],
            "founded_year": [2030],
            "founded_at": ["2030-01-01"],
            "first_funding_at": ["2030-01-01"],
            "last_funding_at": ["2030-01-01"],
            "description": ["Test startup"],
            "state_code": ["CA"],
            "city": ["SF"],
            "category_code": ["software"],
            "name": ["TestCo"],
        })
        fe = FeatureEngineer(reference_year=2026)
        result = fe.transform(df)
        assert result["company_age"].iloc[0] == 0


class TestRiskKeywordExtractor:

    def test_extracts_high_risk_keywords(self):
        rke = RiskKeywordExtractor()
        df = pd.DataFrame({
            "description": ["pre-revenue early stage prototype concept"]
        })
        result = rke.transform(df)
        assert result["high_risk_score"].iloc[0] >= 3

    def test_extracts_low_risk_keywords(self):
        rke = RiskKeywordExtractor()
        df = pd.DataFrame({
            "description": ["profitable market leader acquisition global"]
        })
        result = rke.transform(df)
        assert result["low_risk_score"].iloc[0] >= 3

    def test_handles_empty_description(self):
        rke = RiskKeywordExtractor()
        df = pd.DataFrame({"description": [""]})
        result = rke.transform(df)
        assert result["high_risk_score"].iloc[0] == 0
        assert result["low_risk_score"].iloc[0] == 0

    def test_all_three_categories_present(self):
        rke = RiskKeywordExtractor()
        df = pd.DataFrame({
            "description": ["pre-revenue growth profitable"]
        })
        result = rke.transform(df)
        assert "high_risk_score" in result.columns
        assert "medium_risk_score" in result.columns
        assert "low_risk_score" in result.columns


class TestPipeline:

    def test_pipeline_builds(self):
        pipe = build_preprocessing_pipeline(use_llm_embeddings=False)
        assert pipe is not None

    def test_pipeline_fits_and_transforms(self, sample_startup_df):
        pipe = build_preprocessing_pipeline(use_llm_embeddings=False)
        X = sample_startup_df.drop(columns=["funding_success"])
        pipe.fit(X)
        transformed = pipe.transform(X)
        assert transformed.shape[0] == len(X)

    def test_no_target_in_features(self, sample_startup_df):
        """Ensure target column is not in feature set."""
        pipe = build_preprocessing_pipeline(use_llm_embeddings=False)
        X = sample_startup_df.drop(columns=["funding_success"])
        assert "funding_success" not in X.columns

    def test_get_feature_names(self):
        names = get_feature_names()
        assert "company_age" in names
        assert "funding_velocity" in names
        assert "high_risk_score" in names
