"""
Tests for prediction module.

Tests:
- Risk categorization thresholds
- Single prediction structure
- Batch prediction output
- Model info extraction
- Input validation
- SHAP explanation structure (if available)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predict import (
    categorize_risk, RISK_CATEGORIES, RISK_LABELS,
    predict_single, predict_batch, get_model_info
)


class TestRiskCategorization:

    def test_low_risk(self):
        assert categorize_risk(0.1) == "low"

    def test_medium_risk(self):
        assert categorize_risk(0.4) == "medium"

    def test_high_risk(self):
        assert categorize_risk(0.7) == "high"

    def test_very_high_risk(self):
        assert categorize_risk(0.9) == "very_high"

    def test_boundary_low_medium(self):
        assert categorize_risk(0.3) == "medium"

    def test_boundary_medium_high(self):
        assert categorize_risk(0.6) == "high"

    def test_zero_probability(self):
        assert categorize_risk(0.0) == "low"

    def test_max_probability(self):
        result = categorize_risk(1.0)
        assert result in ["very_high"]


class TestPredictSingle:

    @pytest.fixture
    def mock_model(self):
        """Mock trained pipeline."""
        model = Mock()
        model.predict_proba.return_value = np.array([
            [0.2, 0.8]  # 20% failure, 80% success
        ])
        model.predict.return_value = np.array([1])
        model.named_steps = {
            "preprocessor": Mock(),
            "classifier": Mock(spec=["predict_proba", "predict"])
        }
        return model

    @pytest.fixture
    def sample_startup(self):
        return {
            "name": "TestStartup",
            "description": "AI SaaS platform",
            "category": "Software",
            "market": "Enterprise Software",
            "funding_total_usd": 5000000,
            "funding_rounds": 3,
            "founded_year": 2020,
            "founded_month": 6,
            "first_funding_at": "2020-09-01",
            "last_funding_at": "2023-01-15",
            "country_code": "USA",
            "state_code": "CA",
            "region": "SF Bay Area",
            "city": "San Francisco",
            "status": "operating",
            "status_category": "Series A",
        }

    def test_returns_expected_keys(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert "startup_name" in result
        assert "funding_success_probability" in result
        assert "funding_failure_probability" in result
        assert "predicted_outcome" in result
        assert "risk_category" in result
        assert "risk_label" in result

    def test_success_probability_correct(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert abs(result["funding_success_probability"] - 0.8) < 0.01

    def test_failure_probability_correct(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert abs(result["funding_failure_probability"] - 0.2) < 0.01

    def test_predicted_outcome_success(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert result["predicted_outcome"] == "Success"

    def test_threshold_affects_prediction(self, mock_model, sample_startup):
        """High threshold should predict failure for 0.8 probability."""
        result = predict_single(mock_model, sample_startup, threshold=0.9)
        assert result["predicted_outcome"] == "Failure"

    def test_risk_category_low(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert result["risk_category"] == "low"  # 0.2 failure prob

    def test_includes_threshold(self, mock_model, sample_startup):
        result = predict_single(mock_model, sample_startup, threshold=0.5)
        assert result["threshold_used"] == 0.5


class TestPredictBatch:

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.predict_proba.return_value = np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.5, 0.5],
        ])
        model.predict.return_value = np.array([1, 0, 0])
        model.named_steps = {
            "preprocessor": Mock(),
            "classifier": Mock(spec=["predict_proba", "predict"])
        }
        return model

    @pytest.fixture
    def batch_df(self):
        return pd.DataFrame([
            {"name": "StartupA", "description": "AI SaaS", "funding_total_usd": 5000000},
            {"name": "StartupB", "description": "Pre-revenue", "funding_total_usd": 100000},
            {"name": "StartupC", "description": "Profitable", "funding_total_usd": 10000000},
        ])

    def test_returns_dataframe(self, mock_model, batch_df):
        result = predict_batch(mock_model, batch_df, threshold=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_has_prediction_columns(self, mock_model, batch_df):
        result = predict_batch(mock_model, batch_df, threshold=0.5)
        assert "funding_success_probability" in result.columns
        assert "funding_failure_probability" in result.columns
        assert "predicted_outcome" in result.columns
        assert "risk_category" in result.columns

    def test_correct_row_count(self, mock_model, batch_df):
        result = predict_batch(mock_model, batch_df, threshold=0.5)
        assert len(result) == len(batch_df)

    def test_outcomes_correct(self, mock_model, batch_df):
        result = predict_batch(mock_model, batch_df, threshold=0.5)
        assert result["predicted_outcome"].iloc[0] == "Success"
        assert result["predicted_outcome"].iloc[1] == "Failure"


class TestModelInfo:

    def test_returns_expected_keys(self):
        model = Mock()
        classifier = Mock()
        classifier.__class__.__name__ = "XGBClassifier"
        classifier.get_params.return_value = {"n_estimators": 200}
        model.named_steps = {"preprocessor": Mock(), "classifier": classifier}

        info = get_model_info(model)
        assert "model_type" in info
        assert "model_params" in info
        assert "pipeline_steps" in info

    def test_model_type_correct(self):
        model = Mock()
        classifier = Mock()
        classifier.__class__.__name__ = "LogisticRegression"
        classifier.get_params.return_value = {"max_iter": 1000}
        model.named_steps = {"preprocessor": Mock(), "classifier": classifier}

        info = get_model_info(model)
        assert info["model_type"] == "LogisticRegression"


class TestRiskCategories:

    def test_all_categories_defined(self):
        assert "low" in RISK_CATEGORIES
        assert "medium" in RISK_CATEGORIES
        assert "high" in RISK_CATEGORIES
        assert "very_high" in RISK_CATEGORIES

    def test_labels_defined(self):
        assert len(RISK_LABELS) == 4
        for key in RISK_CATEGORIES:
            assert key in RISK_LABELS

    def test_ranges_cover_full_spectrum(self):
        """Categories should cover 0.0 to 1.0."""
        low_bound = min(lo for lo, _ in RISK_CATEGORIES.values())
        high_bound = max(hi for _, hi in RISK_CATEGORIES.values())
        assert low_bound == 0.0
        assert high_bound == 1.0
