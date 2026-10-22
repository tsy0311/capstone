"""
Inference and prediction module.

Handles:
- Loading trained model
- Single prediction from startup attributes
- Batch prediction from DataFrame/CSV
- Risk categorization
- SHAP-based explanation for individual predictions
- Input preparation with default values for all pipeline-expected columns
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Union
import warnings

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# Risk categories
RISK_CATEGORIES = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 0.8),
    "very_high": (0.8, 1.0),
}

RISK_LABELS = {
    "low": "Low Risk — Likely to succeed",
    "medium": "Medium Risk — Moderate funding success probability",
    "high": "High Risk — Elevated failure probability",
    "very_high": "Very High Risk — Strong failure indicators",
}

# All columns the trained pipeline expects.
# These match the Kaggle startup-success-prediction dataset.
REQUIRED_COLUMNS = {
    # Numeric
    "funding_total_usd": 0,
    "funding_rounds": 0,
    "founded_year": 2015,
    "relationships": 0,
    "milestones": 0,
    "avg_participants": 0.0,
    "age_first_funding_year": 0.0,
    "age_last_funding_year": 0.0,
    "age_first_milestone_year": 0.0,
    "age_last_milestone_year": 0.0,
    # Dates (used by FeatureEngineer)
    "founded_at": "2015-01-01",
    "first_funding_at": "2015-06-01",
    "last_funding_at": "2018-01-01",
    # Nominal
    "state_code": "CA",
    "city": "San Francisco",
    "category_code": "software",
    # Binary flags
    "is_CA": 0, "is_NY": 0, "is_MA": 0, "is_TX": 0, "is_otherstate": 0,
    "is_software": 0, "is_web": 0, "is_mobile": 0, "is_enterprise": 0,
    "is_advertising": 0, "is_gamesvideo": 0, "is_ecommerce": 0,
    "is_biotech": 0, "is_consulting": 0, "is_othercategory": 0,
    "has_VC": 0, "has_angel": 0,
    "has_roundA": 0, "has_roundB": 0, "has_roundC": 0, "has_roundD": 0,
    "is_top500": 0,
    # Text
    "description": "Startup company",
    # Identifier
    "name": "Unknown",
}

# Category code -> binary flag mapping
CATEGORY_FLAGS = {
    "software": "is_software",
    "web": "is_web",
    "mobile": "is_mobile",
    "enterprise": "is_enterprise",
    "advertising": "is_advertising",
    "gamesvideo": "is_gamesvideo",
    "ecommerce": "is_ecommerce",
    "biotech": "is_biotech",
    "consulting": "is_consulting",
    "other": "is_othercategory",
}

# State -> binary flag mapping
STATE_FLAGS = {
    "CA": "is_CA",
    "NY": "is_NY",
    "MA": "is_MA",
    "TX": "is_TX",
}

# Funding round -> binary flag mapping
ROUND_FLAGS = {
    1: ["has_roundA"],
    2: ["has_roundA", "has_roundB"],
    3: ["has_roundA", "has_roundB", "has_roundC"],
    4: ["has_roundA", "has_roundB", "has_roundC", "has_roundD"],
}


def prepare_input(startup_data: Dict) -> Dict:
    """
    Prepare a user-provided input dict by filling in all columns
    the trained pipeline expects with sensible defaults.

    Also auto-derives binary flags from category_code, state_code,
    and funding_rounds if they're not explicitly provided.

    Parameters
    ----------
    startup_data : dict
        User-provided startup attributes (partial).

    Returns
    -------
    dict
        Complete input with all required columns.
    """
    data = dict(REQUIRED_COLUMNS)  # Start with defaults
    data.update(startup_data)      # Override with user values

    # Auto-derive category binary flag if not set
    cat = str(data.get("category_code", "")).lower().strip()
    if cat in CATEGORY_FLAGS:
        flag = CATEGORY_FLAGS[cat]
        data[flag] = 1
    elif cat:
        data["is_othercategory"] = 1

    # Auto-derive state binary flag if not set
    state = str(data.get("state_code", "")).upper().strip()
    if state in STATE_FLAGS:
        flag = STATE_FLAGS[state]
        data[flag] = 1
    elif state:
        data["is_otherstate"] = 1

    # Auto-derive funding round flags from funding_rounds count
    rounds = int(data.get("funding_rounds", 0))
    for flag in ["has_roundA", "has_roundB", "has_roundC", "has_roundD"]:
        data[flag] = 0  # Reset
    if rounds in ROUND_FLAGS:
        for flag in ROUND_FLAGS[rounds]:
            data[flag] = 1
    elif rounds >= 4:
        data["has_roundA"] = 1
        data["has_roundB"] = 1
        data["has_roundC"] = 1
        data["has_roundD"] = 1

    # Ensure founded_at is set if founded_year is provided
    if "founded_year" in startup_data and "founded_at" not in startup_data:
        data["founded_at"] = f"{int(data['founded_year'])}-01-01"

    # Build description from name + category if not provided
    if "description" not in startup_data:
        name = data.get("name", "Startup")
        cat = data.get("category_code", "company")
        data["description"] = f"{name} - {cat} startup"

    return data


def load_model(model_path: str = "models/best_model.joblib"):
    """Load trained pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def categorize_risk(probability: float) -> str:
    """
    Categorize a probability into risk level.

    Parameters
    ----------
    probability : float
        Predicted probability of funding failure (class 0).

    Returns
    -------
    str
        Risk category: low, medium, high, very_high.
    """
    for category, (low, high) in RISK_CATEGORIES.items():
        if low <= probability < high:
            return category
    return "very_high" if probability >= 0.8 else "low"


def predict_single(
    model,
    startup_data: Dict,
    threshold: float = 0.5
) -> Dict:
    """
    Make a single prediction from startup attributes.

    Automatically fills in missing columns with defaults via prepare_input().

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline.
    startup_data : dict
        Dictionary of startup attributes (partial OK).
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Prediction results including probability, class, risk category.
    """
    # Fill in all required columns
    full_data = prepare_input(startup_data)
    df = pd.DataFrame([full_data])

    # Predict probability
    proba = model.predict_proba(df)[0]
    failure_prob = float(proba[0])  # P(failure)
    success_prob = float(proba[1])  # P(success)

    # Class prediction
    predicted_class = int(success_prob >= threshold)

    # Risk category (based on failure probability)
    risk_category = categorize_risk(failure_prob)

    result = {
        "startup_name": startup_data.get("name", "Unknown"),
        "funding_success_probability": round(success_prob, 4),
        "funding_failure_probability": round(failure_prob, 4),
        "predicted_outcome": "Success" if predicted_class == 1 else "Failure",
        "risk_category": risk_category,
        "risk_label": RISK_LABELS.get(risk_category, ""),
        "threshold_used": threshold,
    }

    return result


def predict_batch(
    model,
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Make batch predictions from a DataFrame.

    Fills missing columns with defaults for each row.

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline.
    df : pd.DataFrame
        Startup data.
    threshold : float
        Classification threshold.

    Returns
    -------
    pd.DataFrame
        Original data with prediction columns appended.
    """
    # Fill missing columns for each row
    rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        full_row = prepare_input(row_dict)
        rows.append(full_row)

    df_full = pd.DataFrame(rows)

    proba = model.predict_proba(df_full)
    failure_prob = proba[:, 0]
    success_prob = proba[:, 1]

    results = df.copy()
    results["funding_success_probability"] = success_prob.round(4)
    results["funding_failure_probability"] = failure_prob.round(4)
    results["predicted_outcome"] = np.where(
        success_prob >= threshold, "Success", "Failure"
    )
    results["risk_category"] = np.array([
        categorize_risk(p) for p in failure_prob
    ])

    return results


def explain_prediction_shap(
    model,
    startup_data: Union[Dict, pd.DataFrame],
    background_data: Optional[pd.DataFrame] = None,
    max_display: int = 10
) -> Optional[Dict]:
    """
    Generate SHAP explanation for a single prediction.

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline.
    startup_data : dict or DataFrame
        Startup to explain.
    background_data : DataFrame, optional
        Background dataset for SHAP (for tree explainer).
    max_display : int
        Max features to show.

    Returns
    -------
    dict or None
        SHAP values and feature names, or None if SHAP unavailable.
    """
    if not HAS_SHAP:
        warnings.warn("SHAP not available. Cannot generate explanation.")
        return None

    # Convert single row to DataFrame, filling missing columns
    if isinstance(startup_data, dict):
        full_data = prepare_input(startup_data)
        df = pd.DataFrame([full_data])
    else:
        df = startup_data.copy()

    # Extract preprocessing and model from pipeline
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")

    if preprocessor is None or classifier is None:
        warnings.warn("Pipeline structure not supported for SHAP.")
        return None

    # Transform input
    X_transformed = preprocessor.transform(df)

    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Choose SHAP explainer based on model type
    try:
        if hasattr(classifier, "estimators_") or "XGB" in type(classifier).__name__:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
        else:
            if background_data is not None:
                X_bg = preprocessor.transform(background_data)
                explainer = shap.KernelExplainer(
                    lambda x: classifier.predict_proba(x)[:, 1], X_bg
                )
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.Explainer(classifier, X_transformed)
                shap_values = explainer(X_transformed)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        elif hasattr(shap_values, "values"):
            shap_values = shap_values.values

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:max_display]

        result = {
            "base_value": float(explainer.expected_value)
            if not isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value[1])
            if len(explainer.expected_value) > 1
            else float(explainer.expected_value),
            "feature_contributions": [
                {
                    "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
                    "shap_value": float(shap_values[i]),
                    "value": float(X_transformed[0][i])
                    if hasattr(X_transformed, "__getitem__")
                    else None,
                }
                for i in indices
            ],
        }

        return result

    except Exception as e:
        warnings.warn(f"SHAP explanation failed: {e}")
        return None


def get_model_info(model) -> Dict:
    """Return model metadata."""
    classifier = model.named_steps.get("classifier")
    return {
        "model_type": type(classifier).__name__ if classifier else "Unknown",
        "model_params": classifier.get_params() if classifier and hasattr(classifier, "get_params") else {},
        "pipeline_steps": list(model.named_steps.keys()),
    }


if __name__ == "__main__":
    # Example usage
    model = load_model()
    sample = {
        "name": "ExampleStartup",
        "description": "AI-powered SaaS platform for enterprise workflow automation",
        "category_code": "software",
        "funding_total_usd": 5000000,
        "funding_rounds": 3,
        "founded_year": 2019,
        "first_funding_at": "2019-09-01",
        "last_funding_at": "2022-03-15",
        "state_code": "CA",
        "city": "San Francisco",
        "has_VC": 1,
        "has_angel": 1,
        "relationships": 5,
        "milestones": 2,
        "avg_participants": 3.0,
        "is_top500": 0,
    }
    result = predict_single(model, sample)
    print(result)
