"""
Model training module.

Handles:
- Train/validation/test split with stratification
- Baseline model (Logistic Regression)
- Multiple candidate models (LR, DT, RF, XGBoost, LightGBM)
- Hyperparameter optimization with Optuna
- Cross-validation
- Model saving
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import clone

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Skipping XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Skipping LightGBM model.")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.preprocessing import build_preprocessing_pipeline


RANDOM_SEED = 42


def split_data(
    df: pd.DataFrame,
    target_col: str = "funding_success",
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Target column name.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction of remaining data for validation.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size,
        stratify=y_trainval, random_state=random_state
    )

    print(f"Train:      {len(X_train):,} rows")
    print(f"Validation: {len(X_val):,} rows")
    print(f"Test:       {len(X_test):,} rows")
    print(f"\nTarget distribution (train):")
    print(y_train.value_counts(normalize=True).to_dict())

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_definitions() -> Dict[str, Any]:
    """
    Define all candidate models with default hyperparameters.

    Returns
    -------
    dict
        Model name -> model instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1
        )

    return models


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    Train all candidate models with preprocessing pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.
    use_llm : bool
        Whether to include LLM embeddings in preprocessing.

    Returns
    -------
    dict
        Model name -> fitted Pipeline.
    """
    models = get_model_definitions()
    trained = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Build pipeline: preprocessing + model
        pipeline = SkPipeline([
            ("preprocessor", build_preprocessing_pipeline(use_llm_embeddings=use_llm)),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

        # Quick validation score
        val_score = pipeline.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

    return trained


def optimize_xgboost_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    use_llm: bool = False
) -> Tuple[Any, Dict]:
    """
    Optimize XGBoost hyperparameters using Optuna TPE sampler.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data.
    n_trials : int
        Number of Optuna trials.
    use_llm : bool
        Whether to include LLM embeddings.

    Returns
    -------
    tuple
        (Best pipeline, best params dict)
    """
    if not HAS_OPTUNA:
        warnings.warn("Optuna not available. Skipping optimization.")
        return None, {}

    if not HAS_XGBOOST:
        warnings.warn("XGBoost not available. Skipping optimization.")
        return None, {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBClassifier(
            **params, use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1
        )

        pipeline = SkPipeline([
            ("preprocessor", build_preprocessing_pipeline(use_llm_embeddings=use_llm)),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, y_pred_proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="startup_funding_xgboost"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Train final model with best params
    best_model = XGBClassifier(
        **study.best_params, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1
    )
    best_pipeline = SkPipeline([
        ("preprocessor", build_preprocessing_pipeline(use_llm_embeddings=use_llm)),
        ("classifier", best_model),
    ])
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study.best_params


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    use_llm: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Perform stratified k-fold cross-validation for all models.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature set.
    y : pd.Series
        Full target.
    cv_folds : int
        Number of CV folds.
    use_llm : bool
        Whether to include LLM embeddings.

    Returns
    -------
    dict
        Model name -> {"mean_roc_auc": float, "std_roc_auc": float}
    """
    models = get_model_definitions()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, model in models.items():
        pipeline = SkPipeline([
            ("preprocessor", build_preprocessing_pipeline(use_llm_embeddings=use_llm)),
            ("classifier", model),
        ])

        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {
            "mean_roc_auc": float(scores.mean()),
            "std_roc_auc": float(scores.std()),
        }
        print(f"{name}: ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def save_model(pipeline, path: str = "models/best_model.joblib"):
    """Save the trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def main():
    """Main training entry point."""
    from src.data_loader import load_and_preprocess, load_config

    config = load_config()
    df = load_and_preprocess(config["dataset"]["file"])

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Train all models
    trained_models = train_all_models(X_train, y_train, X_val, y_val, use_llm=False)

    # Optuna optimization
    if HAS_OPTUNA and HAS_XGBOOST:
        best_xgb, best_params = optimize_xgboost_with_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=config["models"]["optuna"]["n_trials"],
            use_llm=False
        )
        if best_xgb:
            trained_models["XGBoost (Optuna)"] = best_xgb

    # Cross-validation
    X = df.drop(columns=["funding_success"])
    y = df["funding_success"]
    cv_results = cross_validate_models(X, y, cv_folds=config["training"]["cv_folds"])

    # Save best model (placeholder — selection based on val ROC-AUC)
    # In production, evaluate all on val set and pick best
    save_model(trained_models.get("XGBoost", list(trained_models.values())[0]))

    print("\nTraining complete. See reports/ for evaluation.")


if __name__ == "__main__":
    main()
