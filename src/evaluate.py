"""
Model evaluation module.

Computes classification metrics, generates confusion matrices,
ROC curves, precision-recall curves, and calibration plots.
Includes cost-sensitive threshold optimization for business use.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss,
    classification_report
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any, Optional
import os
import json


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = "reports/figures/confusion_matrix.png"
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Failure", "Success"],
        yticklabels=["Failure", "Success"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = "reports/figures/roc_curve.png"
):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = "reports/figures/pr_curve.png"
):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_calibration(
    y_true: np.ndarray, y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = "reports/figures/calibration.png"
):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", linewidth=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of successes")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def optimize_threshold(
    y_true: np.ndarray, y_proba: np.ndarray,
    fp_cost: float = 50.0, fn_cost: float = 5000000.0
) -> Tuple[float, float]:
    """
    Find optimal classification threshold minimizing expected cost.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    fp_cost : float
        Cost of false positive (unnecessary due diligence).
    fn_cost : float
        Cost of false negative (missed investment opportunity).

    Returns
    -------
    tuple
        (optimal_threshold, minimum_cost)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fp * fp_cost + fn * fn_cost
        costs.append(total_cost)

    costs = np.array(costs)
    best_idx = np.argmin(costs)
    optimal_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]

    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Expected cost at optimal: ${min_cost:,.0f}")

    return optimal_threshold, min_cost


def plot_threshold_analysis(
    y_true: np.ndarray, y_proba: np.ndarray,
    fp_cost: float = 50.0, fn_cost: float = 5000000.0,
    save_path: str = "reports/figures/threshold_analysis.png"
):
    """Plot cost vs threshold curve."""
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costs.append(fp * fp_cost + fn * fn_cost)

    costs = np.array(costs)
    best_idx = np.argmin(costs)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, linewidth=2)
    ax.axvline(best_t, color="red", linestyle="--", label=f"Optimal: {best_t:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Cost ($)")
    ax.set_title("Cost-Sensitive Threshold Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_model(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str = "Model",
    save_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Full evaluation of a trained model.

    Parameters
    ----------
    pipeline : fitted Pipeline
        Trained sklearn Pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str
        Name for plots and reports.
    save_dir : str
        Directory for saving figures.

    Returns
    -------
    dict
        Evaluation results including metrics and optimal threshold.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Plots
    safe_name = model_name.replace(" ", "_").lower()
    plot_confusion_matrix(
        y_test, y_pred, model_name,
        f"{save_dir}/confusion_matrix_{safe_name}.png"
    )
    plot_roc_curve(
        y_test, y_proba, model_name,
        f"{save_dir}/roc_curve_{safe_name}.png"
    )
    plot_pr_curve(
        y_test, y_proba, model_name,
        f"{save_dir}/pr_curve_{safe_name}.png"
    )
    plot_calibration(
        y_test, y_proba, model_name,
        f"{save_dir}/calibration_{safe_name}.png"
    )

    # Threshold optimization
    optimal_threshold, min_cost = optimize_threshold(y_test, y_proba)
    plot_threshold_analysis(
        y_test, y_proba,
        save_path=f"{save_dir}/threshold_analysis_{safe_name}.png"
    )

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        "model_name": model_name,
        "metrics": metrics,
        "optimal_threshold": optimal_threshold,
        "min_expected_cost": min_cost,
        "classification_report": report,
    }

    print(f"\n{'='*60}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  optimal_threshold: {optimal_threshold:.3f}")
    print(f"  min_expected_cost: ${min_cost:,.0f}")
    print(f"{'='*60}")

    return results


def compare_models(
    trained_models: Dict, X_test: pd.DataFrame, y_test: pd.Series,
    save_dir: str = "reports/figures"
) -> pd.DataFrame:
    """
    Compare all trained models on test set.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model.
    """
    results = []

    for name, pipeline in trained_models.items():
        eval_result = evaluate_model(pipeline, X_test, y_test, name, save_dir)
        eval_result["model_name"] = name
        results.append(eval_result)

    # Build comparison table
    comparison = pd.DataFrame([
        {"Model": r["model_name"], **r["metrics"],
         "Optimal_Threshold": r["optimal_threshold"]}
        for r in results
    ])

    # Save comparison
    comparison.to_csv(f"{save_dir}/model_comparison_test.csv", index=False)
    print(f"\nSaved: {save_dir}/model_comparison_test.csv")

    return comparison


if __name__ == "__main__":
    print("Run src/train.py first to train models, then use evaluate_model().")
