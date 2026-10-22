"""
Data validation module.

Performs comprehensive data quality checks:
- Missing values
- Duplicates
- Invalid ranges
- Inconsistent categories
- Date sanity checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for missing values and flag columns above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    threshold : float
        Fraction of missing values above which to warn.

    Returns
    -------
    dict
        Column -> missing fraction.
    """
    missing = (df.isnull().sum() / len(df)).sort_values(ascending=False)
    missing = missing[missing > 0].to_dict()

    for col, frac in missing.items():
        if frac > threshold:
            warnings.warn(f"Column '{col}' has {frac:.1%} missing values — consider dropping")

    return missing


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """
    Check for duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    subset : list, optional
        Column subset for duplicate detection.

    Returns
    -------
    int
        Number of duplicates found.
    """
    n_duplicates = df.duplicated(subset=subset).sum()
    if n_duplicates > 0:
        warnings.warn(f"{n_duplicates} duplicate rows found (subset={subset})")
    return n_duplicates


def check_funding_ranges(df: pd.DataFrame) -> Dict[str, Tuple]:
    """
    Validate funding_total_usd and funding_rounds ranges.

    Returns
    -------
    dict
        Column -> (min, max) tuple.
    """
    ranges = {}

    if "funding_total_usd" in df.columns:
        ft = df["funding_total_usd"].dropna()
        if len(ft) > 0:
            min_val, max_val = ft.min(), ft.max()
            ranges["funding_total_usd"] = (min_val, max_val)
            if min_val < 0:
                warnings.warn(f"Negative funding_total_usd found: {min_val}")
            if max_val > 1e10:
                warnings.warn(f"Suspiciously high funding_total_usd: {max_val}")

    if "funding_rounds" in df.columns:
        fr = df["funding_rounds"].dropna()
        if len(fr) > 0:
            min_val, max_val = fr.min(), fr.max()
            ranges["funding_rounds"] = (min_val, max_val)
            if min_val < 0:
                warnings.warn(f"Negative funding_rounds found: {min_val}")
            if max_val > 100:
                warnings.warn(f"High funding_rounds count: {max_val}")

    return ranges


def check_date_consistency(df: pd.DataFrame) -> Dict[str, int]:
    """
    Verify date logic:
    - founded_year is reasonable (1990-current)
    - first_funding_at >= founded date
    - last_funding_at >= first_funding_at

    Returns
    -------
    dict
        Check name -> number of violations.
    """
    violations = {}

    if "founded_year" in df.columns:
        current_year = pd.Timestamp.now().year
        bad_year = ((df["founded_year"] < 1990) | (df["founded_year"] > current_year)).sum()
        violations["invalid_founded_year"] = int(bad_year)

    if "first_funding_at" in df.columns and "last_funding_at" in df.columns:
        both_dates = df.dropna(subset=["first_funding_at", "last_funding_at"])
        bad_order = (both_dates["last_funding_at"] < both_dates["first_funding_at"]).sum()
        violations["last_before_first_funding"] = int(bad_order)

    if "founded_year" in df.columns and "first_funding_at" in df.columns:
        df_temp = df.copy()
        df_temp["founded_date"] = pd.to_datetime(
            df_temp["founded_year"].astype(str) + "-01-01", errors="coerce"
        )
        both = df_temp.dropna(subset=["founded_date", "first_funding_at"])
        bad_timing = (both["first_funding_at"] < both["founded_date"] - pd.Timedelta(days=365)).sum()
        violations["funding_before_founding"] = int(bad_timing)

    for check, count in violations.items():
        if count > 0:
            warnings.warn(f"Date consistency check '{check}': {count} violations")

    return violations


def check_categorical_consistency(df: pd.DataFrame, col: str, valid_values: List[str]) -> int:
    """
    Check categorical column for unexpected values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col : str
        Column name to check.
    valid_values : list
        Expected category values.

    Returns
    -------
    int
        Number of rows with invalid categories.
    """
    if col not in df.columns:
        return 0

    invalid = ~df[col].dropna().isin(valid_values)
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        warnings.warn(f"Column '{col}': {n_invalid} rows with unexpected values")
    return n_invalid


def check_target_distribution(df: pd.DataFrame, target_col: str = "funding_success") -> Dict:
    """
    Analyse target distribution and class balance.

    Returns
    -------
    dict
        Distribution stats.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found"}

    counts = df[target_col].value_counts().to_dict()
    total = sum(counts.values())

    stats = {
        "total_samples": total,
        "class_counts": counts,
        "class_fractions": {k: v / total for k, v in counts.items()},
        "imbalance_ratio": max(counts.values()) / min(counts.values()) if counts else 0,
    }

    if stats["imbalance_ratio"] > 3:
        warnings.warn(
            f"Class imbalance detected: ratio = {stats['imbalance_ratio']:.1f}:1. "
            f"Consider class_weight='balanced' or resampling."
        )

    return stats


def validate_all(df: pd.DataFrame, target_col: str = "funding_success") -> Dict:
    """
    Run all validation checks.

    Returns
    -------
    dict
        Complete validation report.
    """
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df),
        "funding_ranges": check_funding_ranges(df),
        "date_consistency": check_date_consistency(df),
        "target_distribution": check_target_distribution(df, target_col),
    }

    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Rows: {report['n_rows']:,}")
    print(f"Columns: {report['n_columns']}")
    print(f"Missing values in {len(report['missing_values'])} columns")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Date violations: {sum(report['date_consistency'].values())}")
    print(f"Imbalance ratio: {report['target_distribution'].get('imbalance_ratio', 'N/A'):.1f}:1")
    print("=" * 60)

    return report


if __name__ == "__main__":
    from src.data_loader import load_and_preprocess, load_config

    config = load_config()
    df = load_and_preprocess(config["dataset"]["file"])
    validate_all(df)
