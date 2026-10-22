"""
Data loading and acquisition module.

Handles:
- Loading startup data from CSV
- Schema validation
- Target encoding (binary: acquired = 1, closed = 0)
- Date parsing and cleaning
- Column mapping to unified schema
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional
import yaml


# Expected columns (flexible — adapts to available data)
EXPECTED_COLUMNS = [
    "name", "category_code", "funding_total_usd", "funding_rounds",
    "founded_at", "first_funding_at", "last_funding_at",
    "state_code", "city", "status",
    "relationships", "milestones", "has_VC", "has_angel",
    "has_roundA", "has_roundB", "has_roundC", "has_roundD",
    "avg_participants", "is_top500"
]

SUCCESS_STATUSES = {"acquired", "ipo", "acquired by", "went public"}
FAILURE_STATUSES = {"closed", "ceased", "dead", "out of business"}
ONGOING_STATUSES = {"operating", "live", "active", "private"}


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw startup data from CSV.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw startup data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Download from Kaggle: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction"
        )

    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path}")
    return df


def encode_target(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    """
    Encode the target variable into binary: funding_success.

    1 = acquired or IPO (success)
    0 = closed (failure)
    NaN = operating (excluded from binary task)

    Parameters
    ----------
    df : pd.DataFrame
        Input data with status column.
    status_col : str
        Column name containing startup status.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'funding_success' binary target column.
    """
    df = df.copy()

    # Normalize status strings
    status_normalized = df[status_col].astype(str).str.lower().str.strip()

    def classify_status(s):
        """Classify a status string into success/failure/ongoing."""
        for kw in SUCCESS_STATUSES:
            if kw in s:
                return 1
        for kw in FAILURE_STATUSES:
            if kw in s:
                return 0
        for kw in ONGOING_STATUSES:
            if kw in s:
                return np.nan
        return np.nan  # Unknown

    df["funding_success"] = status_normalized.apply(classify_status)

    n_success = (df["funding_success"] == 1).sum()
    n_failure = (df["funding_success"] == 0).sum()
    n_ongoing = df["funding_success"].isna().sum()

    print(f"Target encoding complete:")
    print(f"  Success (acquired/IPO): {n_success} ({n_success/(n_success+n_failure)*100:.1f}%)")
    print(f"  Failure (closed):       {n_failure} ({n_failure/(n_success+n_failure)*100:.1f}%)")
    print(f"  Ongoing (excluded):     {n_ongoing}")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns into datetime objects.

    Handles various date formats found in Crunchbase data.
    """
    date_cols = ["first_funding_at", "last_funding_at", "founded_at", "closed_at"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    return df


def extract_founded_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract founded_year from founded_at date column.
    """
    df = df.copy()
    if "founded_at" in df.columns:
        df["founded_year"] = pd.to_datetime(df["founded_at"], errors="coerce", format="mixed").dt.year
    return df


def clean_funding_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the funding_total_usd column.

    Handles:
    - String values like "1,000,000"
    - Placeholder values like " - " or "nan"
    - Converts to numeric
    """
    if "funding_total_usd" in df.columns:
        # Replace common placeholder strings
        placeholders = [" - ", "nan", "None", "N/A", "-", ""]
        df["funding_total_usd"] = df["funding_total_usd"].astype(str)
        df["funding_total_usd"] = df["funding_total_usd"].str.replace(",", "")
        df["funding_total_usd"] = df["funding_total_usd"].replace(placeholders, np.nan)
        df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    return df


def add_description_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic description column from name and category_code
    since the dataset doesn't have an explicit description field.
    """
    df = df.copy()
    parts = []
    if "name" in df.columns:
        parts.append(df["name"].astype(str))
    if "category_code" in df.columns:
        parts.append(df["category_code"].astype(str) + " startup")

    if parts:
        df["description"] = parts[0] + " - " + parts[1] if len(parts) > 1 else parts[0]
        df["description"] = df["description"].fillna("Startup company")
    else:
        df["description"] = "Startup company"

    return df


def load_and_preprocess(file_path: str, config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Full data loading and initial preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV.
    config_path : str
        Path to config YAML.

    Returns
    -------
    pd.DataFrame
        Preprocessed data with binary target.
    """
    df = load_raw_data(file_path)
    df = clean_funding_total(df)
    df = parse_dates(df)
    df = extract_founded_year(df)
    df = add_description_column(df)
    df = encode_target(df)

    # Drop rows with unknown target (ongoing startups)
    df_model = df.dropna(subset=["funding_success"]).copy()
    df_model["funding_success"] = df_model["funding_success"].astype(int)

    print(f"\nModeling dataset: {len(df_model)} rows (ongoing excluded)")
    print(f"  Features available: {len(df_model.columns)}")

    return df_model


if __name__ == "__main__":
    config = load_config()
    df = load_and_preprocess(config["dataset"]["file"])
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nTarget distribution:\n{df['funding_success'].value_counts()}")
