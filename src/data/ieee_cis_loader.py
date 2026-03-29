"""
IEEE-CIS Fraud Detection dataset loader.
Kaggle: https://www.kaggle.com/c/ieee-fraud-detection
Files expected in data/raw/ieee_cis/:
    train_transaction.csv, train_identity.csv
    test_transaction.csv,  test_identity.csv
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ieee_cis"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge transaction + identity tables. Returns (train_df, test_df)."""
    train = pd.read_csv(RAW_DIR / "train_transaction.csv")
    train_id = pd.read_csv(RAW_DIR / "train_identity.csv")
    train = train.merge(train_id, on="TransactionID", how="left")

    test = pd.read_csv(RAW_DIR / "test_transaction.csv")
    test_id = pd.read_csv(RAW_DIR / "test_identity.csv")
    test = test.merge(test_id, on="TransactionID", how="left")

    return train, test


def get_label_column() -> str:
    return "isFraud"
