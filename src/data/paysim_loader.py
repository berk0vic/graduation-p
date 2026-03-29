"""
PaySim dataset loader.
Kaggle: https://www.kaggle.com/ealaxi/paysim1
File expected in data/raw/paysim/:
    PS_20174392719_1491204439457_log.csv  (or renamed to paysim.csv)
Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data/raw/paysim"
FRAUD_TYPES = {"TRANSFER", "CASH_OUT"}
  # only these carry labeled fraud


def load_raw(filename: str = "paysim.csv") -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename)
    return df


def filter_fraud_types(df: pd.DataFrame) -> pd.DataFrame:
    """PaySim fraud only occurs in TRANSFER and CASH_OUT transactions."""
    return df[df["type"].isin(FRAUD_TYPES)].reset_index(drop=True)


def get_label_column() -> str:
    return "isFraud"
