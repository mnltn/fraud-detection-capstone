"""
Feature engineering for the PaySim fraud-detection capstone.

Reproduces the exact feature pipeline used in the analysis notebook
(notebooks/Emmanuel Tan Jr._AIM PGAIML Capstone Project.ipynb), so the
saved models in models/ can be reused without re-running the notebook.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Final feature set selected in Step 3 (order matters: it matches the saved models)
FINAL_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrig', 'errorBalanceDest', 'type_TRANSFER', 'type_CASH_OUT',
    'isMerchantDest',
]

TARGET = 'isFraud'
RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns to a raw PaySim-format DataFrame.

    - errorBalanceOrig: oldbalanceOrg - amount - newbalanceOrig
      (how far the origin account's ending balance is from what the amount implies)
    - errorBalanceDest: oldbalanceDest + amount - newbalanceDest
      (same check on the receiving side; the sign flips because money arrives)
    - isMerchantDest: destination account is a merchant ('M' prefix)
    - type_TRANSFER / type_CASH_OUT: one-hot flags for the only fraud-bearing types
    """
    out = df.copy()
    out['errorBalanceOrig'] = out['oldbalanceOrg'] - out['amount'] - out['newbalanceOrig']
    out['errorBalanceDest'] = out['oldbalanceDest'] + out['amount'] - out['newbalanceDest']
    out['isMerchantDest'] = out['nameDest'].str[0] == 'M'
    # Built directly (not with get_dummies) so both columns always exist,
    # even if a new batch of transactions contains no TRANSFER or CASH_OUT rows.
    out['type_TRANSFER'] = out['type'] == 'TRANSFER'
    out['type_CASH_OUT'] = out['type'] == 'CASH_OUT'
    return out


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and return the model-ready matrix (float32, fixed column order)."""
    return engineer_features(df)[FINAL_FEATURES].astype('float32')


def train_test_data(df: pd.DataFrame):
    """Recreate the notebook's stratified 70/30 split exactly (random_state=42)."""
    X = build_X(df)
    y = df[TARGET].astype('int8')
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
