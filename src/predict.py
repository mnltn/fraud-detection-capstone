"""
Score transactions with the saved fraud-detection models.

Usage (run from the repository root):

    # Score a CSV of transactions (PaySim format) and save predictions
    python src/predict.py --input data/new_transactions.csv --output predictions.csv

    # Reproduce the notebook's test-set results for a model
    python src/predict.py --input data/PS_20174392719_1491204439457_log.csv --test-split

Options:
    --model      random_forest (default), xgboost, or logistic_regression
    --threshold  probability cut-off for flagging fraud (default 0.5)
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score

from features import build_X, train_test_data, TARGET

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_FILES = {
    'random_forest': 'random_forest.joblib',
    'xgboost': 'xgboost.joblib',
    'logistic_regression': 'logistic_regression.joblib',
}


def load_model(name: str):
    model = joblib.load(MODELS_DIR / MODEL_FILES[name])
    # Logistic Regression was trained on scaled features; the tree models were not.
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib') if name == 'logistic_regression' else None
    return model, scaler


def predict_proba(model, scaler, X: pd.DataFrame):
    X_in = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(X_in)[:, 1]


def report(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    print(f"Precision: {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_true, proba):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='CSV of transactions in PaySim format')
    parser.add_argument('--output', help='where to save predictions (CSV)')
    parser.add_argument('--model', default='random_forest', choices=MODEL_FILES)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test-split', action='store_true',
                        help="evaluate on the notebook's 30%% test split instead of all rows")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model, scaler = load_model(args.model)

    if args.test_split:
        _, X_test, _, y_test = train_test_data(df)
        print(f"{args.model} on the test split ({len(X_test):,} rows, {int(y_test.sum())} fraud):")
        report(y_test, predict_proba(model, scaler, X_test), args.threshold)
        return

    proba = predict_proba(model, scaler, build_X(df))
    out = pd.DataFrame({
        'fraud_probability': proba,
        'fraud_flag': (proba >= args.threshold).astype(int),
    }, index=df.index)
    print(f"Scored {len(df):,} transactions; flagged {int(out['fraud_flag'].sum()):,} as fraud.")

    if TARGET in df.columns:
        print("Labels found in input — metrics on these rows:")
        report(df[TARGET], proba, args.threshold)

    if args.output:
        pd.concat([df, out], axis=1).to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()
