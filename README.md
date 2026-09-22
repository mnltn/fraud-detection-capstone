# AI-Powered Fraud Detection — Capstone Project

End-to-end machine learning capstone (Postgraduate Diploma in AI & ML) building a fraud-detection
model on the PaySim synthetic mobile-money transaction dataset — problem framing through model
implementation, explainability, and a bias/fairness audit.

## Problem Statement

Manual, rule-based fraud review cannot scale to millions of daily transactions. The existing
rule-based flag in this dataset (`isFlaggedFraud`) catches only **16 of 8,213** fraud cases — a
0.19% catch rate. This project builds a model that flags high-risk transactions for review before
payout completes.

- **Task type:** Binary classification (fraud vs. legitimate)
- **Primary metrics:** PR-AUC and Recall on the fraud class (not Accuracy — fraud is only 0.129% of
  transactions, so a model predicting "legitimate" every time would be 99.87% "accurate" while
  catching zero fraud)
- **Business KPI:** Fraud $ prevented, net of the cost of reviewing false-positive flags

## Dataset

[PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic mobile-money transaction logs
modelled on one month of real financial data from an African mobile money service. 6.36M
transactions, 11 columns, 0.129% fraud rate.

**The raw CSV (~470MB) is not included in this repo** — GitHub blocks files over 100MB. Download it
directly from the Kaggle link above and place it as `data/PS_20174392719_1491204439457_log.csv`
before running the notebook.

## Repository Structure

```
├── src/              Reusable scripts: features.py (feature engineering), predict.py (scoring)
├── notebooks/        Full analysis notebook (Steps 1-5: problem framing through
│                      explainability & fairness audit)
├── models/            Trained Logistic Regression, Random Forest, XGBoost + scaler (joblib)
├── data/              Data dictionary (raw CSV downloaded separately — see above)
├── presentations/     Technical deck (Jupyter slides + PDF) and business deck (PDF)
└── requirements.txt   Python dependencies
```

## Presentations

- **Technical presentation (Jupyter slides):** [view in browser](https://mnltn.github.io/fraud-detection-capstone/presentations/Emmanuel%20Tan%20Jr._AIM%20PGAIML%20Capstone%20Project_Technical%20Presentation.html),
  or download the `.html` file from `presentations/` and open it in any browser (arrow keys to navigate).
  Source notebook is in the same folder.
- **Technical presentation (PDF)** and **Business presentation (PDF)** — in `presentations/`.

## Key Results

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.023 | 0.940 | 0.544 |
| Random Forest | 1.000 | 0.996 | 0.998 |
| XGBoost (corrected) | 0.999 | 0.996 | 0.997 |

**Two findings matter more than the headline numbers above:**

1. **Performance is substantially an artifact of this dataset's simulation design.** Removing two
   engineered balance-mismatch features (`errorBalanceOrig`, `errorBalanceDest`) collapses PR-AUC
   from 0.997 to ~0.126. PaySim forces the origin balance to exactly 0 on fraudulent transfers,
   making that feature close to a deterministic fraud indicator by construction — not necessarily
   because real-world fraud leaves this clean a trace. This caveat applies to every metric above.
2. **A real fairness gap by account balance.** Recall for the lowest-balance customer quartile is
   0.30, versus ~0.99-1.00 for every other tier — precision stays at 1.0 in every tier, meaning this
   is under-protection of lower-balance customers, not over-flagging. See the notebook's fairness
   section and the limitations writeup for the (partially-confirmed) mechanism and proposed
   mitigations.

## Using the Saved Models (src/)

Install dependencies, then run from the repository root:

```bash
pip install -r requirements.txt

# Reproduce the notebook's test-set results (Random Forest: Precision 1.000, Recall 0.996, PR-AUC 0.998)
python src/predict.py --input data/PS_20174392719_1491204439457_log.csv --test-split

# Score new transactions (PaySim format) and save fraud probabilities + flags
python src/predict.py --input data/new_transactions.csv --output predictions.csv
```

Options: `--model random_forest | xgboost | logistic_regression` and `--threshold 0.5`.

## Reproducing This

1. Open `notebooks/` in Google Colab.
2. Download the dataset from Kaggle (link above) and upload it, or mount your own Google Drive and
   adjust the file paths at the top of the notebook (they currently point to the author's own
   `capstone/data/` and `capstone/models/` Drive folders).
3. Run top to bottom. Model fitting (especially Random Forest) takes several minutes.

## Tech Stack

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`

## Limitations

Full detail in the notebook; summarized:
1. No reliable existing baseline (the current rule-based system catches ~0.19% of fraud)
2. Feature selection included a subjective, domain-knowledge-based judgment call
3. Model performance is sensitive to hyperparameter choice (`scale_pos_weight`)
4. Performance is substantially dependent on two engineered features specific to this dataset
5. Partial dependence plots cannot detect interaction effects
6. Synthetic data limits external validity to real-world transaction patterns

## Author

Emmanuel Tan — Postgraduate Diploma in AI & ML, Capstone Project
