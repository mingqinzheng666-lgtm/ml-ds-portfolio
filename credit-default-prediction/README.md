# Credit Default Prediction — Classical ML vs. Deep Learning

Predicting whether a loan applicant will default, and testing a common assumption head-on: **does a neural network beat a classical ensemble on small, imbalanced tabular data?**

## Problem

Banks need to flag applicants likely to default, but the cost of errors is asymmetric — a **missed default (false negative)** causes direct financial loss and is far worse than wrongly declining a good customer. The [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (1,000 applicants, 16 mixed numerical/categorical features) is also **class-imbalanced (~70% non-default / 30% default)**, so a naive "always predict non-default" model scores 70% accuracy while learning nothing useful. The task was to build models that genuinely separate the two classes and evaluate them with metrics that respect the imbalance and the business cost of false negatives.

## Approach

- **Framed evaluation around imbalance, not raw accuracy.** Chose macro-F1, macro-recall and ROC-AUC as primary metrics, with a stratified train/test split to preserve class balance.
- **Built a clean preprocessing pipeline for mixed data types** — one-hot encoding for the 9 categorical columns, standard scaling for the 7 numerical columns, target encoded 1/0. Exploratory analysis first (class distribution, feature distributions by label, default rate per category) to see which features carried signal.
- **Non-deep-learning baseline: Random Forest** (100 trees, `class_weight='balanced'`) — robust to outliers, handles mixed features with little tuning, gives interpretable feature importances.
- **Deep-learning comparison: MLP** — feed-forward net (hidden layers 64 → 32, ReLU, Adam `lr=0.001`) with early stopping to prevent overfitting on the small dataset.
- **Compared and interpreted** with confusion matrices, ROC curves, a side-by-side metrics table, and RF feature importances tied back to the credit-risk context.

## Results

| Model | Accuracy | Macro-F1 | Macro-Recall | ROC-AUC |
|-------|:--------:|:--------:|:------------:|:-------:|
| **Random Forest** | **0.765** | **0.680** | **0.666** | **0.786** |
| MLP (neural net) | 0.710 | 0.618 | 0.612 | 0.733 |

- **Random Forest outperformed the MLP on every metric.** Both beat the 70% naive baseline / 0.5 random-AUC baseline.
- Confusion matrices exposed the classic credit-risk trade-off: both models identify non-defaulters better than defaulters, leaving residual **false-negative risk** — the costly error for a lender.
- Finding is consistent with the literature: on small structured/tabular data, **ensemble trees typically beat neural networks**, whose advantage needs more data and tuning.

## Tech stack

Python · scikit-learn (RandomForestClassifier, MLPClassifier) · pandas · NumPy · Matplotlib · seaborn

## Data

`German_bank.csv` — 1,000 rows × 17 columns (16 features + binary `default`), no missing values. Public dataset (Statlog German Credit); not committed to the repo. Download it and place it beside the notebook to reproduce.

## Run

```bash
pip install -r requirements.txt
jupyter notebook credit_default_prediction.ipynb
```
