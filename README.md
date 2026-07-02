# Machine Learning & Data Science Portfolio

Selected machine learning and data science projects by **Mingqin (Jason) Zheng** — MSc Financial Technology with Data Science, University of Bristol.

Each project is self-contained in its own folder with a Jupyter notebook, a detailed README (problem → approach → results → tech), and a `requirements.txt`. The focus throughout is on **methodological rigour** — handling class imbalance, choosing honest evaluation metrics, stating hypotheses before evaluating, and interpreting *why* a model behaves as it does — rather than only chasing headline accuracy.

## Projects

| # | Project | Domain | Key techniques | Headline result |
|---|---------|--------|----------------|-----------------|
| 1 | [Credit Default Prediction](./credit-default-prediction) | Classical ML vs. Deep Learning | Random Forest, MLP, class weighting, ROC-AUC | RF ROC-AUC **0.786** > MLP 0.733 on imbalanced tabular data |
| 2 | [Financial News Sentiment (NLP)](./financial-news-sentiment) | NLP / Text classification | TF-IDF (uni+bigram), Logistic Regression | Accuracy **0.71**, macro-F1 0.58; full per-class error analysis |
| 3 | [NanoGPT — Transformer from Scratch](./nanogpt-transformer) | Deep Learning / LLMs | Self-attention, multi-head, causal masking (PyTorch) | 212K-param char-LM + controlled attention-head study |
| 4 | [Quantitative Portfolio Analysis](./quant-portfolio-analysis) | Quant Finance / Time Series | MVO, ARIMA-GARCH, Fama-French 3-factor, VaR/ES & stress testing | Max-utility Sharpe **0.517** vs 0.380; VaR backtest 5.2% at 5% level |

*More projects will be added over time.*

## Tech stack

Python · scikit-learn · PyTorch · statsmodels · arch · SciPy · pandas · NumPy · NLTK · Matplotlib · seaborn · Jupyter

## How to run

Each project runs independently. From a project folder:

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
jupyter notebook          # then open the .ipynb
```

Datasets are not committed (see each project's README for the source and how to obtain them).

## Contact

Mingqin (Jason) Zheng · mingqinzheng666@gmail.com · Bristol, UK
