# Financial News Sentiment Classification (NLP)

An interpretable NLP classifier for financial-news sentiment, built to be **diagnosed rigorously** — the goal was as much about understanding *where and why the model fails* as about the headline score.

## Problem

Financial-news sentiment is a valuable trading and risk signal, but hard to classify: sentiment is context-dependent, the language is domain-specific, and the same vocabulary can be positive, negative, or neutral. The [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) dataset (5,842 sentences) is heavily imbalanced — **53.6% neutral, 31.7% positive, 14.7% negative** — so the negative class, the most decision-relevant for risk, is also the hardest to learn.

## Approach

- **Finance-aware text preprocessing.** Lowercasing, punctuation/number removal, NLTK stopword removal — but deliberately **retained negation/direction words** ("not", "no", "down", "up") because they carry sentiment in finance. Lemmatization to shrink the vocabulary. Reasoned explicitly about the trade-off: a bag-of-words still loses word order, so "not profitable" vs. "profitable" stays hard.
- **TF-IDF with unigrams + bigrams** (`max_features=10000`, `ngram_range=(1,2)`, `min_df=2`) — down-weights common words, up-weights discriminative terms, and bigrams capture phrases like "profit warning" / "record high" to partially handle negation.
- **Interpretable classifier: multinomial Logistic Regression** (`C=1.0`) — works well on high-dimensional sparse text and its coefficients directly show which words push each prediction. Vectorizer fit on training data only, to avoid leakage.
- **Stated a hypothesis before evaluating** — predicted preprocessing + bigrams would help but that the neutral class would stay confusable (it shares vocabulary with both other classes), making the evaluation a genuine test rather than a post-hoc story.
- **Metric, error, and feature analysis** — macro-F1 headline metric, per-class precision/recall, confusion matrix, per-class top-15 TF-IDF features, and manual inspection of misclassified examples.

## Results

- Overall **accuracy 0.713**, **macro-F1 0.58**.
- Per class: **Neutral F1 0.79** (recall 0.88 — easiest), **Positive F1 0.73**, **Negative F1 0.21** (recall 0.15 — hardest) — confirming the hypothesis.
- Most-informative features were semantically coherent (bigrams like "decreased eur" / "rose eur" cleanly separate negative vs. positive), validating the preprocessing and TF-IDF choice.
- **Error analysis:** most errors sit between neutral and the other classes, driven by factual sentences with implicit numeric polarity (e.g. "profit … down eur mn") that a bag-of-words can't contextualise.
- **Next steps identified:** fine-tune a pre-trained language model (BERT / FinBERT) to capture context and word order; address imbalance via oversampling or class weights.

## Tech stack

Python · scikit-learn (TfidfVectorizer, LogisticRegression) · NLTK · pandas · NumPy · Matplotlib · seaborn

## Data

`data.csv` — Financial PhraseBank, 5,842 labelled sentences (`Sentiment`, `Sentence`). Public dataset; not committed. Download and place beside the notebook to reproduce.

## Run

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
jupyter notebook financial_news_sentiment.ipynb
```
