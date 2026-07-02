# Quantitative Portfolio Analysis — Optimisation, Forecasting, Attribution & Risk

An end-to-end quantitative pipeline for a 5-stock equity portfolio (AAPL, DIS, FMC, KO, ORA; ~2,517 trading days, Jan 2013 – Dec 2022), taking it from raw prices to a risk-aware, evidence-backed allocation. Four analytical modules build on one another, each staying critical about what the models actually tell you.

## Problem

Moving a portfolio from an intuition-based allocation to a quantitatively justified one requires answering four connected questions: *how* should capital be allocated, *can* returns and volatility be forecast, *why* does the portfolio behave as it does, and *how bad* could it get in a crisis. A naive approach fails each: an unconstrained optimiser will happily put 73% of capital in one stock, daily returns are near-unpredictable under weak-form efficiency, and a single expected-return view ignores tail risk.

## Approach

**1 · Data cleaning (foundation).** Found 188 isolated NaNs (~7.5% of observations) across the five price series and chose **forward-fill over interpolation**, with explicit reasoning (last traded price is the best fair value; preserves autocorrelation for later ARIMA; interpolation would invent untraded prices). Documented the known side effect — artificial zero-return days slightly understate variance — as a bias flag into the covariance matrix.

**2 · Mean-Variance Optimisation.** Computed annualised μ and Σ from log returns; generated the efficient frontier via 10,000 Monte-Carlo Dirichlet weight draws; solved the **tangency** and **max-utility** portfolios with `scipy.optimize` (SLSQP) under no-shorting / full-investment constraints. Recommended the **max-utility** portfolio over the higher-Sharpe tangency one, arguing its ~33% higher volatility crushes utility at risk-aversion A=5 and that 73% in a single stock is uninvestable in practice.

**3 · Time-series forecasting (ARIMA + GARCH).** Honest 80/20 train/test split; confirmed stationarity with ADF (statistic −16.07); selected **ARIMA(0,0,3)** by ACF/PACF + AIC grid search; diagnosed residuals (Ljung–Box) and correctly identified **volatility clustering**, so fitted **GARCH(1,1)** for conditional variance; evaluated with rolling one-step-ahead forecasts across all five assets.

**4 · Fama-French 3-Factor attribution.** Ran FF3 OLS (`statsmodels`) per stock and at portfolio level, interpreting market/size/value loadings and reconciling them with the MVO weights — showing *why* the optimiser chose a large-cap, value-neutral, defensive tilt.

**5 · Risk & performance evaluation.** Measured **95% VaR / Expected Shortfall** two ways (historical vs. parametric), **backtested** rolling VaR, compared risk-adjusted performance (Sharpe / Sortino / max drawdown) against an equal-weight benchmark, and ran **stress tests** — a historical COVID crash and a designed scenario — with concrete mitigations.

## Results

- **MVO:** max-utility portfolio (AAPL 41.3% / KO 38.7% / ORA 20.0%) → 14.8% return, 18.9% vol, **Sharpe 0.517 vs. 0.380** equal-weight at essentially the same volatility, and far better diversified than the tangency portfolio's 73.4% single-stock concentration.
- **Forecasting:** KO out-of-sample **directional accuracy 49%** (mean is unpredictable, exactly as weak-form efficiency predicts), but the **GARCH variance forecast tracks realised volatility well** — value lives in risk/position-sizing, not return prediction. GARCH persistence > 0.92 across all five stocks (volatility clustering is portfolio-wide).
- **Factor attribution:** portfolio β_market 0.864, β_SMB −0.179, β_HML −0.064, annual α ≈ 6.7%, R² 0.394 — a coherent large-cap, value-neutral profile that explains the MVO weights.
- **Risk:** 95% VaR/ES historical −1.77% / −2.87%; VaR backtest **131 breaches in 2,517 days = 5.20%** (well-calibrated at the 5% level); COVID stress −38.8%, designed scenario −23.2%; max-utility beats equal-weight on Sharpe, Sortino and max drawdown (−34.2% vs. −36.9%).

## Tech stack

Python · NumPy · pandas · SciPy (`optimize`) · statsmodels (OLS, ARIMA) · arch (GARCH) · Matplotlib · seaborn

## Data

Daily adjusted prices for AAPL, DIS, FMC, KO, ORA (Jan 2013 – Dec 2022) and Fama-French daily factors (public, Kenneth French Data Library). Data files are not committed — see the imports section of the notebook for the expected files.

## Run

```bash
pip install -r requirements.txt
jupyter notebook quant_portfolio_analysis.ipynb
```
