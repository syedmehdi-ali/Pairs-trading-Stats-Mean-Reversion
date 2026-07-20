# Statistical Pairs Trading Strategy — KO / PEP

A mean-reversion pairs trading strategy using **Engle-Granger cointegration**, rolling z-score signals, and full historical backtesting. Built with Python.

---

## What This Project Does

Coca-Cola (KO) and PepsiCo (PEP) compete in the same market, so their stock prices tend to move together. When they temporarily diverge, this strategy bets on them converging again — a classic **statistical arbitrage** approach used by quantitative hedge funds.

The pipeline:

1. **Cointegration testing** — statistically verifies the pair has a stable long-run relationship (Engle-Granger two-step method)
2. **Spread construction** — models the price gap using OLS regression to find the hedge ratio
3. **Z-score signal generation** — triggers trades when the spread deviates beyond ±2σ from its rolling mean
4. **Backtesting** — simulates the strategy on out-of-sample data with proper train/test separation
5. **Performance analysis** — computes Sharpe ratio, max drawdown, cumulative returns, and trade count

## Sample Output

| Metric | In-Sample | Out-of-Sample |
|---|---|---|
| Total Return | varies | varies |
| Sharpe Ratio | varies | varies |
| Max Drawdown | varies | varies |

> Results depend on the date range and market conditions. The project is designed to demonstrate methodology, not guarantee returns.

### Generated Charts

The script produces five publication-quality charts in `./output/`:

| Chart | Description |
|---|---|
| `1_prices.png` | KO vs PEP price history with train/test split |
| `2_spread_zscore.png` | Spread and z-score with entry/exit thresholds |
| `3_positions.png` | Long/short position timeline |
| `4_cumulative_returns.png` | Strategy vs buy-and-hold with key metrics |
| `5_drawdown.png` | Underwater plot showing drawdown from peak |

## How It Works

### Cointegration (Why This Pair?)

Two stocks are *cointegrated* if a linear combination of them is stationary — meaning the spread between them fluctuates around a constant mean rather than drifting permanently. The **Engle-Granger test** checks this:

1. Regress KO on PEP via OLS → get the hedge ratio β
2. Compute residuals: `spread = KO − β·PEP − α`
3. Run an Augmented Dickey-Fuller test on the residuals → if p < 0.05, the pair is cointegrated

### Trading Rules

| Condition | Action |
|---|---|
| z-score > +2.0 | **Short the spread** (short KO, long PEP) |
| z-score < −2.0 | **Long the spread** (long KO, short PEP) |
| \|z-score\| < 0.5 | **Close position** (spread has reverted) |

The strategy is **dollar-neutral**: equal capital deployed on each side, so overall market direction doesn't matter.

### Train / Test Split

Parameters (hedge ratio, intercept) are estimated on the **first 60%** of data only. Signals and backtesting are then evaluated on the **remaining 40%** to demonstrate out-of-sample validity — the same approach used in professional quant research.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/pairs-trading-strategy.git
cd pairs-trading-strategy

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python pairs_trading.py
```

Charts and metrics are saved to `./output/`.

## Configuration

All parameters are at the top of `pairs_trading.py`:

```python
STOCK_A    = "KO"           # first ticker
STOCK_B    = "PEP"          # second ticker
START_DATE = "2018-01-01"   # data start
END_DATE   = "2025-12-31"   # data end
Z_ENTRY    = 2.0            # open position threshold
Z_EXIT     = 0.5            # close position threshold
LOOKBACK   = 60             # rolling z-score window (days)
```

You can swap in any pair you suspect is cointegrated (e.g. XOM/CVX, GLD/GDX) — the script will run the cointegration test and report whether the relationship holds.

## Project Structure

```
pairs-trading-strategy/
├── pairs_trading.py     # Full pipeline: data → analysis → backtest → charts
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── output/              # Generated charts and metrics (after running)
    ├── 1_prices.png
    ├── 2_spread_zscore.png
    ├── 3_positions.png
    ├── 4_cumulative_returns.png
    ├── 5_drawdown.png
    └── metrics.csv
```

## Tech Stack

- **yfinance** — historical market data
- **statsmodels** — Engle-Granger cointegration & OLS regression
- **NumPy / Pandas** — numerical computation & data wrangling
- **Matplotlib** — publication-quality visualisations

## Key Concepts for Interviewers

- **Cointegration ≠ Correlation.** Correlation measures co-movement direction; cointegration measures whether a linear combination is mean-reverting. Two stocks can have low correlation but high cointegration.
- **Dollar-neutral.** Equal capital on each leg means the strategy's P&L comes from the *spread*, not from market direction. This is a form of hedging.
- **Out-of-sample testing.** Parameters are fit on training data only. Evaluating on unseen data guards against overfitting — a critical distinction in quantitative finance.
- **Z-score as a signal.** Standardising the spread makes thresholds transferable across pairs and time periods, rather than relying on absolute dollar values.

## References

- Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction*. Econometrica.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
- [QuantStart — Pairs Trading Guide](https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/)

## Disclaimer

This project is for **educational purposes only**. It is not financial advice. Past performance does not indicate future results. Always do your own research before trading.

---

*Built as part of the FinTech course project — [UMASS AMHERST], 2026.*
