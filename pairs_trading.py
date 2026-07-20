"""
Statistical Pairs Trading Strategy — Coca-Cola (KO) & PepsiCo (PEP)
====================================================================
A mean-reversion pairs trading strategy using cointegration analysis,
z-score signals, and full historical backtesting.

Author: [Your Name]
Date:   July 2026
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime
import os

# ── Configuration ────────────────────────────────────────────────────────────

STOCK_A       = "KO"          # Coca-Cola
STOCK_B       = "PEP"         # PepsiCo
START_DATE    = "2018-01-01"
END_DATE      = "2025-12-31"
TRAIN_SPLIT   = 0.6           # 60% training, 40% out-of-sample
Z_ENTRY       = 2.0           # open a position when |z| > 2
Z_EXIT        = 0.5           # close the position when |z| < 0.5
LOOKBACK      = 60            # rolling window for z-score (trading days)
CAPITAL       = 100_000       # notional capital per leg

OUTPUT_DIR    = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_pair(ticker_a: str, ticker_b: str, start: str, end: str) -> pd.DataFrame:
    """
    Generate realistic cointegrated price series for demonstration.
    Uses a shared stochastic trend + individual noise, which is exactly
    how cointegrated pairs behave in the real market.
    """
    print("    ⚠  yfinance unavailable — using synthetic cointegrated data for demo.\n")
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Shared random walk (the common trend)
    common_shocks = np.random.normal(0.0003, 0.012, n)
    common_trend  = 50 + np.cumsum(common_shocks)

    # Stock A ≈ 55-65 range (KO-like)
    noise_a = np.cumsum(np.random.normal(0, 0.004, n))
    price_a = common_trend * 1.15 + noise_a + 5

    # Stock B ≈ 150-180 range (PEP-like)
    noise_b = np.cumsum(np.random.normal(0, 0.004, n))
    price_b = common_trend * 3.0 + noise_b + 15

    # Inject two mean-reverting divergence events for realism
    for centre in [n // 3, 2 * n // 3]:
        bump = np.zeros(n)
        width = 40
        start_i = max(centre - width, 0)
        end_i   = min(centre + width, n)
        bump[start_i:end_i] = 3.0 * np.sin(np.linspace(0, np.pi, end_i - start_i))
        price_a += bump

    prices = pd.DataFrame({ticker_a: price_a, ticker_b: price_b}, index=dates)
    return prices


def fetch_data(ticker_a: str, ticker_b: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for both tickers (falls back to synthetic data)."""
    print(f"📥  Fetching {ticker_a} and {ticker_b} data ({start} → {end})...")
    try:
        raw = yf.download([ticker_a, ticker_b], start=start, end=end, auto_adjust=True)
        prices = raw["Close"][[ticker_a, ticker_b]].dropna()
        if len(prices) < 100:
            raise ValueError("Insufficient data returned")
        print(f"    ✓ {len(prices)} trading days retrieved\n")
        return prices
    except Exception:
        prices = _generate_synthetic_pair(ticker_a, ticker_b, start, end)
        print(f"    ✓ {len(prices)} synthetic trading days generated\n")
        return prices


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COINTEGRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Run the Engle-Granger two-step cointegration test."""
    print("🔬  Running Engle-Granger cointegration test...")

    # Step 1: OLS regression  →  series_a = α + β·series_b + ε
    X = add_constant(series_b)
    model = OLS(series_a, X).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]
    residuals = model.resid

    # Step 2: ADF test on the residuals
    adf_stat, adf_pvalue, *_ = adfuller(residuals, maxlag=1)

    # Also run the statsmodels shortcut for comparison
    coint_stat, coint_pvalue, crit_values = coint(series_a, series_b)

    results = {
        "hedge_ratio":   hedge_ratio,
        "intercept":     intercept,
        "residuals":     residuals,
        "adf_stat":      adf_stat,
        "adf_pvalue":    adf_pvalue,
        "coint_stat":    coint_stat,
        "coint_pvalue":  coint_pvalue,
        "crit_values":   dict(zip(["1%", "5%", "10%"], crit_values)),
    }

    print(f"    Hedge ratio (β):       {hedge_ratio:.4f}")
    print(f"    ADF statistic:         {adf_stat:.4f}")
    print(f"    ADF p-value:           {adf_pvalue:.6f}")
    print(f"    Cointegration p-value: {coint_pvalue:.6f}")
    status = "✓ COINTEGRATED" if coint_pvalue < 0.05 else "✗ NOT cointegrated"
    print(f"    Result (α=0.05):       {status}\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPREAD & Z-SCORE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spread(prices: pd.DataFrame, hedge_ratio: float, intercept: float) -> pd.DataFrame:
    """Build the spread and rolling z-score."""
    df = prices.copy()
    df["spread"] = df[STOCK_A] - hedge_ratio * df[STOCK_B] - intercept
    df["spread_mean"] = df["spread"].rolling(LOOKBACK).mean()
    df["spread_std"]  = df["spread"].rolling(LOOKBACK).std()
    df["z_score"]     = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    return df.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL GENERATION & BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trading rules (dollar-neutral):
      • z > +Z_ENTRY  →  SHORT the spread  (short A, long B)
      • z < -Z_ENTRY  →  LONG  the spread  (long A,  short B)
      • |z| < Z_EXIT  →  CLOSE the position
    """
    df = df.copy()
    df["position"] = 0  # +1 = long spread, -1 = short spread

    pos = 0
    positions = []
    for z in df["z_score"]:
        if pos == 0:
            if z > Z_ENTRY:
                pos = -1   # spread is too wide → short it
            elif z < -Z_ENTRY:
                pos = 1    # spread is too narrow → long it
        else:
            if abs(z) < Z_EXIT:
                pos = 0    # revert to mean → close
        positions.append(pos)

    df["position"] = positions
    return df


def backtest(df: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
    """Compute daily P&L from position changes."""
    df = df.copy()

    # Daily returns for each leg
    df["ret_A"] = df[STOCK_A].pct_change()
    df["ret_B"] = df[STOCK_B].pct_change()

    # Strategy return: position × (ret_A - β·ret_B), dollar-neutral
    df["strategy_ret"] = df["position"].shift(1) * (df["ret_A"] - hedge_ratio * df["ret_B"])
    df["strategy_ret"] = df["strategy_ret"].fillna(0)

    # Cumulative P&L
    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
    df["cum_A"]        = (1 + df["ret_A"]).cumprod()
    df["cum_B"]        = (1 + df["ret_B"]).cumprod()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def performance_summary(df: pd.DataFrame, label: str = "Strategy") -> dict:
    """Compute key risk-return metrics."""
    rets = df["strategy_ret"].dropna()
    trading_days = 252

    total_return   = df["cum_strategy"].iloc[-1] - 1
    annual_return  = (1 + total_return) ** (trading_days / len(rets)) - 1
    annual_vol     = rets.std() * np.sqrt(trading_days)
    sharpe         = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cum = df["cum_strategy"]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # Trade count
    trades = (df["position"].diff().abs() > 0).sum()

    metrics = {
        "label":            label,
        "total_return":     total_return,
        "annual_return":    annual_return,
        "annual_vol":       annual_vol,
        "sharpe_ratio":     sharpe,
        "max_drawdown":     max_dd,
        "num_trades":       int(trades),
        "trading_days":     len(rets),
    }

    print(f"📊  {label} Performance")
    print(f"    Total return:      {total_return:+.2%}")
    print(f"    Annualised return: {annual_return:+.2%}")
    print(f"    Annualised vol:    {annual_vol:.2%}")
    print(f"    Sharpe ratio:      {sharpe:.2f}")
    print(f"    Max drawdown:      {max_dd:.2%}")
    print(f"    Number of trades:  {trades}")
    print()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "blue":   "#1a6fdf",
    "red":    "#e8423f",
    "green":  "#2ca02c",
    "orange": "#ff7f0e",
    "grey":   "#7f7f7f",
    "bg":     "#fafafa",
    "grid":   "#e0e0e0",
}


def _style_axis(ax, title="", ylabel=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def plot_prices(df: pd.DataFrame, split_idx: int):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df[STOCK_A], color=PALETTE["blue"], linewidth=1.2, label=STOCK_A)
    ax.plot(df.index, df[STOCK_B], color=PALETTE["red"],  linewidth=1.2, label=STOCK_B)
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7, label="Train / Test split")
    _style_axis(ax, f"{STOCK_A} vs {STOCK_B} — Adjusted Close Prices", "Price ($)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/1_prices.png", dpi=150)
    plt.close()
    print(f"    📈 Saved: {OUTPUT_DIR}/1_prices.png")


def plot_spread_zscore(df: pd.DataFrame, split_idx: int):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Spread
    ax = axes[0]
    ax.plot(df.index, df["spread"], color=PALETTE["blue"], linewidth=0.9)
    ax.plot(df.index, df["spread_mean"], color=PALETTE["orange"], linewidth=1.0, linestyle="--", label=f"{LOOKBACK}-day mean")
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7)
    _style_axis(ax, "Price Spread (KO − β·PEP − α)", "Spread ($)")
    ax.legend(fontsize=9)

    # Z-score
    ax = axes[1]
    ax.plot(df.index, df["z_score"], color=PALETTE["blue"], linewidth=0.9)
    ax.axhline(Z_ENTRY,  color=PALETTE["red"],   linestyle="--", alpha=0.7, label=f"Entry (±{Z_ENTRY})")
    ax.axhline(-Z_ENTRY, color=PALETTE["red"],   linestyle="--", alpha=0.7)
    ax.axhline(Z_EXIT,   color=PALETTE["green"], linestyle="--", alpha=0.5, label=f"Exit  (±{Z_EXIT})")
    ax.axhline(-Z_EXIT,  color=PALETTE["green"], linestyle="--", alpha=0.5)
    ax.axhline(0,        color=PALETTE["grey"],  linestyle="-",  alpha=0.3)
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7)
    ax.fill_between(df.index, Z_ENTRY, df["z_score"].clip(lower=Z_ENTRY),
                    alpha=0.15, color=PALETTE["red"])
    ax.fill_between(df.index, -Z_ENTRY, df["z_score"].clip(upper=-Z_ENTRY),
                    alpha=0.15, color=PALETTE["green"])
    _style_axis(ax, f"Rolling Z-Score (lookback = {LOOKBACK} days)", "Z-Score (σ)")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/2_spread_zscore.png", dpi=150)
    plt.close()
    print(f"    📈 Saved: {OUTPUT_DIR}/2_spread_zscore.png")


def plot_positions(df: pd.DataFrame, split_idx: int):
    fig, ax = plt.subplots(figsize=(14, 3))
    colors = df["position"].map({1: PALETTE["green"], -1: PALETTE["red"], 0: PALETTE["grey"]})
    ax.bar(df.index, df["position"], color=colors, width=1.5, alpha=0.7)
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7)
    _style_axis(ax, "Trading Positions (+1 = Long Spread, −1 = Short Spread)", "Position")
    ax.set_yticks([-1, 0, 1])
    legend_elements = [Patch(facecolor=PALETTE["green"], alpha=0.7, label="Long spread"),
                       Patch(facecolor=PALETTE["red"],   alpha=0.7, label="Short spread")]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/3_positions.png", dpi=150)
    plt.close()
    print(f"    📈 Saved: {OUTPUT_DIR}/3_positions.png")


def plot_cumulative_returns(df: pd.DataFrame, split_idx: int, metrics_train: dict, metrics_test: dict):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["cum_strategy"], color=PALETTE["blue"],   linewidth=1.5, label="Pairs strategy")
    ax.plot(df.index, df["cum_A"],        color=PALETTE["grey"],   linewidth=0.9, alpha=0.5, label=f"{STOCK_A} buy & hold")
    ax.plot(df.index, df["cum_B"],        color=PALETTE["orange"], linewidth=0.9, alpha=0.5, label=f"{STOCK_B} buy & hold")
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7, label="Train / Test split")
    ax.axhline(1, color=PALETTE["grey"], linestyle="-", alpha=0.3)

    # Annotation box
    box_text = (
        f"IN-SAMPLE          OUT-OF-SAMPLE\n"
        f"Return:  {metrics_train['total_return']:+.1%}        {metrics_test['total_return']:+.1%}\n"
        f"Sharpe:  {metrics_train['sharpe_ratio']:.2f}            {metrics_test['sharpe_ratio']:.2f}\n"
        f"Max DD:  {metrics_train['max_drawdown']:.1%}        {metrics_test['max_drawdown']:.1%}"
    )
    ax.text(0.02, 0.97, box_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.9))

    _style_axis(ax, "Cumulative Returns — Pairs Strategy vs Buy & Hold", "Growth of $1")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/4_cumulative_returns.png", dpi=150)
    plt.close()
    print(f"    📈 Saved: {OUTPUT_DIR}/4_cumulative_returns.png")


def plot_drawdown(df: pd.DataFrame, split_idx: int):
    cum = df["cum_strategy"]
    dd = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.index, dd, 0, color=PALETTE["red"], alpha=0.3)
    ax.plot(df.index, dd, color=PALETTE["red"], linewidth=0.8)
    ax.axvline(df.index[split_idx], color=PALETTE["grey"], linestyle="--", alpha=0.7)
    _style_axis(ax, "Underwater Plot (Drawdown from Peak)", "Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/5_drawdown.png", dpi=150)
    plt.close()
    print(f"    📈 Saved: {OUTPUT_DIR}/5_drawdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  STATISTICAL PAIRS TRADING — COINTEGRATION STRATEGY")
    print(f"  {STOCK_A} / {STOCK_B}  |  {START_DATE} → {END_DATE}")
    print("=" * 70, "\n")

    # ── Fetch data ────────────────────────────────────────────────────────
    prices = fetch_data(STOCK_A, STOCK_B, START_DATE, END_DATE)

    # ── Train / test split ────────────────────────────────────────────────
    split_idx = int(len(prices) * TRAIN_SPLIT)
    train = prices.iloc[:split_idx]
    test  = prices.iloc[split_idx:]
    print(f"📂  Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} days)")
    print(f"    Test:  {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} days)\n")

    # ── Cointegration test (training set only) ────────────────────────────
    coint_results = test_cointegration(train[STOCK_A], train[STOCK_B])

    if coint_results["coint_pvalue"] >= 0.05:
        print("⚠️  Pair is NOT cointegrated at 5% level. Proceeding anyway for demonstration.\n")

    # ── Spread & z-score (full dataset, parameters from train) ────────────
    df = compute_spread(prices, coint_results["hedge_ratio"], coint_results["intercept"])

    # ── Signals & backtest ────────────────────────────────────────────────
    df = generate_signals(df)
    df = backtest(df, coint_results["hedge_ratio"])

    # Recompute split index after dropna
    split_date = train.index[-1]
    split_idx_df = df.index.get_indexer([split_date], method="nearest")[0]

    # ── Performance ───────────────────────────────────────────────────────
    df_train = df.iloc[:split_idx_df]
    df_test  = df.iloc[split_idx_df:]

    metrics_train = performance_summary(df_train, "In-Sample (Train)")
    metrics_test  = performance_summary(df_test,  "Out-of-Sample (Test)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("🎨  Generating charts...")
    plot_prices(prices, split_idx)
    plot_spread_zscore(df, split_idx_df)
    plot_positions(df, split_idx_df)
    plot_cumulative_returns(df, split_idx_df, metrics_train, metrics_test)
    plot_drawdown(df, split_idx_df)

    # ── Save metrics to CSV ───────────────────────────────────────────────
    metrics_df = pd.DataFrame([metrics_train, metrics_test])
    metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
    print(f"\n    📄 Saved: {OUTPUT_DIR}/metrics.csv")

    print("\n" + "=" * 70)
    print("  ✅  Pipeline complete. All outputs saved to ./output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
