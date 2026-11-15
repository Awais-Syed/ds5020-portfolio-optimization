"""Load the price data, build the return matrix X, and describe it.

Run from the project root:
    python src/load_data.py

Uses only numpy, pandas, matplotlib and scipy.stats.
Math link: X in R^{T x n}; row t = trading day, column i = asset,
entry X[t, i] = (p[t, i] - p[t-1, i]) / p[t-1, i].
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prices.csv"
FIGS = ROOT / "figures"


def load_prices(path=DATA):
    """Adjusted closing prices, rows = dates, columns = tickers."""
    return pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()


def simple_returns(prices):
    """R[t,i] = (p[t,i] - p[t-1,i]) / p[t-1,i]; first row is dropped."""
    return prices.pct_change().dropna(how="any")


def describe(prices, returns):
    X = returns.to_numpy()
    T, n = X.shape
    print("=== PRICES ===")
    print(f"Date range : {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Shape      : {prices.shape[0]} rows x {prices.shape[1]} columns")
    print("Dtypes     :", dict(prices.dtypes.astype(str)))
    print("Missing values per column:")
    print(prices.isna().sum().to_string())
    print("\n=== RETURNS (data matrix X) ===")
    print(f"X shape (T x n): {T} x {n}")
    print(returns.describe().T[["mean", "std", "min", "max"]].round(5).to_string())
    print("\nSkewness       :", dict(zip(returns.columns, np.round(stats.skew(X), 3))))
    print("Excess kurtosis:", dict(zip(returns.columns, np.round(stats.kurtosis(X), 3))))

    # Checks tied to the assumptions in the report
    m = X.mean(axis=0)
    Xc = X - m
    S = Xc.T @ Xc / (T - 1)
    eig = np.linalg.eigvalsh(S)
    print("\n=== COVARIANCE CHECKS ===")
    print("Symmetric         :", np.allclose(S, S.T))
    print("Min eigenvalue    :", eig.min())
    print("Condition number  :", eig.max() / eig.min())
    print("Rank of Xc        :", np.linalg.matrix_rank(Xc), "(n =", n, ")")
    print("Matches np.cov    :", np.allclose(S, np.cov(X.T)))
    one = np.ones(n)
    u = np.linalg.solve(S, one)
    v = np.linalg.solve(S, m)
    A, B, C = one @ u, m @ u, m @ v
    print(f"A={A:.4f}  B={B:.4f}  C={C:.4f}  Delta={A * C - B ** 2:.4f}")


def make_plots(prices, returns):
    FIGS.mkdir(exist_ok=True)

    # 1. prices rebased to 100
    (prices / prices.iloc[0] * 100).plot(figsize=(9, 4))
    plt.title("Adjusted close, rebased to 100")
    plt.ylabel("Index level")
    plt.tight_layout()
    plt.savefig(FIGS / "prices_rebased.png", dpi=150)
    plt.close()

    # 2. return histograms
    returns.hist(bins=60, figsize=(10, 6))
    plt.suptitle("Daily simple returns")
    plt.tight_layout()
    plt.savefig(FIGS / "return_histograms.png", dpi=150)
    plt.close()

    # 3. correlation matrix
    cols = list(returns.columns)
    corr = returns.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation of daily returns")
    fig.tight_layout()
    fig.savefig(FIGS / "return_correlation.png", dpi=150)
    plt.close(fig)
    print("Saved figures to", FIGS)


if __name__ == "__main__":
    prices = load_prices()
    returns = simple_returns(prices)
    describe(prices, returns)
    make_plots(prices, returns)