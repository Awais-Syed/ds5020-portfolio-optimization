"""One-time download of adjusted daily closing prices from Yahoo Finance.

Run once from the project root:
    python src/download_data.py

Writes data/prices.csv (Date index, one column per ticker).
This is the ONLY script that needs the extra package `yfinance`
(pip install yfinance). All other scripts use only numpy, pandas,
matplotlib and scipy.stats. If yfinance is not allowed, download the
history for each ticker manually from Yahoo Finance and build
data/prices.csv with columns: Date, AAPL, MSFT, AMZN, GOOGL, WMT.
"""
from pathlib import Path

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "WMT"]
START = "2020-01-01"
END = "2026-01-01"  # end date is exclusive, so this includes 2025-12-31

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prices.csv"


def main():
    # auto_adjust=True makes "Close" the split- and dividend-adjusted close
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"][TICKERS]
    prices.index.name = "Date"
    OUT.parent.mkdir(exist_ok=True)
    prices.to_csv(OUT)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} columns to {OUT}")
    print(prices.head())


if __name__ == "__main__":
    main()