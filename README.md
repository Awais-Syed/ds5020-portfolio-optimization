# DS5020 Final Project: Portfolio Optimization with Covariance Matrices

Team: Soham Bagayatkar, Awais Syed.

## Folders
- `src/`     Python code (data loading, simulation)
- `data/`    price data (not tracked by Git)
- `figures/` generated plots
- `docs/`    Overleaf PDF and Excel project plan

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
Daily adjusted closing prices for AAPL, MSFT, AMZN, GOOGL, WMT, 2020-01-01 to 2025-12-31
(Yahoo Finance). Download once:
```
python src/download_data.py      # writes data/prices.csv
```
Without `yfinance`: download each ticker's history from Yahoo Finance and save
`data/prices.csv` with columns `Date, AAPL, MSFT, AMZN, GOOGL, WMT` (adjusted close).

## Run
```
python src/load_data.py          # describes the data, saves plots to figures/
python src/simulate.py           # simulation check (added next)
```