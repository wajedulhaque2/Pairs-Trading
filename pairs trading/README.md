# Pairs Trading — Implementation Directory

For the project overview, screening result, verified out-of-sample performance and portfolio walkthrough, see the [root README](../README.md).

This directory contains the statistical-arbitrage research pipeline, backtester, notebooks, paper-trading adapter, tests and saved outputs.

## Key files

- `main.py` — complete pair-screening, optimisation and out-of-sample workflow.
- `main.ipynb` — notebook version of the research process.
- `complete_pairs_trading_theory_guide_fixed.ipynb` — theory of correlation, cointegration, stationarity, spreads and mean reversion.
- `pair_data.py` — market-data acquisition and preparation.
- `pair_selection.py` — screening, regression, cointegration, ADF and half-life analysis.
- `pair_strategy.py` — spread/z-score signal generation.
- `pair_backtester.py` — cost-aware two-leg portfolio simulation and trade analytics.
- `research.py` — pair/parameter research orchestration.
- `alpaca_paper.py` — dry-run-first Alpaca paper-trading adapter.
- `tests/` — automated checks for data, selection, signals and backtesting.
- `outputs/` — saved screening, parameters, out-of-sample summaries and diagnostic charts.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Tests:

```bash
pytest
```

The Alpaca adapter remains dry-run-first and requires explicit `--execute-paper` before submitting paper orders.
