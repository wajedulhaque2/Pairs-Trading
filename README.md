# Pairs Trading Research & Paper-Trading System

An end-to-end statistical-arbitrage research project covering **pair screening, cointegration and stationarity testing, hedge-ratio estimation, z-score signals, cost-aware backtesting, out-of-sample evaluation and an optional Alpaca paper-trading bridge**.

**Python · pandas · statsmodels · scikit-learn · yfinance · Alpaca · statistical arbitrage · automated tests**

![Selected pair — normalised prices](pairs%20trading/outputs/selected_pair_normalised_prices.png)

## Project objective

Pairs trading attempts to exploit temporary deviations in the relationship between two historically linked assets. This repository does not begin with a hand-picked pair; it builds a research pipeline that:

1. downloads a candidate universe;
2. screens possible pairs;
3. estimates the spread relationship;
4. tests mean-reversion evidence;
5. selects strategy parameters using training data only;
6. evaluates the result on a separate testing period;
7. optionally translates the selected strategy into a paper-trading order plan.

The project is deliberately transparent about an important outcome: **the strict screen did not identify a fully eligible pair in the saved run**. The system therefore falls back to its highest-ranked candidate for educational analysis instead of pretending the screening conditions were satisfied.

## Research setup

The saved configuration uses 16 liquid US equities:

```text
KO, PEP, XOM, CVX, JPM, BAC, GS, MS,
V, MA, HD, LOW, WMT, TGT, UPS, FDX
```

| Setting | Saved value |
|---|---|
| Research period | 01 Jan 2014 – 01 Jan 2026 |
| Train/test split | 01 Jan 2020 |
| Initial capital | $10,000 |
| Transaction cost | 0.10% |
| Annual short-borrow cost | 3% |
| Execution delay | 1 trading period |
| Minimum absolute return correlation | 0.55 |
| Maximum cointegration p-value | 0.05 |
| Maximum ADF p-value | 0.05 |
| Allowed half-life | 2–120 days |

Pair selection and strategy-parameter selection are performed on the **training period only**.

## Saved pair-selection result

The highest-ranked fallback pair in the stored run is:

```text
UPS / V
```

Its saved diagnostics are:

| Diagnostic | Value |
|---|---:|
| Return correlation | 0.4216 |
| Cointegration p-value | 0.0104 |
| ADF p-value | 0.00215 |
| Regression alpha | 3.2065 |
| Regression beta / hedge coefficient | 0.2466 |
| Estimated half-life | 41.77 days |
| Passed every strict screen? | **No** |

The pair passes the saved cointegration and ADF thresholds, but its **0.4216 return correlation is below the required 0.55**, so `eligible` is stored as `false`.

That distinction matters: the later backtest is best read as a demonstration of the framework and the consequences of using a fallback candidate, not as evidence that the screening process found a production-ready statistical-arbitrage pair.

## Selected strategy parameters

Training-period optimisation selected:

```text
Z-score lookback:       60 days
Entry threshold:        ±2.5
Exit threshold:         0.0
Stop threshold:         ±4.0
Maximum holding period: 60 days
```

The strategy can take long-spread and short-spread positions and includes explicit exit, stop-loss and maximum-holding-period rules.

![Selected pair z-score and thresholds](pairs%20trading/outputs/selected_pair_zscore.png)

## Out-of-sample results

The stored testing output reports:

| Metric | Pairs Strategy | 50/50 Buy & Hold |
|---|---:|---:|
| Final value | $10,107 | $14,951 |
| Total return | **1.07%** | **49.51%** |
| Annualised return | 0.18% | 6.94% |
| Sharpe ratio | 0.084 | 0.398 |
| Sortino ratio | 0.077 | 0.532 |
| Maximum drawdown | -32.19% | -28.01% |
| Completed trades | 19 | 0 |
| Trade win rate | 68.42% | — |
| Average holding period | 31.42 days | — |
| Profit factor | 1.25 | — |

The strategy did **not** outperform the passive benchmark in the saved out-of-sample test. That result is useful rather than something to hide: it shows why cointegration evidence, parameter optimisation and a high trade win rate are not sufficient to guarantee an economically attractive strategy.

## Research workflow

### Pair screening

`pair_selection.py` evaluates candidate pairs using:

- return correlation;
- cointegration tests;
- ADF stationarity tests on the spread;
- regression-based hedge ratios;
- estimated mean-reversion half-life;
- minimum observation requirements.

### Spread construction

The selected relationship is modelled using a regression spread, with optional log-price treatment. The saved spread can be inspected directly:

![Selected regression spread](pairs%20trading/outputs/selected_pair_spread.png)

### Signal generation

`pair_strategy.py` converts the spread into rolling z-scores and target positions using entry, exit, stop and holding-period rules.

### Cost-aware backtesting

`pair_backtester.py` models:

- delayed execution;
- proportional transaction costs;
- annual short-borrow cost;
- gross-normalised two-leg weights;
- turnover and rebalance activity;
- individual completed trades;
- portfolio and trade-level metrics.

### Out-of-sample validation

The project separates the pre-2020 training sample from the later testing sample. Screening and parameter selection occur before the test period, reducing the risk of choosing the pair or parameters based on future results.

## Paper-trading bridge

`alpaca_paper.py` connects the exported pair/strategy settings to an Alpaca **paper** account.

The adapter is dry-run-first: it does **not submit orders unless `--execute-paper` is explicitly supplied**. It requests recent daily bars, calculates the latest target, checks asset tradability/shortability and builds an integer-share order plan.

The script also documents a key execution limitation: the two leg orders are not atomic, so leg risk remains.

## Project structure

```text
Pairs-Trading-/
├── README.md
└── pairs trading/
    ├── main.py
    ├── main.ipynb
    ├── complete_pairs_trading_theory_guide_fixed.ipynb
    ├── pair_data.py
    ├── pair_selection.py
    ├── pair_strategy.py
    ├── pair_backtester.py
    ├── research.py
    ├── alpaca_paper.py
    ├── requirements.txt
    ├── tests/
    ├── data_cache/
    └── outputs/
        ├── pair_screening_results.csv
        ├── selected_pair_and_parameters.json
        ├── strategy_parameter_search.csv
        ├── testing_summary.csv
        ├── trade / signal / daily-result files
        └── saved diagnostic charts
```

## Start here

For a portfolio review:

1. **`README.md`** — methodology, selected pair and honest out-of-sample result.
2. **`pairs trading/main.ipynb`** — complete research workflow.
3. **`pairs trading/complete_pairs_trading_theory_guide_fixed.ipynb`** — statistical-arbitrage theory and assumptions.
4. **`pairs trading/pair_selection.py`** — screening and statistical tests.
5. **`pairs trading/pair_backtester.py`** — cost-aware two-leg backtester.
6. **`pairs trading/alpaca_paper.py`** — dry-run-first paper execution bridge.

## How to run

```bash
cd "pairs trading"
pip install -r requirements.txt
python main.py
```

To run the automated checks:

```bash
pytest
```

The tests cover data handling, pair selection, signal generation and core backtester behaviour.

For the paper adapter, set Alpaca paper-account credentials in environment variables or a local `.env` file. Dry-run behaviour remains the default; paper orders require the explicit `--execute-paper` flag.

## Limitations

- No pair passed every strict screening criterion in the saved run; UPS/V is a fallback research candidate.
- Historical cointegration and stationarity can break down after the estimation window.
- The out-of-sample pairs strategy materially underperformed the stored 50/50 benchmark in this run.
- Shorting introduces borrow costs, shortability constraints and potentially asymmetric losses.
- Transaction and borrow costs are modelled, but market impact and detailed bid/ask execution are not fully reproduced.
- Two-leg paper orders are not atomic, so execution/leg risk remains.
- Historical results do not imply future profitability.

## Skills demonstrated

**Statistical arbitrage · correlation and cointegration · ADF testing · regression hedge ratios · mean-reversion half-life · z-score signals · parameter optimisation · out-of-sample testing · transaction/borrow-cost modelling · two-leg portfolio construction · trade analytics · Alpaca paper trading · automated testing**

> Educational, research and paper-trading project only — not investment advice.
