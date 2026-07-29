# Pairs Trading Research Project

A complete statistical-arbitrage project that:

1. Downloads a candidate stock universe.
2. Finds highly correlated pairs.
3. Tests price relationships for cointegration.
4. Estimates a regression hedge ratio.
5. Builds a stationary-style spread.
6. Trades spread divergence with rolling z-scores.
7. Models two-leg returns, trading costs, and short-borrow costs.
8. Selects parameters using training data only.
9. Reports out-of-sample performance.
10. Exports charts, daily results, trades, pair rankings, and selected parameters.
11. Connects to Alpaca **paper trading** through a dry-run-first adapter.

## Project structure

```text
pairs-trading-project/
├── pair_data.py
├── pair_selection.py
├── pair_strategy.py
├── pair_backtester.py
├── research.py
├── main.py
├── main.ipynb
├── alpaca_paper.py
├── complete_pairs_trading_theory_guide.ipynb
├── requirements.txt
├── .env.example
├── tests/
├── data_cache/
└── outputs/
```

## What each file does

| File | Responsibility |
|---|---|
| `pair_data.py` | Downloads and caches adjusted daily prices |
| `pair_selection.py` | Correlation, OLS hedge ratio, cointegration, ADF, and half-life |
| `pair_strategy.py` | Spread z-score entry, exit, stop, and time-exit logic |
| `pair_backtester.py` | Two-leg weights, P&L, turnover, costs, borrow, trades, and metrics |
| `research.py` | Pair selection, parameter search, period runs, and summary tables |
| `main.py` | Complete train/test workflow and exports |
| `main.ipynb` | Notebook launcher and result inspection |
| `alpaca_paper.py` | Alpaca paper-market-data and dry-run-first order adapter |
| `tests/` | Deterministic tests using synthetic data |

## Core model

The selected pair is fitted on training data with:

```text
log(Y) = alpha + beta × log(X) + spread
```

The strategy trades the residual spread:

```text
Long spread:
    long Y
    short beta-adjusted X

Short spread:
    short Y
    long beta-adjusted X
```

Leg weights are scaled so absolute weights sum to one during an active trade.

## Install

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the research workflow

```bash
python main.py
```

Or open `main.ipynb` and run all cells.

The default workflow:

- Downloads a candidate universe
- Uses data before `2020-01-01` for selection and optimization
- Chooses a pair using correlation, cointegration, ADF, and half-life
- Chooses z-score rules using training-period Sharpe ratio
- Evaluates the frozen pair and rules after the split
- Compares the pairs strategy with a passive 50/50 buy-and-hold benchmark

Edit the **SETTINGS** section at the top of `main.py` to change the universe, dates, screening rules, costs, and parameter grids.

## Outputs

`main.py` writes files into `outputs/`, including:

- `pair_screening_results.csv`
- `strategy_parameter_search.csv`
- `selected_pair_and_parameters.json`
- Full, training, and testing summaries
- Daily strategy results
- Trade histories
- Signal histories
- Normalized-price chart
- Spread chart
- Z-score chart
- Out-of-sample portfolio chart
- Out-of-sample drawdown chart
- Executed-weight chart

## Run the tests

```bash
pytest -q
```

The included test suite checks:

- Hedge-ratio estimation
- Cointegration and ADF diagnostics
- Spread construction
- Long- and short-spread signals
- Execution delay
- Gross-normalized weights
- Two-leg transaction costs
- Borrow costs
- Invalid input rejection
- Data cleaning

## Alpaca paper trading

Create a `.env` file from `.env.example`:

```text
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

First run the research workflow so this file exists:

```text
outputs/selected_pair_and_parameters.json
```

Then run a dry plan:

```bash
python alpaca_paper.py
```

No orders are submitted by default.

To submit the displayed orders to an Alpaca **paper** account:

```bash
python alpaca_paper.py --execute-paper
```

The adapter always constructs `TradingClient(..., paper=True)`.

### Paper-trading warnings

- The two stock orders are separate, not atomic.
- One leg can fill before the other.
- Market orders can fill away from the last daily close.
- Short availability can change.
- Integer-share rounding changes the intended hedge.
- Paper fills do not perfectly reproduce live execution.
- The script should not be used as production trading infrastructure.

## Research warnings

Correlation alone is not enough. Two prices can move together while their difference drifts permanently.

Cointegration tests are statistical evidence, not guarantees. Relationships can break after:

- Business-model changes
- Mergers
- Regulation
- Index changes
- Capital-structure changes
- Commodity or interest-rate regime changes
- Short squeezes
- Market crises

Treat the out-of-sample section as the main historical evidence. Full-period and training results are descriptive and in-sample.

## Important modelling assumptions

- Daily adjusted close data
- One-period execution delay
- Fixed training-period hedge ratio
- Gross exposure normalized to one
- Constant proportional trading costs
- Constant annual short-borrow rate
- Cash earns the configured risk-free rate only while flat
- Fresh capital and cash at the start of each evaluation period
- No intraday execution, liquidity, market impact, taxes, or margin calls
- No atomic pair-order support
- No portfolio allocation across multiple pairs

Read `complete_pairs_trading_theory_guide.ipynb` for a detailed explanation of every formula, code line, assumption, and limitation.

## Official documentation used

- Alpaca-py market data and trading clients
- statsmodels time-series tests, including `coint`
- scikit-learn `LinearRegression`

API behaviour can change, so check the current official documentation before using the paper adapter.
