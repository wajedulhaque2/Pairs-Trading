"""Complete research workflow for the pairs-trading project.

Edit the SETTINGS section, then run:

    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from pair_backtester import PairBacktestConfig
from pair_data import download_adjusted_close_prices
from pair_selection import PairSelectionConfig
from pair_strategy import PairStrategyConfig
from research import (
    choose_pair,
    metrics_table,
    optimize_pair_strategy,
    run_pair_period,
)


# ==================================================
# SETTINGS
# ==================================================

CANDIDATE_UNIVERSE = [
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "MA",
    "HD",
    "LOW",
    "WMT",
    "TGT",
    "UPS",
    "FDX",
]

START_DATE = "2014-01-01"
END_DATE = "2026-01-01"
SPLIT_DATE = "2020-01-01"

OUTPUT_DIRECTORY = Path("outputs")
CACHE_DIRECTORY = Path("data_cache")
SHOW_PLOTS = True
REFRESH_MARKET_DATA = False

PAIR_SELECTION_CONFIG = PairSelectionConfig(
    minimum_observations=756,
    minimum_absolute_return_correlation=0.55,
    maximum_cointegration_pvalue=0.05,
    maximum_adf_pvalue=0.05,
    minimum_half_life=2.0,
    maximum_half_life=120.0,
    use_log_prices=True,
)

BACKTEST_CONFIG = PairBacktestConfig(
    initial_capital=10_000.0,
    transaction_cost_rate=0.001,
    annual_borrow_rate=0.03,
    annual_risk_free_rate=0.0,
    periods_per_year=252,
    execution_delay=1,
)

ZSCORE_LOOKBACKS = [20, 40, 60]
ENTRY_ZSCORES = [1.5, 2.0, 2.5]
EXIT_ZSCORES = [0.0, 0.25, 0.5]
STOP_ZSCORES = [3.5, 4.0]
MAXIMUM_HOLDING_DAYS = [20, 40, 60]
MINIMUM_COMPLETED_TRADES = 5

PERCENT_COLUMNS = [
    "Total Return",
    "Annualized Return",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Daily Win Rate",
    "Market Exposure",
    "Gross Exposure",
    "Absolute Net Exposure",
    "Trade Win Rate",
    "Average Trade Return",
]


# ==================================================
# DISPLAY AND EXPORT HELPERS
# ==================================================


def display_summary(title: str, summary: pd.DataFrame) -> None:
    display = summary.copy()
    for column in PERCENT_COLUMNS:
        if column in display.columns:
            display[column] = display[column] * 100.0

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(display.round(2).to_string())
    print("\nPercentage columns are displayed as percentage points.")


def export_run_group(
    label: str,
    runs: dict[str, dict[str, Any]],
    output_directory: Path,
) -> pd.DataFrame:
    summary = metrics_table(runs)
    summary.to_csv(output_directory / f"{label}_summary.csv")

    for name, run in runs.items():
        safe = name.lower().replace("/", "_").replace(" ", "_")
        run["results"].to_csv(
            output_directory / f"{label}_{safe}_daily_results.csv"
        )
        run["trades"].to_csv(
            output_directory / f"{label}_{safe}_trades.csv",
            index=False,
        )
        if not run["signal"].empty:
            run["signal"].to_csv(
                output_directory / f"{label}_{safe}_signal.csv"
            )

    return summary


def _finish_chart(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_normalised_price_chart(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    split_date: str,
    output_path: Path,
) -> None:
    pair = prices[[symbol_y, symbol_x]].dropna()
    normalised = pair.div(pair.iloc[0]).mul(100.0)

    plt.figure(figsize=(12, 6))
    plt.plot(normalised.index, normalised[symbol_y], label=symbol_y)
    plt.plot(normalised.index, normalised[symbol_x], label=symbol_x)
    plt.axvline(pd.Timestamp(split_date), linestyle="--", label="Train/test split")
    plt.title(f"{symbol_y}/{symbol_x}: normalised adjusted prices")
    plt.xlabel("Date")
    plt.ylabel("Initial value = 100")
    plt.legend()
    plt.grid(True)
    _finish_chart(output_path)


def save_spread_chart(
    signal: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(signal.index, signal["spread"], label="Spread")
    plt.plot(
        signal.index,
        signal["rolling_spread_mean"],
        label="Rolling spread mean",
    )
    plt.title(f"{symbol_y}/{symbol_x}: regression spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True)
    _finish_chart(output_path)


def save_zscore_chart(
    signal: pd.DataFrame,
    strategy_config: PairStrategyConfig,
    symbol_y: str,
    symbol_x: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(signal.index, signal["zscore"], label="Spread z-score")
    plt.axhline(strategy_config.entry_zscore, linestyle="--", label="Short entry")
    plt.axhline(-strategy_config.entry_zscore, linestyle="--", label="Long entry")
    plt.axhline(strategy_config.exit_zscore, linestyle=":", label="Short exit")
    plt.axhline(-strategy_config.exit_zscore, linestyle=":", label="Long exit")
    plt.axhline(strategy_config.stop_zscore, linestyle="-.", label="Short stop")
    plt.axhline(-strategy_config.stop_zscore, linestyle="-.", label="Long stop")
    plt.axhline(0.0)
    plt.title(f"{symbol_y}/{symbol_x}: spread z-score and thresholds")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend(ncol=2)
    plt.grid(True)
    _finish_chart(output_path)


def save_portfolio_chart(
    title: str,
    runs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, run in runs.items():
        plt.plot(run["results"].index, run["results"]["portfolio"], label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(True)
    _finish_chart(output_path)


def save_drawdown_chart(
    title: str,
    runs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, run in runs.items():
        plt.plot(
            run["results"].index,
            run["results"]["drawdown"] * 100.0,
            label=name,
        )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    _finish_chart(output_path)


def save_weights_chart(
    pair_results: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(pair_results.index, pair_results["weight_y"], label=f"{symbol_y} weight")
    plt.plot(pair_results.index, pair_results["weight_x"], label=f"{symbol_x} weight")
    plt.axhline(0.0)
    plt.title(f"{symbol_y}/{symbol_x}: executed leg weights")
    plt.xlabel("Date")
    plt.ylabel("Portfolio weight")
    plt.legend()
    plt.grid(True)
    _finish_chart(output_path)


# ==================================================
# MAIN WORKFLOW
# ==================================================


def main() -> dict[str, Any]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    prices = download_adjusted_close_prices(
        CANDIDATE_UNIVERSE,
        START_DATE,
        END_DATE,
        refresh=REFRESH_MARKET_DATA,
        cache_directory=CACHE_DIRECTORY,
        minimum_coverage=0.90,
    )

    split_timestamp = pd.Timestamp(SPLIT_DATE)
    training_prices = prices[prices.index < split_timestamp].copy()
    testing_prices = prices[prices.index >= split_timestamp].copy()
    if training_prices.empty or testing_prices.empty:
        raise ValueError("SPLIT_DATE must leave observations on both sides")

    print(f"Candidate symbols retained: {list(prices.columns)}")
    print(
        f"Training period: {training_prices.index[0].date()} "
        f"to {training_prices.index[-1].date()}"
    )
    print(
        f"Testing period:  {testing_prices.index[0].date()} "
        f"to {testing_prices.index[-1].date()}"
    )

    selected_pair, pair_ranking = choose_pair(
        training_prices,
        PAIR_SELECTION_CONFIG,
    )
    pair_ranking.to_csv(OUTPUT_DIRECTORY / "pair_screening_results.csv", index=False)

    symbol_y = str(selected_pair["symbol_y"])
    symbol_x = str(selected_pair["symbol_x"])

    if not bool(selected_pair["eligible"]):
        print(
            "\nWARNING: No candidate passed every strict screening rule. "
            "The highest-ranked fallback pair is shown for educational analysis."
        )

    print("\nSelected pair using training data only:")
    print(
        selected_pair[
            [
                "symbol_y",
                "symbol_x",
                "return_correlation",
                "cointegration_pvalue",
                "adf_pvalue",
                "alpha",
                "beta",
                "half_life",
                "eligible",
            ]
        ].to_string()
    )

    parameter_search = optimize_pair_strategy(
        training_prices,
        selected_pair,
        BACKTEST_CONFIG,
        ZSCORE_LOOKBACKS,
        ENTRY_ZSCORES,
        EXIT_ZSCORES,
        STOP_ZSCORES,
        MAXIMUM_HOLDING_DAYS,
        minimum_completed_trades=MINIMUM_COMPLETED_TRADES,
    )
    parameter_search.to_csv(
        OUTPUT_DIRECTORY / "strategy_parameter_search.csv",
        index=False,
    )

    best_parameters = parameter_search.iloc[0]
    selected_strategy_config = PairStrategyConfig(
        zscore_lookback=int(best_parameters["zscore_lookback"]),
        entry_zscore=float(best_parameters["entry_zscore"]),
        exit_zscore=float(best_parameters["exit_zscore"]),
        stop_zscore=float(best_parameters["stop_zscore"]),
        maximum_holding_days=int(best_parameters["maximum_holding_days"]),
    )

    selected_payload = {
        "pair": {
            "symbol_y": symbol_y,
            "symbol_x": symbol_x,
            "alpha": float(selected_pair["alpha"]),
            "beta": float(selected_pair["beta"]),
            "return_correlation": float(selected_pair["return_correlation"]),
            "cointegration_pvalue": float(selected_pair["cointegration_pvalue"]),
            "adf_pvalue": float(selected_pair["adf_pvalue"]),
            "half_life": float(selected_pair["half_life"]),
            "use_log_prices": bool(selected_pair["use_log_prices"]),
            "eligible": bool(selected_pair["eligible"]),
        },
        "strategy": {
            "zscore_lookback": selected_strategy_config.zscore_lookback,
            "entry_zscore": selected_strategy_config.entry_zscore,
            "exit_zscore": selected_strategy_config.exit_zscore,
            "stop_zscore": selected_strategy_config.stop_zscore,
            "maximum_holding_days": selected_strategy_config.maximum_holding_days,
        },
        "backtest": {
            "initial_capital": BACKTEST_CONFIG.initial_capital,
            "transaction_cost_rate": BACKTEST_CONFIG.transaction_cost_rate,
            "annual_borrow_rate": BACKTEST_CONFIG.annual_borrow_rate,
            "annual_risk_free_rate": BACKTEST_CONFIG.annual_risk_free_rate,
            "periods_per_year": BACKTEST_CONFIG.periods_per_year,
            "execution_delay": BACKTEST_CONFIG.execution_delay,
        },
        "research": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "split_date": SPLIT_DATE,
            "selection_was_training_only": True,
        },
    }

    with (OUTPUT_DIRECTORY / "selected_pair_and_parameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selected_payload, file, indent=2)

    print("\nSelected strategy parameters using training-period Sharpe ratio:")
    print(json.dumps(selected_payload["strategy"], indent=2))

    full_runs = run_pair_period(
        prices,
        prices,
        selected_pair,
        selected_strategy_config,
        BACKTEST_CONFIG,
    )
    training_runs = run_pair_period(
        training_prices,
        training_prices,
        selected_pair,
        selected_strategy_config,
        BACKTEST_CONFIG,
    )
    testing_runs = run_pair_period(
        testing_prices,
        prices,
        selected_pair,
        selected_strategy_config,
        BACKTEST_CONFIG,
    )

    full_summary = export_run_group("full", full_runs, OUTPUT_DIRECTORY)
    training_summary = export_run_group("training", training_runs, OUTPUT_DIRECTORY)
    testing_summary = export_run_group("testing", testing_runs, OUTPUT_DIRECTORY)

    display_summary("FULL-PERIOD RESULTS — DESCRIPTIVE", full_summary)
    display_summary("TRAINING RESULTS — IN SAMPLE", training_summary)
    display_summary("TESTING RESULTS — OUT OF SAMPLE", testing_summary)

    save_normalised_price_chart(
        prices,
        symbol_y,
        symbol_x,
        SPLIT_DATE,
        OUTPUT_DIRECTORY / "selected_pair_normalised_prices.png",
    )
    save_spread_chart(
        full_runs["Pairs Strategy"]["signal"],
        symbol_y,
        symbol_x,
        OUTPUT_DIRECTORY / "selected_pair_spread.png",
    )
    save_zscore_chart(
        full_runs["Pairs Strategy"]["signal"],
        selected_strategy_config,
        symbol_y,
        symbol_x,
        OUTPUT_DIRECTORY / "selected_pair_zscore.png",
    )
    save_portfolio_chart(
        f"{symbol_y}/{symbol_x}: out-of-sample portfolio comparison",
        testing_runs,
        OUTPUT_DIRECTORY / "testing_portfolio_comparison.png",
    )
    save_drawdown_chart(
        f"{symbol_y}/{symbol_x}: out-of-sample drawdown comparison",
        testing_runs,
        OUTPUT_DIRECTORY / "testing_drawdown_comparison.png",
    )
    save_weights_chart(
        testing_runs["Pairs Strategy"]["results"],
        symbol_y,
        symbol_x,
        OUTPUT_DIRECTORY / "testing_executed_weights.png",
    )

    print(f"\nFiles saved to: {OUTPUT_DIRECTORY.resolve()}")
    print(
        "Treat the testing section as the main evidence. Pair relationships can "
        "break, shorting has operational risks, and historical profitability does "
        "not guarantee future performance."
    )

    return {
        "prices": prices,
        "training_prices": training_prices,
        "testing_prices": testing_prices,
        "pair_ranking": pair_ranking,
        "selected_pair": selected_pair,
        "parameter_search": parameter_search,
        "strategy_config": selected_strategy_config,
        "runs": {
            "full": full_runs,
            "training": training_runs,
            "testing": testing_runs,
        },
        "summaries": {
            "full": full_summary,
            "training": training_summary,
            "testing": testing_summary,
        },
        "selected_payload": selected_payload,
    }


if __name__ == "__main__":
    main()
