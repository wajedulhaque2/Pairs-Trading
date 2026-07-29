"""Research workflow helpers for pair selection and parameter optimisation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from pair_backtester import (
    PairBacktestConfig,
    run_equal_weight_buy_and_hold,
    run_pair_backtest,
)
from pair_data import align_pair_prices
from pair_selection import PairSelectionConfig, screen_pairs, select_best_pair
from pair_strategy import PairStrategyConfig, build_pair_signal


def choose_pair(
    training_prices: pd.DataFrame,
    selection_config: PairSelectionConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Screen the training universe and return the highest-ranked pair."""

    ranking = screen_pairs(training_prices, selection_config)
    selected = select_best_pair(ranking)
    return selected, ranking


def optimize_pair_strategy(
    training_prices: pd.DataFrame,
    selected_pair: pd.Series,
    backtest_config: PairBacktestConfig,
    zscore_lookbacks: Iterable[int],
    entry_zscores: Iterable[float],
    exit_zscores: Iterable[float],
    stop_zscores: Iterable[float],
    maximum_holding_days_values: Iterable[int],
    *,
    minimum_completed_trades: int = 5,
) -> pd.DataFrame:
    """Grid-search spread-trading parameters using training data only."""

    symbol_y = str(selected_pair["symbol_y"])
    symbol_x = str(selected_pair["symbol_x"])
    alpha = float(selected_pair["alpha"])
    beta = float(selected_pair["beta"])
    use_log_prices = bool(selected_pair.get("use_log_prices", True))
    pair_prices = align_pair_prices(training_prices, symbol_y, symbol_x)

    rows: list[dict[str, Any]] = []

    for lookback in zscore_lookbacks:
        for entry in entry_zscores:
            for exit_value in exit_zscores:
                for stop in stop_zscores:
                    for maximum_holding_days in maximum_holding_days_values:
                        if exit_value >= entry or stop <= entry:
                            continue

                        strategy_config = PairStrategyConfig(
                            zscore_lookback=int(lookback),
                            entry_zscore=float(entry),
                            exit_zscore=float(exit_value),
                            stop_zscore=float(stop),
                            maximum_holding_days=int(maximum_holding_days),
                        )
                        signal = build_pair_signal(
                            pair_prices,
                            symbol_y,
                            symbol_x,
                            alpha,
                            beta,
                            strategy_config,
                            use_log_prices=use_log_prices,
                        )
                        _, metrics, _ = run_pair_backtest(
                            pair_prices,
                            symbol_y,
                            symbol_x,
                            signal["target_position"],
                            beta,
                            backtest_config,
                            signal_frame=signal,
                        )

                        rows.append(
                            {
                                **asdict(strategy_config),
                                "total_return": metrics["total_return"],
                                "annualized_return": metrics["annualized_return"],
                                "annualized_volatility": metrics[
                                    "annualized_volatility"
                                ],
                                "sharpe_ratio": metrics["sharpe_ratio"],
                                "sortino_ratio": metrics["sortino_ratio"],
                                "max_drawdown": metrics["max_drawdown"],
                                "completed_trades": metrics[
                                    "number_of_completed_trades"
                                ],
                                "trade_win_rate": metrics["trade_win_rate"],
                                "profit_factor": metrics["profit_factor"],
                                "total_turnover": metrics["total_turnover"],
                            }
                        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("strategy parameter search produced no results")

    results = results.replace([np.inf, -np.inf], np.nan)
    eligible = results[
        results["completed_trades"] >= minimum_completed_trades
    ].copy()
    if eligible.empty:
        eligible = results.copy()

    eligible = eligible.dropna(subset=["sharpe_ratio"])
    if eligible.empty:
        raise ValueError("strategy parameter search produced no valid Sharpe ratios")

    return eligible.sort_values(
        by=["sharpe_ratio", "annualized_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_pair_period(
    period_prices: pd.DataFrame,
    signal_history_prices: pd.DataFrame,
    selected_pair: pd.Series,
    strategy_config: PairStrategyConfig,
    backtest_config: PairBacktestConfig,
) -> dict[str, dict[str, Any]]:
    """Run the pair strategy and passive benchmark for one evaluation period.

    Indicators are created from ``signal_history_prices`` and then aligned to the
    requested period. The backtest itself starts from fresh capital and cash.
    """

    symbol_y = str(selected_pair["symbol_y"])
    symbol_x = str(selected_pair["symbol_x"])
    alpha = float(selected_pair["alpha"])
    beta = float(selected_pair["beta"])
    use_log_prices = bool(selected_pair.get("use_log_prices", True))

    history_pair = align_pair_prices(signal_history_prices, symbol_y, symbol_x)
    period_pair = align_pair_prices(period_prices, symbol_y, symbol_x)

    full_signal = build_pair_signal(
        history_pair,
        symbol_y,
        symbol_x,
        alpha,
        beta,
        strategy_config,
        use_log_prices=use_log_prices,
    )
    period_signal = full_signal.reindex(period_pair.index)

    pair_results, pair_metrics, pair_trades = run_pair_backtest(
        period_pair,
        symbol_y,
        symbol_x,
        period_signal["target_position"].fillna(0.0),
        beta,
        backtest_config,
        signal_frame=period_signal,
    )

    benchmark_results, benchmark_metrics = run_equal_weight_buy_and_hold(
        period_pair,
        symbol_y,
        symbol_x,
        backtest_config,
    )

    return {
        "Pairs Strategy": {
            "results": pair_results,
            "metrics": pair_metrics,
            "trades": pair_trades,
            "signal": period_signal,
        },
        "50/50 Buy and Hold": {
            "results": benchmark_results,
            "metrics": benchmark_metrics,
            "trades": pd.DataFrame(),
            "signal": pd.DataFrame(index=period_pair.index),
        },
    }


def metrics_table(runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        metrics = run["metrics"]
        rows[name] = {
            "Final Value": metrics["final_value"],
            "Total Return": metrics["total_return"],
            "Annualized Return": metrics["annualized_return"],
            "Annualized Volatility": metrics["annualized_volatility"],
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Sortino Ratio": metrics["sortino_ratio"],
            "Maximum Drawdown": metrics["max_drawdown"],
            "Daily Win Rate": metrics["daily_win_rate"],
            "Market Exposure": metrics["market_exposure"],
            "Gross Exposure": metrics["average_gross_exposure"],
            "Absolute Net Exposure": metrics["average_absolute_net_exposure"],
            "Total Turnover": metrics["total_turnover"],
            "Rebalance Days": metrics["number_of_rebalance_days"],
            "Leg Orders": metrics["number_of_leg_orders"],
            "Completed Trades": metrics["number_of_completed_trades"],
            "Trade Win Rate": metrics["trade_win_rate"],
            "Average Trade Return": metrics["average_trade_return"],
            "Average Holding Days": metrics["average_holding_days"],
            "Profit Factor": metrics["profit_factor"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")
