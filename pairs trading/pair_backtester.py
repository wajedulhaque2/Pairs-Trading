"""Two-leg daily pairs-trading backtester.

The engine accepts a pair price frame and a spread-position series:

    1  = long spread  (long Y, short beta-adjusted X)
    0  = flat
   -1  = short spread (short Y, long beta-adjusted X)

Leg weights are scaled so their absolute values sum to one while a pair trade is
active. This creates one unit of gross exposure rather than two units of gross
leverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pair_data import align_pair_prices


@dataclass(frozen=True)
class PairBacktestConfig:
    """Capital, cost, borrow, timing, and annualisation assumptions."""

    initial_capital: float = 10_000.0
    transaction_cost_rate: float = 0.001
    annual_borrow_rate: float = 0.03
    annual_risk_free_rate: float = 0.0
    periods_per_year: int = 252
    execution_delay: int = 1

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between 0 and 1")
        if self.annual_borrow_rate < 0:
            raise ValueError("annual_borrow_rate cannot be negative")
        if self.annual_risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be greater than -100%")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if self.execution_delay < 0:
            raise ValueError("execution_delay cannot be negative")


def calculate_leg_weights(spread_position: pd.Series, beta: float) -> pd.DataFrame:
    """Convert spread direction into gross-normalised Y and X leg weights."""

    if not isinstance(spread_position, pd.Series):
        raise TypeError("spread_position must be a pandas Series")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")

    position = pd.to_numeric(spread_position, errors="coerce").fillna(0.0)
    allowed = (
        np.isclose(position, -1.0)
        | np.isclose(position, 0.0)
        | np.isclose(position, 1.0)
    )
    if not bool(np.all(allowed)):
        raise ValueError("spread positions must contain only -1, 0, or 1")

    scale = 1.0 + abs(float(beta))
    weight_y = position / scale
    weight_x = -position * float(beta) / scale

    return pd.DataFrame(
        {
            "weight_y": weight_y.astype(float),
            "weight_x": weight_x.astype(float),
        },
        index=position.index,
    )


def calculate_drawdown(portfolio: pd.Series) -> pd.Series:
    running_peak = portfolio.cummax()
    return portfolio.div(running_peak).sub(1.0)


def _elapsed_years(index: pd.Index, periods_per_year: int) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
        return max(float(years), 1.0 / periods_per_year)
    return max(len(index) / periods_per_year, 1.0 / periods_per_year)


def extract_pair_trades(results: pd.DataFrame) -> pd.DataFrame:
    """Convert the executed spread position into individual pair trades."""

    required = {
        "spread_position",
        "strategy_return",
        "price_y",
        "price_x",
        "zscore",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    columns = [
        "entry_date",
        "exit_date",
        "side",
        "entry_price_y",
        "exit_price_y",
        "entry_price_x",
        "exit_price_x",
        "entry_zscore",
        "exit_zscore",
        "holding_days",
        "trade_return",
        "status",
    ]

    trades: list[dict[str, Any]] = []
    current_side = 0
    entry_date: Any | None = None
    entry_price_y: float | None = None
    entry_price_x: float | None = None
    entry_zscore: float = np.nan
    accumulated_returns: list[float] = []
    holding_days = 0
    previous_date: Any | None = None
    previous_price_y: float | None = None
    previous_price_x: float | None = None
    previous_zscore: float = np.nan

    def close_trade(
        exit_date: Any,
        exit_price_y: float,
        exit_price_x: float,
        exit_zscore: float,
        status: str,
    ) -> None:
        nonlocal current_side, entry_date, entry_price_y, entry_price_x
        nonlocal entry_zscore, accumulated_returns, holding_days

        if current_side == 0 or entry_date is None:
            return

        trade_return = float(np.prod(1.0 + np.asarray(accumulated_returns)) - 1.0)
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "side": "Long Spread" if current_side == 1 else "Short Spread",
                "entry_price_y": float(entry_price_y),
                "exit_price_y": float(exit_price_y),
                "entry_price_x": float(entry_price_x),
                "exit_price_x": float(exit_price_x),
                "entry_zscore": float(entry_zscore),
                "exit_zscore": float(exit_zscore),
                "holding_days": int(holding_days),
                "trade_return": trade_return,
                "status": status,
            }
        )

        current_side = 0
        entry_date = None
        entry_price_y = None
        entry_price_x = None
        entry_zscore = np.nan
        accumulated_returns = []
        holding_days = 0

    for date, row in results.iterrows():
        position = int(row["spread_position"])
        daily_return = float(row["strategy_return"])
        price_y = float(row["price_y"])
        price_x = float(row["price_x"])
        zscore = float(row["zscore"]) if pd.notna(row["zscore"]) else np.nan

        if current_side == 0:
            if position != 0:
                current_side = position
                entry_date = date
                entry_price_y = price_y
                entry_price_x = price_x
                entry_zscore = zscore
                accumulated_returns = [daily_return]
                holding_days = 1

        elif position == current_side:
            accumulated_returns.append(daily_return)
            holding_days += 1

        elif position == 0:
            accumulated_returns.append(daily_return)
            close_trade(date, price_y, price_x, zscore, "Closed")

        else:
            if (
                previous_date is not None
                and previous_price_y is not None
                and previous_price_x is not None
            ):
                close_trade(
                    previous_date,
                    previous_price_y,
                    previous_price_x,
                    previous_zscore,
                    "Closed",
                )

            current_side = position
            entry_date = date
            entry_price_y = price_y
            entry_price_x = price_x
            entry_zscore = zscore
            accumulated_returns = [daily_return]
            holding_days = 1

        previous_date = date
        previous_price_y = price_y
        previous_price_x = price_x
        previous_zscore = zscore

    if (
        current_side != 0
        and previous_date is not None
        and previous_price_y is not None
        and previous_price_x is not None
    ):
        close_trade(
            previous_date,
            previous_price_y,
            previous_price_x,
            previous_zscore,
            "Open",
        )

    return pd.DataFrame(trades, columns=columns)


def calculate_pair_metrics(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    config: PairBacktestConfig,
) -> dict[str, float | int]:
    strategy_returns = results["strategy_return"]
    portfolio = results["portfolio"]
    spread_position = results["spread_position"]

    final_value = float(portfolio.iloc[-1])
    total_return = final_value / config.initial_capital - 1.0
    years = _elapsed_years(results.index, config.periods_per_year)
    annualized_return = (final_value / config.initial_capital) ** (1.0 / years) - 1.0

    volatility = float(strategy_returns.std(ddof=1))
    annualized_volatility = volatility * np.sqrt(config.periods_per_year)

    daily_risk_free = (1.0 + config.annual_risk_free_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    excess_returns = strategy_returns - daily_risk_free

    sharpe_ratio = (
        float(excess_returns.mean()) / volatility * np.sqrt(config.periods_per_year)
        if volatility > 0 and np.isfinite(volatility)
        else np.nan
    )

    negative_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = float(negative_returns.std(ddof=1))
    sortino_ratio = (
        float(excess_returns.mean())
        / downside_deviation
        * np.sqrt(config.periods_per_year)
        if len(negative_returns) >= 2
        and downside_deviation > 0
        and np.isfinite(downside_deviation)
        else np.nan
    )

    active_returns = strategy_returns[spread_position != 0]
    daily_win_rate = (
        float((active_returns > 0).mean()) if not active_returns.empty else np.nan
    )

    closed_trades = trades[trades["status"] == "Closed"] if not trades.empty else trades
    winning = (
        closed_trades[closed_trades["trade_return"] > 0]
        if not closed_trades.empty
        else closed_trades
    )
    losing = (
        closed_trades[closed_trades["trade_return"] < 0]
        if not closed_trades.empty
        else closed_trades
    )

    gross_profit = float(winning["trade_return"].sum()) if not winning.empty else 0.0
    gross_loss = float(-losing["trade_return"].sum()) if not losing.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(results["drawdown"].min()),
        "daily_win_rate": float(daily_win_rate),
        "market_exposure": float((spread_position != 0).mean()),
        "average_gross_exposure": float(results["gross_exposure"].mean()),
        "average_absolute_net_exposure": float(results["net_exposure"].abs().mean()),
        "total_turnover": float(results["turnover"].sum()),
        "number_of_rebalance_days": int((results["turnover"] > 1e-12).sum()),
        "number_of_leg_orders": int(
            (results["turnover_y"] > 1e-12).sum()
            + (results["turnover_x"] > 1e-12).sum()
        ),
        "number_of_completed_trades": int(len(closed_trades)),
        "trade_win_rate": (
            float((closed_trades["trade_return"] > 0).mean())
            if not closed_trades.empty
            else np.nan
        ),
        "average_trade_return": (
            float(closed_trades["trade_return"].mean())
            if not closed_trades.empty
            else np.nan
        ),
        "best_trade_return": (
            float(closed_trades["trade_return"].max())
            if not closed_trades.empty
            else np.nan
        ),
        "worst_trade_return": (
            float(closed_trades["trade_return"].min())
            if not closed_trades.empty
            else np.nan
        ),
        "average_holding_days": (
            float(closed_trades["holding_days"].mean())
            if not closed_trades.empty
            else np.nan
        ),
        "profit_factor": float(profit_factor),
    }


def run_pair_backtest(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    target_positions: pd.Series,
    beta: float,
    config: PairBacktestConfig | None = None,
    *,
    signal_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    """Run a close-to-close two-leg spread backtest."""

    config = config or PairBacktestConfig()
    config.validate()
    pair = align_pair_prices(prices, symbol_y, symbol_x)
    symbol_y = symbol_y.upper()
    symbol_x = symbol_x.upper()

    target = (
        pd.to_numeric(target_positions, errors="coerce")
        .reindex(pair.index)
        .fillna(0.0)
        .astype(float)
    )
    allowed = (
        np.isclose(target, -1.0)
        | np.isclose(target, 0.0)
        | np.isclose(target, 1.0)
    )
    if not bool(np.all(allowed)):
        raise ValueError("target_positions must contain only -1, 0, or 1")

    spread_position = target.shift(config.execution_delay).fillna(0.0)
    weights = calculate_leg_weights(spread_position, beta)

    returns_y = pair[symbol_y].pct_change(fill_method=None).fillna(0.0)
    returns_x = pair[symbol_x].pct_change(fill_method=None).fillna(0.0)

    previous_weight_y = weights["weight_y"].shift(1).fillna(0.0)
    previous_weight_x = weights["weight_x"].shift(1).fillna(0.0)
    turnover_y = weights["weight_y"].sub(previous_weight_y).abs()
    turnover_x = weights["weight_x"].sub(previous_weight_x).abs()
    turnover = turnover_y + turnover_x

    gross_pair_return = (
        weights["weight_y"] * returns_y + weights["weight_x"] * returns_x
    )

    gross_exposure = weights.abs().sum(axis=1)
    net_exposure = weights.sum(axis=1)
    short_notional = (-weights).clip(lower=0.0).sum(axis=1)

    transaction_cost = turnover * config.transaction_cost_rate
    daily_borrow_rate = (1.0 + config.annual_borrow_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    borrow_cost = short_notional * daily_borrow_rate

    daily_risk_free_rate = (1.0 + config.annual_risk_free_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    cash_return = (gross_exposure == 0.0).astype(float) * daily_risk_free_rate

    strategy_return = gross_pair_return + cash_return - transaction_cost - borrow_cost
    if (strategy_return <= -1.0).any():
        raise ValueError("a daily net return was -100% or worse")

    portfolio = config.initial_capital * (1.0 + strategy_return).cumprod()
    drawdown = calculate_drawdown(portfolio)

    zscore = pd.Series(np.nan, index=pair.index, dtype=float)
    spread = pd.Series(np.nan, index=pair.index, dtype=float)
    target_event = pd.Series("", index=pair.index, dtype=object)
    if signal_frame is not None:
        if "zscore" in signal_frame:
            zscore = signal_frame["zscore"].reindex(pair.index)
        if "spread" in signal_frame:
            spread = signal_frame["spread"].reindex(pair.index)
        if "signal_event" in signal_frame:
            target_event = signal_frame["signal_event"].reindex(pair.index).fillna("")

    results = pd.DataFrame(
        {
            "price_y": pair[symbol_y],
            "price_x": pair[symbol_x],
            "target_position": target,
            "spread_position": spread_position,
            "weight_y": weights["weight_y"],
            "weight_x": weights["weight_x"],
            "return_y": returns_y,
            "return_x": returns_x,
            "gross_pair_return": gross_pair_return,
            "cash_return": cash_return,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "short_notional": short_notional,
            "turnover_y": turnover_y,
            "turnover_x": turnover_x,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "strategy_return": strategy_return,
            "portfolio": portfolio,
            "drawdown": drawdown,
            "spread": spread,
            "zscore": zscore,
            "target_event": target_event,
        },
        index=pair.index,
    )

    trades = extract_pair_trades(results)
    metrics = calculate_pair_metrics(results, trades, config)
    return results, metrics, trades


def run_equal_weight_buy_and_hold(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    config: PairBacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Run a passive 50/50 initial-allocation benchmark without rebalancing."""

    config = config or PairBacktestConfig()
    config.validate()
    pair = align_pair_prices(prices, symbol_y, symbol_x)
    symbol_y = symbol_y.upper()
    symbol_x = symbol_x.upper()

    initial_cost = config.transaction_cost_rate
    investable_capital = config.initial_capital * (1.0 - initial_cost)
    shares_y = 0.5 * investable_capital / float(pair[symbol_y].iloc[0])
    shares_x = 0.5 * investable_capital / float(pair[symbol_x].iloc[0])

    portfolio = shares_y * pair[symbol_y] + shares_x * pair[symbol_x]
    portfolio.iloc[0] = investable_capital
    strategy_return = portfolio.pct_change(fill_method=None).fillna(-initial_cost)
    drawdown = calculate_drawdown(portfolio)

    results = pd.DataFrame(
        {
            "portfolio": portfolio,
            "strategy_return": strategy_return,
            "drawdown": drawdown,
        },
        index=pair.index,
    )

    volatility = float(strategy_return.std(ddof=1))
    daily_rf = (1 + config.annual_risk_free_rate) ** (1 / config.periods_per_year) - 1
    excess = strategy_return - daily_rf
    years = _elapsed_years(results.index, config.periods_per_year)
    final_value = float(portfolio.iloc[-1])

    negative = strategy_return[strategy_return < 0]
    downside = float(negative.std(ddof=1))

    metrics: dict[str, float | int] = {
        "final_value": final_value,
        "total_return": float(final_value / config.initial_capital - 1.0),
        "annualized_return": float(
            (final_value / config.initial_capital) ** (1 / years) - 1
        ),
        "annualized_volatility": float(volatility * np.sqrt(config.periods_per_year)),
        "sharpe_ratio": float(
            excess.mean() / volatility * np.sqrt(config.periods_per_year)
        ) if volatility > 0 else np.nan,
        "sortino_ratio": float(
            excess.mean() / downside * np.sqrt(config.periods_per_year)
        ) if len(negative) >= 2 and downside > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "daily_win_rate": float((strategy_return[1:] > 0).mean()),
        "market_exposure": 1.0,
        "average_gross_exposure": 1.0,
        "average_absolute_net_exposure": 1.0,
        "total_turnover": 1.0,
        "number_of_rebalance_days": 1,
        "number_of_leg_orders": 2,
        "number_of_completed_trades": 0,
        "trade_win_rate": np.nan,
        "average_trade_return": np.nan,
        "best_trade_return": np.nan,
        "worst_trade_return": np.nan,
        "average_holding_days": np.nan,
        "profit_factor": np.nan,
    }
    return results, metrics
