"""Spread construction and z-score signal generation for pairs trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pair_data import align_pair_prices
from pair_selection import calculate_spread


@dataclass(frozen=True)
class PairStrategyConfig:
    """Entry, exit, stop, and holding-period rules for the spread."""

    zscore_lookback: int = 40
    entry_zscore: float = 2.0
    exit_zscore: float = 0.25
    stop_zscore: float = 4.0
    maximum_holding_days: int = 60

    def validate(self) -> None:
        if self.zscore_lookback <= 2:
            raise ValueError("zscore_lookback must be greater than two")
        if self.entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if self.exit_zscore < 0:
            raise ValueError("exit_zscore cannot be negative")
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("exit_zscore must be smaller than entry_zscore")
        if self.stop_zscore <= self.entry_zscore:
            raise ValueError("stop_zscore must exceed entry_zscore")
        if self.maximum_holding_days <= 0:
            raise ValueError("maximum_holding_days must be positive")


def build_pair_signal(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    alpha: float,
    beta: float,
    config: PairStrategyConfig | None = None,
    *,
    use_log_prices: bool = True,
) -> pd.DataFrame:
    """Create a stateful long-spread/short-spread target-position series.

    Position meanings
    -----------------
    1:
        Long the spread: long Y and short beta-adjusted X.
    0:
        Flat.
    -1:
        Short the spread: short Y and long beta-adjusted X.
    """

    config = config or PairStrategyConfig()
    config.validate()
    pair = align_pair_prices(prices, symbol_y, symbol_x)

    spread = calculate_spread(
        pair[symbol_y.upper()],
        pair[symbol_x.upper()],
        alpha,
        beta,
        use_log_prices=use_log_prices,
    )
    rolling_mean = spread.rolling(
        config.zscore_lookback,
        min_periods=config.zscore_lookback,
    ).mean()
    rolling_std = spread.rolling(
        config.zscore_lookback,
        min_periods=config.zscore_lookback,
    ).std(ddof=0)
    zscore = spread.sub(rolling_mean).div(rolling_std.replace(0.0, np.nan))

    target_positions: list[float] = []
    events: list[str] = []
    holding_days_series: list[int] = []

    current_position = 0.0
    holding_days = 0

    for value in zscore:
        event = ""

        if pd.isna(value):
            current_position = 0.0
            holding_days = 0

        elif current_position == 0.0:
            holding_days = 0
            if value <= -config.entry_zscore:
                current_position = 1.0
                holding_days = 1
                event = "Enter Long Spread"
            elif value >= config.entry_zscore:
                current_position = -1.0
                holding_days = 1
                event = "Enter Short Spread"

        elif current_position == 1.0:
            holding_days += 1
            if value <= -config.stop_zscore:
                current_position = 0.0
                holding_days = 0
                event = "Stop Long Spread"
            elif value >= -config.exit_zscore:
                current_position = 0.0
                holding_days = 0
                event = "Exit Long Spread"
            elif holding_days >= config.maximum_holding_days:
                current_position = 0.0
                holding_days = 0
                event = "Time Exit Long Spread"

        elif current_position == -1.0:
            holding_days += 1
            if value >= config.stop_zscore:
                current_position = 0.0
                holding_days = 0
                event = "Stop Short Spread"
            elif value <= config.exit_zscore:
                current_position = 0.0
                holding_days = 0
                event = "Exit Short Spread"
            elif holding_days >= config.maximum_holding_days:
                current_position = 0.0
                holding_days = 0
                event = "Time Exit Short Spread"

        target_positions.append(current_position)
        events.append(event)
        holding_days_series.append(holding_days)

    return pd.DataFrame(
        {
            symbol_y.upper(): pair[symbol_y.upper()],
            symbol_x.upper(): pair[symbol_x.upper()],
            "spread": spread,
            "rolling_spread_mean": rolling_mean,
            "rolling_spread_std": rolling_std,
            "zscore": zscore,
            "target_position": pd.Series(
                target_positions,
                index=pair.index,
                dtype=float,
            ),
            "signal_event": events,
            "signal_holding_days": holding_days_series,
        },
        index=pair.index,
    )
