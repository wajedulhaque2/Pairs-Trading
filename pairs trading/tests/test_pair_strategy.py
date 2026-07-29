import numpy as np
import pandas as pd
import pytest

from pair_strategy import PairStrategyConfig, build_pair_signal


def prices_from_spread(spread_values):
    observations = len(spread_values)
    dates = pd.date_range("2025-01-01", periods=observations, freq="B")
    log_x = np.log(100.0) + np.linspace(0, 0.02, observations)
    log_y = log_x + np.asarray(spread_values, dtype=float)
    return pd.DataFrame({"Y": np.exp(log_y), "X": np.exp(log_x)}, index=dates)


def test_strategy_can_enter_long_and_short_spreads():
    spread = [0, 0.01, -0.01, 0, -0.10, -0.12, 0.00, 0.10, 0.12, 0.00]
    prices = prices_from_spread(spread)
    config = PairStrategyConfig(
        zscore_lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.2,
        stop_zscore=5.0,
        maximum_holding_days=20,
    )

    result = build_pair_signal(prices, "Y", "X", 0.0, 1.0, config)

    assert 1.0 in result["target_position"].values
    assert -1.0 in result["target_position"].values
    assert set(result["target_position"].unique()).issubset({-1.0, 0.0, 1.0})


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        PairStrategyConfig(
            entry_zscore=1.0,
            exit_zscore=1.0,
            stop_zscore=2.0,
        ).validate()
