import numpy as np
import pandas as pd
import pytest

from pair_backtester import (
    PairBacktestConfig,
    calculate_leg_weights,
    run_pair_backtest,
)


def make_prices(y_values, x_values):
    dates = pd.date_range("2025-01-01", periods=len(y_values), freq="B")
    return pd.DataFrame({"Y": y_values, "X": x_values}, index=dates, dtype=float)


def test_leg_weights_have_unit_gross_exposure():
    position = pd.Series([1.0, -1.0, 0.0])
    weights = calculate_leg_weights(position, beta=1.5)

    np.testing.assert_allclose(weights.abs().sum(axis=1), [1.0, 1.0, 0.0])
    assert weights.iloc[0]["weight_y"] > 0
    assert weights.iloc[0]["weight_x"] < 0


def test_execution_delay_prevents_same_day_return():
    prices = make_prices([100, 110, 110], [100, 100, 100])
    target = pd.Series([1.0, 1.0, 1.0], index=prices.index)
    config = PairBacktestConfig(
        transaction_cost_rate=0.0,
        annual_borrow_rate=0.0,
        execution_delay=1,
    )

    results, _, _ = run_pair_backtest(prices, "Y", "X", target, beta=1.0, config=config)

    assert results["spread_position"].iloc[0] == 0.0
    assert results["strategy_return"].iloc[1] > 0
    assert results["strategy_return"].iloc[0] == 0.0


def test_entry_and_exit_charge_both_legs():
    prices = make_prices([100, 100, 100, 100], [100, 100, 100, 100])
    target = pd.Series([1.0, 1.0, 0.0, 0.0], index=prices.index)
    config = PairBacktestConfig(
        transaction_cost_rate=0.001,
        annual_borrow_rate=0.0,
        execution_delay=1,
    )

    results, metrics, _ = run_pair_backtest(prices, "Y", "X", target, beta=1.0, config=config)

    # Gross-normalised entry and exit each create total turnover of one.
    assert results["turnover"].sum() == pytest.approx(2.0)
    assert results["transaction_cost"].sum() == pytest.approx(0.002)
    assert metrics["number_of_leg_orders"] == 4


def test_short_leg_borrow_cost_is_charged():
    prices = make_prices([100, 100, 100], [100, 100, 100])
    target = pd.Series([1.0, 1.0, 1.0], index=prices.index)
    config = PairBacktestConfig(
        transaction_cost_rate=0.0,
        annual_borrow_rate=0.10,
        execution_delay=0,
    )

    results, _, _ = run_pair_backtest(prices, "Y", "X", target, beta=1.0, config=config)

    assert (results["borrow_cost"] > 0).all()
    assert (results["strategy_return"] < 0).all()


def test_fractional_spread_position_is_rejected():
    prices = make_prices([100, 101, 102], [100, 100, 100])
    target = pd.Series([0.0, 0.5, 0.0], index=prices.index)

    with pytest.raises(ValueError):
        run_pair_backtest(prices, "Y", "X", target, beta=1.0)
