import numpy as np
import pandas as pd

from pair_selection import (
    PairSelectionConfig,
    calculate_spread,
    evaluate_pair,
    fit_hedge_ratio,
)


def synthetic_cointegrated_prices(seed: int = 7, observations: int = 900):
    rng = np.random.default_rng(seed)
    log_x = np.log(100.0) + np.cumsum(rng.normal(0, 0.008, observations))

    spread = np.zeros(observations)
    for index in range(1, observations):
        spread[index] = 0.85 * spread[index - 1] + rng.normal(0, 0.01)

    alpha = 0.25
    beta = 0.92
    log_y = alpha + beta * log_x + spread
    dates = pd.date_range("2018-01-01", periods=observations, freq="B")

    return (
        pd.Series(np.exp(log_y), index=dates, name="Y"),
        pd.Series(np.exp(log_x), index=dates, name="X"),
        alpha,
        beta,
    )


def test_linear_regression_recovers_hedge_ratio():
    y, x, alpha, beta = synthetic_cointegrated_prices()
    fitted = fit_hedge_ratio(y, x, use_log_prices=True)

    assert abs(fitted["alpha"] - alpha) < 0.10
    assert abs(fitted["beta"] - beta) < 0.03
    assert fitted["r_squared"] > 0.95


def test_spread_matches_regression_residual():
    y, x, alpha, beta = synthetic_cointegrated_prices()
    spread = calculate_spread(y, x, alpha, beta, use_log_prices=True)

    expected = np.log(y) - alpha - beta * np.log(x)
    np.testing.assert_allclose(spread, expected)


def test_cointegrated_pair_passes_diagnostics():
    y, x, _, _ = synthetic_cointegrated_prices()
    prices = pd.concat([y, x], axis=1)
    config = PairSelectionConfig(
        minimum_observations=500,
        minimum_absolute_return_correlation=0.0,
        maximum_cointegration_pvalue=0.10,
        maximum_adf_pvalue=0.10,
        minimum_half_life=1.0,
        maximum_half_life=200.0,
        use_log_prices=True,
    )

    result = evaluate_pair(prices, "Y", "X", config)

    assert result["cointegration_pvalue"] < 0.10
    assert result["adf_pvalue"] < 0.10
    assert 1.0 <= result["half_life"] <= 200.0
