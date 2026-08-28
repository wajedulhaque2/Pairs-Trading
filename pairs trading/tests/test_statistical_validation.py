import numpy as np
import pandas as pd
import pytest

from pair_selection import PairSelectionConfig
from statistical_validation import (
    apply_fdr_control,
    benjamini_hochberg,
    relationship_stability_summary,
    rolling_pair_diagnostics,
    select_tradeable_pair,
    walk_forward_pair_candidates,
)


def test_benjamini_hochberg_known_values():
    p = pd.Series([0.001, 0.01, 0.04, 0.20])
    q = benjamini_hochberg(p)
    assert q.tolist() == pytest.approx([0.004, 0.02, 0.0533333333, 0.20])


def test_strict_selector_returns_none_when_no_pair_survives():
    screened = pd.DataFrame(
        {
            "cointegration_pvalue": [0.02, 0.20],
            "eligible": [False, False],
            "symbol_y": ["A", "C"],
            "symbol_x": ["B", "D"],
        }
    )
    assert select_tradeable_pair(screened) is None


def test_fdr_can_remove_raw_significance_from_large_family():
    pvalues = [0.01] + [0.20] * 19
    screened = pd.DataFrame(
        {
            "cointegration_pvalue": pvalues,
            "eligible": [True] + [False] * 19,
        }
    )
    adjusted = apply_fdr_control(screened, alpha=0.05)
    assert adjusted.loc[0, "cointegration_qvalue"] == pytest.approx(0.20)
    assert not bool(adjusted.loc[0, "eligible_fdr"])


def synthetic_universe(seed=17, observations=720):
    rng = np.random.default_rng(seed)
    log_x = np.log(100.0) + np.cumsum(rng.normal(0, 0.008, observations))
    spread = np.zeros(observations)
    for index in range(1, observations):
        spread[index] = 0.80 * spread[index - 1] + rng.normal(0, 0.008)
    log_y = 0.2 + 0.95 * log_x + spread
    log_z = np.log(90.0) + np.cumsum(rng.normal(0, 0.012, observations))
    dates = pd.date_range("2018-01-01", periods=observations, freq="B")
    return pd.DataFrame(
        {"X": np.exp(log_x), "Y": np.exp(log_y), "Z": np.exp(log_z)},
        index=dates,
    )


def test_rolling_diagnostics_quantify_relationship_stability():
    prices = synthetic_universe()
    config = PairSelectionConfig(
        minimum_observations=200,
        minimum_absolute_return_correlation=0.0,
        maximum_cointegration_pvalue=0.10,
        maximum_adf_pvalue=0.10,
        minimum_half_life=0.5,
        maximum_half_life=100.0,
        use_log_prices=True,
    )
    diagnostics = rolling_pair_diagnostics(
        prices, "Y", "X", window=250, step=100, config=config
    )
    summary = relationship_stability_summary(diagnostics)
    assert len(diagnostics) >= 4
    assert np.isfinite(diagnostics["beta"]).all()
    assert 0 <= summary["eligible_window_fraction"] <= 1
    assert abs(summary["beta_mean"] - 0.95) < 0.10


def test_walk_forward_reselects_without_using_test_window():
    prices = synthetic_universe(observations=760)
    config = PairSelectionConfig(
        minimum_observations=250,
        minimum_absolute_return_correlation=0.0,
        maximum_cointegration_pvalue=0.10,
        maximum_adf_pvalue=0.10,
        minimum_half_life=0.5,
        maximum_half_life=100.0,
        use_log_prices=True,
    )
    result = walk_forward_pair_candidates(
        prices,
        config,
        train_periods=400,
        test_periods=100,
        step_periods=100,
        fdr_alpha=0.10,
    )
    assert len(result) >= 3
    for index in range(len(result)):
        assert result.loc[index, "train_end"] < result.loc[index, "test_start"]
    selected = result[~result["no_trade"]]
    assert not selected.empty
    assert any(
        set(row) == {"X", "Y"}
        for row in selected[["symbol_y", "symbol_x"]].itertuples(
            index=False, name=None
        )
    )
