"""Multiple-testing control and relationship-stability checks for pair research."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pair_data import align_pair_prices
from pair_selection import PairSelectionConfig, evaluate_pair, screen_pairs


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    values = pd.to_numeric(pvalues, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result
    if ((valid < 0) | (valid > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")

    ordered = valid.sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1, dtype=float)
    raw_adjusted = ordered.to_numpy() * m / ranks
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted = np.clip(monotone, 0.0, 1.0)
    result.loc[ordered.index] = adjusted
    return result


def apply_fdr_control(
    screening_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add cointegration q-values and an FDR-aware eligibility flag.

    Engle-Granger cointegration is treated as the primary family of hypothesis
    tests across candidate pairs. The existing ADF/correlation/half-life rules
    remain part of ``eligible`` rather than being treated as an independent
    second family of tests.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    required = {"cointegration_pvalue", "eligible"}
    missing = required.difference(screening_results.columns)
    if missing:
        raise ValueError(f"screening_results is missing: {sorted(missing)}")

    results = screening_results.copy()
    results["cointegration_qvalue"] = benjamini_hochberg(
        results["cointegration_pvalue"]
    )
    results["cointegration_fdr_reject"] = (
        results["cointegration_qvalue"] <= alpha
    )
    results["eligible_fdr"] = (
        results["eligible"].astype(bool)
        & results["cointegration_fdr_reject"].astype(bool)
    )
    return results


def select_tradeable_pair(
    screening_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | None:
    """Return the best FDR-valid pair, or ``None`` when evidence is insufficient."""
    results = (
        screening_results
        if "eligible_fdr" in screening_results.columns
        else apply_fdr_control(screening_results, alpha=alpha)
    )
    eligible = results[results["eligible_fdr"]]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def rolling_pair_diagnostics(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    window: int = 504,
    step: int = 21,
    config: PairSelectionConfig | None = None,
) -> pd.DataFrame:
    """Re-estimate the pair relationship on rolling windows to expose drift."""
    if window < 30:
        raise ValueError("window must be at least 30 observations")
    if step < 1:
        raise ValueError("step must be positive")
    base = config or PairSelectionConfig()
    window_config = replace(
        base,
        minimum_observations=min(base.minimum_observations, window),
    )

    pair = align_pair_prices(prices, symbol_y, symbol_x)
    if len(pair) < window:
        raise ValueError("not enough observations for one rolling window")

    rows = []
    for end in range(window, len(pair) + 1, step):
        sample = pair.iloc[end - window : end]
        diagnostic = evaluate_pair(sample, symbol_y, symbol_x, window_config)
        rows.append(
            {
                "window_start": sample.index[0],
                "window_end": sample.index[-1],
                **diagnostic,
            }
        )
    return pd.DataFrame(rows)


def relationship_stability_summary(
    diagnostics: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise how consistently a historical pair relationship survives."""
    required = {"beta", "eligible", "cointegration_pvalue", "half_life"}
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"diagnostics is missing: {sorted(missing)}")
    if diagnostics.empty:
        raise ValueError("diagnostics is empty")

    beta = pd.to_numeric(diagnostics["beta"], errors="coerce")
    half_life = pd.to_numeric(diagnostics["half_life"], errors="coerce")
    return {
        "windows": int(len(diagnostics)),
        "eligible_window_fraction": float(diagnostics["eligible"].mean()),
        "cointegrated_window_fraction": float(
            (diagnostics["cointegration_pvalue"] <= 0.05).mean()
        ),
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std(ddof=1)) if len(beta) > 1 else 0.0,
        "beta_range": float(beta.max() - beta.min()),
        "median_half_life": float(half_life.median()),
    }


def walk_forward_pair_candidates(
    prices: pd.DataFrame,
    selection_config: PairSelectionConfig,
    train_periods: int = 756,
    test_periods: int = 252,
    step_periods: int = 252,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Re-screen the universe before each unseen test window.

    A window explicitly records ``no_trade=True`` when no pair survives the
    original diagnostics plus FDR control.
    """
    if min(train_periods, test_periods, step_periods) <= 0:
        raise ValueError("walk-forward window lengths must be positive")
    clean = prices.sort_index()
    needed = train_periods + test_periods
    if len(clean) < needed:
        raise ValueError(f"at least {needed} observations are required")

    rows = []
    max_start = len(clean) - needed
    for start in range(0, max_start + 1, step_periods):
        train = clean.iloc[start : start + train_periods]
        test = clean.iloc[
            start + train_periods : start + train_periods + test_periods
        ]
        screened = apply_fdr_control(
            screen_pairs(train, selection_config), fdr_alpha
        )
        selected = select_tradeable_pair(screened, fdr_alpha)
        rows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "no_trade": selected is None,
                "symbol_y": None if selected is None else selected["symbol_y"],
                "symbol_x": None if selected is None else selected["symbol_x"],
                "cointegration_qvalue": (
                    np.nan
                    if selected is None
                    else float(selected["cointegration_qvalue"])
                ),
                "beta": np.nan if selected is None else float(selected["beta"]),
            }
        )
    return pd.DataFrame(rows)
