"""Pair screening, hedge-ratio estimation, and stationarity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, coint

from pair_data import align_pair_prices, clean_price_frame


@dataclass(frozen=True)
class PairSelectionConfig:
    """Rules used to screen and rank candidate stock pairs."""

    minimum_observations: int = 504
    minimum_absolute_return_correlation: float = 0.60
    maximum_cointegration_pvalue: float = 0.05
    maximum_adf_pvalue: float = 0.05
    minimum_half_life: float = 2.0
    maximum_half_life: float = 120.0
    use_log_prices: bool = True

    def validate(self) -> None:
        if self.minimum_observations < 30:
            raise ValueError("minimum_observations must be at least 30")
        if not 0 <= self.minimum_absolute_return_correlation <= 1:
            raise ValueError(
                "minimum_absolute_return_correlation must be between 0 and 1"
            )
        if not 0 < self.maximum_cointegration_pvalue <= 1:
            raise ValueError("maximum_cointegration_pvalue must be in (0, 1]")
        if not 0 < self.maximum_adf_pvalue <= 1:
            raise ValueError("maximum_adf_pvalue must be in (0, 1]")
        if self.minimum_half_life <= 0:
            raise ValueError("minimum_half_life must be positive")
        if self.maximum_half_life <= self.minimum_half_life:
            raise ValueError("maximum_half_life must exceed minimum_half_life")


def transform_prices(prices: pd.Series, use_log_prices: bool = True) -> pd.Series:
    clean = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if (clean <= 0).any():
        raise ValueError("prices must be positive")
    return np.log(clean) if use_log_prices else clean


def fit_hedge_ratio(
    price_y: pd.Series,
    price_x: pd.Series,
    *,
    use_log_prices: bool = True,
) -> dict[str, float | int]:
    """Fit ``Y = alpha + beta * X`` using scikit-learn OLS.

    ``beta`` is the hedge ratio used to form the spread:

        spread = Y - alpha - beta * X
    """

    aligned = pd.concat(
        [price_y.rename("y"), price_x.rename("x")], axis=1
    ).dropna()
    if len(aligned) < 3:
        raise ValueError("at least three aligned observations are required")

    y = transform_prices(aligned["y"], use_log_prices=use_log_prices).to_numpy()
    x = transform_prices(aligned["x"], use_log_prices=use_log_prices).to_numpy().reshape(-1, 1)

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    return {
        "alpha": float(model.intercept_),
        "beta": float(model.coef_[0]),
        "r_squared": float(model.score(x, y)),
        "observations": int(len(aligned)),
    }


def calculate_spread(
    price_y: pd.Series,
    price_x: pd.Series,
    alpha: float,
    beta: float,
    *,
    use_log_prices: bool = True,
) -> pd.Series:
    """Construct the regression residual used as the trading spread."""

    aligned = pd.concat(
        [price_y.rename("y"), price_x.rename("x")], axis=1
    ).dropna()
    y = transform_prices(aligned["y"], use_log_prices=use_log_prices)
    x = transform_prices(aligned["x"], use_log_prices=use_log_prices)
    spread = y - float(alpha) - float(beta) * x
    spread.name = "spread"
    return spread


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from an AR(1)-style regression."""

    clean = pd.to_numeric(spread, errors="coerce").dropna().astype(float)
    lagged = clean.shift(1)
    delta = clean.diff()
    regression_data = pd.concat(
        [delta.rename("delta"), lagged.rename("lagged")], axis=1
    ).dropna()

    if len(regression_data) < 10:
        return math.inf

    model = LinearRegression(fit_intercept=True)
    model.fit(regression_data[["lagged"]], regression_data["delta"])
    speed = float(model.coef_[0])

    if speed >= 0 or np.isclose(speed, 0.0):
        return math.inf

    return float(-math.log(2.0) / speed)


def evaluate_pair(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
    config: PairSelectionConfig | None = None,
) -> dict[str, float | int | str | bool]:
    """Calculate correlation, cointegration, ADF, hedge, and half-life statistics."""

    config = config or PairSelectionConfig()
    config.validate()
    pair = align_pair_prices(prices, symbol_y, symbol_x)

    if len(pair) < config.minimum_observations:
        raise ValueError(
            f"{symbol_y}/{symbol_x} has {len(pair)} observations; "
            f"{config.minimum_observations} are required"
        )

    returns = np.log(pair).diff().dropna()
    return_correlation = float(returns.iloc[:, 0].corr(returns.iloc[:, 1]))

    hedge = fit_hedge_ratio(
        pair.iloc[:, 0],
        pair.iloc[:, 1],
        use_log_prices=config.use_log_prices,
    )
    spread = calculate_spread(
        pair.iloc[:, 0],
        pair.iloc[:, 1],
        float(hedge["alpha"]),
        float(hedge["beta"]),
        use_log_prices=config.use_log_prices,
    )

    transformed_y = transform_prices(
        pair.iloc[:, 0], use_log_prices=config.use_log_prices
    )
    transformed_x = transform_prices(
        pair.iloc[:, 1], use_log_prices=config.use_log_prices
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cointegration_stat, cointegration_pvalue, _ = coint(
            transformed_y,
            transformed_x,
            trend="c",
            autolag="aic",
        )
        adf_stat, adf_pvalue, *_ = adfuller(spread, regression="c", autolag="AIC")

    half_life = estimate_half_life(spread)

    eligible = bool(
        abs(return_correlation) >= config.minimum_absolute_return_correlation
        and float(cointegration_pvalue) <= config.maximum_cointegration_pvalue
        and float(adf_pvalue) <= config.maximum_adf_pvalue
        and config.minimum_half_life <= half_life <= config.maximum_half_life
    )

    safe_coint_p = max(float(cointegration_pvalue), 1e-12)
    safe_adf_p = max(float(adf_pvalue), 1e-12)
    stationarity_score = (
        -math.log10(safe_coint_p)
        - math.log10(safe_adf_p)
        + abs(return_correlation)
    )

    return {
        "symbol_y": symbol_y.upper(),
        "symbol_x": symbol_x.upper(),
        "observations": int(len(pair)),
        "return_correlation": return_correlation,
        "absolute_return_correlation": abs(return_correlation),
        "cointegration_statistic": float(cointegration_stat),
        "cointegration_pvalue": float(cointegration_pvalue),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "alpha": float(hedge["alpha"]),
        "beta": float(hedge["beta"]),
        "regression_r_squared": float(hedge["r_squared"]),
        "half_life": float(half_life),
        "stationarity_score": float(stationarity_score),
        "eligible": eligible,
        "use_log_prices": bool(config.use_log_prices),
    }


def screen_pairs(
    prices: pd.DataFrame,
    config: PairSelectionConfig | None = None,
) -> pd.DataFrame:
    """Evaluate every pair in a universe and rank the strongest candidates."""

    config = config or PairSelectionConfig()
    config.validate()
    prices = clean_price_frame(prices, minimum_coverage=0.50)

    rows: list[dict[str, float | int | str | bool]] = []
    for symbol_a, symbol_b in combinations(prices.columns, 2):
        pair = prices[[symbol_a, symbol_b]].dropna()
        if len(pair) < config.minimum_observations:
            continue

        try:
            rows.append(evaluate_pair(pair, symbol_a, symbol_b, config))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("pair screening produced no valid candidates")

    return results.sort_values(
        by=[
            "eligible",
            "stationarity_score",
            "absolute_return_correlation",
            "half_life",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_best_pair(screening_results: pd.DataFrame) -> pd.Series:
    """Return the highest-ranked eligible pair, or the best fallback candidate."""

    if screening_results.empty:
        raise ValueError("screening_results is empty")

    eligible = screening_results[screening_results["eligible"]]
    if not eligible.empty:
        return eligible.iloc[0]
    return screening_results.iloc[0]
