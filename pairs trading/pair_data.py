"""Historical market-data helpers for the pairs-trading project.

The default research path uses adjusted daily prices from yfinance because it
works without brokerage credentials. Alpaca market data is used by the paper-
trading adapter in ``alpaca_paper.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalise_symbols(symbols: Iterable[str]) -> list[str]:
    cleaned = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    if len(cleaned) < 2:
        raise ValueError("at least two unique symbols are required")
    return cleaned


def _cache_path(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_directory: Path,
) -> Path:
    payload = "|".join(symbols + [start, end or "latest"]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return cache_directory / f"adjusted_close_{digest}.csv"


def _extract_close_frame(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("the market-data provider returned no rows")

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = set(raw.columns.get_level_values(0))
        level_1 = set(raw.columns.get_level_values(1))

        if "Close" in level_0:
            close = raw["Close"].copy()
        elif "Close" in level_1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("downloaded data does not contain a Close field")
    else:
        if "Close" not in raw.columns:
            raise ValueError("downloaded data does not contain a Close field")
        close = raw[["Close"]].copy()
        if len(symbols) == 1:
            close.columns = symbols

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    close.columns = [str(column).upper() for column in close.columns]
    close = close.reindex(columns=[symbol for symbol in symbols if symbol in close.columns])
    return close


def clean_price_frame(
    prices: pd.DataFrame,
    minimum_coverage: float = 0.90,
) -> pd.DataFrame:
    """Validate, sort, and filter a multi-asset adjusted-price frame."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")

    clean = prices.copy()
    clean.columns = [str(column).strip().upper() for column in clean.columns]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    if isinstance(clean.index, pd.DatetimeIndex) and clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)

    clean = clean.dropna(how="all")
    if clean.empty:
        raise ValueError("prices contains no usable observations")

    coverage = clean.notna().mean()
    keep = coverage[coverage >= minimum_coverage].index.tolist()
    clean = clean[keep]

    if clean.shape[1] < 2:
        raise ValueError(
            "fewer than two symbols met the minimum data-coverage requirement"
        )
    if (clean <= 0).any().any():
        raise ValueError("all available prices must be greater than zero")

    return clean


def download_adjusted_close_prices(
    symbols: Iterable[str],
    start: str,
    end: str | None = None,
    *,
    refresh: bool = False,
    cache_directory: str | Path = "data_cache",
    minimum_coverage: float = 0.90,
) -> pd.DataFrame:
    """Download adjusted daily close prices for a candidate universe.

    Results are cached as CSV so repeated research runs use the same local data
    unless ``refresh=True`` is requested.
    """

    symbols = _normalise_symbols(symbols)
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(symbols, start, end, cache_directory)

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return clean_price_frame(cached, minimum_coverage=minimum_coverage)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install yfinance before downloading market data"
        ) from exc

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    close = _extract_close_frame(raw, symbols)
    close.index.name = "Date"
    close = clean_price_frame(close, minimum_coverage=minimum_coverage)
    close.to_csv(cache_path)
    return close


def align_pair_prices(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
) -> pd.DataFrame:
    """Return two strictly aligned positive price series."""

    symbol_y = symbol_y.upper()
    symbol_x = symbol_x.upper()
    missing = {symbol_y, symbol_x}.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing symbols: {sorted(missing)}")

    pair = prices[[symbol_y, symbol_x]].dropna().astype(float).sort_index()
    if len(pair) < 3:
        raise ValueError("the selected pair has too few overlapping observations")
    if (pair <= 0).any().any():
        raise ValueError("pair prices must be strictly positive")

    pair.columns = [symbol_y, symbol_x]
    return pair
