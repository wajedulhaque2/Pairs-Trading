import pandas as pd
import pytest

from pair_data import clean_price_frame


def test_price_cleaner_sorts_dates_and_removes_low_coverage_symbol():
    dates = pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"])
    prices = pd.DataFrame(
        {
            "A": [103, 100, 101],
            "B": [203, 200, 201],
            "C": [None, 50, None],
        },
        index=dates,
    )

    clean = clean_price_frame(prices, minimum_coverage=0.80)

    assert list(clean.columns) == ["A", "B"]
    assert clean.index.is_monotonic_increasing


def test_price_cleaner_requires_two_symbols():
    prices = pd.DataFrame(
        {"A": [100, 101, 102], "B": [None, None, 100]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    with pytest.raises(ValueError):
        clean_price_frame(prices, minimum_coverage=0.90)
