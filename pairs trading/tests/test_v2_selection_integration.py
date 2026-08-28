import pandas as pd
import pytest

import research
from alpaca_paper import _require_tradeable_selection
from pair_selection import PairSelectionConfig


def _selection_config():
    return PairSelectionConfig(
        minimum_observations=30,
        minimum_absolute_return_correlation=0.0,
        maximum_cointegration_pvalue=0.05,
        maximum_adf_pvalue=0.05,
        minimum_half_life=1.0,
        maximum_half_life=100.0,
    )


def test_choose_pair_marks_fdr_rejected_raw_candidate_as_research_fallback(monkeypatch):
    ranking = pd.DataFrame(
        {
            "symbol_y": ["A"] + [f"Y{i}" for i in range(1, 20)],
            "symbol_x": ["B"] + [f"X{i}" for i in range(1, 20)],
            "cointegration_pvalue": [0.01] + [0.20] * 19,
            "eligible": [True] + [False] * 19,
        }
    )
    monkeypatch.setattr(research, "screen_pairs", lambda *_args, **_kwargs: ranking)

    selected, validated = research.choose_pair(pd.DataFrame(), _selection_config())

    assert selected["symbol_y"] == "A"
    assert bool(selected["eligible_raw"])
    assert not bool(selected["eligible"])
    assert selected["selection_mode"] == "research_fallback"
    assert validated.loc[0, "cointegration_qvalue"] == pytest.approx(0.20)
    assert not bool(validated.loc[0, "eligible_fdr"])


def test_choose_pair_keeps_strict_fdr_valid_candidate_tradeable(monkeypatch):
    ranking = pd.DataFrame(
        {
            "symbol_y": ["A", "C"],
            "symbol_x": ["B", "D"],
            "cointegration_pvalue": [0.001, 0.90],
            "eligible": [True, False],
        }
    )
    monkeypatch.setattr(research, "screen_pairs", lambda *_args, **_kwargs: ranking)

    selected, validated = research.choose_pair(pd.DataFrame(), _selection_config())

    assert selected["symbol_y"] == "A"
    assert bool(selected["eligible"])
    assert selected["selection_mode"] == "strict_fdr"
    assert validated.loc[0, "cointegration_qvalue"] == pytest.approx(0.002)
    assert bool(validated.loc[0, "eligible_fdr"])


def test_paper_adapter_rejects_research_fallback_before_execution():
    with pytest.raises(RuntimeError, match="research fallback"):
        _require_tradeable_selection({"eligible": False})


def test_paper_adapter_accepts_strict_tradeable_pair():
    _require_tradeable_selection({"eligible": True})
