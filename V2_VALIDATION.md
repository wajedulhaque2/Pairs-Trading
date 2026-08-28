# Pairs Trading V2: Statistical Validation

This upgrade strengthens the research process without changing the original saved UPS/V case study. The new code is in `pairs trading/statistical_validation.py`.

## Multiple-testing control

Screening 16 stocks produces 120 candidate pairs. Testing many cointegration hypotheses increases the chance of finding apparently significant p-values by luck. `apply_fdr_control()` therefore applies the Benjamini-Hochberg false-discovery-rate procedure to the family of Engle-Granger cointegration tests.

The original `eligible` flag still contains the correlation, cointegration, ADF and half-life rules. V2 adds `eligible_fdr`, which requires the original rules **and** survival of the cross-sectional FDR correction.

ADF remains a diagnostic on the fitted spread and is not presented as an independent second family of multiple tests.

```python
from statistical_validation import apply_fdr_control, select_tradeable_pair

screened = screen_pairs(training_prices, selection_config)
validated = apply_fdr_control(screened, alpha=0.05)
selected = select_tradeable_pair(validated)

if selected is None:
    print("No trade: no pair has sufficient evidence.")
```

The strict selector deliberately returns `None` rather than forcing a fallback candidate.

## Rolling relationship diagnostics

`rolling_pair_diagnostics()` repeatedly re-estimates correlation, hedge ratio, cointegration, ADF evidence and half-life on rolling windows. `relationship_stability_summary()` then reports how often the pair remains eligible and how much beta varies.

This is intended to answer a different question from the original full-sample test: **does the relationship stay economically and statistically similar through time?**

## Walk-forward pair discovery

`walk_forward_pair_candidates()` re-screens the entire candidate universe before each unseen test window. Each row records either the FDR-valid pair selected from the training sample or an explicit `no_trade=True` state.

This separates pair discovery from future evaluation and provides the basis for a later portfolio-level implementation in the multi-asset Backtesting Engine.

## Verification

Synthetic tests plant a cointegrated X/Y relationship with beta near 0.95 and an unrelated Z series. The tests verify known Benjamini-Hochberg values, demonstrate loss of raw significance after multiplicity correction, recover the planted rolling beta and enforce train-before-test chronology. GitHub Actions runs these tests together with the repository's existing pair-data, selection, signal and backtester tests.

## Scope

This V2 does not claim that a pair which passes statistical tests is profitable. It strengthens the evidence required before the strategy is allowed to trade. Dynamic hedge ratios such as Kalman filtering and a diversified portfolio of simultaneously eligible pairs remain natural later extensions once the stricter validation pipeline is established.
