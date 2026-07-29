# Pairs Trading Research and Paper-Trading System

An end-to-end statistical-arbitrage research project covering pair selection, spread modelling, strategy testing and optional paper-trading execution.

## Overview

Pairs trading attempts to identify two historically related assets whose prices temporarily move away from their usual relationship.

Instead of predicting whether the overall market will rise or fall, the strategy trades the relative movement between the two assets:

* Buy the relatively undervalued asset
* Short the relatively overvalued asset
* Exit when their relationship moves back toward its historical level

This project examines the full research process rather than applying a strategy to a manually selected pair.

## Start Here

Readers who are mainly interested in the theory and results should begin with:

1. **`complete_pairs_trading_theory_guide_fixed.ipynb`**
   Explains correlation, cointegration, stationarity, hedge ratios, spread construction, z-scores, mean reversion and the risks of pairs trading.

2. **`main.ipynb`**
   Runs the complete workflow, including market-data preparation, pair screening, parameter selection, backtesting and out-of-sample evaluation.

3. **`alpaca_paper.py`**
   Shows how the selected strategy can be connected to an Alpaca paper-trading account. It operates in dry-run mode unless paper execution is explicitly enabled.

The remaining Python files contain the pair-selection, signal-generation, research and backtesting modules used by the main notebook.

## Research Workflow

The system:

1. Downloads and cleans adjusted historical prices
2. Screens possible asset pairs
3. Measures return correlation
4. Tests for cointegration and spread stationarity
5. Estimates the hedge ratio using linear regression
6. Calculates the spread’s estimated mean-reversion half-life
7. Generates z-score-based trading signals
8. Optimises strategy parameters using training data
9. Evaluates the selected pair on separate out-of-sample data
10. Produces an optional paper-trading order plan

## Key Features

* Automated pair screening and ranking
* Correlation, cointegration and ADF testing
* Regression-based hedge-ratio estimation
* Mean-reversion half-life estimation
* Long-spread and short-spread signals
* Entry, exit, stop-loss and maximum-holding-period rules
* Delayed signal execution
* Transaction-cost and short-borrow-cost modelling
* Gross-normalised two-leg portfolio weights
* Trade-level and portfolio-level performance metrics
* Training-only parameter optimisation
* Out-of-sample strategy evaluation
* Passive benchmark comparison
* Dry-run-first Alpaca paper-trading integration

## Purpose

This project demonstrates how a statistical trading idea can be developed from theory into a structured research and execution workflow.

It also highlights important limitations. Historical relationships can break down, cointegration does not guarantee profitability, short positions introduce additional risks and two separate orders may not execute simultaneously.

The system is intended for education, research and paper trading—not live financial advice or production trading.
