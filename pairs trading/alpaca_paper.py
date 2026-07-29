"""Dry-run-first Alpaca paper-trading adapter for the selected pair.

The script never submits an order unless ``--execute-paper`` is supplied. It
loads the pair and strategy parameters exported by ``main.py``, requests recent
daily bars from Alpaca, calculates the latest target, and proposes integer-share
orders for the two legs.

This is an educational paper-trading bridge, not production execution code.
The two stock orders are not atomic, so leg risk remains.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pair_backtester import calculate_leg_weights
from pair_strategy import PairStrategyConfig, build_pair_signal


def _load_selected_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run main.py before using the paper adapter."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret_key:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env file"
        )
    return api_key, secret_key


def _download_alpaca_daily_prices(
    api_key: str,
    secret_key: str,
    symbols: list[str],
    calendar_days: int,
) -> pd.DataFrame:
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise RuntimeError("Install alpaca-py before running this script") from exc

    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=calendar_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request)
    frame = bars.df.copy()
    if frame.empty:
        raise RuntimeError("Alpaca returned no daily bars")

    frame = frame.reset_index()
    if "symbol" not in frame.columns or "timestamp" not in frame.columns:
        raise RuntimeError("unexpected Alpaca bar response shape")

    close = frame.pivot(index="timestamp", columns="symbol", values="close")
    close = close.reindex(columns=symbols).dropna().astype(float).sort_index()
    if close.empty:
        raise RuntimeError("the two Alpaca price histories do not overlap")

    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


def _signed_current_quantities(trading_client: Any, symbols: list[str]) -> dict[str, int]:
    quantities = {symbol: 0 for symbol in symbols}
    for position in trading_client.get_all_positions():
        symbol = str(position.symbol).upper()
        if symbol not in quantities:
            continue

        quantity = abs(float(position.qty))
        side_text = str(position.side).lower()
        quantities[symbol] = int(round(-quantity if "short" in side_text else quantity))
    return quantities


def _validate_assets(trading_client: Any, target_quantities: dict[str, int]) -> None:
    for symbol, quantity in target_quantities.items():
        asset = trading_client.get_asset(symbol)
        if not bool(asset.tradable):
            raise RuntimeError(f"{symbol} is not tradable in this Alpaca account")
        if quantity < 0 and not bool(asset.shortable):
            raise RuntimeError(f"{symbol} is not currently marked shortable")


def _build_order_plan(
    current_quantities: dict[str, int],
    target_quantities: dict[str, int],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for symbol, target in target_quantities.items():
        current = current_quantities.get(symbol, 0)
        delta = int(target - current)
        if delta == 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "current_quantity": current,
                "target_quantity": target,
                "delta_quantity": delta,
                "side": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta),
            }
        )

    # Reductions are placed before orders that increase absolute exposure.
    def reduction_first(order: dict[str, Any]) -> tuple[int, str]:
        current = int(order["current_quantity"])
        target = int(order["target_quantity"])
        reducing = abs(target) < abs(current) or (current != 0 and np.sign(target) != np.sign(current))
        return (0 if reducing else 1, str(order["symbol"]))

    return sorted(orders, key=reduction_first)


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-json",
        type=Path,
        default=Path("outputs/selected_pair_and_parameters.json"),
    )
    parser.add_argument(
        "--gross-exposure-fraction",
        type=float,
        default=0.20,
        help="Fraction of paper-account equity allocated to gross pair exposure",
    )
    parser.add_argument(
        "--calendar-days",
        type=int,
        default=500,
        help="Calendar days of daily bars requested for indicator warm-up",
    )
    parser.add_argument(
        "--execute-paper",
        action="store_true",
        help="Actually submit orders to Alpaca paper trading",
    )
    args = parser.parse_args()

    if not 0 < args.gross_exposure_fraction <= 1:
        raise ValueError("gross-exposure-fraction must be in (0, 1]")
    if args.calendar_days < 100:
        raise ValueError("calendar-days must be at least 100")

    payload = _load_selected_payload(args.selected_json)
    api_key, secret_key = _require_credentials()

    pair_info = payload["pair"]
    strategy_info = payload["strategy"]
    symbol_y = str(pair_info["symbol_y"]).upper()
    symbol_x = str(pair_info["symbol_x"]).upper()
    symbols = [symbol_y, symbol_x]

    prices = _download_alpaca_daily_prices(
        api_key,
        secret_key,
        symbols,
        args.calendar_days,
    )
    strategy_config = PairStrategyConfig(**strategy_info)
    signal = build_pair_signal(
        prices,
        symbol_y,
        symbol_x,
        float(pair_info["alpha"]),
        float(pair_info["beta"]),
        strategy_config,
        use_log_prices=bool(pair_info.get("use_log_prices", True)),
    )

    latest_target = int(signal["target_position"].iloc[-1])
    latest_zscore = float(signal["zscore"].iloc[-1])
    latest_prices = prices.iloc[-1]

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
    except ImportError as exc:
        raise RuntimeError("Install alpaca-py before running this script") from exc

    trading_client = TradingClient(api_key, secret_key, paper=True)
    account = trading_client.get_account()
    equity = float(account.equity)
    gross_dollars = equity * args.gross_exposure_fraction

    latest_weights = calculate_leg_weights(
        pd.Series([latest_target], index=[prices.index[-1]], dtype=float),
        float(pair_info["beta"]),
    ).iloc[-1]

    target_quantities = {
        symbol_y: int(round(gross_dollars * latest_weights["weight_y"] / latest_prices[symbol_y])),
        symbol_x: int(round(gross_dollars * latest_weights["weight_x"] / latest_prices[symbol_x])),
    }

    _validate_assets(trading_client, target_quantities)
    current_quantities = _signed_current_quantities(trading_client, symbols)
    order_plan = _build_order_plan(current_quantities, target_quantities)

    print("\nALPACA PAPER-TRADING PAIR PLAN")
    print("=" * 70)
    print(f"Pair: {symbol_y}/{symbol_x}")
    print(f"Latest completed daily bar: {prices.index[-1].date()}")
    print(f"Latest z-score: {latest_zscore:.3f}")
    print(f"Target spread position: {latest_target}")
    print(f"Paper account equity: ${equity:,.2f}")
    print(f"Gross allocation: ${gross_dollars:,.2f}")
    print(f"Current quantities: {current_quantities}")
    print(f"Target quantities:  {target_quantities}")

    if not order_plan:
        print("No orders are required.")
        return {
            "target_position": latest_target,
            "zscore": latest_zscore,
            "orders": [],
        }

    print("\nProposed orders:")
    print(pd.DataFrame(order_plan).to_string(index=False))

    if not args.execute_paper:
        print("\nDRY RUN ONLY — no orders were submitted.")
        print("Rerun with --execute-paper to submit to the paper account.")
        return {
            "target_position": latest_target,
            "zscore": latest_zscore,
            "orders": order_plan,
        }

    print(
        "\nSubmitting separate market orders to PAPER trading. "
        "The legs are not atomic and can fill at different times."
    )
    submitted: list[dict[str, Any]] = []
    for order in order_plan:
        request = MarketOrderRequest(
            symbol=order["symbol"],
            qty=order["quantity"],
            side=OrderSide.BUY if order["side"] == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"pairs-{uuid.uuid4().hex[:20]}",
        )
        response = trading_client.submit_order(order_data=request)
        submitted.append(
            {
                **order,
                "order_id": str(response.id),
                "status": str(response.status),
            }
        )
        print(f"Submitted {order['side']} {order['quantity']} {order['symbol']}")

    return {
        "target_position": latest_target,
        "zscore": latest_zscore,
        "orders": submitted,
    }


if __name__ == "__main__":
    main()
