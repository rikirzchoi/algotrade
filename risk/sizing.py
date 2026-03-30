"""
risk.sizing — position sizing using the fixed-fractional method.

The fixed-fractional method risks a fixed percentage of current account
equity on every trade.  Given:

    risk_pct      — fraction of equity to risk (e.g. 0.01 for 1 %)
    account_equity — current net liquidation value from IB
    entry_price   — planned entry price
    stop_price    — planned initial stop price

The formula is:

    risk_per_share = |entry_price - stop_price|
    dollar_risk    = account_equity × risk_pct
    shares         = floor(dollar_risk / risk_per_share)

The result is capped by the per-symbol maximum from config and by the
maximum notional order value enforced by RiskManager.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal

logger = logging.getLogger(__name__)


def fixed_fractional(
    capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_price: float,
    max_position_size: int,
) -> int:
    """
    Calculate position size using the fixed-fractional method.

    Used by the backtester and any caller that does not have a RiskManager
    instance.

    Parameters
    ----------
    capital           : total account capital to risk against
    risk_per_trade_pct: fraction of capital to risk per trade (e.g. 0.01)
    entry_price       : planned fill price
    stop_price        : initial stop-loss price
    max_position_size : hard cap on position size

    Returns
    -------
    int
        Number of shares, minimum 1, maximum max_position_size.
    """
    risk_dollars = capital * risk_per_trade_pct
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return 1

    qty = math.floor(risk_dollars / risk_per_share)
    return max(1, min(qty, max_position_size))


def fixed_fractional_size(
    account_equity: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
    risk_pct: float,
    max_shares: int,
    min_shares: int = 1,
) -> int:
    """
    Calculate position size using the fixed-fractional method.

    Parameters
    ----------
    account_equity : current net liquidation value of the account
    entry_price    : planned fill price
    stop_price     : initial stop-loss price
    risk_pct       : fraction of equity to risk per trade (e.g. 0.01)
    max_shares     : hard cap on position size (from config / risk manager)
    min_shares     : minimum viable size; return 0 if calc falls below this

    Returns
    -------
    int
        Number of shares to trade, or 0 if the trade cannot be sized safely.
    """
    try:
        risk_per_share = abs(float(entry_price) - float(stop_price))
        if risk_per_share <= 0:
            logger.warning(
                "fixed_fractional_size: zero risk_per_share "
                "(entry=%s stop=%s) — returning 0",
                entry_price,
                stop_price,
            )
            return 0

        dollar_risk = float(account_equity) * risk_pct
        shares = math.floor(dollar_risk / risk_per_share)
        shares = min(shares, max_shares)

        if shares < min_shares:
            return 0
        return shares
    except Exception:
        logger.exception("fixed_fractional_size: unexpected error")
        return 0


def round_to_lot(shares: int, lot_size: int = 1) -> int:
    """
    Round *shares* down to the nearest *lot_size* multiple.

    Most US equities trade in lots of 1, but some ETFs and options have
    minimum lot requirements.
    """
    if lot_size <= 0:
        return shares
    return (shares // lot_size) * lot_size


def notional_value(shares: int, price: Decimal) -> Decimal:
    """Return the total notional (shares × price) for a given order."""
    return Decimal(shares) * price
