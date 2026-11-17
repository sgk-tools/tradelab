"""Trading cost models.

The presets are *assumptions* representing typical retail conditions. They are
not quotes from any broker; replace them with the actual fee schedule of the
venue you intend to use before trusting a result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostModel:
    """Costs and trading constraints applied by the backtest engine.

    commission_rate    Fraction of traded notional charged per fill.
    min_commission     Minimum commission per fill (quote currency).
    slippage_rate      Adverse price move per fill as a fraction of price.
                       Use it for half the bid/ask spread plus slippage.
    lot_size           Minimum tradable unit. 0 means fractional units.
    allow_short        Whether negative positions are permitted.
    max_leverage       Maximum |position notional| / equity.
    annual_borrow_rate Yearly rate charged on short notional and borrowed cash.
    """

    commission_rate: float = 0.0
    min_commission: float = 0.0
    slippage_rate: float = 0.0
    lot_size: float = 0.0
    allow_short: bool = True
    max_leverage: float = 1.0
    annual_borrow_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_rate", "min_commission", "slippage_rate",
                     "lot_size", "annual_borrow_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.slippage_rate >= 1:
            raise ValueError("slippage_rate must be < 1")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")


PRESETS: dict[str, CostModel] = {
    # No costs at all. Only for sanity checks; never for decisions.
    "zero": CostModel(),
    # Japanese cash equities: zero-commission online broker, 100-share lots.
    "jp_stock": CostModel(commission_rate=0.0, slippage_rate=0.0005,
                          lot_size=100, allow_short=False),
    # US equities via a Japanese broker: ~0.495% commission, 1-share lots.
    "us_stock": CostModel(commission_rate=0.00495, slippage_rate=0.0005,
                          lot_size=1, allow_short=False),
    # Spot FX (majors): cost is the spread; swap approximated by borrow rate.
    "fx": CostModel(commission_rate=0.0, slippage_rate=0.00005, lot_size=1,
                    allow_short=True, max_leverage=1.0,
                    annual_borrow_rate=0.0),
    # Spot crypto on an exchange: ~0.1% taker fee, fractional units.
    "crypto": CostModel(commission_rate=0.001, slippage_rate=0.0005,
                        lot_size=0.0, allow_short=False),
}


def get_cost_model(market: str, **overrides) -> CostModel:
    """Return the preset for ``market`` with optional field overrides."""
    try:
        base = PRESETS[market]
    except KeyError:
        raise ValueError(
            f"unknown market '{market}'. choices: {', '.join(PRESETS)}") from None
    return replace(base, **overrides) if overrides else base
