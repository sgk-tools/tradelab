"""tradelab: market-agnostic backtesting lab for trading strategy research."""

from .costs import PRESETS, CostModel, get_cost_model
from .data import fetch_ohlcv, load_csv, synthetic_ohlcv
from .engine import BacktestResult, backtest
from .metrics import compute_metrics
from .strategies import STRATEGIES, make_signal
from .validate import grid_search, walk_forward

__all__ = [
    "PRESETS", "CostModel", "get_cost_model",
    "fetch_ohlcv", "load_csv", "synthetic_ohlcv",
    "BacktestResult", "backtest", "compute_metrics",
    "STRATEGIES", "make_signal", "grid_search", "walk_forward",
]
__version__ = "0.5.0"
