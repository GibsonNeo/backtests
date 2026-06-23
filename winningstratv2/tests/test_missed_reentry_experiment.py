import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("missed_reentry_experiment", HERE / "missed_reentry_experiment.py")
exp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exp
spec.loader.exec_module(exp)


def _prices_with_sma20_reclaim():
    idx = pd.bdate_range("2024-01-01", periods=45)
    px = pd.Series(100.0, index=idx)
    px.iloc[20:25] = 105.0
    px.iloc[25] = 98.0
    px.iloc[26] = 102.0
    px.iloc[27:] = 103.0
    return px


def _baseline_long_window(index, start=20, end=35):
    base = pd.Series(0.0, index=index)
    base.iloc[start:end] = 1.0
    return base


def test_sma_reclaim_waits_for_touch_then_enters_on_reclaim():
    px = _prices_with_sma20_reclaim()
    base = _baseline_long_window(px.index)
    pos = exp.missed_reentry_position(px, base, exp.Fallback("test", "sma_reclaim", 20))

    assert pos.iloc[24] == 0.0
    assert pos.iloc[25] == 0.0
    assert pos.iloc[26] == 1.0
    assert pos.iloc[34] == 1.0
    assert pos.iloc[35] == 0.0


def test_sma_touch_enters_on_first_pullback_to_level():
    px = _prices_with_sma20_reclaim()
    base = _baseline_long_window(px.index)
    pos = exp.missed_reentry_position(px, base, exp.Fallback("test", "sma_touch", 20))

    assert pos.iloc[24] == 0.0
    assert pos.iloc[25] == 1.0
    assert pos.iloc[34] == 1.0
    assert pos.iloc[35] == 0.0


def test_reclaim_any_enters_on_first_reclaimed_level():
    px = _prices_with_sma20_reclaim()
    base = _baseline_long_window(px.index)
    pos = exp.missed_reentry_position(px, base, exp.Fallback("test", "sma_reclaim_any", (20, 50, 100)))

    assert pos.iloc[25] == 0.0
    assert pos.iloc[26] == 1.0
    assert pos.iloc[35] == 0.0
