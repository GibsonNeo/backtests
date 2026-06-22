import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rob = _load("robustness", "robustness.py")


def _series(start, n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    drift = np.linspace(0.0, 0.7, n)
    noise = rng.normal(0, 0.01, n).cumsum()
    return pd.Series(100.0 * np.exp(drift + noise), index=idx)


def _cash(start, n):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(np.linspace(100.0, 110.0, n), index=idx)


def _cfg():
    return {
        "end": "today",
        "cash_chain": ["SGOV", "BIL"],
        "fixed_variant": {"long_window": 200, "short_window": 20, "entry_days_long": 3,
                          "entry_days_short": 3, "exit_days_long": 2, "exit_days_short": 1},
        "tier2": {"random_samples": 15, "random_min_years": 3, "random_max_years": "auto",
                  "random_seed": 49, "structured_buckets": [3, 5, 7, 10],
                  "structured_step_fraction": 0.5, "structured_tail_overlap": 0.6},
    }


def _cache():
    # OLD started in 2005 (tradable through the 2008 GFC); YOUNG started in 2019.
    return {
        "OLD": {"sig": _series("2005-01-03", 5200, 1), "tr": _series("2005-01-03", 5200, 1)},
        "YOUNG": {"sig": _series("2019-01-02", 1900, 2), "tr": _series("2019-01-02", 1900, 2)},
        "SGOV": {"sig": _cash("2005-01-03", 5200), "tr": _cash("2005-01-03", 5200)},
        "BIL": {"sig": _cash("2005-01-03", 5200), "tr": _cash("2005-01-03", 5200)},
    }


def test_full_history_flags_gfc_coverage():
    fh = rob.run_full_history_robustness(_cfg(), _cache(), ["OLD", "YOUNG"],
                                         {"OLD": "old_theme", "YOUNG": "young_theme"})
    assert {"fh_avg_sharpe_delta", "fh_sharpe_beat_rate", "fh_windows",
            "history_years", "covers_gfc"}.issubset(fh.columns)
    old = fh.set_index("ticker").loc["OLD"]
    young = fh.set_index("ticker").loc["YOUNG"]
    assert bool(old["covers_gfc"]) is True
    assert bool(young["covers_gfc"]) is False
    assert old["history_years"] > young["history_years"]
    assert old["fh_windows"] > young["fh_windows"]  # longer history -> more rolling windows


def test_attach_full_history_adds_shallow_flag_and_preserves_order():
    fh = rob.run_full_history_robustness(_cfg(), _cache(), ["OLD", "YOUNG"],
                                         {"OLD": "old_theme", "YOUNG": "young_theme"})
    shared = pd.DataFrame({"ticker": ["YOUNG", "OLD"], "robust_rank": [1, 2]})
    shared.attrs["shared_start"] = "2019-01-02"
    merged = rob.attach_full_history(shared, fh)
    assert list(merged["ticker"]) == ["YOUNG", "OLD"]        # left (rank) order preserved
    assert merged.attrs.get("shared_start") == "2019-01-02"  # attrs preserved through merge
    by_ticker = merged.set_index("ticker")
    assert int(by_ticker.loc["YOUNG", "shallow_history"]) == 1
    assert int(by_ticker.loc["OLD", "shallow_history"]) == 0
