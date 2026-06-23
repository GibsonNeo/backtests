#!/usr/bin/env python3
"""Sharpe / CAGR / MaxDD for individual tickers under the locked winningstratv2
strategy, each over its OWN full history (Tier-1 style, standalone)."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
def _load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
core = _load("strat_core"); scr = _load("screen")

V = dict(long_window=200, short_window=20, entry_days_long=3, entry_days_short=3,
         exit_days_long=2, exit_days_short=1)
TICKERS = ["QQQ", "VUG", "IGM", "IWY", "IWF", "SPMO", "FDMO"]

cache = scr.build_cache(sorted(set(TICKERS) | {"SGOV", "BIL"}), scr._resolve_end("today"))
cash_daily = core.build_cash_chain(cache["SGOV"]["tr"], cache["BIL"]["tr"])

rows = []
for t in TICKERS:
    if t not in cache:
        print(f"  !! {t} unavailable"); continue
    sig, tr = cache[t]["sig"], cache[t]["tr"]
    pos = core.hybrid_position(sig, **V)
    m = core.metrics_from_returns(core.daily_from_pos(pos, tr, cash_daily))
    rows.append({"ticker": t, "start": tr.index.min().strftime("%Y-%m-%d"),
                 "end": tr.index.max().strftime("%Y-%m-%d"), "years": round(len(tr)/252, 1),
                 "Sharpe": round(m["Sharpe"], 3), "CAGR": f'{m["CAGR"]*100:.2f}%',
                 "MaxDD": f'{m["MaxDD"]*100:.1f}%'})
df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
print(df.to_string(index=False))
