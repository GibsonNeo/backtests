#!/usr/bin/env python3
"""Apples-to-apples: Sharpe / CAGR / MaxDD for each ticker under the locked
winningstratv2 strategy, all measured over the SAME shared window (the common
period where every ticker is fully warmed-up and active)."""
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

# Per-ticker strat daily returns on FULL history (so SMA200 warms up properly).
strat = {}
for t in TICKERS:
    pos = core.hybrid_position(cache[t]["sig"], **V)
    strat[t] = core.daily_from_pos(pos, cache[t]["tr"], cash_daily).loc[pos.index.min():]

shared_start = max(s.index.min() for s in strat.values())
shared_end = min(s.index.max() for s in strat.values())

rows = []
for t in TICKERS:
    d = strat[t].loc[shared_start:shared_end]
    bh = cache[t]["tr"].pct_change().loc[shared_start:shared_end].fillna(0.0)
    m = core.metrics_from_returns(d); b = core.metrics_from_returns(bh)
    rows.append({"ticker": t,
                 "Sharpe": round(m["Sharpe"], 3), "CAGR": f'{m["CAGR"]*100:.2f}%',
                 "MaxDD": f'{m["MaxDD"]*100:.1f}%', "Calmar": round(m["Calmar"], 2),
                 "bh_Sharpe": round(b["Sharpe"], 3), "bh_MaxDD": f'{b["MaxDD"]*100:.1f}%'})
df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
print(f"Shared window: {shared_start.date()} -> {shared_end.date()} "
      f"({round((shared_end-shared_start).days/365.25,1)}y), identical for all tickers\n")
print(df.to_string(index=False))
