#!/usr/bin/env python3
"""Profit-taking / "exit at the top" experiment.

Adds a profit-take overlay to the locked hybrid strat and measures whether it
improves or degrades performance vs the baseline (no profit take). The overlay
is built INTO the baseline state machine so that, with profit-taking disabled,
it reproduces strat_core.hybrid_position exactly (asserted at startup).

Profit-take families tested:
  target  : exit when price >= entry_price * (1 + p)      (hard profit target)
  trail   : exit when price <= peak_since_entry * (1 - p)  (trailing stop)
  stretch : exit when price >= SMA200 * (1 + p)            (sell the froth)
  rsi     : exit when RSI(14) >= p                          (overbought)

After ANY profit-take exit, re-entry is BLOCKED until price cools off (closes
below SMA20), then the normal entry rule re-arms. This models "I took profits,
I'll get back in on a pullback" rather than instantly re-buying the next bar.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)
_scr = importlib.util.spec_from_file_location("screen", HERE / "screen.py")
scr = importlib.util.module_from_spec(_scr)
_scr.loader.exec_module(scr)


def wilder_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def hybrid_position_pt(price_signal, *, long_window=200, short_window=20,
                       entry_days_long=3, entry_days_short=3,
                       exit_days_long=2, exit_days_short=1,
                       pt_mode=None, pt_param=None):
    sma_long = core.rolling_sma(price_signal, long_window)
    sma_short = core.rolling_sma(price_signal, short_window)
    above_long = price_signal > sma_long
    above_short = price_signal > sma_short
    slope_up = sma_short > sma_short.shift(1)
    entry_long_ok = core.consec_true(above_long, entry_days_long)
    entry_short_ok = core.consec_true(above_short & slope_up, entry_days_short)
    exit_long_ok = core.consec_true(~above_long, exit_days_long)
    exit_short_ok = core.consec_true(~above_short, exit_days_short)
    short_below_long = sma_short < sma_long
    rsi = wilder_rsi(price_signal) if pt_mode == "rsi" else None

    p_ = price_signal.to_numpy()
    sl_ = sma_long.to_numpy()
    ss_ = sma_short.to_numpy()
    rsi_ = rsi.to_numpy() if rsi is not None else None

    idx = price_signal.index
    pos = pd.Series(0.0, index=idx)
    in_pos = 0.0
    entry_price = np.nan
    peak = np.nan
    blocked = False

    for i in range(len(idx)):
        use_short = bool(short_below_long.iat[i])
        entry_ok = entry_short_ok.iat[i] if use_short else entry_long_ok.iat[i]
        exit_ok = exit_short_ok.iat[i] if use_short else exit_long_ok.iat[i]
        price = p_[i]

        if in_pos == 1.0:
            peak = price if np.isnan(peak) else max(peak, price)
            if exit_ok:                      # normal SMA exit (no cool-off block)
                in_pos = 0.0
            elif pt_mode is not None:        # profit-take overlay
                take = False
                if pt_mode == "target":
                    take = price >= entry_price * (1.0 + pt_param)
                elif pt_mode == "trail":
                    take = price <= peak * (1.0 - pt_param)
                elif pt_mode == "stretch":
                    take = (not np.isnan(sl_[i])) and price >= sl_[i] * (1.0 + pt_param)
                elif pt_mode == "rsi":
                    take = (not np.isnan(rsi_[i])) and rsi_[i] >= pt_param
                if take:
                    in_pos = 0.0
                    blocked = True
        else:
            if blocked and (not np.isnan(ss_[i])) and price < ss_[i]:
                blocked = False              # cooled off -> re-arm normal entry
            if (not blocked) and entry_ok:
                in_pos = 1.0
                entry_price = price
                peak = price
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0.0)
    fv_long = sma_long.dropna().index.min()
    fv_short = sma_short.dropna().index.min()
    cands = [x for x in [fv_long, fv_short] if x is not None]
    if cands:
        pos = pos[pos.index >= max(cands)]
    return pos


VARIANTS = [
    ("baseline", None, None),
    ("target +50%", "target", 0.50),
    ("target +30%", "target", 0.30),
    ("trail -15%", "trail", 0.15),
    ("trail -10%", "trail", 0.10),
    ("stretch +20% >200SMA", "stretch", 0.20),
    ("stretch +30% >200SMA", "stretch", 0.30),
    ("RSI(14) >= 80", "rsi", 80.0),
]

V = dict(long_window=200, short_window=20, entry_days_long=3, entry_days_short=3,
         exit_days_long=2, exit_days_short=1)


def sleeve_daily(cache, ticker, cash_daily, pt_mode, pt_param):
    sig = cache[ticker]["sig"]
    tr = cache[ticker]["tr"]
    pos = hybrid_position_pt(sig, pt_mode=pt_mode, pt_param=pt_param, **V)
    return core.daily_from_pos(pos, tr, cash_daily)


def metrics_row(name, daily):
    m = core.metrics_from_returns(daily)
    return dict(variant=name, CAGR=m["CAGR"], Sharpe=m["Sharpe"],
                MaxDD=m["MaxDD"], Calmar=m["Calmar"], TotMult=m["TotalMultiple"])


def fmt(df):
    d = df.copy()
    d["CAGR"] = (d["CAGR"] * 100).map("{:.2f}%".format)
    d["MaxDD"] = (d["MaxDD"] * 100).map("{:.1f}%".format)
    d["Sharpe"] = d["Sharpe"].map("{:.3f}".format)
    d["Calmar"] = d["Calmar"].map("{:.2f}".format)
    d["TotMult"] = d["TotMult"].map("{:.2f}x".format)
    return d.to_string(index=False)


def main():
    tickers = ["SPMO", "IWY", "QQQ", "DGRW", "IWF", "IGM", "EFG"]
    need = sorted(set(tickers) | {"SGOV", "BIL"})
    print(f"Fetching {need} ...")
    cache = scr.build_cache(need, scr._resolve_end("today"))
    cash_daily = core.build_cash_chain(cache["SGOV"]["tr"], cache["BIL"]["tr"])

    # Sanity: pt disabled == strat_core.hybrid_position
    for t in ["IWF", "IGM"]:
        a = hybrid_position_pt(cache[t]["sig"], pt_mode=None, **V)
        b = core.hybrid_position(cache[t]["sig"], **V)
        assert (a.reindex(b.index).fillna(0).to_numpy() == b.to_numpy()).all(), f"baseline mismatch {t}"
    print("OK: profit-take-disabled reproduces baseline exactly.\n")

    combos = {
        "RECOMMENDED combo (SPMO, IWF, IGM, EFG)": ["SPMO", "IWF", "IGM", "EFG"],
        "INCUMBENT combo (SPMO, IWY, QQQ, DGRW)": ["SPMO", "IWY", "QQQ", "DGRW"],
    }
    for title, members in combos.items():
        rows = []
        for name, mode, param in VARIANTS:
            sleeves = {t: sleeve_daily(cache, t, cash_daily, mode, param) for t in members}
            blend = core.combo_blend(sleeves)
            rows.append(metrics_row(name, blend))
        df = pd.DataFrame(rows)
        base = df.iloc[0]
        df["dSharpe"] = (df["Sharpe"] - base["Sharpe"]).map("{:+.3f}".format)
        df["dCAGR"] = ((df["CAGR"] - base["CAGR"]) * 100).map("{:+.2f}pp".format)
        win = blend.index.min().strftime("%Y-%m-%d")
        print(f"== {title}  [window {win} -> today, equal-weight] ==")
        print(fmt(df.drop(columns=["dSharpe", "dCAGR"])) )
        print(df[["variant", "dSharpe", "dCAGR"]].to_string(index=False))
        print()

    # Deep-history single sleeves: the only series old enough to have lived
    # through the dot-com (2000) and GFC (2008) tops -- the real test of "exit at the top".
    for t in ["IWF", "IGM"]:
        rows = []
        for name, mode, param in VARIANTS:
            rows.append(metrics_row(name, sleeve_daily(cache, t, cash_daily, mode, param)))
        df = pd.DataFrame(rows)
        base = df.iloc[0]
        df["dSharpe"] = (df["Sharpe"] - base["Sharpe"]).map("{:+.3f}".format)
        df["dCAGR"] = ((df["CAGR"] - base["CAGR"]) * 100).map("{:+.2f}pp".format)
        start = cache[t]["tr"].index.min().strftime("%Y-%m-%d")
        print(f"== DEEP HISTORY single sleeve {t}  [{start} -> today, incl. dot-com + GFC] ==")
        print(fmt(df.drop(columns=["dSharpe", "dCAGR"])))
        print(df[["variant", "dSharpe", "dCAGR"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
