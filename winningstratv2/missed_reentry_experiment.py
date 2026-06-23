#!/usr/bin/env python3
"""Missed re-entry fallback experiment for the locked winningstratv2 rule.

The baseline strategy enters whenever the locked hybrid SMA rule flips from
cash to long. This experiment assumes every such entry is missed ("stayed in
SGOV/BIL") and tests fallback rules that can join the trade later while the
baseline signal remains long.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
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

PORTFOLIO = ["IWF", "SPMO", "OEF", "DGRO", "XLI"]

V = dict(
    long_window=200,
    short_window=20,
    entry_days_long=3,
    entry_days_short=3,
    exit_days_long=2,
    exit_days_short=1,
)


@dataclass(frozen=True)
class Fallback:
    name: str
    kind: str
    value: float | int | tuple[int, ...] | None = None
    max_wait: int | None = None


FALLBACKS = [
    Fallback("baseline: follow original signal", "baseline"),
    Fallback("reclaim SMA20 after pullback/touch", "sma_reclaim", 20),
    Fallback("reclaim SMA50 after pullback/touch", "sma_reclaim", 50),
    Fallback("reclaim SMA100 after pullback/touch", "sma_reclaim", 100),
    Fallback("reclaim SMA200 after pullback/touch", "sma_reclaim", 200),
    Fallback("touch/undercut SMA20", "sma_touch", 20),
    Fallback("touch/undercut SMA50", "sma_touch", 50),
    Fallback("touch/undercut SMA100", "sma_touch", 100),
    Fallback("touch/undercut SMA200", "sma_touch", 200),
    Fallback("above rising SMA20", "above_rising_sma", 20),
    Fallback("above rising SMA50", "above_rising_sma", 50),
    Fallback("above rising SMA100", "above_rising_sma", 100),
    Fallback("first reclaim among SMA20/50/100", "sma_reclaim_any", (20, 50, 100)),
    Fallback("first touch among SMA20/50/100", "sma_touch_any", (20, 50, 100)),
    Fallback("no fallback: stay bonds until next full cycle", "never"),
]


def baseline_position(price_signal: pd.Series) -> pd.Series:
    return core.hybrid_position(price_signal, **V)


def missed_reentry_position(price_signal: pd.Series, baseline_pos: pd.Series, rule: Fallback) -> pd.Series:
    """Return fallback position after missing every baseline 0->1 entry.

    Once fallback enters, it stays long until the baseline exits. If the
    baseline exits before fallback entry, the episode is skipped.
    """
    if rule.kind == "baseline":
        return baseline_pos.copy()

    windows = sorted({20, 50, 100, 200})
    if isinstance(rule.value, int):
        windows = sorted(set(windows) | {rule.value})
    elif isinstance(rule.value, tuple):
        windows = sorted(set(windows) | set(rule.value))
    smas = {w: core.rolling_sma(price_signal, w) for w in windows}
    baseline_pos = baseline_pos.reindex(price_signal.index).fillna(0.0)
    out = pd.Series(0.0, index=price_signal.index)

    active_miss = False
    joined = False
    missed_price = np.nan
    miss_i = -1
    armed_reclaim = {w: False for w in windows}
    prev_base = 0.0

    def _sma_value(window: int, loc: int) -> float:
        value = smas[window].iat[loc]
        return float(value) if not pd.isna(value) else np.nan

    def _touch(window: int, loc: int) -> bool:
        sma = _sma_value(window, loc)
        return (not np.isnan(sma)) and price <= sma

    def _reclaim(window: int, loc: int) -> bool:
        sma = _sma_value(window, loc)
        if (not np.isnan(sma)) and price <= sma:
            armed_reclaim[window] = True
        return armed_reclaim[window] and (not np.isnan(sma)) and price > sma

    def _above_rising(window: int, loc: int) -> bool:
        sma = _sma_value(window, loc)
        prev = _sma_value(window, loc - 1) if loc > 0 else np.nan
        return (not np.isnan(sma)) and (not np.isnan(prev)) and price > sma and sma > prev

    for i, dt in enumerate(price_signal.index):
        base = float(baseline_pos.iat[i])
        price = float(price_signal.iat[i])

        if base == 0.0:
            active_miss = False
            joined = False
            armed_reclaim = {w: False for w in windows}
            out.iat[i] = 0.0
            prev_base = base
            continue

        if prev_base == 0.0 and base == 1.0:
            active_miss = True
            joined = False
            armed_reclaim = {w: False for w in windows}
            missed_price = price
            miss_i = i

        if active_miss and not joined:
            days_waited = i - miss_i
            enter = False
            if rule.kind == "never":
                enter = False
            elif rule.kind == "wait_days":
                enter = days_waited >= int(rule.value)
            elif rule.kind == "pullback":
                enter = price <= missed_price * (1.0 - float(rule.value))
            elif rule.kind == "sma_touch":
                enter = _touch(int(rule.value), i)
            elif rule.kind == "sma_touch_any":
                enter = any(_touch(w, i) for w in rule.value)
            elif rule.kind == "sma_reclaim":
                enter = _reclaim(int(rule.value), i)
            elif rule.kind == "sma_reclaim_any":
                enter = any(_reclaim(w, i) for w in rule.value)
            elif rule.kind == "above_rising_sma":
                enter = _above_rising(int(rule.value), i)
            elif rule.kind == "pullback_or_wait":
                enter = (
                    price <= missed_price * (1.0 - float(rule.value))
                    or days_waited >= int(rule.max_wait or 0)
                )
            else:  # pragma: no cover - defensive guard for new experiments
                raise ValueError(f"Unknown fallback kind: {rule.kind}")
            if enter:
                joined = True

        out.iat[i] = 1.0 if joined else 0.0
        prev_base = base

    return out.loc[baseline_pos.index]


def sleeve_daily(cache: dict, ticker: str, cash_daily: pd.Series, rule: Fallback) -> pd.Series:
    sig = cache[ticker]["sig"]
    tr = cache[ticker]["tr"]
    base = baseline_position(sig)
    pos = missed_reentry_position(sig, base, rule)
    return core.daily_from_pos(pos, tr, cash_daily)


def count_entry_episodes(cache: dict, ticker: str) -> int:
    pos = baseline_position(cache[ticker]["sig"])
    return int(((pos == 1.0) & (pos.shift(1).fillna(0.0) == 0.0)).sum())


def summarize_current(cache: dict, ticker: str) -> dict:
    sig = cache[ticker]["sig"]
    tr = cache[ticker]["tr"]
    tr_aligned = tr.reindex(sig.index).ffill()
    base = baseline_position(sig)
    entries = base[(base == 1.0) & (base.shift(1).fillna(0.0) == 0.0)]
    exits = base[(base == 0.0) & (base.shift(1).fillna(0.0) == 1.0)]
    last_entry = entries.index.max() if not entries.empty else pd.NaT
    last_exit = exits.index.max() if not exits.empty else pd.NaT
    latest = sig.index.max()
    in_signal = bool(base.loc[latest] == 1.0)
    entry_px = float(tr_aligned.loc[last_entry]) if pd.notna(last_entry) else np.nan
    last_px = float(tr_aligned.loc[latest])
    signal_px = float(sig.loc[latest])
    smas = {w: core.rolling_sma(sig, w) for w in [20, 50, 100, 200]}
    row = {
        "ticker": ticker,
        "latest_date": latest.strftime("%Y-%m-%d"),
        "in_signal_now": in_signal,
        "last_entry": last_entry.strftime("%Y-%m-%d") if pd.notna(last_entry) else "",
        "last_exit": last_exit.strftime("%Y-%m-%d") if pd.notna(last_exit) else "",
        "days_since_entry": int((sig.index.get_loc(latest) - sig.index.get_loc(last_entry))) if pd.notna(last_entry) else np.nan,
        "entry_adj_price": entry_px,
        "last_adj_price": last_px,
        "pct_from_entry": (last_px / entry_px - 1.0) if entry_px and not np.isnan(entry_px) else np.nan,
        "signal_price": signal_px,
    }
    for w, sma in smas.items():
        value = float(sma.loc[latest]) if not pd.isna(sma.loc[latest]) else np.nan
        row[f"sma{w}"] = value
        row[f"pct_vs_sma{w}"] = (signal_px / value - 1.0) if value and not np.isnan(value) else np.nan
    return row


def metrics_row(rule: Fallback, daily: pd.Series, baseline: dict | None = None) -> dict:
    m = core.metrics_from_returns(daily)
    row = {
        "variant": rule.name,
        "CAGR": m["CAGR"],
        "Sharpe": m["Sharpe"],
        "MaxDD": m["MaxDD"],
        "Calmar": m["Calmar"],
        "UlcerIndex": m["UlcerIndex"],
        "TotalMultiple": m["TotalMultiple"],
    }
    if baseline:
        row["dCAGR_pp"] = (m["CAGR"] - baseline["CAGR"]) * 100.0
        row["dSharpe"] = m["Sharpe"] - baseline["Sharpe"]
        row["dMaxDD_pp"] = (m["MaxDD"] - baseline["MaxDD"]) * 100.0
    return row


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def write_report(outdir: Path, summary: pd.DataFrame, current: pd.DataFrame, episodes: pd.DataFrame, start: str, end: str) -> None:
    display = summary.copy()
    for col in ["CAGR", "MaxDD"]:
        display[col] = (display[col] * 100).map("{:.2f}%".format)
    display["Sharpe"] = display["Sharpe"].map("{:.3f}".format)
    display["Calmar"] = display["Calmar"].map("{:.2f}".format)
    display["UlcerIndex"] = display["UlcerIndex"].map("{:.2f}".format)
    display["TotalMultiple"] = display["TotalMultiple"].map("{:.2f}x".format)
    for col in ["dCAGR_pp", "dMaxDD_pp"]:
        display[col] = display[col].map("{:+.2f}pp".format)
    display["dSharpe"] = display["dSharpe"].map("{:+.3f}".format)

    cur = current.copy()
    for col in ["entry_adj_price", "last_adj_price", "signal_price", "sma20", "sma50", "sma100", "sma200"]:
        cur[col] = cur[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    for col in ["pct_from_entry", "pct_vs_sma20", "pct_vs_sma50", "pct_vs_sma100", "pct_vs_sma200"]:
        cur[col] = cur[col].map(fmt_pct)

    lines = [
        "# Missed Re-entry Fallback Experiment",
        "",
        f"Portfolio sleeves: {', '.join(PORTFOLIO)}. Equal-weight blend. Window: {start} to {end}.",
        "",
        "Method: every historical baseline cash-to-long transition is intentionally missed. "
        "The fallback variant stays in the SGOV/BIL cash chain until its SMA-based rule "
        "fires while the original signal remains long; after joining, it follows the "
        "original exit.",
        "",
        "This is a research backtest, not investment advice.",
        "",
        "## Portfolio Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Current Sleeve State",
        "",
        cur.to_markdown(index=False),
        "",
        "## Baseline Entry Episode Count",
        "",
        episodes.to_markdown(index=False),
        "",
    ]
    (outdir / "missed_reentry_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    outdir = HERE / "outputs_missed_reentry"
    outdir.mkdir(parents=True, exist_ok=True)

    needed = sorted(set(PORTFOLIO) | {"SGOV", "BIL"})
    print(f"Fetching {needed} ...")
    cache = scr.build_cache(needed, scr._resolve_end("today"))
    missing = [t for t in needed if t not in cache]
    if missing:
        raise SystemExit(f"Missing required tickers: {missing}")
    cash_daily = core.build_cash_chain(cache["SGOV"]["tr"], cache["BIL"]["tr"])

    rows = []
    baseline_metrics = None
    blended_by_rule = {}
    for rule in FALLBACKS:
        sleeves = {t: sleeve_daily(cache, t, cash_daily, rule) for t in PORTFOLIO}
        blend = core.combo_blend(sleeves)
        blended_by_rule[rule.name] = blend
        if rule.kind == "baseline":
            baseline_metrics = core.metrics_from_returns(blend)
        rows.append(metrics_row(rule, blend, baseline_metrics))

    summary = pd.DataFrame(rows)
    summary["score"] = (
        summary["dSharpe"].fillna(0.0).abs() * 2.0
        + summary["dCAGR_pp"].fillna(0.0).abs() / 10.0
        + summary["dMaxDD_pp"].fillna(0.0).clip(upper=0.0).abs() / 10.0
    )
    summary = pd.concat([summary.iloc[[0]], summary.iloc[1:].sort_values("score")], ignore_index=True)
    summary.to_csv(outdir / "missed_reentry_summary.csv", index=False)

    curves = pd.DataFrame({name: (1.0 + daily).cumprod() for name, daily in blended_by_rule.items()}).dropna()
    curves.to_csv(outdir / "missed_reentry_equity_curves.csv")

    current = pd.DataFrame([summarize_current(cache, t) for t in PORTFOLIO])
    current.to_csv(outdir / "current_sleeve_state.csv", index=False)

    episodes = pd.DataFrame({
        "ticker": PORTFOLIO,
        "baseline_entry_episodes": [count_entry_episodes(cache, t) for t in PORTFOLIO],
    })
    episodes.to_csv(outdir / "entry_episode_counts.csv", index=False)

    write_report(
        outdir,
        summary.drop(columns=["score"]),
        current,
        episodes,
        curves.index.min().strftime("%Y-%m-%d"),
        curves.index.max().strftime("%Y-%m-%d"),
    )
    print(f"Wrote {outdir / 'missed_reentry_report.md'}")
    print(summary[["variant", "CAGR", "Sharpe", "MaxDD", "dCAGR_pp", "dSharpe", "dMaxDD_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
