#!/usr/bin/env python3
"""Tier 1: run the locked strat on every candidate over its own full history."""
from __future__ import annotations

import importlib.util
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)


def _resolve_end(value) -> str:
    if value is None or str(value).strip().lower() == "today":
        return date.today().strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_config(path: str = None) -> dict:
    path = path or str(HERE / "config.yml")
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_universe(cfg: dict) -> dict:
    """Return {ticker: theme}. First theme wins if a ticker appears twice."""
    out = {}
    for theme, tickers in cfg["universe"].items():
        for t in tickers:
            out.setdefault(t, theme)
    return out


def build_cache(tickers, end: str, warmup_days: int = 965) -> dict:
    """Fetch each ticker from its earliest available date (auto-inception)."""
    cache = {}
    early = "1990-01-01"
    for t in tickers:
        try:
            sig, tr = core.fetch_series(t, early, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN skipping {t}: {exc}")
            continue
        if sig.empty or tr.empty:
            print(f"  WARN skipping {t}: empty series")
            continue
        cache[t] = {"sig": sig, "tr": tr}
    return cache


def cash_chain_from_cache(cfg: dict, cache: dict) -> pd.Series:
    sgov_name, bil_name = cfg["cash_chain"]
    return core.build_cash_chain(cache[sgov_name]["tr"], cache[bil_name]["tr"])


def run_screen(cfg: dict, cache: dict = None) -> tuple[pd.DataFrame, dict]:
    end = _resolve_end(cfg.get("end"))
    theme_of = flatten_universe(cfg)
    variant = cfg["fixed_variant"]
    needed = sorted(set(theme_of) | set(cfg["cash_chain"]))
    if cache is None:
        print("Fetching price history...")
        cache = build_cache(needed, end)
    for name in cfg["cash_chain"]:
        if name not in cache:
            raise ValueError(f"Cash chain ticker {name} failed to download")
    cash_daily = cash_chain_from_cache(cfg, cache)

    rows = []
    for ticker, theme in theme_of.items():
        if ticker not in cache:
            continue
        sig = cache[ticker]["sig"]
        tr = cache[ticker]["tr"]
        pos = core.hybrid_position(
            sig,
            long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"],
        )
        strat_daily = core.daily_from_pos(pos, tr, cash_daily)
        bh_daily = tr.pct_change().fillna(0.0)
        sm = core.metrics_from_returns(strat_daily)
        bm = core.metrics_from_returns(bh_daily)
        rows.append({
            "ticker": ticker, "theme": theme,
            "start": tr.index.min().strftime("%Y-%m-%d"), "end": tr.index.max().strftime("%Y-%m-%d"),
            "observations": int(strat_daily.dropna().shape[0]),
            "strat_sharpe": sm["Sharpe"], "strat_cagr": sm["CAGR"], "strat_maxdd": sm["MaxDD"],
            "strat_calmar": sm["Calmar"], "strat_ulcer": sm["UlcerIndex"],
            "bh_sharpe": bm["Sharpe"], "bh_cagr": bm["CAGR"], "bh_maxdd": bm["MaxDD"],
            "sharpe_delta": sm["Sharpe"] - bm["Sharpe"],
            "cagr_delta": sm["CAGR"] - bm["CAGR"],
            "maxdd_delta": sm["MaxDD"] - bm["MaxDD"],
        })

    df = pd.DataFrame(rows)
    df["combined_rank_score"] = core.combined_rank(df, "strat_sharpe", "strat_cagr")
    df = df.sort_values(["combined_rank_score", "strat_sharpe"], ascending=[True, False]).reset_index(drop=True)
    df["combined_rank"] = df.index + 1

    # Survivor gate.
    min_days = int(cfg.get("min_history_days", 1500))
    incumbents = set(cfg.get("incumbents", []))
    gate = (df["observations"] >= min_days)
    if cfg.get("gate_require_positive_sharpe_delta", True):
        gate = gate & (df["sharpe_delta"] > 0)
    df["survivor"] = (gate | df["ticker"].isin(incumbents)).astype(int)
    return df, cache


def main():
    cfg = load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    df, _ = run_screen(cfg)
    path = outdir / "tier1_screen.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}  ({df['survivor'].sum()} survivors of {len(df)})")


if __name__ == "__main__":
    main()
