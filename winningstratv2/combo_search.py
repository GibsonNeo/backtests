#!/usr/bin/env python3
"""Tier 3: exhaustive equal-weight 4-sleeve combos among Tier-2 finalists,
scored on a blended robust metric, with a correlation guard and the incumbent
combo printed as the bar to beat."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)


def sleeve_daily_returns(cache: dict, tickers, variant, cash_daily) -> dict:
    out = {}
    for t in tickers:
        pos = core.hybrid_position(
            cache[t]["sig"], long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"])
        out[t] = core.daily_from_pos(pos, cache[t]["tr"], cash_daily)
    return out


def _avg_pairwise_corr(cache: dict, tickers) -> float:
    rets = pd.DataFrame({t: cache[t]["tr"].pct_change() for t in tickers}).dropna()
    c = rets.corr().to_numpy()
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(c[iu]))


def run_combo_search(cfg: dict, cache: dict, finalists: list, cash_daily) -> pd.DataFrame:
    variant = cfg["fixed_variant"]
    sw = cfg["tier3"]["score_weights"]
    size = int(cfg["tier3"].get("combo_size", 4))
    sleeves = sleeve_daily_returns(cache, finalists, variant, cash_daily)

    rows = []
    for combo in itertools.combinations(sorted(finalists), size):
        blend = core.combo_blend({t: sleeves[t] for t in combo})
        m = core.metrics_from_returns(blend)
        rows.append({"combo": ",".join(combo), "n_obs": int(blend.dropna().shape[0]),
                     "start": blend.index.min().strftime("%Y-%m-%d"), "end": blend.index.max().strftime("%Y-%m-%d"),
                     "sharpe": m["Sharpe"], "cagr": m["CAGR"], "maxdd": m["MaxDD"],
                     "calmar": m["Calmar"], "avg_corr": _avg_pairwise_corr(cache, combo)})
    df = pd.DataFrame(rows)

    # Blended robust score: z-score each metric (maxdd higher=better since less negative).
    for col, sign in [("sharpe", 1), ("cagr", 1), ("maxdd", 1)]:
        z = (df[col] - df[col].mean()) / (df[col].std(ddof=0) or 1.0)
        df[f"z_{col}"] = sign * z
    df["blended_score"] = sw["sharpe"] * df["z_sharpe"] + sw["cagr"] * df["z_cagr"] + sw["maxdd"] * df["z_maxdd"]
    df = df.sort_values("blended_score", ascending=False).reset_index(drop=True)
    df["combo_rank"] = df.index + 1
    return df


def main():
    import screen as scr
    import robustness as rob
    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    t2 = pd.read_csv(outdir / "tier2_robustness.csv")
    fin = t2.sort_values("robust_rank").head(int(cfg["n_finalists"]))["ticker"].tolist()
    for inc in cfg["incumbents"]:
        if inc not in fin:
            fin.append(inc)
    cache = scr.build_cache(sorted(set(fin) | set(cfg["cash_chain"])), scr._resolve_end(cfg.get("end")))
    cash_daily = scr.cash_chain_from_cache(cfg, cache)
    fin = [t for t in fin if t in cache]
    df = run_combo_search(cfg, cache, fin, cash_daily)
    df.to_csv(outdir / "tier3_combos.csv", index=False)
    inc_combo = ",".join(sorted(cfg["incumbents"]))
    inc_row = df[df["combo"] == inc_combo]
    print(df.head(int(cfg["tier3"]["top_n_report"]))[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "avg_corr", "blended_score"]].to_string(index=False))
    if not inc_row.empty:
        print("\nIncumbents (bar to beat):")
        print(inc_row[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "avg_corr", "blended_score"]].to_string(index=False))
    print(f"\nSaved {outdir/'tier3_combos.csv'}")


if __name__ == "__main__":
    main()
