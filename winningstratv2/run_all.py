#!/usr/bin/env python3
"""Run Tier 1 -> 2 -> 3 with one shared price cache; write report.md + charts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

HERE = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, str(HERE / f))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    core = _load("strat_core", "strat_core.py")
    scr = _load("screen", "screen.py")
    rob = _load("robustness", "robustness.py")
    cs = _load("combo_search", "combo_search.py")

    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    (outdir / "charts").mkdir(parents=True, exist_ok=True)
    theme_of = scr.flatten_universe(cfg)

    # One shared cache for the entire universe + cash chain.
    needed = sorted(set(theme_of) | set(cfg["cash_chain"]))
    cache = scr.build_cache(needed, scr._resolve_end(cfg.get("end")))
    cash_daily = scr.cash_chain_from_cache(cfg, cache)

    t1, cache = scr.run_screen(cfg, cache=cache)
    t1.to_csv(outdir / "tier1_screen.csv", index=False)
    survivors = [t for t in t1[t1["survivor"] == 1]["ticker"] if t in cache]

    t2 = rob.run_robustness(cfg, cache, survivors, theme_of)
    t2.to_csv(outdir / "tier2_robustness.csv", index=False)
    fin = rob.finalists(t2, cfg)
    fin = [t for t in fin if t in cache]

    t3 = cs.run_combo_search(cfg, cache, fin, cash_daily)
    t3.to_csv(outdir / "tier3_combos.csv", index=False)

    # Charts: incumbents vs best combo.
    best = t3.iloc[0]["combo"].split(",")
    inc = sorted(cfg["incumbents"])
    sleeves = cs.sleeve_daily_returns(cache, sorted(set(best) | set(inc)), cfg["fixed_variant"], cash_daily)
    best_curve = (1 + core.combo_blend({t: sleeves[t] for t in best})).cumprod()
    inc_curve = (1 + core.combo_blend({t: sleeves[t] for t in inc})).cumprod()
    if plt is not None:
        eq = pd.DataFrame({"best_combo": best_curve, "incumbents": inc_curve}).dropna()
        ax = eq.plot(figsize=(12, 7), title="Best robust combo vs incumbents (growth of $1)")
        ax.set_ylabel("Growth of $1"); ax.grid(True, alpha=0.25); ax.figure.tight_layout()
        ax.figure.savefig(outdir / "charts" / "equity_incumbents_vs_best.png", dpi=160)
        dd = eq / eq.cummax() - 1.0
        ax2 = dd.plot(figsize=(12, 7), title="Drawdown: best robust combo vs incumbents")
        ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.25); ax2.figure.tight_layout()
        ax2.figure.savefig(outdir / "charts" / "drawdown_incumbents_vs_best.png", dpi=160)

    # Report.
    top = t3.head(int(cfg["tier3"]["top_n_report"]))
    inc_combo = ",".join(inc)
    inc_row = t3[t3["combo"] == inc_combo]
    lines = ["# winningstratv2 — Ticker-Swap Study Report", "",
             f"Universe screened: {len(t1)} tickers. Survivors to Tier 2: {len(survivors)}. "
             f"Finalists to Tier 3: {len(fin)}.",
             f"Tier-2 shared window: {t2.attrs.get('shared_start')} to {t2.attrs.get('shared_end')}.", "",
             "## Top individual candidates (Tier 1, by combined rank)", "",
             t1.head(15)[["combined_rank", "ticker", "theme", "strat_sharpe", "strat_cagr",
                          "sharpe_delta", "cagr_delta"]].to_markdown(index=False), "",
             "## Robustness finalists (Tier 2)", "",
             t2.head(int(cfg["n_finalists"]))[["robust_rank", "ticker", "theme", "avg_sharpe_delta",
                          "sharpe_beat_rate", "cagr_beat_rate"]].to_markdown(index=False), "",
             "## Best 4-sleeve combos (Tier 3)", "",
             top[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "calmar", "avg_corr",
                  "blended_score"]].to_markdown(index=False), ""]
    if not inc_row.empty:
        lines += ["## Incumbents (bar to beat)", "",
                  inc_row[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "calmar", "avg_corr",
                           "blended_score"]].to_markdown(index=False), ""]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outdir/'report.md'} and charts/. Best combo: {best}")


if __name__ == "__main__":
    main()
