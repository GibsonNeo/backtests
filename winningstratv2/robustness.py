#!/usr/bin/env python3
"""Tier 2: random + structured window sampling on Tier-1 survivors, ranked by
cross-window consistency (avg delta + beat-rate) vs buy-hold."""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)
TRADING_DAYS = core.TRADING_DAYS


# --- window generators copied verbatim from v1 (pure index math) ---

def generate_random_windows(common_index, num_samples, min_years, max_years, seed):
    min_days = max(2, int(round(min_years * TRADING_DAYS)))
    if max_years in (None, "", "auto"):
        max_days = len(common_index)
    else:
        max_days = min(len(common_index), int(round(float(max_years) * TRADING_DAYS)))

    if max_days < min_days:
        raise ValueError("Sampling range is too short for requested min_years/max_years")

    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
    windows = []
    seen = set()
    attempts = 0
    max_attempts = max(num_samples * 100, 1000)

    while len(windows) < num_samples and attempts < max_attempts:
        attempts += 1
        span_days = int(rng.integers(min_days, max_days + 1))
        max_start_idx = len(common_index) - span_days
        if max_start_idx < 0:
            continue

        start_idx = int(rng.integers(0, max_start_idx + 1))
        end_idx = start_idx + span_days - 1

        start = pd.Timestamp(common_index[start_idx])
        end = pd.Timestamp(common_index[end_idx])
        key = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if key in seen:
            continue
        seen.add(key)

        windows.append(
            {
                "window_id": f"sample_{len(windows) + 1:03d}",
                "start": key[0],
                "end": key[1],
                "trading_days": int(span_days),
                "calendar_days": int((end - start).days),
                "span_years": float(span_days / TRADING_DAYS),
            }
        )

    if len(windows) < num_samples:
        raise ValueError(f"Could only generate {len(windows)} unique windows out of {num_samples} requested")

    return pd.DataFrame(windows)


def _align_next(index, target):
    pos = index.searchsorted(pd.Timestamp(target), side="left")
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos])


def _align_prev(index, target):
    pos = index.searchsorted(pd.Timestamp(target), side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(index[pos])


def _window_overlap_fraction(index, start_a, end_a, start_b, end_b):
    a0 = int(index.searchsorted(pd.Timestamp(start_a), side="left"))
    a1 = int(index.searchsorted(pd.Timestamp(end_a), side="right")) - 1
    b0 = int(index.searchsorted(pd.Timestamp(start_b), side="left"))
    b1 = int(index.searchsorted(pd.Timestamp(end_b), side="right")) - 1

    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    len_a = max(1, a1 - a0 + 1)
    len_b = max(1, b1 - b0 + 1)
    return overlap / min(len_a, len_b)


def _make_bucket_windows(common_index, length_years, step_fraction, append_tail, max_tail_overlap):
    windows = []
    seen = set()
    start_anchor = pd.Timestamp(common_index.min())
    end_anchor = pd.Timestamp(common_index.max())

    step_months = max(1, int(round(length_years * 12 * step_fraction)))
    start_target = start_anchor

    while True:
        start = _align_next(common_index, start_target)
        if start is None:
            break

        end_target = start + pd.DateOffset(years=length_years) - pd.Timedelta(days=1)
        if end_target > end_anchor:
            break

        end = _align_prev(common_index, end_target)
        if end is None or end <= start:
            break

        key = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if key not in seen:
            seen.add(key)
            windows.append(
                {
                    "bucket_years": length_years,
                    "start": key[0],
                    "end": key[1],
                    "trading_days": int(common_index.searchsorted(end, side="right") - common_index.searchsorted(start, side="left")),
                    "calendar_days": int((end - start).days),
                    "span_years": float(
                        (common_index.searchsorted(end, side="right") - common_index.searchsorted(start, side="left"))
                        / TRADING_DAYS
                    ),
                    "window_source": "rolling",
                }
            )

        start_target = start_target + pd.DateOffset(months=step_months)

    if append_tail:
        tail_start_target = end_anchor - pd.DateOffset(years=length_years) + pd.Timedelta(days=1)
        tail_start = _align_next(common_index, tail_start_target)
        tail_end = end_anchor
        if tail_start is not None and tail_start < tail_end:
            key = (tail_start.strftime("%Y-%m-%d"), tail_end.strftime("%Y-%m-%d"))
            if key not in seen:
                max_overlap = 0.0
                for existing in windows:
                    overlap = _window_overlap_fraction(
                        common_index,
                        tail_start,
                        tail_end,
                        existing["start"],
                        existing["end"],
                    )
                    max_overlap = max(max_overlap, overlap)

                if max_overlap <= max_tail_overlap:
                    seen.add(key)
                    windows.append(
                        {
                            "bucket_years": length_years,
                            "start": key[0],
                            "end": key[1],
                            "trading_days": int(common_index.searchsorted(tail_end, side="right") - common_index.searchsorted(tail_start, side="left")),
                            "calendar_days": int((tail_end - tail_start).days),
                            "span_years": float(
                                (common_index.searchsorted(tail_end, side="right") - common_index.searchsorted(tail_start, side="left"))
                                / TRADING_DAYS
                            ),
                            "window_source": "tail",
                        }
                    )

    windows = sorted(windows, key=lambda row: (row["bucket_years"], row["start"], row["end"]))
    for idx, row in enumerate(windows, start=1):
        row["window_id"] = f"{int(row['bucket_years']):02d}y_{idx:02d}"
    return windows


def generate_structured_windows(common_index, lengths, step_fraction, append_tail, max_tail_overlap):
    rows = []
    for length_years in lengths:
        rows.extend(
            _make_bucket_windows(
                common_index=common_index,
                length_years=length_years,
                step_fraction=step_fraction,
                append_tail=append_tail,
                max_tail_overlap=max_tail_overlap,
            )
        )
    return pd.DataFrame(rows)


def _shared_index(cache: dict, tickers, shared_start, end) -> pd.DatetimeIndex:
    common = None
    for t in tickers:
        idx = cache[t]["tr"].loc[shared_start:end].index
        common = idx if common is None else common.intersection(idx)
    return common.sort_values()


def _eval_window(cache, tickers, variant, cash_daily, run_start, run_end):
    out = []
    for t in tickers:
        sig = cache[t]["sig"].loc[run_start:run_end]
        tr = cache[t]["tr"].loc[run_start:run_end]
        pos = core.hybrid_position(
            sig, long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"],
        )
        strat = core.metrics_from_returns(core.daily_from_pos(pos, tr, cash_daily))
        bh = core.metrics_from_returns(tr.pct_change().fillna(0.0))
        out.append({"ticker": t,
                    "sharpe_delta": strat["Sharpe"] - bh["Sharpe"],
                    "cagr_delta": strat["CAGR"] - bh["CAGR"],
                    "beat_sharpe": float(strat["Sharpe"] > bh["Sharpe"]),
                    "beat_cagr": float(strat["CAGR"] > bh["CAGR"]),
                    "strat_sharpe": strat["Sharpe"], "strat_cagr": strat["CAGR"]})
    return out


def run_robustness(cfg: dict, cache: dict, survivors: list, theme_of: dict) -> pd.DataFrame:
    t2 = cfg["tier2"]
    end = pd.Timestamp(date.today() if str(cfg.get("end", "today")).lower() == "today" else cfg["end"])
    if str(t2.get("shared_start", "auto")).lower() == "auto":
        shared_start = max(cache[t]["tr"].index.min() for t in survivors)
    else:
        shared_start = pd.Timestamp(t2["shared_start"])
    cash_daily = core.build_cash_chain(cache[cfg["cash_chain"][0]]["tr"], cache[cfg["cash_chain"][1]]["tr"])

    common = _shared_index(cache, survivors, shared_start, end)
    rand_windows = generate_random_windows(common, t2["random_samples"], t2["random_min_years"],
                                           t2["random_max_years"], t2.get("random_seed"))
    struct_windows = generate_structured_windows(common, t2["structured_buckets"],
                                                 t2["structured_step_fraction"], True, t2["structured_tail_overlap"])

    records = []
    for w in pd.concat([rand_windows.assign(scheme="random"),
                        struct_windows.assign(scheme="structured")], ignore_index=True).itertuples(index=False):
        for r in _eval_window(cache, survivors, cfg["fixed_variant"], cash_daily,
                              pd.Timestamp(w.start), pd.Timestamp(w.end)):
            r["scheme"] = w.scheme
            records.append(r)

    per = pd.DataFrame(records)
    summary = per.groupby("ticker", as_index=False).agg(
        avg_sharpe_delta=("sharpe_delta", "mean"), avg_cagr_delta=("cagr_delta", "mean"),
        sharpe_beat_rate=("beat_sharpe", "mean"), cagr_beat_rate=("beat_cagr", "mean"),
        avg_strat_sharpe=("strat_sharpe", "mean"), avg_strat_cagr=("strat_cagr", "mean"))
    summary["theme"] = summary["ticker"].map(theme_of)
    summary["delta_sharpe_rank"] = summary["avg_sharpe_delta"].rank(ascending=False, method="min")
    summary["delta_cagr_rank"] = summary["avg_cagr_delta"].rank(ascending=False, method="min")
    summary["combined_robust_score"] = (t2["rank_sharpe_weight"] * summary["delta_sharpe_rank"]
                                        + t2["rank_cagr_weight"] * summary["delta_cagr_rank"])
    summary = summary.sort_values(["combined_robust_score", "avg_sharpe_delta"],
                                  ascending=[True, False]).reset_index(drop=True)
    summary["robust_rank"] = summary.index + 1
    summary.attrs["shared_start"] = common.min().strftime("%Y-%m-%d")
    summary.attrs["shared_end"] = common.max().strftime("%Y-%m-%d")
    return summary


def finalists(summary: pd.DataFrame, cfg: dict) -> list:
    n = int(cfg.get("n_finalists", 12))
    picks = summary.head(n)["ticker"].tolist()
    for inc in cfg.get("incumbents", []):
        if inc not in picks and inc in set(summary["ticker"]):
            picks.append(inc)
    return picks


def main():
    import screen as scr
    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(outdir / "tier1_screen.csv")
    survivors = t1[t1["survivor"] == 1]["ticker"].tolist()
    theme_of = scr.flatten_universe(cfg)
    cache = scr.build_cache(sorted(set(survivors) | set(cfg["cash_chain"])), scr._resolve_end(cfg.get("end")))
    summary = run_robustness(cfg, cache, [t for t in survivors if t in cache], theme_of)
    summary.to_csv(outdir / "tier2_robustness.csv", index=False)
    print(f"Saved {outdir/'tier2_robustness.csv'}  finalists: {finalists(summary, cfg)}")


if __name__ == "__main__":
    main()
