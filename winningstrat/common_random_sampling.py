#!/usr/bin/env python3
import importlib.util
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "winningstrat.py"

spec = importlib.util.spec_from_file_location("winningstrat_core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

TRADING_DAYS = core.TRADING_DAYS


def _resolve_date(value):
    if value is None:
        return date.today()

    if isinstance(value, date):
        return value

    text = str(value).strip().lower()
    if text == "today":
        return date.today()
    if text == "yesterday":
        return date.today() - timedelta(days=1)
    return pd.Timestamp(value).date()


def _load_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", [])
    if not tickers:
        raise ValueError("Config requires tickers")

    strategy = dict(cfg.get("strategy", {}))
    if not strategy:
        raise ValueError("Config requires strategy")

    shared_start = pd.Timestamp(_resolve_date(cfg.get("shared_start")))
    end = pd.Timestamp(_resolve_date(cfg.get("end", "yesterday")))
    if end < shared_start:
        raise ValueError("end must be on or after shared_start")

    outdir = cfg.get("outdir", "outputs_common_random_sampling")
    if not os.path.isabs(outdir):
        outdir = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), outdir)

    rand_cfg = cfg.get("random_sampling", {})
    rank_cfg = cfg.get("ranking_weights", {})

    return {
        "tickers": tickers,
        "strategy": strategy,
        "shared_start": shared_start,
        "end": end,
        "outdir": outdir,
        "cash_rate_percent": float(cfg.get("cash_rate_percent", 0.0)),
        "sharpe_rf_percent": float(cfg.get("sharpe_rf_percent", 0.0)),
        "num_samples": int(rand_cfg.get("num_samples", 100)),
        "min_years": float(rand_cfg.get("min_years", 3)),
        "max_years": rand_cfg.get("max_years", "auto"),
        "random_seed": rand_cfg.get("random_seed"),
        "rank_sharpe_weight": float(rank_cfg.get("sharpe", 0.6)),
        "rank_cagr_weight": float(rank_cfg.get("cagr", 0.4)),
    }


def _generate_windows(common_index, num_samples, min_years, max_years, seed):
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


def _build_common_index(data, shared_start, end):
    common_index = None
    for ticker_data in data.values():
        ticker_index = ticker_data["tr"].loc[shared_start:end].index
        common_index = ticker_index if common_index is None else common_index.intersection(ticker_index)
    return common_index.sort_values()


def _mean_metrics(frame, prefix):
    cols = ["CAGR", "Sharpe", "MaxDD", "AnnVol", "AnnReturn", "UlcerIndex", "TotalReturn"]
    grouped = frame.groupby("ticker", as_index=False)[cols].mean()
    grouped = grouped.rename(columns={col: f"{prefix}_{col.lower()}" for col in cols})
    return grouped


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "common_random_sampling.yml")
    cfg = _load_config(cfg_path)

    os.makedirs(cfg["outdir"], exist_ok=True)

    daily_cash = (cfg["cash_rate_percent"] / 100.0) / TRADING_DAYS
    sharpe_rf = cfg["sharpe_rf_percent"] / 100.0
    strategy = cfg["strategy"]
    warmup_days = core._warmup_days([strategy])

    shared_start = cfg["shared_start"]
    end = cfg["end"]
    ext_start = (shared_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    print("Fetching price history...")
    data = {}
    for ticker in cfg["tickers"]:
        sig, tr = core.fetch_series(ticker, ext_start, end.strftime("%Y-%m-%d"))
        data[ticker] = {"sig": sig, "tr": tr}

    common_index = _build_common_index(data, shared_start, end)
    if common_index.empty:
        raise ValueError("No shared trading dates found across tickers")

    windows = _generate_windows(
        common_index=common_index,
        num_samples=cfg["num_samples"],
        min_years=cfg["min_years"],
        max_years=cfg["max_years"],
        seed=cfg["random_seed"],
    )

    actual_shared_start = common_index.min()
    actual_shared_end = common_index.max()

    window_path = os.path.join(cfg["outdir"], "sample_windows.csv")
    windows.to_csv(window_path, index=False)

    results = []
    for row in windows.itertuples(index=False):
        run_start = pd.Timestamp(row.start)
        run_end = pd.Timestamp(row.end)
        print(f"Running {row.window_id}: {row.start} to {row.end}")

        for ticker in cfg["tickers"]:
            sig_px = data[ticker]["sig"].loc[run_start:run_end]
            tr_px = data[ticker]["tr"].loc[run_start:run_end]

            pos = core._variant_position(sig_px, strategy)
            strat_daily = core._daily_from_pos(pos, tr_px, daily_cash)
            strat_metrics = core.metrics_from_returns(strat_daily, sharpe_rf)

            bh_daily = tr_px.pct_change().fillna(0.0)
            bh_metrics = core.metrics_from_returns(bh_daily, sharpe_rf)

            results.append(
                {
                    "window_id": row.window_id,
                    "start": row.start,
                    "end": row.end,
                    "trading_days": row.trading_days,
                    "span_years": row.span_years,
                    "ticker": ticker,
                    "variant": strategy["name"],
                    "baseline_variant": "baseline_buyhold",
                    **{f"strategy_{k.lower()}": v for k, v in strat_metrics.items()},
                    **{f"buyhold_{k.lower()}": v for k, v in bh_metrics.items()},
                }
            )

    per_window = pd.DataFrame(results)
    per_window["sharpe_delta"] = per_window["strategy_sharpe"] - per_window["buyhold_sharpe"]
    per_window["cagr_delta"] = per_window["strategy_cagr"] - per_window["buyhold_cagr"]
    per_window["maxdd_delta"] = per_window["strategy_maxdd"] - per_window["buyhold_maxdd"]
    per_window["strategy_beats_buyhold_sharpe"] = (per_window["strategy_sharpe"] > per_window["buyhold_sharpe"]).astype(float)
    per_window["strategy_beats_buyhold_cagr"] = (per_window["strategy_cagr"] > per_window["buyhold_cagr"]).astype(float)
    per_window["strategy_better_maxdd"] = (per_window["strategy_maxdd"] > per_window["buyhold_maxdd"]).astype(float)

    per_window_path = os.path.join(cfg["outdir"], "per_ticker_window_results.csv")
    per_window.to_csv(per_window_path, index=False)

    strategy_avg = _mean_metrics(
        per_window.rename(
            columns={
                "strategy_cagr": "CAGR",
                "strategy_sharpe": "Sharpe",
                "strategy_maxdd": "MaxDD",
                "strategy_annvol": "AnnVol",
                "strategy_annreturn": "AnnReturn",
                "strategy_ulcerindex": "UlcerIndex",
                "strategy_totalreturn": "TotalReturn",
            }
        ),
        "strategy",
    )
    buyhold_avg = _mean_metrics(
        per_window.rename(
            columns={
                "buyhold_cagr": "CAGR",
                "buyhold_sharpe": "Sharpe",
                "buyhold_maxdd": "MaxDD",
                "buyhold_annvol": "AnnVol",
                "buyhold_annreturn": "AnnReturn",
                "buyhold_ulcerindex": "UlcerIndex",
                "buyhold_totalreturn": "TotalReturn",
            }
        ),
        "buyhold",
    )

    beat_rates = (
        per_window.groupby("ticker", as_index=False)[
            [
                "strategy_beats_buyhold_sharpe",
                "strategy_beats_buyhold_cagr",
                "strategy_better_maxdd",
                "sharpe_delta",
                "cagr_delta",
                "maxdd_delta",
            ]
        ]
        .mean()
        .rename(
            columns={
                "strategy_beats_buyhold_sharpe": "sharpe_beat_rate",
                "strategy_beats_buyhold_cagr": "cagr_beat_rate",
                "strategy_better_maxdd": "better_maxdd_rate",
                "sharpe_delta": "avg_sharpe_delta",
                "cagr_delta": "avg_cagr_delta",
                "maxdd_delta": "avg_maxdd_delta",
            }
        )
    )

    summary = strategy_avg.merge(buyhold_avg, on="ticker", how="inner").merge(beat_rates, on="ticker", how="inner")
    summary["strategy_sharpe_rank"] = summary["strategy_sharpe"].rank(ascending=False, method="min").astype(int)
    summary["strategy_cagr_rank"] = summary["strategy_cagr"].rank(ascending=False, method="min").astype(int)
    summary["delta_sharpe_rank"] = summary["avg_sharpe_delta"].rank(ascending=False, method="min").astype(int)
    summary["delta_cagr_rank"] = summary["avg_cagr_delta"].rank(ascending=False, method="min").astype(int)
    summary["combined_rank_score"] = (
        cfg["rank_sharpe_weight"] * summary["delta_sharpe_rank"]
        + cfg["rank_cagr_weight"] * summary["delta_cagr_rank"]
    )
    summary = summary.sort_values(
        ["combined_rank_score", "delta_sharpe_rank", "delta_cagr_rank", "avg_sharpe_delta", "avg_cagr_delta"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    summary["combined_rank"] = summary.index + 1

    summary["configured_shared_start"] = shared_start.strftime("%Y-%m-%d")
    summary["actual_shared_start"] = actual_shared_start.strftime("%Y-%m-%d")
    summary["actual_shared_end"] = actual_shared_end.strftime("%Y-%m-%d")
    summary["num_windows"] = cfg["num_samples"]
    summary["sampling_min_years"] = cfg["min_years"]
    summary["sampling_max_years"] = (
        float(cfg["max_years"]) if cfg["max_years"] not in (None, "", "auto") else round(len(common_index) / TRADING_DAYS, 3)
    )
    summary["rank_sharpe_weight"] = cfg["rank_sharpe_weight"]
    summary["rank_cagr_weight"] = cfg["rank_cagr_weight"]

    summary_path = os.path.join(cfg["outdir"], "ticker_summary.csv")
    ranking_path = os.path.join(cfg["outdir"], "ticker_ranking_vs_buyhold.csv")
    summary.to_csv(summary_path, index=False)
    summary.to_csv(ranking_path, index=False)

    metadata = pd.DataFrame(
        [
            {
                "strategy": strategy["name"],
                "tickers": ",".join(cfg["tickers"]),
                "configured_shared_start": shared_start.strftime("%Y-%m-%d"),
                "actual_shared_start": actual_shared_start.strftime("%Y-%m-%d"),
                "actual_shared_end": actual_shared_end.strftime("%Y-%m-%d"),
                "configured_end": end.strftime("%Y-%m-%d"),
                "num_windows": cfg["num_samples"],
                "min_years": cfg["min_years"],
                "max_years": cfg["max_years"],
                "random_seed": cfg["random_seed"],
                "rank_sharpe_weight": cfg["rank_sharpe_weight"],
                "rank_cagr_weight": cfg["rank_cagr_weight"],
            }
        ]
    )
    metadata_path = os.path.join(cfg["outdir"], "run_metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    print(f"Saved {window_path}")
    print(f"Saved {per_window_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {ranking_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
