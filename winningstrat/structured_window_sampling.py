#!/usr/bin/env python3
import importlib.util
import os
import sys
from datetime import date, timedelta
from pathlib import Path

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

    outdir = cfg.get("outdir", "outputs_structured_window_sampling")
    if not os.path.isabs(outdir):
        outdir = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), outdir)

    lengths = [int(v) for v in cfg.get("window_lengths_years", [])]
    if not lengths:
        raise ValueError("Config requires window_lengths_years")

    rank_cfg = cfg.get("ranking_weights", {})

    return {
        "tickers": tickers,
        "strategy": strategy,
        "shared_start": shared_start,
        "end": end,
        "outdir": outdir,
        "cash_rate_percent": float(cfg.get("cash_rate_percent", 0.0)),
        "sharpe_rf_percent": float(cfg.get("sharpe_rf_percent", 0.0)),
        "window_lengths_years": lengths,
        "step_fraction_of_length": float(cfg.get("step_fraction_of_length", 0.5)),
        "max_overlap_fraction_for_tail": float(cfg.get("max_overlap_fraction_for_tail", 0.6)),
        "append_tail_window": bool(cfg.get("append_tail_window", True)),
        "rank_sharpe_weight": float(rank_cfg.get("sharpe", 0.6)),
        "rank_cagr_weight": float(rank_cfg.get("cagr", 0.4)),
    }


def _build_common_index(data, shared_start, end):
    common_index = None
    for ticker_data in data.values():
        ticker_index = ticker_data["tr"].loc[shared_start:end].index
        common_index = ticker_index if common_index is None else common_index.intersection(ticker_index)
    return common_index.sort_values()


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


def _generate_windows(common_index, lengths, step_fraction, append_tail, max_tail_overlap):
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


def _mean_metrics(frame, prefix):
    cols = ["CAGR", "Sharpe", "MaxDD", "AnnVol", "AnnReturn", "UlcerIndex", "TotalReturn"]
    grouped = frame.groupby(["ticker", "bucket_years"], as_index=False)[cols].mean()
    grouped = grouped.rename(columns={col: f"{prefix}_{col.lower()}" for col in cols})
    return grouped


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "structured_window_sampling.yml")
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
        lengths=cfg["window_lengths_years"],
        step_fraction=cfg["step_fraction_of_length"],
        append_tail=cfg["append_tail_window"],
        max_tail_overlap=cfg["max_overlap_fraction_for_tail"],
    )
    if windows.empty:
        raise ValueError("No structured windows were generated")

    actual_shared_start = common_index.min()
    actual_shared_end = common_index.max()

    windows_path = os.path.join(cfg["outdir"], "structured_windows.csv")
    windows.to_csv(windows_path, index=False)

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
                    "bucket_years": row.bucket_years,
                    "window_source": row.window_source,
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

    strategy_bucket = _mean_metrics(
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
    buyhold_bucket = _mean_metrics(
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
    beat_bucket = (
        per_window.groupby(["ticker", "bucket_years"], as_index=False)[
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

    bucket_summary = strategy_bucket.merge(buyhold_bucket, on=["ticker", "bucket_years"], how="inner").merge(
        beat_bucket, on=["ticker", "bucket_years"], how="inner"
    )
    bucket_summary = bucket_summary.sort_values(["bucket_years", "ticker"]).reset_index(drop=True)
    bucket_summary_path = os.path.join(cfg["outdir"], "ticker_length_bucket_summary.csv")
    bucket_summary.to_csv(bucket_summary_path, index=False)

    summary = (
        bucket_summary.groupby("ticker", as_index=False)[
            [
                "strategy_cagr",
                "strategy_sharpe",
                "strategy_maxdd",
                "strategy_annvol",
                "strategy_annreturn",
                "strategy_ulcerindex",
                "strategy_totalreturn",
                "buyhold_cagr",
                "buyhold_sharpe",
                "buyhold_maxdd",
                "buyhold_annvol",
                "buyhold_annreturn",
                "buyhold_ulcerindex",
                "buyhold_totalreturn",
                "sharpe_beat_rate",
                "cagr_beat_rate",
                "better_maxdd_rate",
                "avg_sharpe_delta",
                "avg_cagr_delta",
                "avg_maxdd_delta",
            ]
        ]
        .mean()
    )

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

    bucket_counts = windows.groupby("bucket_years", as_index=False).size().rename(columns={"size": "windows_in_bucket"})
    summary["configured_shared_start"] = shared_start.strftime("%Y-%m-%d")
    summary["actual_shared_start"] = actual_shared_start.strftime("%Y-%m-%d")
    summary["actual_shared_end"] = actual_shared_end.strftime("%Y-%m-%d")
    summary["rank_sharpe_weight"] = cfg["rank_sharpe_weight"]
    summary["rank_cagr_weight"] = cfg["rank_cagr_weight"]
    summary["bucket_count"] = len(cfg["window_lengths_years"])

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
                "window_lengths_years": ",".join(str(v) for v in cfg["window_lengths_years"]),
                "step_fraction_of_length": cfg["step_fraction_of_length"],
                "append_tail_window": cfg["append_tail_window"],
                "max_overlap_fraction_for_tail": cfg["max_overlap_fraction_for_tail"],
                "total_windows": len(windows),
                "rank_sharpe_weight": cfg["rank_sharpe_weight"],
                "rank_cagr_weight": cfg["rank_cagr_weight"],
            }
        ]
    )
    metadata_path = os.path.join(cfg["outdir"], "run_metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    bucket_counts_path = os.path.join(cfg["outdir"], "window_counts_by_bucket.csv")
    bucket_counts.to_csv(bucket_counts_path, index=False)

    print(f"Saved {windows_path}")
    print(f"Saved {per_window_path}")
    print(f"Saved {bucket_summary_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {ranking_path}")
    print(f"Saved {bucket_counts_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
