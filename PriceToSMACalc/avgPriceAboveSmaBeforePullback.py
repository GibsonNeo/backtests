#!/usr/bin/env python3
"""
Average percent excursion above SMA until price returns to the SMA
Reads settings from config.yml by default

Features
• Median and percentiles for peak excursion
• Run length stats in trading days
• Regime filter, only count runs that begin with price above a regime SMA
• Now versus history panel, compares the latest percent above SMA to the historical distribution
• Simple interpretation for selected longer SMAs, shows percentile rank, mean reversion probability,
  potential upside, and potential downside to the SMA
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not available, install with pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}, create config.yml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_series(col) -> pd.Series:
    if isinstance(col, pd.DataFrame):
        if col.shape[1] == 0:
            return pd.Series(dtype=float)
        return col.iloc[:, 0].astype(float)
    return pd.Series(col, dtype=float)


def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not available, install with pip install yfinance")
    # yfinance end is exclusive, pad a bit to include the end date
    df = yf.download(ticker, start=start, end=pd.to_datetime(end) + pd.Timedelta(days=2),
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data for {ticker} between {start} and {end}")
    price = None
    if isinstance(df.columns, pd.MultiIndex):
        for col in ["Adj Close", "Close"]:
            try:
                price = df.loc[:, (col, slice(None))]
                break
            except KeyError:
                continue
    else:
        if "Adj Close" in df.columns:
            price = df["Adj Close"]
        elif "Close" in df.columns:
            price = df["Close"]
    if price is None:
        raise ValueError("Could not find Adj Close or Close in downloaded data")
    price = _ensure_series(price).dropna()
    price = price[price.index <= pd.to_datetime(end)]
    if price.empty:
        raise ValueError("Price series is empty after dropna and end date filter")
    return price.astype(float)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def excursions_above_sma(price: pd.Series,
                         sma_run: pd.Series,
                         sma_regime: pd.Series | None) -> List[Dict[str, float]]:
    """
    Build completed runs that begin when price crosses from at or below SMA(window) to above,
    end when price touches or falls back to that SMA,
    and record peak percent above SMA and run length in trading days.

    If sma_regime is provided, require that the run begins with price above sma_regime
    For example, sma_regime equals SMA 200
    """
    runs: List[Dict[str, float]] = []
    in_run = False
    peak_pct = 0.0
    start_idx = None

    valid_idx = price.index.intersection(sma_run.dropna().index)

    for i, dt in enumerate(valid_idx):
        p = float(price.at[dt])
        s = float(sma_run.at[dt])
        if not np.isfinite(p) or not np.isfinite(s) or s == 0.0:
            continue

        gap_pct = (p - s) / s * 100.0
        above = p > s

        if not in_run:
            if i == 0:
                continue
            prev_dt = valid_idx[i - 1]
            p_prev = float(price.at[prev_dt])
            s_prev = float(sma_run.at[prev_dt])
            if not np.isfinite(p_prev) or not np.isfinite(s_prev):
                continue
            crossed_up = (p_prev <= s_prev) and above
            if crossed_up:
                # Regime filter, require price above regime SMA at the start of the run
                if sma_regime is not None and prev_dt in sma_regime.index:
                    s_reg_prev = float(sma_regime.at[prev_dt])
                    if np.isfinite(s_reg_prev):
                        if not (p_prev > s_reg_prev):
                            continue
                in_run = True
                peak_pct = max(0.0, gap_pct)
                start_idx = i
        else:
            if above:
                if gap_pct > peak_pct:
                    peak_pct = gap_pct
            else:
                # completed run
                length_days = float(i - start_idx + 1) if start_idx is not None else 0.0
                runs.append({"peak_pct": peak_pct, "length_days": length_days})
                in_run = False
                peak_pct = 0.0
                start_idx = None

    return runs


def summarize_numbers(values: List[float], percentiles: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        out = {"count": 0, "avg": 0.0, "median": 0.0}
        for q in percentiles:
            out[f"p{int(q)}"] = 0.0
        return out
    arr = np.asarray(values, dtype=float)
    out = {
        "count": int(arr.size),
        "avg": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }
    qs = np.percentile(arr, percentiles)
    for q, v in zip(percentiles, qs):
        out[f"p{int(q)}"] = float(v)
    return out


def percentile_rank(sample: np.ndarray, value: float) -> float:
    """
    Percentile rank of value within sample, in [0, 100]
    Inclusive on ties
    """
    if sample.size == 0 or not np.isfinite(value):
        return 0.0
    return float(100.0 * np.mean(sample <= value))


def simple_view_for_window(res_entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Build simple interpretation metrics for a single SMA window
    Returns keys
      gap_now_pct, perc_rank, mean_rev_prob, pot_upside_pct, pot_downside_pct
    """
    n = res_entry["now"]
    gap_now = float(n["gap_pct"]) if np.isfinite(n["gap_pct"]) else np.nan
    pot_downside = max(0.0, gap_now) if np.isfinite(gap_now) else 0.0

    runs = res_entry["runs"]
    peaks = np.asarray([r["peak_pct"] for r in runs], dtype=float) if len(runs) > 0 else np.asarray([])

    if peaks.size > 0 and np.isfinite(gap_now):
        perc = percentile_rank(peaks, gap_now)
        frac_exceed = float(np.mean(peaks > gap_now))
        pot_up = float(n["avg_additional_upside_if_exceeds_pct"])
        mean_rev = float(1.0 - frac_exceed)
    else:
        perc = 0.0
        pot_up = 0.0
        mean_rev = 1.0 if np.isfinite(gap_now) else 0.0

    return dict(
        gap_now_pct=gap_now,
        perc_rank=perc,
        mean_rev_prob=mean_rev,
        pot_upside_pct=pot_up,
        pot_downside_pct=pot_downside,
    )


def analyze(ticker: str, start: str, end: str, windows: List[int],
            regime_window: int,
            percentiles: List[float]) -> Dict[int, Dict[str, Any]]:
    price = fetch_prices(ticker, start, end)
    sma_regime = compute_sma(price, regime_window) if regime_window else None

    results: Dict[int, Dict[str, Any]] = {}

    for w in windows:
        sma_w = compute_sma(price, w)
        runs = excursions_above_sma(price, sma_w, sma_regime)

        peak_list = [r["peak_pct"] for r in runs]
        len_list = [r["length_days"] for r in runs]

        peak_stats = summarize_numbers(peak_list, percentiles)
        len_stats = summarize_numbers(len_list, percentiles)

        # Now versus history metrics, based on last available bar
        last_dt = price.index[-1]
        p_now = float(price.iloc[-1])
        s_now = float(sma_w.iloc[-1]) if np.isfinite(sma_w.iloc[-1]) else np.nan
        gap_series = (price - sma_w) / sma_w * 100.0
        gap_series = gap_series.replace([np.inf, -np.inf], np.nan)
        gap_now = float(gap_series.iloc[-1]) if np.isfinite(gap_series.iloc[-1]) else np.nan

        if len(peak_list) > 0 and np.isfinite(gap_now):
            arr = np.asarray(peak_list, dtype=float)
            frac_exceed = float(np.mean(arr > gap_now))
            extra_upside = arr - gap_now
            extra_upside = extra_upside[extra_upside > 0]
            avg_extra_up = float(np.mean(extra_upside)) if extra_upside.size > 0 else 0.0
        else:
            frac_exceed = 0.0
            avg_extra_up = 0.0

        results[w] = {
            "runs": runs,
            "peak_stats": peak_stats,
            "length_stats": len_stats,
            "gap_series": gap_series,
            "now": {
                "date": str(last_dt.date()),
                "price": p_now,
                "sma": s_now,
                "gap_pct": gap_now,
                "frac_hist_peaks_exceeding_current": frac_exceed,
                "avg_additional_upside_if_exceeds_pct": avg_extra_up,
            },
        }

    return results


def build_percentile_frame(results: Dict[int, Dict[str, Any]], windows: List[int]) -> pd.DataFrame:
    """
    Assemble a dataframe of percentile ranks for each SMA window.
    Percentile rank is computed using the full sample.
    """
    series_list = []
    for w in windows:
        entry = results.get(w)
        if not entry:
            continue
        gap_series = entry.get("gap_series")
        if gap_series is None or gap_series.empty:
            continue
        perc_series = gap_series.rank(pct=True, method="max") * 100.0
        perc_series = perc_series.rename(f"SMA {w}")
        series_list.append(perc_series)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    df = df.dropna(how="all")
    return df.sort_index()


def build_gap_frame(results: Dict[int, Dict[str, Any]], windows: List[int]) -> pd.DataFrame:
    """
    Assemble a dataframe of percent gap values for each SMA window.
    """
    series_list = []
    for w in windows:
        entry = results.get(w)
        if not entry:
            continue
        gap_series = entry.get("gap_series")
        if gap_series is None or gap_series.empty:
            continue
        series_list.append(gap_series.rename(f"SMA {w}"))
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    df = df.dropna(how="all")
    return df.sort_index()


def plot_sma_percentiles(results: Dict[int, Dict[str, Any]],
                         windows: List[int],
                         out_dir: str,
                         months: int) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available, install with pip install matplotlib")
    df = build_percentile_frame(results, windows)
    if df.empty:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    months = max(months, 1)
    cutoff = df.index.max() - pd.DateOffset(months=months)
    recent = df[df.index >= cutoff]
    if recent.empty:
        recent = df.tail(1)

    # Combined plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in recent.columns:
        ax.plot(recent.index, recent[col], label=col)
    ax.set_title(f"Price/SMA gap percentile (last {months} months)")
    ax.set_ylabel("Percentile rank")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path / "sma_percentile_combined.png", dpi=150)
    plt.close(fig)

    # Individual plots
    for col in recent.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(recent.index, recent[col], color="tab:blue")
        ax.set_title(f"{col} percentile (last {months} months)")
        ax.set_ylabel("Percentile rank")
        ax.set_ylim(0, 100)
        fig.autofmt_xdate()
        fig.tight_layout()
        sma_label = col.lower().replace(" ", "_")
        fig.savefig(out_path / f"{sma_label}_percentile.png", dpi=150)
        plt.close(fig)


def plot_gap_series(results: Dict[int, Dict[str, Any]],
                    windows: List[int],
                    out_dir: str,
                    months: int) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available, install with pip install matplotlib")
    df = build_gap_frame(results, windows)
    if df.empty:
        return

    out_path = Path(out_dir) / "gap_pct"
    out_path.mkdir(parents=True, exist_ok=True)

    months = max(months, 1)
    cutoff = df.index.max() - pd.DateOffset(months=months)
    recent = df[df.index >= cutoff]
    if recent.empty:
        recent = df.tail(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in recent.columns:
        ax.plot(recent.index, recent[col], label=col)
    ax.set_title(f"Price/SMA gap percent (last {months} months)")
    ax.set_ylabel("Gap (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    ax.legend(loc="best")
    fig.savefig(out_path / "sma_gap_combined.png", dpi=150)
    plt.close(fig)

    for col in recent.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(recent.index, recent[col], color="tab:orange")
        ax.set_title(f"{col} gap percent (last {months} months)")
        ax.set_ylabel("Gap (%)")
        fig.autofmt_xdate()
        fig.tight_layout()
        sma_label = col.lower().replace(" ", "_")
        fig.savefig(out_path / f"{sma_label}_gap.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average peak percent above SMA until retracement, with stats")
    p.add_argument("--config", default="config.yml", help="Path to config file")
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--windows", nargs="+", type=int)
    p.add_argument("--regime_window", type=int)
    p.add_argument("--percentiles", nargs="+", type=float)
    p.add_argument("--plot_dir", help="Directory to write PNG charts (enables plotting when provided)")
    p.add_argument("--plot_months", type=int, help="Number of months to display in the percentile charts")
    p.add_argument("--plot_windows", nargs="+", type=int, help="SMA windows to include in the percentile charts")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    ticker = args.ticker or cfg.get("ticker")
    start = args.start or cfg.get("start")
    end = args.end or cfg.get("end")
    windows = args.windows or cfg.get("windows", [50, 100, 150, 200])
    regime_window = args.regime_window if args.regime_window is not None else cfg.get("regime_window", 200)
    percentiles = args.percentiles or cfg.get("percentiles", [25, 50, 75, 90])
    plot_dir = args.plot_dir or cfg.get("plot_dir")
    plot_months = args.plot_months if args.plot_months is not None else cfg.get("plot_months", 6)
    plot_windows = args.plot_windows or cfg.get("plot_windows", [20, 50, 100, 200])

    if not ticker or not start or not end:
        raise ValueError("Config is missing required keys, provide ticker, start, end")

    res = analyze(ticker, start, end, windows, regime_window, percentiles)

    print(f"Ticker, {ticker}")
    print(f"Range, {start} to {end}")
    print(f"Regime filter, runs must begin with price above SMA {regime_window}")
    print("")

    # Table one, peak stats
    print("SMA, Runs, Avg peak pct, Median peak pct, " + ", ".join([f"P{int(q)} peak pct" for q in percentiles]))
    for w in sorted(res.keys()):
        ps = res[w]["peak_stats"]
        row = [w, ps["count"], ps["avg"], ps["median"]] + [ps[f"p{int(q)}"] for q in percentiles]
        fmt = f"{row[0]}, {row[1]}, {row[2]:.2f}, {row[3]:.2f}" + "".join([f", {v:.2f}" for v in row[4:]])
        print(fmt)

    print("")
    # Table two, run length stats
    print("SMA, Avg length days, Median length days, " + ", ".join([f"P{int(q)} length days" for q in percentiles]))
    for w in sorted(res.keys()):
        ls = res[w]["length_stats"]
        row = [w, ls["avg"], ls["median"]] + [ls[f"p{int(q)}"] for q in percentiles]
        fmt = f"{row[0]}, {row[1]:.1f}, {row[2]:.1f}" + "".join([f", {v:.1f}" for v in row[3:]])
        print(fmt)

    print("")
    # Table three, now versus history
    print("SMA, Last date, Price, SMA value, Gap pct now, Fraction peaks exceeding gap, Avg additional upside pct")
    for w in sorted(res.keys()):
        n = res[w]["now"]
        if np.isfinite(n["sma"]) and np.isfinite(n["gap_pct"]):
            print(f"{w}, {n['date']}, {n['price']:.2f}, {n['sma']:.2f}, {n['gap_pct']:.2f}, {n['frac_hist_peaks_exceeding_current']:.2f}, {n['avg_additional_upside_if_exceeds_pct']:.2f}")
        else:
            print(f"{w}, {n['date']}, {n['price']:.2f}, N/A, N/A, 0.00, 0.00")

    # Simple interpretation for longer SMAs, vertical layout
    print("")
    long_windows = cfg.get("simple_summary_windows", [150, 200])
    print("Simple interpretation, longer SMAs")
    for w in sorted(res.keys()):
        if w not in long_windows:
            continue
        sv = simple_view_for_window(res[w])
        print(f"SMA: {w}")
        if np.isfinite(sv["gap_now_pct"]):
            print(f"  Gap now pct: {sv['gap_now_pct']:.2f}")
            print(f"  Percentile rank: {sv['perc_rank']:.0f}")
            print(f"  Mean reversion probability: {sv['mean_rev_prob']:.2f}")
            print(f"  Potential upside pct: {sv['pot_upside_pct']:.2f}")
            print(f"  Potential downside pct: {sv['pot_downside_pct']:.2f}")
        else:
            print("  Gap now pct: N/A")
            print("  Percentile rank: N/A")
            print("  Mean reversion probability: N/A")
            print("  Potential upside pct: N/A")
            print("  Potential downside pct: N/A")
        print("")

    if plot_dir:
        selected_windows = [w for w in plot_windows if w in res]
        if selected_windows:
            try:
                plot_sma_percentiles(res, selected_windows, plot_dir, plot_months)
                plot_gap_series(res, selected_windows, plot_dir, plot_months)
                print(f"Saved percentile and gap charts to {plot_dir}")
            except RuntimeError as exc:
                print(f"Plotting skipped: {exc}")
        else:
            print("Plotting skipped: no overlap between requested plot windows and analysis windows")


if __name__ == "__main__":
    main()
