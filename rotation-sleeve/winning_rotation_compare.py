#!/usr/bin/env python3
"""Compare the CEF rotation model with the winningstrat top-ticker strategy."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cef_rotation import (
    StrategySpec,
    build_monthly_prices_from_daily,
    drawdown_series,
    fetch_adjusted_prices,
    run_rotation_strategy,
    summarize_returns,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


TRADING_DAYS = 252
ORIGINAL_CEFS = ("ADX", "RQI", "KYN", "NMZ")
TOP_WINNING_TICKERS = ("SPMO", "IWY", "QQQ", "DGRW")
FALLBACK = "BIL"
LOOKBACKS = (3, 7)
DOWNLOAD_START = "2005-01-01"


@dataclass
class DailyComparisonResult:
    name: str
    daily_returns: pd.Series
    monthly_returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    summary: dict[str, float | str | int]


def _get_col(df: pd.DataFrame, col_name: str, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_signal_and_total_return(tickers: Iterable[str], start: str, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install dependencies with: pip install -r requirements.txt")
    signal_cols = {}
    tr_cols = {}
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, actions=True)
        if raw.empty:
            raise ValueError(f"No data for {ticker}")
        close = _get_col(raw, "Close", ticker).astype(float)
        splits = _get_col(raw, "Stock Splits", ticker).fillna(0.0).astype(float)
        split_factor = splits.replace(0.0, 1.0)
        split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        signal_cols[ticker] = close * split_adj

        tr = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        tr_cols[ticker] = _get_col(tr, "Close", ticker).astype(float)
    return pd.DataFrame(signal_cols).sort_index(), pd.DataFrame(tr_cols).sort_index()


def _rolling_sma(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).mean()


def _consec_true(mask: pd.Series, days: int) -> pd.Series:
    count = 0
    values = []
    for value in mask.fillna(False).to_numpy():
        count = count + 1 if bool(value) else 0
        values.append(count >= days)
    return pd.Series(values, index=mask.index)


def winningstrat_position(
    price_signal: pd.Series,
    long_window: int = 200,
    short_window: int = 20,
    entry_days_long: int = 3,
    entry_days_short: int = 2,
    exit_days_long: int = 2,
    exit_days_short: int = 1,
) -> pd.Series:
    sma_long = _rolling_sma(price_signal, long_window)
    sma_short = _rolling_sma(price_signal, short_window)
    short_below_long = sma_short < sma_long

    entry_long_ok = _consec_true(price_signal > sma_long, entry_days_long)
    entry_short_ok = _consec_true((price_signal > sma_short) & (sma_short > sma_short.shift(1)), entry_days_short)
    exit_long_ok = _consec_true(price_signal <= sma_long, exit_days_long)
    exit_short_ok = _consec_true(price_signal <= sma_short, exit_days_short)

    pos = pd.Series(0.0, index=price_signal.index)
    in_pos = 0.0
    for i in range(len(price_signal)):
        use_short = bool(short_below_long.iat[i])
        entry_ok = entry_short_ok.iat[i] if use_short else entry_long_ok.iat[i]
        exit_ok = exit_short_ok.iat[i] if use_short else exit_long_ok.iat[i]
        if in_pos == 1.0 and exit_ok:
            in_pos = 0.0
        elif in_pos == 0.0 and entry_ok:
            in_pos = 1.0
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0.0)
    first_valid = max(sma_long.dropna().index.min(), sma_short.dropna().index.min())
    return pos[pos.index >= first_valid]


def build_winning_variant_grid() -> list[dict[str, int | str]]:
    variants = []
    for entry_long in (2, 3):
        for entry_short in (2, 3):
            for exit_long in (1, 2):
                variants.append(
                    {
                        "name": f"hybridL200_S20_eL{entry_long}_eS{entry_short}_xL{exit_long}_xS1",
                        "entry_days_long": entry_long,
                        "entry_days_short": entry_short,
                        "exit_days_long": exit_long,
                        "exit_days_short": 1,
                    }
                )
    return variants


def build_monthly_gate(
    monthly_prices: pd.DataFrame,
    assets: tuple[str, ...],
    cash_proxy: str = FALLBACK,
    lookbacks: tuple[int, ...] = LOOKBACKS,
) -> pd.DataFrame:
    gates = {}
    for asset in assets:
        asset_scores = []
        cash_scores = []
        for lookback in lookbacks:
            asset_scores.append(monthly_prices[asset] / monthly_prices[asset].shift(lookback) - 1.0)
            cash_scores.append(monthly_prices[cash_proxy] / monthly_prices[cash_proxy].shift(lookback) - 1.0)
        asset_score = pd.concat(asset_scores, axis=1).mean(axis=1)
        cash_score = pd.concat(cash_scores, axis=1).mean(axis=1)
        raw = (asset_score > cash_score).astype(float)
        raw[pd.concat(asset_scores, axis=1).isna().any(axis=1)] = np.nan
        gates[asset] = raw.shift(1).fillna(0.0)
    return pd.DataFrame(gates, index=monthly_prices.index)


def equal_weight_daily_with_fallback(
    asset_returns: pd.DataFrame,
    fallback_returns: pd.Series,
    positions: pd.DataFrame,
) -> pd.Series:
    aligned_assets = asset_returns.reindex(positions.index).fillna(0.0)
    fallback = fallback_returns.reindex(positions.index).fillna(0.0)
    sleeve_returns = []
    for asset in positions.columns:
        pos = positions[asset].reindex(aligned_assets.index).fillna(0.0)
        sleeve_returns.append(pos * aligned_assets[asset] + (1.0 - pos) * fallback)
    return sum(sleeve_returns) / len(sleeve_returns)


def combine_monthly_gate_with_daily_trend(
    asset_returns: pd.DataFrame,
    fallback_returns: pd.Series,
    monthly_gate: pd.DataFrame,
    daily_trend: pd.DataFrame,
) -> pd.Series:
    daily_month = (asset_returns.index.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)).normalize()
    gate_daily = monthly_gate.reindex(daily_month).set_index(asset_returns.index).fillna(0.0)
    combined_position = gate_daily.reindex(columns=daily_trend.columns).fillna(0.0) * daily_trend.reindex(asset_returns.index).fillna(0.0)
    return equal_weight_daily_with_fallback(asset_returns, fallback_returns, combined_position)


def combine_monthly_gate_or_daily_trend(
    asset_returns: pd.DataFrame,
    fallback_returns: pd.Series,
    monthly_gate: pd.DataFrame,
    daily_trend: pd.DataFrame,
) -> pd.Series:
    daily_month = (asset_returns.index.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)).normalize()
    gate_daily = monthly_gate.reindex(daily_month).set_index(asset_returns.index).fillna(0.0)
    combined_position = (
        gate_daily.reindex(columns=daily_trend.columns).fillna(0.0)
        .clip(0.0, 1.0)
        .combine(daily_trend.reindex(asset_returns.index).fillna(0.0).clip(0.0, 1.0), np.maximum)
    )
    return equal_weight_daily_with_fallback(asset_returns, fallback_returns, combined_position)


def monthly_from_daily(daily_returns: pd.Series) -> pd.Series:
    return daily_returns.resample("ME").apply(lambda s: (1.0 + s).prod() - 1.0)


def summarize_daily(name: str, daily_returns: pd.Series, fallback_daily: pd.Series) -> DailyComparisonResult:
    daily = daily_returns.dropna()
    monthly = monthly_from_daily(daily)
    rf_monthly = monthly_from_daily(fallback_daily.reindex(daily.index).fillna(0.0))
    equity = (1.0 + daily).cumprod()
    summary = summarize_returns(name, monthly, rf_monthly=rf_monthly)
    daily_vol = daily.std(ddof=0) * math.sqrt(TRADING_DAYS)
    summary["daily_ann_volatility"] = float(daily_vol)
    summary["daily_max_drawdown"] = float((equity / equity.cummax() - 1.0).min())
    return DailyComparisonResult(name, daily, monthly, equity, drawdown_series(equity), summary)


def align_results(results: dict[str, DailyComparisonResult]) -> dict[str, DailyComparisonResult]:
    common = None
    for result in results.values():
        common = result.daily_returns.index if common is None else common.intersection(result.daily_returns.index)
    if common is None or common.empty:
        raise ValueError("No common overlap across comparison results")
    common = common.sort_values()
    return {
        name: summarize_daily(name, result.daily_returns.reindex(common), pd.Series(0.0, index=common))
        for name, result in results.items()
    }


def plot_comparison(results: dict[str, DailyComparisonResult], outdir: Path) -> None:
    if plt is None:
        return
    chart_dir = outdir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.DataFrame({name: result.equity for name, result in results.items()})
    drawdowns = pd.DataFrame({name: result.drawdown for name, result in results.items()})
    for frame, title, file_name, ylabel in [
        (equity, "Strategy Comparison Equity", "comparison_equity.png", "Growth of $1"),
        (drawdowns, "Strategy Comparison Drawdowns", "comparison_drawdowns.png", "Drawdown"),
    ]:
        ax = frame.plot(figsize=(12, 7), title=title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / file_name, dpi=160)
        plt.close(ax.figure)


def write_report(outdir: Path, summary: pd.DataFrame, top_tickers: tuple[str, ...]) -> None:
    report = f"""# Winningstrat vs Rotation Comparison

Top winningstrat tickers used: {", ".join(top_tickers)}

`QQQ` is used instead of `QQQM` to avoid limiting the test to QQQM's 2020 inception.

## Summary

{summary.to_markdown(index=False)}

## Strategy Definitions

- `rotation_original_cefs_bil`: monthly 3+7 momentum rotation on `ADX, RQI, KYN, NMZ`, fallback `BIL`.
- `rotation_top4_bil`: same monthly 3+7 momentum rotation on `{", ".join(top_tickers)}`, fallback `BIL`.
- `winningstrat_top4_zero_cash`: daily `hybridL200_S20_eL3_eS2_xL2_xS1` on `{", ".join(top_tickers)}`, zero-return cash when out.
- `winningstrat_top4_bil_cash`: same daily rule, but out-of-market sleeves hold `BIL`.
- `hybrid_rotation_gate_plus_winning_daily_bil`: top-four sleeves hold a ticker only when the prior month-end 3+7 momentum gate is long and the daily winningstrat state is long; otherwise the sleeve holds `BIL`.
- `hybrid_rotation_gate_or_winning_daily_bil`: top-four sleeves hold a ticker when either the prior month-end 3+7 momentum gate or the daily winningstrat state is long; otherwise the sleeve holds `BIL`.

## Files

- `comparison_summary.csv`
- `comparison_monthly_returns.csv`
- `comparison_daily_returns.csv`
- `comparison_equity_curves.csv`
- `comparison_drawdowns.csv`
- `winningstrat_variant_grid_bil.csv`
- `charts/comparison_equity.png`
- `charts/comparison_drawdowns.png`
"""
    (outdir / "winning_rotation_comparison_report.md").write_text(report, encoding="utf-8")


def run_comparison(start: str, end: str | None, outdir: Path, top_tickers: tuple[str, ...] = TOP_WINNING_TICKERS) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_tickers = tuple(dict.fromkeys([*ORIGINAL_CEFS, *top_tickers, FALLBACK]))

    signal_prices, tr_prices = fetch_signal_and_total_return(all_tickers, DOWNLOAD_START, end)
    daily_returns = tr_prices.pct_change().fillna(0.0)
    monthly_prices = build_monthly_prices_from_daily(tr_prices).loc[start:]

    daily_trend = pd.DataFrame({ticker: winningstrat_position(signal_prices[ticker]) for ticker in top_tickers})
    common_daily_start = max(
        [
            pd.Timestamp(start),
            *[tr_prices[ticker].dropna().index.min() for ticker in [*ORIGINAL_CEFS, *top_tickers, FALLBACK]],
            *[daily_trend[ticker].dropna().index.min() for ticker in top_tickers],
        ]
    )
    daily_returns = daily_returns.loc[common_daily_start:]
    daily_trend = daily_trend.loc[common_daily_start:]
    monthly_prices = monthly_prices.loc[common_daily_start:]

    original_rotation = run_rotation_strategy(StrategySpec("rotation_original_cefs_bil", ORIGINAL_CEFS, FALLBACK, LOOKBACKS), monthly_prices)
    top_rotation = run_rotation_strategy(StrategySpec("rotation_top4_bil", top_tickers, FALLBACK, LOOKBACKS), monthly_prices)

    top_asset_returns = daily_returns[list(top_tickers)]
    fallback_daily = daily_returns[FALLBACK]
    winning_zero = equal_weight_daily_with_fallback(top_asset_returns, pd.Series(0.0, index=daily_returns.index), daily_trend)
    winning_bil = equal_weight_daily_with_fallback(top_asset_returns, fallback_daily, daily_trend)
    monthly_gate = build_monthly_gate(monthly_prices, top_tickers, FALLBACK, LOOKBACKS)
    hybrid = combine_monthly_gate_with_daily_trend(top_asset_returns, fallback_daily, monthly_gate, daily_trend)
    hybrid_or = combine_monthly_gate_or_daily_trend(top_asset_returns, fallback_daily, monthly_gate, daily_trend)

    daily_results = {
        "winningstrat_top4_zero_cash": summarize_daily("winningstrat_top4_zero_cash", winning_zero, fallback_daily),
        "winningstrat_top4_bil_cash": summarize_daily("winningstrat_top4_bil_cash", winning_bil, fallback_daily),
        "hybrid_rotation_gate_plus_winning_daily_bil": summarize_daily("hybrid_rotation_gate_plus_winning_daily_bil", hybrid, fallback_daily),
        "hybrid_rotation_gate_or_winning_daily_bil": summarize_daily("hybrid_rotation_gate_or_winning_daily_bil", hybrid_or, fallback_daily),
    }

    # Convert monthly rotation results to daily-shaped month-end series for common summary outputs.
    monthly_results = {
        "rotation_original_cefs_bil": original_rotation,
        "rotation_top4_bil": top_rotation,
    }
    common_monthly = None
    for result in [*monthly_results.values(), *daily_results.values()]:
        idx = result.monthly_returns.index
        common_monthly = idx if common_monthly is None else common_monthly.intersection(idx)
    common_monthly = common_monthly.sort_values()

    summary_rows = []
    monthly_return_cols = {}
    equity_cols = {}
    drawdown_cols = {}
    for name, result in monthly_results.items():
        r = result.monthly_returns.reindex(common_monthly).dropna()
        refreshed = summarize_returns(name, r, rf_monthly=monthly_from_daily(fallback_daily).reindex(r.index).fillna(0.0))
        summary_rows.append(refreshed)
        monthly_return_cols[name] = r
        equity = (1.0 + r).cumprod()
        equity_cols[name] = equity
        drawdown_cols[name] = drawdown_series(equity)
    for name, result in daily_results.items():
        r = result.monthly_returns.reindex(common_monthly).dropna()
        refreshed = summarize_returns(name, r, rf_monthly=monthly_from_daily(fallback_daily).reindex(r.index).fillna(0.0))
        summary_rows.append(refreshed)
        monthly_return_cols[name] = r
        equity = (1.0 + r).cumprod()
        equity_cols[name] = equity
        drawdown_cols[name] = drawdown_series(equity)

    summary = pd.DataFrame(summary_rows).sort_values(["sharpe", "cagr"], ascending=[False, False])
    variant_grid_rows = []
    for variant in build_winning_variant_grid():
        variant_trend = pd.DataFrame(
            {
                ticker: winningstrat_position(
                    signal_prices[ticker],
                    entry_days_long=int(variant["entry_days_long"]),
                    entry_days_short=int(variant["entry_days_short"]),
                    exit_days_long=int(variant["exit_days_long"]),
                    exit_days_short=int(variant["exit_days_short"]),
                )
                for ticker in top_tickers
            }
        ).loc[common_daily_start:]
        variant_returns = equal_weight_daily_with_fallback(top_asset_returns, fallback_daily, variant_trend)
        variant_monthly = monthly_from_daily(variant_returns).reindex(common_monthly).dropna()
        row = summarize_returns(str(variant["name"]), variant_monthly, rf_monthly=monthly_from_daily(fallback_daily).reindex(variant_monthly.index).fillna(0.0))
        row.update(
            {
                "entry_days_long": int(variant["entry_days_long"]),
                "entry_days_short": int(variant["entry_days_short"]),
                "exit_days_long": int(variant["exit_days_long"]),
                "exit_days_short": int(variant["exit_days_short"]),
            }
        )
        variant_grid_rows.append(row)
    variant_grid = pd.DataFrame(variant_grid_rows).sort_values(["sharpe", "cagr"], ascending=[False, False])

    pd.DataFrame(monthly_return_cols).to_csv(outdir / "comparison_monthly_returns.csv")
    pd.DataFrame({name: result.daily_returns for name, result in daily_results.items()}).to_csv(outdir / "comparison_daily_returns.csv")
    pd.DataFrame(equity_cols).to_csv(outdir / "comparison_equity_curves.csv")
    pd.DataFrame(drawdown_cols).to_csv(outdir / "comparison_drawdowns.csv")
    summary.to_csv(outdir / "comparison_summary.csv", index=False)
    variant_grid.to_csv(outdir / "winningstrat_variant_grid_bil.csv", index=False)

    plot_ready = {
        name: DailyComparisonResult(name, pd.Series(dtype=float), monthly_return_cols[name], equity_cols[name], drawdown_cols[name], {})
        for name in equity_cols
    }
    plot_comparison(plot_ready, outdir)
    write_report(outdir, summary, top_tickers)
    print(f"Wrote comparison outputs to {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare winningstrat top tickers with the CEF rotation model.")
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", default="outputs_winning_compare")
    p.add_argument("--top-tickers", default=",".join(TOP_WINNING_TICKERS), help="Comma-separated replacement sleeve tickers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = tuple(t.strip().upper() for t in args.top_tickers.split(",") if t.strip())
    run_comparison(args.start, args.end, Path(args.outdir), tickers)


if __name__ == "__main__":
    main()
