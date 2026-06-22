#!/usr/bin/env python3
"""Backtest the Seeking Alpha 4 CEF rotation sleeve and robustness variants."""

from __future__ import annotations

import argparse
import math
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime
    plt = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - handled at runtime
    yf = None


MONTHS_PER_YEAR = 12
DEFAULT_START = "2006-01-01"
DOWNLOAD_START = "1990-01-01"
ARTICLE_TARGETS = {"CAGR": 0.114, "MaxDD": -0.17, "AnnVol": 0.104}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    assets: tuple[str, ...]
    fallback: str
    lookbacks: tuple[int, ...]


@dataclass
class StrategyResult:
    spec: StrategySpec
    monthly_returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    allocations: pd.DataFrame
    sleeve_positions: pd.DataFrame
    switches: int
    summary: dict[str, float | str | int]


def pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


def build_monthly_prices_from_daily(daily_prices: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_prices.sort_index().resample("ME").last()
    if len(monthly):
        today = pd.Timestamp.today().normalize()
        if monthly.index[-1].normalize() > today:
            monthly = monthly.iloc[:-1]
    return monthly.dropna(how="all")


def fetch_adjusted_prices(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install dependencies with: pip install -r requirements.txt")
    tickers = sorted(set(tickers))
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers
    close = close.apply(pd.to_numeric, errors="coerce").sort_index()
    return close


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if len(dd) else math.nan


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def annualized_return(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return math.nan
    total = float((1.0 + r).prod())
    years = len(r) / MONTHS_PER_YEAR
    return total ** (1.0 / years) - 1.0 if years > 0 else math.nan


def summarize_returns(
    name: str,
    monthly_returns: pd.Series,
    rf_monthly: pd.Series | None = None,
    switches: int = 0,
    exposure: pd.Series | None = None,
) -> dict[str, float | str | int]:
    r = monthly_returns.dropna()
    equity = (1.0 + r).cumprod()
    if rf_monthly is None:
        rf = pd.Series(0.0, index=r.index)
    else:
        rf = rf_monthly.reindex(r.index).fillna(0.0)
    excess = r - rf
    downside = excess[excess < 0.0]
    ann_vol = float(r.std(ddof=0) * math.sqrt(MONTHS_PER_YEAR)) if len(r) else math.nan
    sharpe = float(excess.mean() / r.std(ddof=0) * math.sqrt(MONTHS_PER_YEAR)) if r.std(ddof=0) > 0 else math.nan
    sortino = float(excess.mean() / downside.std(ddof=0) * math.sqrt(MONTHS_PER_YEAR)) if len(downside) and downside.std(ddof=0) > 0 else math.nan
    mdd = max_drawdown(equity)
    cagr = annualized_return(r)
    by_year = r.groupby(r.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
    out: dict[str, float | str | int] = {
        "strategy": name,
        "start": r.index.min().strftime("%Y-%m-%d") if len(r) else "",
        "end": r.index.max().strftime("%Y-%m-%d") if len(r) else "",
        "months": int(len(r)),
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else math.nan,
        "best_year": float(by_year.max()) if len(by_year) else math.nan,
        "worst_year": float(by_year.min()) if len(by_year) else math.nan,
        "monthly_win_rate": float((r > 0).mean()) if len(r) else math.nan,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else math.nan,
        "switches": int(switches),
    }
    if exposure is not None:
        for asset, weight in exposure.items():
            out[f"exposure_{asset}"] = float(weight)
    return out


def run_rotation_strategy(spec: StrategySpec, monthly_prices: pd.DataFrame, cash_proxy: str = "BIL") -> StrategyResult:
    needed = list(dict.fromkeys([*spec.assets, spec.fallback, cash_proxy]))
    prices = monthly_prices.reindex(columns=needed).dropna(how="all")
    returns = prices.pct_change()

    cash_returns = returns[cash_proxy].fillna(0.0) if cash_proxy in returns else pd.Series(0.0, index=returns.index)
    sleeve_signals: dict[str, pd.Series] = {}
    for asset in spec.assets:
        lookback_scores = []
        cash_scores = []
        for lookback in spec.lookbacks:
            asset_ret = prices[asset] / prices[asset].shift(lookback) - 1.0
            cash_ret = prices[cash_proxy] / prices[cash_proxy].shift(lookback) - 1.0 if cash_proxy in prices else pd.Series(0.0, index=prices.index)
            lookback_scores.append(asset_ret)
            cash_scores.append(cash_ret.fillna(0.0))
        avg_momentum = pd.concat(lookback_scores, axis=1).mean(axis=1)
        avg_cash = pd.concat(cash_scores, axis=1).mean(axis=1)
        signal = (avg_momentum > avg_cash).astype(float)
        signal[pd.concat(lookback_scores, axis=1).isna().any(axis=1)] = np.nan
        sleeve_signals[asset] = signal

    signal_df = pd.DataFrame(sleeve_signals, index=prices.index)
    trade_signal = signal_df.shift(1).fillna(0.0)
    sleeve_positions = trade_signal.apply(lambda s: np.where(s > 0.0, s.name, spec.fallback), axis=0, result_type="expand")
    sleeve_positions.columns = spec.assets

    weight = 1.0 / len(spec.assets)
    allocations = pd.DataFrame(0.0, index=prices.index, columns=sorted(set([*spec.assets, spec.fallback])))
    strategy_returns = pd.Series(0.0, index=prices.index, name=spec.name)
    for asset in spec.assets:
        asset_leg = trade_signal[asset] * returns[asset].fillna(0.0)
        fallback_leg = (1.0 - trade_signal[asset]) * returns[spec.fallback].fillna(0.0)
        strategy_returns += weight * (asset_leg + fallback_leg)
        allocations[asset] += weight * trade_signal[asset]
        allocations[spec.fallback] += weight * (1.0 - trade_signal[asset])

    valid = signal_df.notna().any(axis=1).shift(1).fillna(False)
    strategy_returns = strategy_returns[valid].round(10)
    allocations = allocations.loc[strategy_returns.index]
    sleeve_positions = sleeve_positions.loc[strategy_returns.index]
    switches = int((sleeve_positions != sleeve_positions.shift(1)).sum().sum())
    equity = (1.0 + strategy_returns).cumprod()
    dd = drawdown_series(equity)
    rf_monthly = cash_returns.reindex(strategy_returns.index).fillna(0.0)
    summary = summarize_returns(spec.name, strategy_returns, rf_monthly, switches, allocations.mean())
    return StrategyResult(spec, strategy_returns, equity, dd, allocations, sleeve_positions, switches, summary)


def equal_weight_buy_hold(name: str, assets: tuple[str, ...], monthly_prices: pd.DataFrame, rf_monthly: pd.Series | None = None) -> StrategyResult:
    returns = monthly_prices[list(assets)].pct_change().dropna(how="all").fillna(0.0)
    monthly = returns.mean(axis=1)
    equity = (1.0 + monthly).cumprod()
    allocations = pd.DataFrame(1.0 / len(assets), index=monthly.index, columns=assets)
    summary = summarize_returns(name, monthly, rf_monthly, exposure=allocations.mean())
    return StrategyResult(StrategySpec(name, assets, "NONE", ()), monthly, equity, drawdown_series(equity), allocations, allocations, 0, summary)


def bootstrap_monthly_returns(monthly_returns: pd.Series, runs: int = 2000, seed: int = 42) -> dict[str, float]:
    r = monthly_returns.dropna().to_numpy()
    if len(r) == 0:
        return {"bootstrap_median_cagr": math.nan, "bootstrap_5pct_cagr": math.nan, "bootstrap_95pct_cagr": math.nan, "bootstrap_median_max_drawdown": math.nan}
    rng = np.random.default_rng(seed)
    cagrs = []
    mdds = []
    for _ in range(runs):
        sample = rng.choice(r, size=len(r), replace=True)
        s = pd.Series(sample)
        equity = (1.0 + s).cumprod()
        cagrs.append(annualized_return(s))
        mdds.append(max_drawdown(equity))
    return {
        "bootstrap_median_cagr": float(np.median(cagrs)),
        "bootstrap_5pct_cagr": float(np.percentile(cagrs, 5)),
        "bootstrap_95pct_cagr": float(np.percentile(cagrs, 95)),
        "bootstrap_median_max_drawdown": float(np.median(mdds)),
    }


def regime_table(results: dict[str, StrategyResult], regimes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        for regime, (start, end) in regimes.items():
            r = result.monthly_returns.loc[start:end]
            if r.empty:
                continue
            row = summarize_returns(name, r)
            row["regime"] = regime
            if not result.allocations.empty:
                alloc = result.allocations.reindex(r.index).fillna(0.0)
                defensive_cols = [c for c in alloc.columns if c in ("IEF", "TLT", "SHY", "BIL", "SGOV", "Cash")]
                row["defensive_exposure"] = float(alloc[defensive_cols].sum(axis=1).mean()) if defensive_cols else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def data_integrity_table(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in monthly_prices.columns:
        s = monthly_prices[ticker].dropna()
        if s.empty:
            rows.append({"ticker": ticker, "first_month": "", "last_month": "", "months": 0, "missing_inside_history": math.nan})
            continue
        interior = monthly_prices.loc[s.index.min():s.index.max(), ticker]
        rows.append(
            {
                "ticker": ticker,
                "first_month": s.index.min().strftime("%Y-%m-%d"),
                "last_month": s.index.max().strftime("%Y-%m-%d"),
                "months": int(len(s)),
                "missing_inside_history": int(interior.isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker")


def plot_outputs(results: dict[str, StrategyResult], outdir: Path) -> None:
    if plt is None:
        return
    chart_dir = outdir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.DataFrame({name: r.equity for name, r in results.items()})
    drawdowns = pd.DataFrame({name: r.drawdown for name, r in results.items()})
    for frame, title, fname, ylabel in [
        (equity, "Equity Curves", "equity_curves.png", "Growth of $1"),
        (drawdowns, "Drawdowns", "drawdowns.png", "Drawdown"),
    ]:
        ax = frame.dropna(how="all").plot(figsize=(12, 7), linewidth=1.4, title=title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / fname, dpi=160)
        plt.close(ax.figure)
    for name, result in results.items():
        if result.allocations.empty:
            continue
        ax = result.allocations.plot.area(figsize=(12, 6), title=f"{name} Exposure", linewidth=0)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / f"exposure_{safe_name(name)}.png", dpi=160)
        plt.close(ax.figure)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def write_report(
    outdir: Path,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    regimes: pd.DataFrame,
    bootstrap: pd.DataFrame,
    integrity: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    original = summary[summary["strategy"] == "A_original_cefs_ief"]
    original_line = "Original strategy did not produce results."
    if not original.empty:
        row = original.iloc[0]
        original_line = (
            f"Original reproduction: CAGR {pct(row['cagr'])}, max drawdown {pct(row['max_drawdown'])}, "
            f"volatility {pct(row['annualized_volatility'])}. Article targets were about CAGR "
            f"{pct(ARTICLE_TARGETS['CAGR'])}, max drawdown {pct(ARTICLE_TARGETS['MaxDD'])}, volatility {pct(ARTICLE_TARGETS['AnnVol'])}."
        )
    answers = build_key_answers(summary, sensitivity, regimes, attribution)
    report = f"""# 4 CEF Rotation Model Report

Generated by `cef_rotation.py`.

## Reproduction Check

{original_line}

## Key Answers

{answers}

## Data Integrity Notes

The run uses Yahoo Finance adjusted close data (`auto_adjust=True`) and month-end total return prices. Missing-value and inception checks are written to `data_integrity.csv`.

{integrity.to_markdown(index=False)}

## Main Strategy Comparison

{summary.to_markdown(index=False)}

## Attribution Proxies

{attribution.to_markdown(index=False)}

## Sensitivity Analysis

{sensitivity.to_markdown(index=False)}

## Historical Regimes

{regimes.to_markdown(index=False)}

## Bootstrap Robustness

{bootstrap.to_markdown(index=False)}

## Files

- `monthly_returns.csv`: monthly returns for every strategy and benchmark.
- `equity_curves.csv`: growth of $1 for every strategy and benchmark.
- `drawdowns.csv`: drawdown series.
- `summary_comparison.csv`: primary metrics, switches, and exposures.
- `sensitivity_analysis.csv`: lookback/defensive asset grid.
- `regime_analysis.csv`: period splits requested in the objective.
- `bootstrap_analysis.csv`: bootstrap monthly return robustness.
- `attribution_analysis.csv`: proxy decomposition for trend, duration, and fund selection.
- `charts/`: equity, drawdown, and exposure PNGs.
"""
    (outdir / "report.md").write_text(report, encoding="utf-8")


def build_key_answers(summary: pd.DataFrame, sensitivity: pd.DataFrame, regimes: pd.DataFrame, attribution: pd.DataFrame) -> str:
    def val(strategy: str, col: str) -> float | None:
        rows = summary[summary["strategy"] == strategy]
        return None if rows.empty else float(rows.iloc[0][col])

    orig = val("A_original_cefs_ief", "cagr")
    bil = val("B_original_cefs_bil", "cagr")
    etf_ief = val("C_etfs_ief", "cagr")
    etf_bil = val("D_etfs_bil", "cagr")
    spy = val("SPY_buy_hold", "cagr")
    lines = [
        f"1. Article reproduction: compare the original run against the target row above; Yahoo data/vendor timing can create modest drift.",
        f"2. IEF contribution proxy: original IEF CAGR minus original BIL CAGR is {pct((orig - bil) if orig is not None and bil is not None else None)}.",
        f"3. Replacing IEF with BIL {'reduced' if orig is not None and bil is not None and bil < orig else 'did not reduce'} CAGR in this run.",
        f"4. Momentum contribution is proxied in `attribution_analysis.csv` by timing variants versus equal-weight buy-and-hold.",
        f"5. CEF selection is proxied by original CEF variants versus broad ETF variants; ETF/BIL CAGR is {pct(etf_bil)}.",
        f"6. SPY buy-and-hold CAGR is {pct(spy)}; compare it with the ETF variants before assuming CEF-specific edge.",
        "7. Overfit risk is highest if only the exact (3m + 7m)/2 and IEF combination stands out in `sensitivity_analysis.csv`.",
        "8. Forward robustness should favor variants that remain competitive with shorter-duration or cash-like defensive assets across regimes.",
    ]
    return "\n".join(lines)


def run_all(start: str, end: str | None, outdir: Path, bootstrap_runs: int) -> None:
    primary_specs = [
        StrategySpec("A_original_cefs_ief", ("ADX", "RQI", "KYN", "NMZ"), "IEF", (3, 7)),
        StrategySpec("B_original_cefs_bil", ("ADX", "RQI", "KYN", "NMZ"), "BIL", (3, 7)),
        StrategySpec("B2_original_cefs_sgov", ("ADX", "RQI", "KYN", "NMZ"), "SGOV", (3, 7)),
        StrategySpec("C_etfs_ief", ("SPY", "VNQ", "XLE", "MUB"), "IEF", (3, 7)),
        StrategySpec("D_etfs_bil", ("SPY", "VNQ", "XLE", "MUB"), "BIL", (3, 7)),
        StrategySpec("D2_etfs_sgov", ("SPY", "VNQ", "XLE", "MUB"), "SGOV", (3, 7)),
        StrategySpec("E_original_cefs_6m_ief", ("ADX", "RQI", "KYN", "NMZ"), "IEF", (6,)),
        StrategySpec("F_original_cefs_12m_ief", ("ADX", "RQI", "KYN", "NMZ"), "IEF", (12,)),
        StrategySpec("G_original_cefs_3_6_12m_ief", ("ADX", "RQI", "KYN", "NMZ"), "IEF", (3, 6, 12)),
    ]
    lookback_grid = [(3,), (6,), (7,), (9,), (12,), (3, 7), (3, 12), (6, 12)]
    defensive_grid = ["IEF", "TLT", "SHY", "BIL", "SGOV", "Cash"]
    all_tickers = {"ADX", "RQI", "KYN", "NMZ", "IEF", "BIL", "SGOV", "SPY", "VNQ", "XLE", "MUB", "TLT", "SHY"}
    daily = fetch_adjusted_prices(all_tickers, DOWNLOAD_START, end)
    monthly = build_monthly_prices_from_daily(daily)
    monthly["Cash"] = 1.0
    monthly = monthly.loc[start:]
    outdir.mkdir(parents=True, exist_ok=True)

    integrity = data_integrity_table(monthly)
    integrity.to_csv(outdir / "data_integrity.csv", index=False)

    rf_monthly = monthly["BIL"].pct_change().fillna(0.0) if "BIL" in monthly else pd.Series(0.0, index=monthly.index)
    results: dict[str, StrategyResult] = {}
    for spec in primary_specs:
        results[spec.name] = run_rotation_strategy(spec, monthly)
    results["SPY_buy_hold"] = equal_weight_buy_hold("SPY_buy_hold", ("SPY",), monthly, rf_monthly)
    results["CEF_equal_weight_buy_hold"] = equal_weight_buy_hold("CEF_equal_weight_buy_hold", ("ADX", "RQI", "KYN", "NMZ"), monthly, rf_monthly)
    results["ETF_equal_weight_buy_hold"] = equal_weight_buy_hold("ETF_equal_weight_buy_hold", ("SPY", "VNQ", "XLE", "MUB"), monthly, rf_monthly)

    sensitivity_results = []
    for lbs in lookback_grid:
        for defensive in defensive_grid:
            name = f"sens_cefs_{'_'.join(map(str, lbs))}m_{defensive}"
            spec = StrategySpec(name, ("ADX", "RQI", "KYN", "NMZ"), defensive, lbs)
            r = run_rotation_strategy(spec, monthly)
            row = r.summary.copy()
            row["lookbacks"] = "+".join(map(str, lbs))
            row["defensive_asset"] = defensive
            sensitivity_results.append(row)
    sensitivity = pd.DataFrame(sensitivity_results)

    summary = pd.DataFrame([r.summary for r in results.values()])
    attribution = build_attribution(summary)
    regimes = regime_table(
        results,
        {
            "2006_to_2012": ("2006-01-01", "2012-12-31"),
            "2013_to_2019": ("2013-01-01", "2019-12-31"),
            "2020_to_2021": ("2020-01-01", "2021-12-31"),
            "2022_to_latest": ("2022-01-01", "2099-12-31"),
        },
    )
    bootstrap = pd.DataFrame(
        [{"strategy": name, **bootstrap_monthly_returns(result.monthly_returns, runs=bootstrap_runs)} for name, result in results.items()]
    )

    pd.DataFrame({name: r.monthly_returns for name, r in results.items()}).to_csv(outdir / "monthly_returns.csv")
    pd.DataFrame({name: r.equity for name, r in results.items()}).to_csv(outdir / "equity_curves.csv")
    pd.DataFrame({name: r.drawdown for name, r in results.items()}).to_csv(outdir / "drawdowns.csv")
    summary.to_csv(outdir / "summary_comparison.csv", index=False)
    sensitivity.to_csv(outdir / "sensitivity_analysis.csv", index=False)
    regimes.to_csv(outdir / "regime_analysis.csv", index=False)
    bootstrap.to_csv(outdir / "bootstrap_analysis.csv", index=False)
    attribution.to_csv(outdir / "attribution_analysis.csv", index=False)
    plot_outputs(results, outdir)
    write_report(outdir, summary, sensitivity, regimes, bootstrap, integrity, attribution)


def build_attribution(summary: pd.DataFrame) -> pd.DataFrame:
    def cagr(name: str) -> float:
        rows = summary[summary["strategy"] == name]
        return math.nan if rows.empty else float(rows.iloc[0]["cagr"])

    rows = [
        {
            "component_proxy": "trend_filter_plus_IEF_on_CEFs",
            "comparison": "A_original_cefs_ief CAGR minus CEF_equal_weight_buy_hold CAGR",
            "estimated_cagr_delta": cagr("A_original_cefs_ief") - cagr("CEF_equal_weight_buy_hold"),
        },
        {
            "component_proxy": "IEF_duration_exposure",
            "comparison": "A_original_cefs_ief CAGR minus B_original_cefs_bil CAGR",
            "estimated_cagr_delta": cagr("A_original_cefs_ief") - cagr("B_original_cefs_bil"),
        },
        {
            "component_proxy": "CEF_selection_vs_broad_etfs_with_IEF",
            "comparison": "A_original_cefs_ief CAGR minus C_etfs_ief CAGR",
            "estimated_cagr_delta": cagr("A_original_cefs_ief") - cagr("C_etfs_ief"),
        },
        {
            "component_proxy": "CEF_selection_vs_broad_etfs_without_duration",
            "comparison": "B_original_cefs_bil CAGR minus D_etfs_bil CAGR",
            "estimated_cagr_delta": cagr("B_original_cefs_bil") - cagr("D_etfs_bil"),
        },
        {
            "component_proxy": "SGOV_vs_BIL_defensive_asset",
            "comparison": "B2_original_cefs_sgov CAGR minus B_original_cefs_bil CAGR",
            "estimated_cagr_delta": cagr("B2_original_cefs_sgov") - cagr("B_original_cefs_bil"),
        },
        {
            "component_proxy": "distribution_capture",
            "comparison": "Implicitly included through Yahoo adjusted close total return data; not separately decomposed without price-only NAV/market data.",
            "estimated_cagr_delta": math.nan,
        },
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Reproduce and robustness-test the Seeking Alpha 4 CEF Rotation Model.

            Example:
              python cef_rotation.py --start 2006-01-01 --outdir outputs
            """
        ),
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=None, help="Optional exclusive Yahoo end date, YYYY-MM-DD.")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--bootstrap-runs", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args.start, args.end, Path(args.outdir), args.bootstrap_runs)
    print(f"Wrote results to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
