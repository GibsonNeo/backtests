# Strategy, first and last trading week only
# Author: ChatGPT
# Usage examples:
#   python strategy_spy_month_edges.py --years 10
#   python strategy_spy_month_edges.py --start 2010-01-01 --end 2025-10-10 --window 5 --rf 0.0 --cash_proxy NONE
#   python strategy_spy_month_edges.py --config config.yml
#
# Notes:
# Uses Adjusted Close for total return
# Holding rule: long SPY only on dates that fall within the first N or last N trading days of each month
# Complement rule: long SPY only on dates that are NOT in those windows
# Outside the market we sit in cash, optional cash proxy ETF if provided

import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class Config:
    ticker: str = "SPY"
    cash_proxy: str = "NONE"  # "NONE" means zero return cash
    risk_free_rate: float = 0.0  # annualized, as decimal, 0.015 means 1.5 percent
    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None
    years: Optional[int] = 10  # lookback in years if start is None
    window: int = 5  # number of trading days from start and from end of month to include
    outdir: str = "outputs"


def parse_cli() -> Config:
    p = argparse.ArgumentParser(description="Hold SPY only during first and last trading weeks, then compare performance.")
    p.add_argument("--config", type=str, help="Path to config.yml", default=None)
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--cash_proxy", type=str, default=None, help='ETF for cash return, for example "BIL". Use "NONE" for zero return cash.')
    p.add_argument("--rf", type=float, default=None, help="Risk free rate, annualized decimal, example 0.03 for 3 percent")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--window", type=int, default=None, help="Number of trading days to include at each edge of the month")
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.config:
        try:
            import yaml  # PyYAML
            with open(args.config, "r") as f:
                y = yaml.safe_load(f) or {}
            for k, v in y.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            print(f"Could not read config file, falling back to CLI and defaults. Reason: {e}")

    # Apply CLI overrides when provided
    if args.ticker is not None:
        cfg.ticker = args.ticker
    if args.cash_proxy is not None:
        cfg.cash_proxy = args.cash_proxy
    if args.rf is not None:
        cfg.risk_free_rate = args.rf
    if args.start is not None:
        cfg.start = args.start
    if args.end is not None:
        cfg.end = args.end
    if args.years is not None:
        cfg.years = args.years
    if args.window is not None:
        cfg.window = args.window
    if args.outdir is not None:
        cfg.outdir = args.outdir

    return cfg


def fetch_prices(ticker: str, start: Optional[str], end: Optional[str], years: Optional[int]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed in this environment. Please install it with: pip install yfinance")
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if start is None:
        end_ts = pd.Timestamp(end)
        start_ts = end_ts - pd.DateOffset(years=years or 10)
        start = start_ts.strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    data = data.dropna()
    if "Adj Close" in data.columns:
        data = data.rename(columns={"Adj Close": "AdjClose"})
    if "AdjClose" not in data.columns:
        data["AdjClose"] = data["Close"]
    return data[["AdjClose"]]


def trading_day_windows(index: pd.DatetimeIndex, window: int) -> pd.Series:
    """
    Returns a boolean Series aligned with index.
    True for dates that are within the first N or last N trading days of their month.
    """
    df = pd.DataFrame(index=index)
    df["year"] = index.year
    df["month"] = index.month

    # Rank day number within month from start
    df["rank_start"] = df.groupby(["year", "month"]).cumcount() + 1
    # Rank from end
    df["rank_end"] = df.groupby(["year", "month"])["rank_start"].transform("max") - df["rank_start"] + 1

    in_window = (df["rank_start"] <= window) | (df["rank_end"] <= window)
    in_window.name = "edge_window"
    return in_window


def daily_returns_from_adjclose(adjclose: pd.Series) -> pd.Series:
    ret = adjclose.pct_change().fillna(0.0)
    ret.name = "ret"
    return ret


def cash_series(index: pd.DatetimeIndex, rf: float, cash_proxy: str) -> pd.Series:
    if cash_proxy and cash_proxy.upper() != "NONE":
        # Try to fetch a cash proxy ETF, for example BIL
        try:
            px = fetch_prices(cash_proxy, start=index[0].strftime("%Y-%m-%d"), end=index[-1].strftime("%Y-%m-%d"), years=None)
            r = daily_returns_from_adjclose(px["AdjClose"])
            return r.reindex(index).fillna(0.0)
        except Exception:
            pass
    # Fallback to flat daily risk free rate
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    return pd.Series(daily_rf, index=index, name="cash_ret")


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    min_idx = dd.idxmin()
    trough = min_idx
    peak = equity.loc[:min_idx].idxmax()
    return dd.min(), peak, trough


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index, based on daily closes. Square root of mean of squared drawdowns."""
    rollmax = equity.cummax()
    dd = (equity / rollmax - 1.0) * 100.0  # percent drawdown
    return float(np.sqrt(np.mean(np.square(dd)))) if len(dd) else float("nan")


def annualized_stats(daily_ret: pd.Series, rf: float = 0.0) -> dict:
    mu = daily_ret.mean() * 252.0
    sigma = daily_ret.std(ddof=0) * 252.0**0.5
    sharpe = (mu - rf) / sigma if sigma > 0 else np.nan

    equity = (1.0 + daily_ret).cumprod()
    mdd, peak, trough = max_drawdown(equity)
    ui = ulcer_index(equity)
    total_return = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (252.0 / len(daily_ret)) - 1.0 if len(daily_ret) else np.nan

    return {
        "CAGR": cagr,
        "AnnReturn": mu,
        "AnnVol": sigma,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "UlcerIndex": ui,
        "TotalReturn": total_return,
    }


def by_calendar_year(daily_ret: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    df = daily_ret.to_frame("ret")
    out = []
    for y, grp in df.groupby(df.index.year):
        s = annualized_stats(grp["ret"], rf=rf)
        s["Year"] = int(y)
        out.append(s)
    res = pd.DataFrame(out).set_index("Year").sort_index()
    return res


def month_of_year_breakdown(daily_ret: pd.Series) -> pd.DataFrame:
    df = daily_ret.to_frame("ret")
    out = df.groupby(df.index.month)["ret"].mean() * 252.0
    out.index.name = "Month"
    return out.to_frame("AvgAnnReturnFromDailyMean")


def build_strategy(cfg: Config):
    px = fetch_prices(cfg.ticker, cfg.start, cfg.end, cfg.years)
    spy_ret = daily_returns_from_adjclose(px["AdjClose"])
    idx = spy_ret.index

    # Window mask
    edge_mask = trading_day_windows(idx, cfg.window)

    # Cash return series
    cash_ret = cash_series(idx, cfg.risk_free_rate, cfg.cash_proxy)

    # Strategy returns, hold SPY on edge_mask dates, else hold cash
    strat_ret = np.where(edge_mask.values, spy_ret.values, cash_ret.values)
    strat_ret = pd.Series(strat_ret, index=idx, name="edge_only")

    # Complement returns, hold SPY only when not in edge window
    comp_ret = np.where(~edge_mask.values, spy_ret.values, cash_ret.values)
    comp_ret = pd.Series(comp_ret, index=idx, name="middle_only")

    # Buy and hold returns
    bah_ret = spy_ret.rename("buy_and_hold")

    return strat_ret, comp_ret, bah_ret, edge_mask


def summarize_and_save(cfg: Config):
    from pathlib import Path

    strat_ret, comp_ret, bah_ret, edge_mask = build_strategy(cfg)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Overall stats
    overall = pd.DataFrame({
        "EdgeOnly": pd.Series(annualized_stats(strat_ret, rf=cfg.risk_free_rate)),
        "MiddleOnly": pd.Series(annualized_stats(comp_ret, rf=cfg.risk_free_rate)),
        "BuyAndHold": pd.Series(annualized_stats(bah_ret, rf=cfg.risk_free_rate)),
    })
    overall.to_csv(outdir / "overall_stats.csv")

    # Year by year
    by_year = pd.concat([
        by_calendar_year(strat_ret, rf=cfg.risk_free_rate).add_prefix("Edge_"),
        by_calendar_year(comp_ret, rf=cfg.risk_free_rate).add_prefix("Middle_"),
        by_calendar_year(bah_ret, rf=cfg.risk_free_rate).add_prefix("BAH_"),
    ], axis=1)
    by_year.to_csv(outdir / "by_year_stats.csv")

    # Month of year breakdown
    moy_edge = month_of_year_breakdown(strat_ret).rename(columns={"AvgAnnReturnFromDailyMean": "Edge_AvgAnnReturn"})
    moy_middle = month_of_year_breakdown(comp_ret).rename(columns={"AvgAnnReturnFromDailyMean": "Middle_AvgAnnReturn"})
    moy_bah = month_of_year_breakdown(bah_ret).rename(columns={"AvgAnnReturnFromDailyMean": "BAH_AvgAnnReturn"})
    moy = moy_edge.join(moy_middle).join(moy_bah)
    moy.to_csv(outdir / "month_of_year_breakdown.csv")

    # Save mask calendar for audit
    mask_df = pd.DataFrame({"EdgeWindow": edge_mask.astype(int)})
    mask_df.to_csv(outdir / "edge_calendar.csv")

    # Save equity curves
    eq = pd.DataFrame({
        "EdgeOnly": (1.0 + strat_ret).cumprod(),
        "MiddleOnly": (1.0 + comp_ret).cumprod(),
        "BuyAndHold": (1.0 + bah_ret).cumprod(),
    })
    eq.to_csv(outdir / "equity_curves.csv")

    # Console preview
    print("\nOverall stats")
    print(overall.round(4))
    print("\nBy calendar year, head")
    print(by_year.head().round(4))
    print("\nMonth of year average annualized returns from daily mean")
    print(moy.round(4))


def main():
    cfg = parse_cli()
    summarize_and_save(cfg)


if __name__ == "__main__":
    main()
