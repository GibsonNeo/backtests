# strategy_sma_timing.py
# SMA timing, enter above SMA, exit below SMA
# Usage examples:
#   python strategy_sma_timing.py --years 10
#   python strategy_sma_timing.py --start 2010-01-01 --end 2025-10-10 --rf 0.03 --cash_proxy BIL
#   python strategy_sma_timing.py --config config.yml

import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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
    cash_proxy: str = "NONE"     # "NONE" means flat daily cash return
    risk_free_rate: float = 0.0  # annual, decimal
    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None
    years: Optional[int] = 10
    sma_windows: tuple = (100, 150, 200)
    outdir: str = "outputs"


def parse_cli() -> Config:
    p = argparse.ArgumentParser(description="SMA timing, enter above SMA, exit below SMA.")
    p.add_argument("--config", type=str, default=None, help="Path to config.yml")
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--cash_proxy", type=str, default=None, help='ETF for cash, for example "BIL", or "NONE"')
    p.add_argument("--rf", type=float, default=None, help="Annual risk free rate, decimal, example 0.03")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--sma_windows", type=str, default=None, help="Comma list, example 100,150,200")
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.config:
        try:
            import yaml
            with open(args.config, "r") as f:
                y = yaml.safe_load(f) or {}
            for k, v in y.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            print(f"Could not read config file, using defaults and CLI. Reason: {e}")

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
    if args.sma_windows is not None:
        cfg.sma_windows = tuple(int(x.strip()) for x in args.sma_windows.split(",") if x.strip())
    if args.outdir is not None:
        cfg.outdir = args.outdir

    return cfg


def fetch_prices(ticker: str, start: Optional[str], end: Optional[str], years: Optional[int]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. pip install yfinance")
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


def daily_returns_from_adjclose(adjclose: pd.Series) -> pd.Series:
    ret = adjclose.pct_change().fillna(0.0)
    ret.name = "ret"
    return ret


def cash_series(index: pd.DatetimeIndex, rf: float, cash_proxy: str) -> pd.Series:
    if cash_proxy and cash_proxy.upper() != "NONE":
        try:
            px = fetch_prices(cash_proxy, start=index[0].strftime("%Y-%m-%d"), end=index[-1].strftime("%Y-%m-%d"), years=None)
            r = daily_returns_from_adjclose(px["AdjClose"])
            return r.reindex(index).fillna(0.0)
        except Exception:
            pass
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    return pd.Series(daily_rf, index=index, name="cash_ret")


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return dd.min(), peak, trough


def ulcer_index(equity: pd.Series) -> float:
    rollmax = equity.cummax()
    dd = (equity / rollmax - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(dd)))) if len(dd) else float("nan")


def annualized_stats(daily_ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
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


def build_sma_strategy(adjclose: pd.Series, sma_window: int, cash_ret: pd.Series) -> pd.Series:
    sma = adjclose.rolling(sma_window, min_periods=sma_window).mean()
    invested = adjclose > sma
    underlying_ret = adjclose.pct_change().fillna(0.0)
    strat_ret = np.where(invested, underlying_ret, cash_ret)
    return pd.Series(strat_ret, index=adjclose.index, name=f"SMA{str(sma_window)}")


def by_calendar_year(daily_ret: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    df = daily_ret.to_frame("ret")
    rows = []
    for y, grp in df.groupby(df.index.year):
        s = annualized_stats(grp["ret"], rf=rf)
        s["Year"] = int(y)
        rows.append(s)
    return pd.DataFrame(rows).set_index("Year").sort_index()


def run(cfg: Config):
    from pathlib import Path

    px = fetch_prices(cfg.ticker, cfg.start, cfg.end, cfg.years)
    adj = px["AdjClose"]
    idx = adj.index
    cash_ret = cash_series(idx, cfg.risk_free_rate, cfg.cash_proxy)
    bah_ret = adj.pct_change().fillna(0.0).rename("BuyAndHold")

    strategies = {f"SMA{w}": build_sma_strategy(adj, w, cash_ret) for w in cfg.sma_windows}

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame({
        **{k: pd.Series(annualized_stats(v, rf=cfg.risk_free_rate)) for k, v in strategies.items()},
        "BuyAndHold": pd.Series(annualized_stats(bah_ret, rf=cfg.risk_free_rate)),
    })
    overall.to_csv(outdir / "overall_stats.csv")

    by_year_list = [by_calendar_year(v, rf=cfg.risk_free_rate).add_prefix(f"{k}_") for k, v in strategies.items()]
    by_year = pd.concat(by_year_list + [by_calendar_year(bah_ret, rf=cfg.risk_free_rate).add_prefix("BAH_")], axis=1)
    by_year.to_csv(outdir / "by_year_stats.csv")

    eq = pd.DataFrame({k: (1.0 + v).cumprod() for k, v in strategies.items()})
    eq["BuyAndHold"] = (1.0 + bah_ret).cumprod()
    eq.to_csv(outdir / "equity_curves.csv")

    print("\nOverall stats")
    print(overall.round(4))
    print("\nBy calendar year, head")
    print(by_year.head().round(4))


def main():
    cfg = parse_cli()
    run(cfg)


if __name__ == "__main__":
    main()
