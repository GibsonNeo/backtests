# smatimingstrat.py
# SMA timing strategy, enter when price > SMA using prior day data, exit to cash otherwise.
# Prefetch to seed SMAs, no lookahead, clean Buy and Hold, sanity check, auto load config.yml.

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

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
    cash_proxy: str = "NONE"      # "NONE" means flat daily cash return
    risk_free_rate: float = 0.0   # annual, decimal
    start: Optional[str] = None   # "YYYY-MM-DD"
    end: Optional[str] = None
    years: Optional[int] = 10
    sma_windows: tuple = (100, 150, 200)
    outdir: str = "outputs"


def load_yaml_into(cfg: Config, path: str) -> None:
    try:
        import yaml
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        for k, v in y.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    except Exception as e:
        print(f"Could not read config file at {path}. Reason: {e}")


def parse_cli() -> Config:
    p = argparse.ArgumentParser(description="SMA timing without lookahead.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--cash_proxy", type=str, default=None)
    p.add_argument("--rf", type=float, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--sma_windows", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    cfg = Config()

    # Auto load config.yml if present
    if args.config:
        load_yaml_into(cfg, args.config)
    else:
        default_cfg_path = os.path.join(os.getcwd(), "config.yml")
        if os.path.isfile(default_cfg_path):
            load_yaml_into(cfg, default_cfg_path)

    # CLI overrides
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


# ---------- core helpers ----------

def fetch_prices(ticker: str, start: Optional[str], end: Optional[str], years: Optional[int], prewindow: int = 200) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # pull extra history to seed SMA
    if start is not None:
        start_dt = pd.to_datetime(start) - pd.Timedelta(days=int(prewindow * 1.5))
        start = start_dt.strftime("%Y-%m-%d")
    else:
        end_ts = pd.Timestamp(end)
        start_ts = end_ts - pd.DateOffset(years=years or 10)
        start = start_ts.strftime("%Y-%m-%d")

    # no auto adjust, we will pick one adjusted series explicitly
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False).dropna(how="any")

    # Handle both single level and multi level columns
    if isinstance(data.columns, pd.MultiIndex):
        # For a single ticker requested as a list, columns look like
        # level 0, Open High Low Close Adj Close Volume
        # level 1, TICKER
        cols_lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in cols_lvl0:
            adj = data.loc[:, ("Adj Close", slice(None))]
        else:
            adj = data.loc[:, ("Close", slice(None))]
        # squeeze to a Series if there is only one second level column
        if isinstance(adj, pd.DataFrame) and adj.shape[1] == 1:
            adj = adj.iloc[:, 0]
    else:
        if "Adj Close" in data.columns:
            adj = data["Adj Close"]
        else:
            adj = data["Close"]

    # at this point adj may still be a DataFrame if multiple symbols slipped in
    if isinstance(adj, pd.DataFrame):
        # take the first column deterministically
        adj = adj.iloc[:, 0]

    adj = pd.to_numeric(adj, errors="coerce").dropna()
    adj.name = "AdjClose"
    return pd.DataFrame(adj)



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
    rollmax = equity.cummax()
    dd = equity / rollmax - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return dd.min(), peak, trough


def ulcer_index(equity: pd.Series) -> float:
    rollmax = equity.cummax()
    dd = (equity / rollmax - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(dd)))) if len(dd) else float("nan")


def annualized_stats(daily_ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    mu = daily_ret.mean() * 252.0
    sigma = daily_ret.std(ddof=0) * (252.0 ** 0.5)
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
    signal = adjclose.shift(1) > sma.shift(1)        # today position uses prior day info
    underlying_ret = adjclose.pct_change().fillna(0.0)
    strat_ret = np.where(signal.fillna(False), underlying_ret.values, cash_ret.values)
    strat_ret = pd.Series(strat_ret, index=adjclose.index, name=f"SMA{str(sma_window)}")

    # Start once SMA is fully valid on the prior day
    valid_mask = sma.shift(1).notna()
    first_valid = valid_mask.idxmax() if valid_mask.any() else adjclose.index[0]
    return strat_ret.loc[first_valid:]


def by_calendar_year(daily_ret: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    df = daily_ret.to_frame("ret")
    rows = []
    for y, grp in df.groupby(df.index.year):
        if len(grp) == 0:
            continue
        s = annualized_stats(grp["ret"], rf=rf)
        s["Year"] = int(y)
        rows.append(s)
    return pd.DataFrame(rows).set_index("Year").sort_index()


def align_to_eval_window(series_dict: Dict[str, pd.Series], start: Optional[str], end: Optional[str]) -> Dict[str, pd.Series]:
    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None
    out = {}
    for k, s in series_dict.items():
        s2 = s
        if start_dt is not None:
            s2 = s2.loc[s2.index >= start_dt]
        if end_dt is not None:
            s2 = s2.loc[s2.index <= end_dt]
        out[k] = s2.dropna()
    return out


def sanity_check_bah(adj: pd.Series, eval_start: pd.Timestamp, eval_end: pd.Timestamp, bah_ret: pd.Series):
    px = adj.loc[(adj.index >= eval_start) & (adj.index <= eval_end)]
    if len(px) < 2 or len(bah_ret) < 2:
        print("Sanity check skipped, not enough data")
        return
    price_ratio = float(px.iloc[-1] / px.iloc[0]) - 1.0
    equity = float((1.0 + bah_ret).prod()) - 1.0
    print(f"Sanity, price ratio {price_ratio:.4f}, cumprod {equity:.4f}, delta {equity - price_ratio:.6f}")


# ---------- main run ----------

def run(cfg: Config):
    from pathlib import Path

    max_window = max(cfg.sma_windows)
    px = fetch_prices(cfg.ticker, cfg.start, cfg.end, cfg.years, prewindow=max_window)
    adj = px["AdjClose"]
    idx = adj.index

    cash_ret_full = cash_series(idx, cfg.risk_free_rate, cfg.cash_proxy)
    bah_ret_full = adj.pct_change().fillna(0.0).rename("BuyAndHold")

    # Build strategies
    strategies = {f"SMA{w}": build_sma_strategy(adj, w, cash_ret_full) for w in cfg.sma_windows}

    # Align to evaluation window without dict union operator
    combined = dict(strategies)
    combined["BuyAndHold"] = bah_ret_full
    aligned = align_to_eval_window(combined, cfg.start, cfg.end)
    bah_ret = aligned.pop("BuyAndHold")
    strategies = aligned

    # Effective window and sanity
    all_series = list(strategies.values()) + [bah_ret]
    common_start = min(s.index[0] for s in all_series)
    common_end = max(s.index[-1] for s in all_series)
    print(f"Effective evaluation window, {common_start.date()} to {common_end.date()}")
    sanity_check_bah(adj, common_start, common_end, bah_ret)

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
