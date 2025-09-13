
#!/usr/bin/env python3
"""
Tiered Triple MA backtest
Position sizing in thirds, based on price vs 20, 50, 100 period MAs
Full when above all, two thirds when below 20 and still above 50 and 100,
one third when below 50 and still above 100,
flat when below 100

Includes options for EMA vs SMA, start and end dates, ticker, timeframe,
entry and exit confirmation bars, slippage and flat per trade fees.
Outputs a single CSV that is easy to use in Google Sheets.
Optional plot with --plot.
"""

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd

# quiet future downcasting behavior
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class Params:
    start_date: Optional[str] = "2001-01-01"
    end_date: Optional[str] = "2025-09-11"
    ticker: Optional[str] = "wmt"
    use_ema: bool = True
    short_len: int = 50
    med_len: int = 100
    long_len: int = 200
    entry_confirm_bars: int = 1
    exit_confirm_bars: int = 1
    allow_equal_on_price: bool = False
    timeframe: str = "1D"
    initial_capital: float = 100000.0
    slippage_pct: float = 0.0
    fee_per_trade: float = 0.0
    seed: Optional[int] = None


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = (rule or "1D").upper()
    if r in ("1D", "D", "DAY", "DAILY"):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(r).agg(ohlc).dropna(how="any")


def _ma(series: pd.Series, length: int, use_ema: bool) -> pd.Series:
    if use_ema:
        return series.ewm(span=length, adjust=False, min_periods=length).mean()
    return series.rolling(window=length, min_periods=length).mean()


def _confirm(seq: pd.Series, bars: int) -> pd.Series:
    if bars <= 1:
        return seq.fillna(False).astype("boolean")
    out = seq.rolling(window=bars, min_periods=bars).sum() == bars
    return out.astype("boolean")


def _find_col_like(df: pd.DataFrame, target: str):
    t = target.lower()
    for c in df.columns:
        if str(c).strip().lower() == t:
            return c
    for c in df.columns:
        if t in str(c).strip().lower():
            return c
    return None


def _normalize_ohlcv(data: pd.DataFrame, ticker: Optional[str]) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        cols = data.columns
        lev0 = cols.get_level_values(0)
        lev1 = cols.get_level_values(1)
        fields = {"open", "high", "low", "close", "adj close", "volume"}
        count0 = sum(str(v).lower() in fields for v in lev0)
        count1 = sum(str(v).lower() in fields for v in lev1)
        field_level = 0 if count0 >= count1 else 1
        ticker_level = 1 - field_level
        if ticker is not None and ticker in cols.get_level_values(ticker_level):
            sub = data.xs(ticker, axis=1, level=ticker_level, drop_level=True)
        else:
            tick_vals = cols.levels[ticker_level]
            sub = data.xs(tick_vals[0], axis=1, level=ticker_level, drop_level=True)
        df = pd.DataFrame(index=data.index)
        for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            c = _find_col_like(sub, f)
            if c is not None:
                df[f] = pd.to_numeric(sub[c], errors="coerce")
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].dropna(how="any")
    df = pd.DataFrame(index=data.index)
    for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        c = _find_col_like(data, f)
        if c is not None:
            df[f] = pd.to_numeric(data[c], errors="coerce")
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna(how="any")


def load_data(ticker: Optional[str], start: Optional[str], end: Optional[str], csv: Optional[str]) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv)
        cols = {c.lower(): c for c in df.columns}
        need = ["date", "open", "high", "low", "close", "volume"]
        for k in need:
            if k not in cols:
                raise ValueError(f"CSV is missing required column {k}")
        df = df.rename(columns={
            cols["date"]: "Date",
            cols["open"]: "Open",
            cols["high"]: "High",
            cols["low"]: "Low",
            cols["close"]: "Close",
            cols["volume"]: "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")

    if yf is None:
        raise RuntimeError("yfinance is not available, install it or provide a CSV file")
    if not ticker:
        raise ValueError("Provide ticker or csv path")

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No data returned from yfinance")

    data = _normalize_ohlcv(raw, ticker)
    data.index = pd.to_datetime(data.index)
    return data


def build_tier_positions(df: pd.DataFrame, p: Params):
    price = df["Close"]
    sMA = _ma(price, p.short_len, p.use_ema)
    mMA = _ma(price, p.med_len, p.use_ema)
    lMA = _ma(price, p.long_len, p.use_ema)

    if p.allow_equal_on_price:
        above20 = price >= sMA
        above50 = price >= mMA
        above100 = price >= lMA
    else:
        above20 = price > sMA
        above50 = price > mMA
        above100 = price > lMA

    a20_c = _confirm(above20, p.entry_confirm_bars)
    a50_c = _confirm(above50, p.entry_confirm_bars)
    a100_c = _confirm(above100, p.entry_confirm_bars)

    b20_c = _confirm(~above20, p.exit_confirm_bars)
    b50_c = _confirm(~above50, p.exit_confirm_bars)
    b100_c = _confirm(~above100, p.exit_confirm_bars)

    n = len(df.index)
    pos = np.zeros(n, dtype=float)

    tier1 = False  # above 100
    tier2 = False  # above 50
    tier3 = False  # above 20

    for i in range(n):
        if bool(b100_c.iloc[i]):
            tier1 = tier2 = tier3 = False
        else:
            if bool(b50_c.iloc[i]):
                tier2 = False
                tier3 = False
            else:
                if bool(b20_c.iloc[i]):
                    tier3 = False

        if bool(a100_c.iloc[i]):
            tier1 = True
        if tier1 and bool(a50_c.iloc[i]):
            tier2 = True
        if tier2 and bool(a20_c.iloc[i]):
            tier3 = True

        pos[i] = (int(tier1) + int(tier2) + int(tier3)) / 3.0

    ctx = pd.DataFrame(index=df.index)
    ctx["sMA"] = sMA
    ctx["mMA"] = mMA
    ctx["lMA"] = lMA
    ctx["above20"] = above20
    ctx["above50"] = above50
    ctx["above100"] = above100
    ctx["pos_target"] = pos
    return pos, ctx


def backtest(df: pd.DataFrame, p: Params):
    pos_target, ctx = build_tier_positions(df, p)

    pos = pd.Series(pos_target, index=df.index)
    pos_shift = pos.shift(1).fillna(0.0)

    close = pd.to_numeric(df["Close"], errors="coerce")
    openp = pd.to_numeric(df["Open"], errors="coerce")
    ret_close = close.pct_change().fillna(0.0)

    strat_ret = pos_shift * ret_close

    delta = pos - pos_shift
    trades_per_day = (delta.abs() * 3.0).round().astype(int)
    slip_factor = 1.0 - (p.slippage_pct * delta.abs())
    slip_factor = slip_factor.clip(lower=0.0)

    equity_gross = p.initial_capital * (1.0 + strat_ret).cumprod()
    slip_cum = pd.Series(slip_factor.values, index=df.index).cumprod()

    fee_flow = -(p.fee_per_trade * trades_per_day)
    fee_cum = fee_flow.cumsum()

    equity_net = equity_gross * slip_cum + fee_cum

    equity = pd.DataFrame(index=df.index)
    equity["position"] = pos_shift
    equity["ret_close"] = ret_close
    equity["strat_ret"] = strat_ret
    equity["equity"] = equity_net
    equity["delta_pos"] = delta
    equity["trades_count"] = trades_per_day

    ch = delta[delta != 0.0]
    trades = []
    for ts, d in ch.items():
        trades.append({
            "dt": ts,
            "side": "BUY" if d > 0 else "SELL",
            "change_frac": float(abs(d)),
            "new_pos_frac": float(pos.loc[ts]),
            "price_open": float(openp.loc[ts])
        })
    trades_df = pd.DataFrame(trades)

    return equity, trades_df, ctx


def _annual_returns(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    first = eq.groupby(eq.index.year).first()
    last = eq.groupby(eq.index.year).last()
    ann = last / first - 1.0
    ann.index.name = "year"
    ann.name = "annual_return"
    return ann


def metrics(equity: pd.Series) -> dict:
    eq = equity.dropna()
    if eq.empty:
        return {}
    rets = eq.pct_change().dropna()
    start_val = float(eq.iloc[0])
    end_val = float(eq.iloc[-1])
    days = max(1, (eq.index[-1] - eq.index[0]).days)
    cagr = (end_val / start_val) ** (365.25 / days) - 1.0 if start_val > 0 else 0.0
    vol = float(rets.std()) * math.sqrt(252.0)
    sharpe = float(rets.mean()) * 252.0 / vol if vol > 0 else 0.0
    neg = rets[rets < 0.0]
    sort_den = float(np.sqrt((neg ** 2).mean()) * np.sqrt(252.0)) if not neg.empty else 0.0
    sortino = float(rets.mean()) * 252.0 / sort_den if sort_den > 0 else 0.0
    roll_max = eq.cummax()
    ddown = (eq / roll_max) - 1.0
    max_dd = float(ddown.min())
    exposure = float((rets != 0).mean())
    return {
        "start_value": start_val,
        "end_value": end_val,
        "cagr": cagr,
        "vol_annual": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "exposure_fraction": exposure,
    }


def to_all_in_one(prefix: str, equity: pd.DataFrame, trades_df: pd.DataFrame, ctx: pd.DataFrame, buy_hold: pd.DataFrame, m_strat: dict, m_bh: dict):
    metrics_rows = []
    for k in sorted(m_strat.keys()):
        metrics_rows.append({"section":"metrics", "metric": k, "strategy": m_strat.get(k), "buy_hold": m_bh.get(k)})
    metrics_df = pd.DataFrame(metrics_rows)

    ann_strat = _annual_returns(equity["equity"])
    ann_bh = _annual_returns(buy_hold["buy_hold_equity"])
    ann_df = pd.concat([ann_strat, ann_bh], axis=1)
    ann_df.columns = ["strategy_annual_return", "buy_hold_annual_return"]
    ann_df = ann_df.reset_index().rename(columns={"index":"year"})
    ann_df.insert(0, "section", "annual_returns")

    trades_out = trades_df.copy()
    if not trades_out.empty:
        trades_out.insert(0, "section", "trades")

    eq_out = equity[["equity", "position", "delta_pos", "trades_count"]].copy()
    eq_out = eq_out.join(buy_hold, how="left")
    eq_out = eq_out.reset_index().rename(columns={"index":"Date"})
    eq_out.insert(0, "section", "equity")

    all_in_one = pd.concat([metrics_df, ann_df, trades_out, eq_out], axis=0, ignore_index=True)
    all_in_one.to_csv(f"{prefix}_all_in_one.csv", index=False)
    return all_in_one


def buy_and_hold(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    entry_px = float(pd.to_numeric(df["Open"].iloc[0], errors="coerce"))
    if entry_px == 0.0 or np.isnan(entry_px):
        entry_px = float(pd.to_numeric(df["Close"].iloc[0], errors="coerce"))
    ret_close = df["Close"].pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + ret_close).cumprod()
    first_adj = float(df["Close"].iloc[0]) / entry_px if entry_px > 0 else 1.0
    equity.iloc[0] = initial_capital * first_adj
    return equity.to_frame(name="buy_hold_equity")


def maybe_plot(df: pd.DataFrame, ctx: pd.DataFrame, equity: pd.DataFrame, title: str):
    if plt is None:
        print("matplotlib not available, skipping plot")
        return
    plt.figure(figsize=(11, 6))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(ctx.index, ctx["sMA"], label="MA20")
    plt.plot(ctx.index, ctx["mMA"], label="MA50")
    plt.plot(ctx.index, ctx["lMA"], label="MA100")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(equity.index, equity["position"], label="Position fraction", alpha=0.5)
    ax.legend(loc="upper left")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Tiered triple MA backtest, one file output")
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--short_len", type=int, default=None)
    ap.add_argument("--med_len", type=int, default=None)
    ap.add_argument("--long_len", type=int, default=None)
    ap.add_argument("--entry_confirm_bars", type=int, default=None)
    ap.add_argument("--exit_confirm_bars", type=int, default=None)
    ap.add_argument("--allow_equal_on_price", action="store_true")
    ap.add_argument("--timeframe", type=str, default=None)
    ap.add_argument("--initial_capital", type=float, default=None)
    ap.add_argument("--slippage_pct", type=float, default=None)
    ap.add_argument("--fee_per_trade", type=float, default=None)
    ap.add_argument("--save_all", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--print_trades", action="store_true", help="Print position change log to terminal")

    args = ap.parse_args(argv)

    p = Params()
    if args.ticker is not None:
        p.ticker = args.ticker
    if args.start is not None:
        p.start_date = args.start
    if args.end is not None:
        p.end_date = args.end
    if args.use_ema:
        p.use_ema = True
    if args.short_len is not None:
        p.short_len = args.short_len
    if args.med_len is not None:
        p.med_len = args.med_len
    if args.long_len is not None:
        p.long_len = args.long_len
    if args.entry_confirm_bars is not None:
        p.entry_confirm_bars = args.entry_confirm_bars
    if args.exit_confirm_bars is not None:
        p.exit_confirm_bars = args.exit_confirm_bars
    if args.allow_equal_on_price:
        p.allow_equal_on_price = True
    if args.timeframe is not None:
        p.timeframe = args.timeframe
    if args.initial_capital is not None:
        p.initial_capital = args.initial_capital
    if args.slippage_pct is not None:
        p.slippage_pct = args.slippage_pct
    if args.fee_per_trade is not None:
        p.fee_per_trade = args.fee_per_trade

    df = load_data(p.ticker, p.start_date, p.end_date, args.csv)
    df = _resample_ohlcv(df, p.timeframe)

    equity, trades_df, ctx = backtest(df, p)
    buy_hold = buy_and_hold(df, p.initial_capital)

    m_strat = metrics(equity["equity"])
    m_bh = metrics(buy_hold["buy_hold_equity"])

    tag = f"{p.ticker or 'data'}_{'EMA' if p.use_ema else 'SMA'}_{p.short_len}-{p.med_len}-{p.long_len}_{p.timeframe}".replace(" ", "")
    prefix = f"tiered_{tag}"

    to_all_in_one(prefix, equity, trades_df, ctx, buy_hold, m_strat, m_bh)

    if args.save_all:
        equity.to_csv(f"{prefix}_equity.csv")
        trades_df.to_csv(f"{prefix}_trades.csv", index=False)
        ctx.to_csv(f"{prefix}_context.csv")
        buy_hold.to_csv(f"{prefix}_buyhold.csv")

    # console metrics only, unless --print_trades is passed
    def _print_metrics(title: str, m: dict):
        print("\n" + title)
        print("=" * len(title))
        for k, v in m.items():
            if isinstance(v, float):
                if "drawdown" in k or "fraction" in k:
                    print(f"{k:>20}: {v:.2%}")
                elif "cagr" in k or "sharpe" in k or "sortino" in k or "vol" in k:
                    print(f"{k:>20}: {v:.4f}")
                else:
                    print(f"{k:>20}: {v:.4f}")
            else:
                print(f"{k:>20}: {v}")

    _print_metrics("Strategy metrics", m_strat)
    _print_metrics("Buy and hold metrics", m_bh)

    if args.print_trades:
        if trades_df.empty:
            print("\nNo position changes on this sample")
        else:
            print("\nPosition change log")
            print("===================")
            print(trades_df.to_string(index=False))

    if args.plot:
        maybe_plot(df, ctx, equity, f"Tiered 20 50 100, {p.ticker}")


if __name__ == "__main__":
    main()
