
#!/usr/bin/env python3
"""
Single MA rule backtest
Choose one moving average (SMA, EMA, TEMA) at a chosen length
Enter when the entry rule is true (slope up or price above)
Exit when the exit rule is true (slope down or price below)

Trade logic uses next bar open, returns are applied with position held during the bar
Outputs one all_in_one CSV that combines metrics, annual returns, trades, equity
Optional plot with --plot
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
    ticker: Optional[str] = "msft"
    ma_type: str = "sma"             # sma, ema, tema
    ma_len: int = 200
    use_ema_for_tema: bool = True     # TEMA always uses EMA internally
    entry_rule: str = "slope_up"      # slope_up or price_above
    exit_rule: str = "price_below"    # slope_down or price_below
    entry_confirm_bars: int =1
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


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    t = (ma_type or "sma").strip().lower()
    if t == "sma":
        return series.rolling(window=length, min_periods=length).mean()
    if t == "ema":
        return _ema(series, length)
    if t == "tema":
        # TEMA per Kaufman: TEMA = 3*EMA1 - 3*EMA2 + EMA3
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        e3 = _ema(e2, length)
        tema = 3.0 * e1 - 3.0 * e2 + e3
        tema.name = "TEMA"
        return tema
    raise ValueError("ma_type must be sma, ema, or tema")


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


def build_rules(df: pd.DataFrame, p: Params):
    price = df["Close"]
    ma = _ma(price, p.ma_len, p.ma_type)

    if p.allow_equal_on_price:
        above = price >= ma
        below = price <= ma
    else:
        above = price > ma
        below = price < ma

    slope_up = ma > ma.shift(1)
    slope_down = ma < ma.shift(1)

    if p.entry_rule == "slope_up":
        entry_raw = slope_up
    elif p.entry_rule == "price_above":
        entry_raw = above
    else:
        raise ValueError("entry_rule must be slope_up or price_above")

    if p.exit_rule == "slope_down":
        exit_raw = slope_down
    elif p.exit_rule == "price_below":
        exit_raw = below
    else:
        raise ValueError("exit_rule must be slope_down or price_below")

    entry_ready = _confirm(entry_raw, p.entry_confirm_bars)
    exit_ready = _confirm(exit_raw, p.exit_confirm_bars)

    ctx = pd.DataFrame(index=df.index)
    ctx["MA"] = ma
    ctx["price_above_MA"] = above
    ctx["price_below_MA"] = below
    ctx["slope_up"] = slope_up
    ctx["slope_down"] = slope_down
    ctx["entry_ready"] = entry_ready
    ctx["exit_ready"] = exit_ready
    return entry_ready, exit_ready, ctx


def backtest(df: pd.DataFrame, p: Params):
    entry_ready, exit_ready, ctx = build_rules(df, p)

    # trade at next open
    entry_arr = entry_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)
    exit_arr  = exit_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)

    open_arr = pd.to_numeric(df["Open"], errors="coerce").to_numpy()
    close_arr = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
    idx = df.index.to_numpy()

    in_pos = False
    trades = []
    entry_idx = None

    for i in range(len(idx)):
        if (not in_pos) and entry_arr[i]:
            entry_idx = i
            in_pos = True
        elif in_pos and exit_arr[i]:
            e_i = entry_idx
            x_i = i
            entry_dt = idx[e_i]
            exit_dt = idx[x_i]
            entry_px = float(open_arr[e_i])
            exit_px = float(open_arr[x_i])
            bars = int(x_i - e_i)
            ret_pct = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
            trades.append({
                "entry_dt": entry_dt, "entry_px": entry_px,
                "exit_dt": exit_dt, "exit_px": exit_px,
                "bars": bars, "ret_pct": ret_pct
            })
            in_pos = False
            entry_idx = None

    if in_pos and entry_idx is not None:
        e_i = entry_idx
        x_i = len(idx) - 1
        entry_dt = idx[e_i]
        exit_dt = idx[x_i]
        entry_px = float(open_arr[e_i])
        exit_px = float(close_arr[x_i])
        bars = int(x_i - e_i)
        ret_pct = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
        trades.append({
            "entry_dt": entry_dt, "entry_px": entry_px,
            "exit_dt": exit_dt, "exit_px": exit_px,
            "bars": bars, "ret_pct": ret_pct
        })

    ret_close = pd.Series(close_arr, index=df.index).pct_change().fillna(0.0)
    position = pd.Series(0.0, index=df.index)
    for t in trades:
        start = df.index.get_loc(t["entry_dt"])
        end = df.index.get_loc(t["exit_dt"])
        if end > start:
            position.iloc[start:end] = 1.0

    strat_ret = position.shift(1).fillna(0.0) * ret_close

    # slippage and fees on entries and exits
    delta_pos = position - position.shift(1).fillna(0.0)
    slip = (1.0 - p.slippage_pct * delta_pos.abs()).clip(lower=0.0)
    slip_cum = slip.cumprod()

    equity_gross = p.initial_capital * (1.0 + strat_ret).cumprod()

    fee_flow = pd.Series(0.0, index=df.index)
    if p.fee_per_trade and p.fee_per_trade != 0.0:
        changes = delta_pos[delta_pos != 0.0]
        for ts, _ in changes.items():
            fee_flow.loc[ts] -= p.fee_per_trade
    fee_cum = fee_flow.cumsum()

    equity_net = equity_gross * slip_cum + fee_cum

    equity = pd.DataFrame(index=df.index)
    equity["position"] = position
    equity["ret_close"] = ret_close
    equity["strat_ret"] = strat_ret
    equity["equity"] = equity_net

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["pnl"] = trades_df["ret_pct"] * p.initial_capital

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

    eq_out = equity[["equity", "position"]].copy()
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


def maybe_plot(df: pd.DataFrame, ctx: pd.DataFrame, equity: pd.DataFrame, p: Params):
    if plt is None:
        print("matplotlib not available, skipping plot")
        return
    plt.figure(figsize=(11, 6))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(ctx.index, ctx["MA"], label=f"{p.ma_type.upper()} {p.ma_len}")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(equity.index, equity["position"], label="Position", alpha=0.5)
    ax.legend(loc="upper left")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title(f"{p.ticker} single MA rule")
    plt.tight_layout()
    plt.show()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Single MA backtest, slope or price rules")
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--ma_type", type=str, default=None, choices=["sma","ema","tema"])
    ap.add_argument("--ma_len", type=int, default=None)
    ap.add_argument("--entry_rule", type=str, default=None, choices=["slope_up","price_above"])
    ap.add_argument("--exit_rule", type=str, default=None, choices=["slope_down","price_below"])
    ap.add_argument("--entry_confirm_bars", type=int, default=None)
    ap.add_argument("--exit_confirm_bars", type=int, default=None)
    ap.add_argument("--allow_equal_on_price", action="store_true")
    ap.add_argument("--timeframe", type=str, default=None)
    ap.add_argument("--initial_capital", type=float, default=None)
    ap.add_argument("--slippage_pct", type=float, default=None)
    ap.add_argument("--fee_per_trade", type=float, default=None)
    ap.add_argument("--save_all", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--print_trades", action="store_true", help="Print trade list")

    args = ap.parse_args(argv)

    p = Params()
    if args.ticker is not None:
        p.ticker = args.ticker
    if args.start is not None:
        p.start_date = args.start
    if args.end is not None:
        p.end_date = args.end
    if args.ma_type is not None:
        p.ma_type = args.ma_type
    if args.ma_len is not None:
        p.ma_len = args.ma_len
    if args.entry_rule is not None:
        p.entry_rule = args.entry_rule
    if args.exit_rule is not None:
        p.exit_rule = args.exit_rule
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

    tag = f"{p.ticker or 'data'}_{p.ma_type.upper()}_{p.ma_len}_{p.entry_rule}2{p.exit_rule}_{p.timeframe}".replace(" ", "")
    prefix = f"singleMA_{tag}"

    to_all_in_one(prefix, equity, trades_df, ctx, buy_hold, m_strat, m_bh)

    if args.save_all:
        equity.to_csv(f"{prefix}_equity.csv")
        trades_df.to_csv(f"{prefix}_trades.csv", index=False)
        ctx.to_csv(f"{prefix}_context.csv")
        buy_hold.to_csv(f"{prefix}_buyhold.csv")

    # metrics only in console, unless print_trades is asked
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
            print("\nNo completed trades on this sample")
        else:
            print("\nTrades")
            print("======")
            print(trades_df.to_string(index=False))

    if args.plot:
        maybe_plot(df, ctx, equity, p)


if __name__ == "__main__":
    main()
