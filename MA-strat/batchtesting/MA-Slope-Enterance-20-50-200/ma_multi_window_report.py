
#!/usr/bin/env python3
"""
MA multi-window portfolio report
Window-isolated backtests, start flat in each window, force close on last bar.
Compares strategies S1, S2, S3 across a ticker universe.
Includes equal slices, full period, and rolling three year windows stepping yearly.
Market regime tagging uses SPY inside each window.
Everything uses Close for apples to apples signals and buy and hold.
Cash earns zero when out of market.
No trade ledger is written, only trade count summaries.

Usage
  pip install pandas numpy yfinance xlsxwriter pyyaml
  python ma_multi_window_report.py --config batch_backtest.yml
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import yaml  # PyYAML
except Exception:
    yaml = None


# ========== Core helpers ==========

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    t = (ma_type or "sma").strip().lower()
    if t == "sma":
        return series.rolling(window=length, min_periods=length).mean()
    if t == "ema":
        return _ema(series, length)
    if t == "tema":
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        e3 = _ema(e2, length)
        return 3.0 * e1 - 3.0 * e2 + e3
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


def load_data(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Install it or modify the script to point to local CSVs.")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned from yfinance for {ticker}")
    df = _normalize_ohlcv(raw, ticker)
    df.index = pd.to_datetime(df.index)
    return df


# ========== Strategy and backtest ==========

@dataclass
class StratParams:
    name: str
    ma_type: str = "sma"
    ma_len: int = 200
    ma_len_price: Optional[int] = 200
    ma_len_slope: Optional[int] = 20    # S1 default is 20
    entry_rule: str = "slope_up"        # slope_up or price_above
    exit_rule: str = "price_below"      # slope_down or price_below
    entry_confirm_bars: int = 1
    exit_confirm_bars: int = 1


@dataclass
class RunParams:
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    slippage_pct: float = 0.0
    fee_per_trade: float = 0.0
    min_data_fraction: float = 0.8      # require this fraction of trading days to be present per window per ticker


def build_rules(price: pd.Series, p: StratParams):
    len_price = p.ma_len_price if p.ma_len_price is not None else p.ma_len
    len_slope = p.ma_len_slope if p.ma_len_slope is not None else p.ma_len

    ma_price = _ma(price, len_price, p.ma_type)
    ma_slope = _ma(price, len_slope, p.ma_type)

    above = price > ma_price
    below = price < ma_price
    slope_up = ma_slope > ma_slope.shift(1)
    slope_down = ma_slope < ma_slope.shift(1)

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

    ctx = pd.DataFrame(index=price.index)
    ctx["MA_price"] = ma_price
    ctx["MA_slope"] = ma_slope
    ctx["price_above_MAprice"] = above
    ctx["price_below_MAprice"] = below
    ctx["slope_up_MAslope"] = slope_up
    ctx["slope_down_MAslope"] = slope_down
    ctx["entry_ready"] = entry_ready
    ctx["exit_ready"] = exit_ready
    return entry_ready, exit_ready, ctx


def backtest_one(df: pd.DataFrame, p: StratParams, run: RunParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return equity DataFrame, trades DataFrame, and context DataFrame for the given df.
    Starts flat, trades at next open, forces close on last bar.
    """
    price = pd.to_numeric(df["Close"], errors="coerce")
    entry_ready, exit_ready, ctx = build_rules(price, p)

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

    # force close on last bar if still in position
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

    equity = pd.DataFrame(index=df.index)
    equity["position"] = position
    equity["ret_close"] = ret_close
    equity["strat_ret"] = strat_ret
    equity["equity"] = run.initial_capital * (1.0 + strat_ret).cumprod()

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["pnl"] = trades_df["ret_pct"] * run.initial_capital

    return equity, trades_df, ctx


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


def buy_and_hold_close(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Buy at first Close and hold, apples to apples on Close only."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    if close.empty or close.iloc[0] <= 0:
        return pd.Series(dtype=float)
    eq = initial_capital * (close / close.iloc[0])
    eq.name = "buy_hold_equity"
    return eq


# ========== Windows ==========

def build_equal_slices(index: pd.DatetimeIndex, n_slices: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    if n_slices <= 0:
        return []
    dates = index.sort_values()
    start = dates[0]
    end = dates[-1]
    total_days = (end - start).days
    bounds = [start + pd.Timedelta(days=int(round(i * total_days / n_slices))) for i in range(n_slices)]
    bounds.append(end)
    windows = []
    for i in range(n_slices):
        s_raw = bounds[i]
        e_raw = bounds[i + 1]
        s = dates[dates >= s_raw][0]
        e = dates[dates <= e_raw][-1]
        windows.append((s, e, f"W{str(i+1).zfill(2)}"))
    return windows


def build_full_window(index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    dates = index.sort_values()
    return [(dates[0], dates[-1], "FULL")]


def build_rolling_3y(index: pd.DatetimeIndex, step_years: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    dates = index.sort_values()
    windows = []
    first_y = dates[0].year
    last_y = dates[-1].year
    for y in range(first_y, last_y - 2, step_years):
        s_raw = pd.Timestamp(year=y, month=1, day=1)
        e_raw = s_raw + pd.DateOffset(years=3) - pd.Timedelta(days=1)
        s = dates[dates >= s_raw]
        e = dates[dates <= e_raw]
        if len(s) == 0 or len(e) == 0:
            continue
        s = s[0]
        e = e[-1]
        win = dates[(dates >= s) & (dates <= e)]
        if len(win) >= 252:
            windows.append((s, e, f"R{y}"))
    return windows


# ========== Portfolio ==========

def portfolio_equality_active(rets_by_ticker: Dict[str, pd.Series], pos_by_ticker: Dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.Series:
    """Equal weight across active names, weights sum to one when any active exists, zero when none active."""
    aligned_rets = {t: rets.reindex(index).fillna(0.0) for t, rets in rets_by_ticker.items()}
    aligned_pos  = {t: pos.reindex(index).fillna(0.0) for t, pos in pos_by_ticker.items()}
    active_mask = {t: (aligned_pos[t].shift(1).fillna(0.0) > 0.0).astype(float) for t in aligned_pos.keys()}
    actives_df = pd.DataFrame(active_mask)
    k = actives_df.sum(axis=1)
    weights = actives_df.div(k.replace(0.0, np.nan), axis=0).fillna(0.0)
    rets_df = pd.DataFrame(aligned_rets).fillna(0.0)
    port_ret = (weights * rets_df).sum(axis=1)
    port_ret.name = "portfolio_ret"
    return port_ret


def equal_weight_buy_hold(rets_by_ticker: Dict[str, pd.Series], index: pd.DatetimeIndex, initial_capital: float) -> pd.Series:
    """Equal weight across all tickers, static weights, simple average of daily returns."""
    aligned = [s.reindex(index).fillna(0.0) for s in rets_by_ticker.values()]
    if not aligned:
        return pd.Series(dtype=float)
    rets_df = pd.concat(aligned, axis=1)
    ew_ret = rets_df.mean(axis=1).fillna(0.0)
    eq = initial_capital * (1.0 + ew_ret).cumprod()
    eq.name = "ew_buy_hold_equity"
    return eq


# ========== Regime tagging ==========

def max_drawdown_from_equity(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    roll_max = eq.cummax()
    ddown = (eq / roll_max) - 1.0
    return float(ddown.min())


def tag_regime_spy(spy_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, initial_capital: float) -> Tuple[str, float, float]:
    w = spy_df[(spy_df.index >= start) & (spy_df.index <= end)]
    if w.empty:
        return ("Unknown", 0.0, 0.0)
    eq = buy_and_hold_close(w, initial_capital)
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    mdd = max_drawdown_from_equity(eq)
    if ret > 0.0 and mdd > -0.15:
        return ("Bull", ret, mdd)
    if ret < 0.0 and mdd <= -0.20:
        return ("Bear", ret, mdd)
    return ("Sideways", ret, mdd)


# ========== Workbook writer ==========

def _auto_width_and_formats(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame):
    ws = writer.sheets[sheet_name]
    wb = writer.book
    pct = wb.add_format({'num_format': '0.00%'})
    ratio = wb.add_format({'num_format': '0.0000'})
    datefmt = wb.add_format({'num_format': 'yyyy-mm-dd'})
    ws.freeze_panes(1, 1)
    for i, col in enumerate(df.columns):
        series = df[col]
        max_len = max([len(str(col))] + [len(str(v)) for v in series.head(2000)])
        max_len = min(max_len + 2, 48)
        ws.set_column(i, i, max_len)
        name = str(col).lower()
        if "date" in name:
            ws.set_column(i, i, max_len, datefmt)
        if "return" in name or "drawdown" in name or "cagr" in name or "fraction" in name or "excess" in name:
            ws.set_column(i, i, max_len, pct)
        if "sharpe" in name or "sortino" in name or "vol_annual" in name:
            ws.set_column(i, i, max_len, ratio)


def build_workbook(
    out_path: str,
    summary_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    per_ticker_df: pd.DataFrame,
    portfolio_df: pd.DataFrame
):
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _auto_width_and_formats(writer, "Summary", summary_df)

        windows_df.to_excel(writer, sheet_name="Windows", index=False)
        _auto_width_and_formats(writer, "Windows", windows_df)

        per_ticker_df.to_excel(writer, sheet_name="Per_Ticker", index=False)
        _auto_width_and_formats(writer, "Per_Ticker", per_ticker_df)

        portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)
        _auto_width_and_formats(writer, "Portfolio", portfolio_df)


# ========== Orchestrator ==========

def compute_all(
    universe_label: str,
    tickers: List[str],
    run_params: RunParams,
    strategies: List[StratParams],
    include_rolling_3y: bool,
    equal_slice_count: int,
    rolling_step_years: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # fetch once
    data_by_ticker: Dict[str, pd.DataFrame] = {}
    for t in sorted(set(tickers + ["SPY"])):
        data_by_ticker[t] = load_data(t, run_params.start_date, run_params.end_date)

    spy = data_by_ticker["SPY"]
    shared_index = spy.index

    # windows
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    windows.extend(build_full_window(shared_index))
    if equal_slice_count and equal_slice_count > 0:
        windows.extend(build_equal_slices(shared_index, equal_slice_count))
    if include_rolling_3y:
        windows.extend(build_rolling_3y(shared_index, step_years=rolling_step_years))

    windows_df_rows = []
    for s, e, wid in windows:
        label, spy_ret, spy_mdd = tag_regime_spy(spy, s, e, run_params.initial_capital)
        windows_df_rows.append({
            "window_id": wid,
            "start": s.date().isoformat(),
            "end": e.date().isoformat(),
            "days": int((e - s).days) + 1,
            "spy_return": spy_ret,
            "spy_max_drawdown": spy_mdd,
            "regime": label,
        })
    windows_df = pd.DataFrame(windows_df_rows)

    # SPY buy and hold metrics per window for portfolio level benchmarks
    spy_bh_map: Dict[str, dict] = {}
    for s, e, wid in windows:
        spy_w = spy[(spy.index >= s) & (spy.index <= e)]
        spy_eq = buy_and_hold_close(spy_w, run_params.initial_capital)
        spy_bh_map[wid] = metrics(spy_eq)

    per_ticker_rows = []
    portfolio_rows = []

    # per window, window-isolated runs
    for s, e, wid in windows:
        spy_len = len(spy[(spy.index >= s) & (spy.index <= e)])
        if spy_len == 0:
            continue

        # per ticker metrics and trade summaries
        for strat in strategies:
            for t in tickers:
                df_full = data_by_ticker[t]
                df_w = df_full[(df_full.index >= s) & (df_full.index <= e)]
                if len(df_w) < 10:
                    continue
                if len(df_w) / spy_len < run_params.min_data_fraction:
                    continue

                eq_w, trades_w, ctx_w = backtest_one(df_w, strat, run_params)

                m = metrics(eq_w["equity"])
                bh_eq = buy_and_hold_close(df_w, run_params.initial_capital)
                m_bh = metrics(bh_eq)

                pos = eq_w["position"].fillna(0.0)
                entries = int(((pos > 0.0) & (pos.shift(1).fillna(0.0) == 0.0)).sum())
                exits = int(((pos == 0.0) & (pos.shift(1).fillna(0.0) > 0.0)).sum())
                hold_bars = int((pos > 0.0).sum())

                per_ticker_rows.append({
                    "window_id": wid,
                    "start": s.date().isoformat(),
                    "end": e.date().isoformat(),
                    "ticker": t,
                    "strategy": strat.name,
                    "entries": entries,
                    "exits": exits,
                    "holding_bars": hold_bars,
                    **{f"strat_{k}": v for k, v in m.items()},
                    **{f"bh_{k}": v for k, v in m_bh.items()},
                    "beat_bh_sharpe": (m.get("sharpe", 0.0) > m_bh.get("sharpe", 0.0)),
                    "beat_bh_cagr": (m.get("cagr", 0.0) > m_bh.get("cagr", 0.0)),
                    "beat_bh_mdd": (m.get("max_drawdown", -1.0) > m_bh.get("max_drawdown", -1.0)),
                    "mdd_diff_vs_bh": float(m.get("max_drawdown", np.nan)) - float(m_bh.get("max_drawdown", np.nan)),
                })

        # portfolio per strategy for this window
        index_w = spy[(spy.index >= s) & (spy.index <= e)].index
        for strat in strategies:
            rets_by_ticker = {}
            pos_by_ticker = {}
            bh_rets_by_ticker = {}

            for t in tickers:
                df_full = data_by_ticker[t]
                df_w = df_full[(df_full.index >= s) & (df_full.index <= e)]
                if len(df_w) < 10:
                    continue
                if len(df_w) / spy_len < run_params.min_data_fraction:
                    continue

                eq_w, trades_w, ctx_w = backtest_one(df_w, strat, run_params)
                rets_by_ticker[t] = eq_w["ret_close"].reindex(index_w).fillna(0.0)
                pos_by_ticker[t] = eq_w["position"].reindex(index_w).fillna(0.0)

                close_w = pd.to_numeric(df_w["Close"], errors="coerce").reindex(index_w)
                bh_rets_by_ticker[t] = close_w.pct_change().fillna(0.0)

            if not rets_by_ticker:
                continue

            port_ret = portfolio_equality_active(rets_by_ticker, pos_by_ticker, index_w)
            port_eq = run_params.initial_capital * (1.0 + port_ret).cumprod()
            port_m = metrics(port_eq)

            ew_bh_eq = equal_weight_buy_hold(bh_rets_by_ticker, index_w, run_params.initial_capital)
            ew_bh_m = metrics(ew_bh_eq)

            portfolio_rows.append({
                "window_id": wid,
                "start": s.date().isoformat(),
                "end": e.date().isoformat(),
                "strategy": strat.name,
                **{f"portfolio_{k}": v for k, v in port_m.items()},
                **{f"ew_bh_{k}": v for k, v in ew_bh_m.items()},
                "beat_bh_sharpe": (port_m.get("sharpe", 0.0) > ew_bh_m.get("sharpe", 0.0)),
                "beat_bh_cagr": (port_m.get("cagr", 0.0) > ew_bh_m.get("cagr", 0.0)),
                "beat_bh_mdd": (port_m.get("max_drawdown", -1.0) > ew_bh_m.get("max_drawdown", -1.0)),
                "mdd_diff_vs_bh": float(port_m.get("max_drawdown", np.nan)) - float(ew_bh_m.get("max_drawdown", np.nan)),
            })

    per_ticker_df = pd.DataFrame(per_ticker_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)

    # winners by Sharpe, tie break smaller max drawdown, then higher CAGR
    winners = []
    if not portfolio_df.empty:
        g_all = portfolio_df.dropna(subset=["portfolio_sharpe"]).copy()
        if not g_all.empty:
            g_all["rank_key"] = list(zip(-g_all["portfolio_sharpe"], -g_all["portfolio_max_drawdown"], -g_all["portfolio_cagr"]))
            for wid, group in g_all.groupby("window_id"):
                gg = group.sort_values("rank_key")
                best = gg.iloc[0]
                winners.append({"window_id": wid, "winner_strategy": best["strategy"]})
    winners_df = pd.DataFrame(winners)

    # Summary build
    windows_df = windows_df  # already built
    windows_map = windows_df[["window_id", "regime"]]
    port_join = portfolio_df.merge(winners_df, on="window_id", how="left").merge(windows_map, on="window_id", how="left")

    summary_rows = []
    for strat in [s.name for s in strategies]:
        g = port_join[port_join["strategy"] == strat].copy()
        if g.empty:
            continue

        # add SPY BH metrics into the joined frame
        g["spy_sharpe"] = g["window_id"].map(lambda w: spy_bh_map.get(w, {}).get("sharpe", np.nan))
        g["spy_cagr"]   = g["window_id"].map(lambda w: spy_bh_map.get(w, {}).get("cagr", np.nan))
        g["spy_mdd"]    = g["window_id"].map(lambda w: spy_bh_map.get(w, {}).get("max_drawdown", np.nan))

        # equal weight BH comparisons
        beat_ew_bh_sharpe_pct = float(g["beat_bh_sharpe"].mean())
        beat_ew_bh_cagr_pct   = float(g["beat_bh_cagr"].mean())
        beat_ew_bh_mdd_pct    = float(g["beat_bh_mdd"].mean())
        avg_excess_sharpe_vs_ew_bh = float((g["portfolio_sharpe"] - g["ew_bh_sharpe"]).mean())
        avg_excess_cagr_vs_ew_bh   = float((g["portfolio_cagr"] - g["ew_bh_cagr"]).mean())
        avg_mdd_diff_vs_ew_bh      = float(g["mdd_diff_vs_bh"].mean())
        med_mdd_diff_vs_ew_bh      = float(g["mdd_diff_vs_bh"].median())

        # SPY BH comparisons
        beat_spy_sharpe_pct = float((g["portfolio_sharpe"] > g["spy_sharpe"]).mean())
        beat_spy_cagr_pct   = float((g["portfolio_cagr"] > g["spy_cagr"]).mean())
        beat_spy_mdd_pct    = float((g["portfolio_max_drawdown"] > g["spy_mdd"]).mean())
        avg_excess_sharpe_vs_spy = float((g["portfolio_sharpe"] - g["spy_sharpe"]).mean())
        avg_excess_cagr_vs_spy   = float((g["portfolio_cagr"] - g["spy_cagr"]).mean())
        avg_mdd_diff_vs_spy      = float((g["portfolio_max_drawdown"] - g["spy_mdd"]).mean())
        med_mdd_diff_vs_spy      = float((g["portfolio_max_drawdown"] - g["spy_mdd"]).median())

        summary_rows.append({
            "strategy": strat,
            "windows_evaluated": int(len(g)),
            "wins_total": int((g["winner_strategy"] == strat).sum()),
            "wins_bull": int(((g["winner_strategy"] == strat) & (g["regime"] == "Bull")).sum()),
            "wins_bear": int(((g["winner_strategy"] == strat) & (g["regime"] == "Bear")).sum()),
            "wins_sideways": int(((g["winner_strategy"] == strat) & (g["regime"] == "Sideways")).sum()),
            "avg_sharpe": float(g["portfolio_sharpe"].mean()),
            "avg_sortino": float(g["portfolio_sortino"].mean()),
            "median_max_drawdown": float(g["portfolio_max_drawdown"].median()),
            "avg_cagr": float(g["portfolio_cagr"].mean()),
            "beat_ew_bh_sharpe_pct": beat_ew_bh_sharpe_pct,
            "beat_ew_bh_cagr_pct": beat_ew_bh_cagr_pct,
            "beat_ew_bh_mdd_pct": beat_ew_bh_mdd_pct,
            "avg_excess_sharpe_vs_ew_bh": avg_excess_sharpe_vs_ew_bh,
            "avg_excess_cagr_vs_ew_bh": avg_excess_cagr_vs_ew_bh,
            "avg_mdd_diff_vs_ew_bh": avg_mdd_diff_vs_ew_bh,
            "median_mdd_diff_vs_ew_bh": med_mdd_diff_vs_ew_bh,
            "beat_spy_sharpe_pct": beat_spy_sharpe_pct,
            "beat_spy_cagr_pct": beat_spy_cagr_pct,
            "beat_spy_mdd_pct": beat_spy_mdd_pct,
            "avg_excess_sharpe_vs_spy": avg_excess_sharpe_vs_spy,
            "avg_excess_cagr_vs_spy": avg_excess_cagr_vs_spy,
            "avg_mdd_diff_vs_spy": avg_mdd_diff_vs_spy,
            "median_mdd_diff_vs_spy": med_mdd_diff_vs_spy,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("wins_total", ascending=False)

    return summary_df, windows_df, per_ticker_df, portfolio_df


def main(argv=None):
    ap = argparse.ArgumentParser(description="MA multi-window portfolio report")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config, example batch_backtest.yml")
    args = ap.parse_args(argv)

    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    universe_label = str(cfg.get("universe_label", "UNIVERSE"))
    tickers = list(cfg["tickers"])

    overall = cfg.get("overall", {})
    start = overall.get("start", "2001-01-01")
    end = overall.get("end", "2025-09-01")

    common = cfg.get("common", {})
    run_params = RunParams(
        start_date=start,
        end_date=end,
        initial_capital=float(common.get("initial_capital", 100000.0)),
        slippage_pct=float(common.get("slippage_pct", 0.0)),
        fee_per_trade=float(common.get("fee_per_trade", 0.0)),
        min_data_fraction=float(common.get("min_data_fraction", 0.8)),
    )

    strategies_cfg = cfg.get("strategies", [])
    strategies: List[StratParams] = []
    for s in strategies_cfg:
        strategies.append(StratParams(
            name=str(s["name"]),
            ma_type=str(s.get("ma_type", "sma")),
            ma_len=int(s.get("ma_len", 200)),
            ma_len_price=int(s.get("ma_len_price", 200)) if s.get("ma_len_price", None) is not None else None,
            ma_len_slope=int(s.get("ma_len_slope", 20)) if s.get("ma_len_slope", None) is not None else None,
            entry_rule=str(s.get("entry_rule", "slope_up")),
            exit_rule=str(s.get("exit_rule", "price_below")),
            entry_confirm_bars=int(s.get("entry_confirm_bars", 1)),
            exit_confirm_bars=int(s.get("exit_confirm_bars", 1)),
        ))

    windows_cfg = cfg.get("windows", {})
    equal_slice_count = int(windows_cfg.get("equal_slices", 10))
    include_rolling_3y = bool(windows_cfg.get("include_rolling_3y", True))
    rolling_step_years = int(windows_cfg.get("rolling_step_years", 1))

    summary_df, windows_df, per_ticker_df, portfolio_df = compute_all(
        universe_label=universe_label,
        tickers=tickers,
        run_params=run_params,
        strategies=strategies,
        include_rolling_3y=include_rolling_3y,
        equal_slice_count=equal_slice_count,
        rolling_step_years=rolling_step_years
    )

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.get("output", {}).get("workbook_path", f"ma_report_{universe_label}_{ts}.xlsx")
    build_workbook(out_path, summary_df, windows_df, per_ticker_df, portfolio_df)
    print(f"Wrote workbook to {out_path}")


if __name__ == "__main__":
    main()
