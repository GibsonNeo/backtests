
#!/usr/bin/env python3
"""
S3 baseline plus momentum overlay portfolio variants, v2
Changes, remove deprecated fillna(method) and pct_change default fill, switch resample rules to ME and QE,
speed up momentum overlay by vectorizing selection and weight assignment.
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
    import yaml
except Exception:
    yaml = None


# ---------- helpers ----------

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
    if bars is None or bars <= 1:
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
        raise RuntimeError("yfinance is not available. Install it or modify the script to pull from CSVs.")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned from yfinance for {ticker}")
    df = _normalize_ohlcv(raw, ticker)
    df.index = pd.to_datetime(df.index)
    return df


# ---------- S3 daily signals ----------

@dataclass
class S3Params:
    ma_type: str = "sma"
    ma_len: int = 200
    entry_confirm_bars: int = 1
    exit_confirm_bars: int = 1


def build_s3_signals(price: pd.Series, p: S3Params) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    ma200 = _ma(price, p.ma_len, p.ma_type)
    slope_up = ma200 > ma200.shift(1)
    slope_down = ma200 < ma200.shift(1)
    above = price > ma200
    entry_raw = slope_up
    exit_raw = slope_down | (~above)
    entry_ready = _confirm(entry_raw, p.entry_confirm_bars)
    exit_ready = _confirm(exit_raw, p.exit_confirm_bars)
    ctx = pd.DataFrame(index=price.index)
    ctx["MA200"] = ma200
    ctx["slope_up"] = slope_up
    ctx["slope_down"] = slope_down
    ctx["price_above"] = above
    ctx["entry_ready"] = entry_ready
    ctx["exit_ready"] = exit_ready
    return entry_ready, exit_ready, ctx


def backtest_s3(df: pd.DataFrame, p: S3Params, initial_capital: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.to_numeric(df["Close"], errors="coerce")
    entry_ready, exit_ready, _ = build_s3_signals(price, p)

    entry_arr = entry_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)
    exit_arr  = exit_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
    idx = df.index.to_numpy()

    in_pos = False
    entry_idx = None
    trades = []

    for i in range(len(idx)):
        if (not in_pos) and entry_arr[i]:
            entry_idx = i
            in_pos = True
        elif in_pos and exit_arr[i]:
            e_i = entry_idx
            x_i = i
            entry_dt = idx[e_i]
            exit_dt = idx[x_i]
            entry_px = float(close[e_i])
            exit_px = float(close[x_i])
            bars = int(x_i - e_i)
            ret_pct = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
            trades.append({"entry_dt": entry_dt, "exit_dt": exit_dt, "bars": bars, "ret_pct": ret_pct})
            in_pos = False
            entry_idx = None

    if in_pos and entry_idx is not None:
        e_i = entry_idx
        x_i = len(idx) - 1
        entry_dt = idx[e_i]
        exit_dt = idx[x_i]
        entry_px = float(close[e_i])
        exit_px = float(close[x_i])
        bars = int(x_i - e_i)
        ret_pct = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
        trades.append({"entry_dt": entry_dt, "exit_dt": exit_dt, "bars": bars, "ret_pct": ret_pct})

    ret_close = pd.Series(close, index=df.index).pct_change(fill_method=None).fillna(0.0)
    position = pd.Series(0.0, index=df.index)
    for t in trades:
        s = df.index.get_loc(t["entry_dt"])
        e = df.index.get_loc(t["exit_dt"])
        if e > s:
            position.iloc[s:e] = 1.0
    strat_ret = position.shift(1).fillna(0.0) * ret_close
    equity = pd.DataFrame(index=df.index)
    equity["ret_close"] = ret_close
    equity["position"] = position
    equity["equity"] = initial_capital * (1.0 + strat_ret).cumprod()
    trades_df = pd.DataFrame(trades)
    return equity, trades_df


# ---------- momentum overlay portfolio ----------

def momentum_12m_1m(close: pd.Series) -> pd.Series:
    return close.shift(21) / close.shift(252) - 1.0


def rebalance_points(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "W":
        return index.to_series().resample("W-FRI").last().dropna().index
    if freq == "M":
        return index.to_series().resample("ME").last().dropna().index
    if freq == "Q":
        return index.to_series().resample("QE").last().dropna().index
    if freq == "6M":
        months = index.to_series().resample("ME").last().dropna().index
        return months[::6]
    raise ValueError("freq must be one of W, M, Q, 6M")


def portfolio_equal_weight_active(rets_by_ticker: Dict[str, pd.Series], pos_by_ticker: Dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.Series:
    aligned_rets = {t: r.reindex(index).fillna(0.0) for t, r in rets_by_ticker.items()}
    aligned_pos  = {t: p.reindex(index).fillna(0.0) for t, p in pos_by_ticker.items()}
    actives = pd.DataFrame({t: (aligned_pos[t].shift(1).fillna(0.0) > 0.0).astype(float) for t in aligned_pos})
    k = actives.sum(axis=1)
    weights = actives.div(k.replace(0.0, np.nan), axis=0).fillna(0.0)
    rets = pd.DataFrame(aligned_rets).fillna(0.0)
    port_ret = (weights * rets).sum(axis=1)
    port_ret.name = "portfolio_ret"
    return port_ret


def portfolio_momentum_overlay(
    close_by_ticker: Dict[str, pd.Series],
    rets_by_ticker: Dict[str, pd.Series],
    pos_by_ticker: Dict[str, pd.Series],
    index: pd.DatetimeIndex,
    reserved: List[str],
    top_k: int,
    freq: str
) -> pd.Series:
    # build momentum matrix aligned to index for all names
    momo_df = pd.DataFrame({t: momentum_12m_1m(close_by_ticker[t]) for t in close_by_ticker.keys()}).reindex(index).ffill()

    # rebalance dates
    rebal_dates = rebalance_points(index, freq)
    rebal_dates = rebal_dates[(rebal_dates >= index[0]) & (rebal_dates <= index[-1])]
    if len(rebal_dates) == 0 or rebal_dates[0] > index[0]:
        rebal_dates = pd.Index([index[0]]).append(rebal_dates)

    # pre create weights DataFrame, faster than per series dict
    weights_df = pd.DataFrame(0.0, index=index, columns=list(close_by_ticker.keys()))

    # assign per segment with slice based .loc
    for i, d in enumerate(rebal_dates):
        start = d
        end = rebal_dates[i+1] if i+1 < len(rebal_dates) else index[-1]
        seg = slice(start, end)

        pool = [t for t in close_by_ticker.keys() if t not in reserved]
        # choose top momentum at date d
        if d in momo_df.index:
            series_at_d = momo_df.loc[d, pool]
        else:
            # use most recent available before d
            series_at_d = momo_df.loc[:d, pool].iloc[-1]
        top = series_at_d.dropna().nlargest(top_k).index.tolist()

        # fixed sleeves for SPY and QQQ
        if "SPY" in weights_df.columns:
            weights_df.loc[seg, "SPY"] = 0.10
        if "QQQ" in weights_df.columns:
            weights_df.loc[seg, "QQQ"] = 0.10

        # remaining to top names
        if len(top) > 0:
            w = 0.80 / float(len(top))
            weights_df.loc[seg, top] = w

    # compute portfolio return with S3 position mask
    port_ret = pd.Series(0.0, index=index)
    for t in close_by_ticker.keys():
        r = rets_by_ticker[t].reindex(index).fillna(0.0)
        pos = pos_by_ticker[t].reindex(index).fillna(0.0).shift(1).fillna(0.0) > 0.0
        w = weights_df[t]
        port_ret = port_ret + r * w * pos.astype(float)
    port_ret.name = "portfolio_ret"
    return port_ret


# ---------- metrics ----------

def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if eq.empty:
        return {}
    rets = eq.pct_change(fill_method=None).dropna()
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
    exposure = float((rets != 0.0).mean())
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
    close = pd.to_numeric(df["Close"], errors="coerce")
    if close.empty or close.iloc[0] <= 0:
        return pd.Series(dtype=float)
    eq = initial_capital * (close / close.iloc[0])
    eq.name = "buy_hold_equity"
    return eq


# ---------- regimes ----------

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
        # else
    return ("Sideways", ret, mdd)


# ---------- workbook formatting ----------

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


def build_workbook(out_path: str,
                   summary_df: pd.DataFrame,
                   windows_df: pd.DataFrame,
                   per_ticker_df: pd.DataFrame,
                   portfolio_df: pd.DataFrame,
                   head_to_head_df: pd.DataFrame,
                   regime_winrates_df: pd.DataFrame,
                   summary_plus_df: pd.DataFrame):
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _auto_width_and_formats(writer, "Summary", summary_df)

        summary_plus_df.to_excel(writer, sheet_name="SummaryPlus", index=False)
        _auto_width_and_formats(writer, "SummaryPlus", summary_plus_df)

        windows_df.to_excel(writer, sheet_name="Windows", index=False)
        _auto_width_and_formats(writer, "Windows", windows_df)

        per_ticker_df.to_excel(writer, sheet_name="Per_Ticker", index=False)
        _auto_width_and_formats(writer, "Per_Ticker", per_ticker_df)

        portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)
        _auto_width_and_formats(writer, "Portfolio", portfolio_df)

        head_to_head_df.to_excel(writer, sheet_name="HeadToHead", index=False)
        _auto_width_and_formats(writer, "HeadToHead", head_to_head_df)

        regime_winrates_df.to_excel(writer, sheet_name="Regime_WinRates", index=False)
        _auto_width_and_formats(writer, "Regime_WinRates", regime_winrates_df)


# ---------- orchestrator ----------

@dataclass
class RunParams:
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    min_data_fraction: float = 0.8


@dataclass
class StratSpec:
    name: str
    portfolio_mode: str  # "equal_weight_active" or "momo_overlay"
    momo_freq: Optional[str] = None  # "W", "M", "Q", "6M"
    momo_top_k: int = 5


def compute_all(universe_label: str,
                tickers: List[str],
                run: RunParams,
                strategies: List[StratSpec]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Load data
    data_by = {t: load_data(t, run.start_date, run.end_date) for t in sorted(set(tickers + ["SPY", "QQQ"]))}
    idx_ref = data_by["SPY"].index

    # Windows, full, ten equal slices, rolling 3 years with one year step
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

    def build_rolling_3y(index: pd.DatetimeIndex, step_years: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
        dates = index.sort_values()
        windows = []
        for y in range(dates[0].year, dates[-1].year - 2, step_years):
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

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    windows.append((idx_ref[0], idx_ref[-1], "FULL"))
    windows.extend(build_equal_slices(idx_ref, 10))
    windows.extend(build_rolling_3y(idx_ref, 1))

    # Tag regimes from SPY
    def tag_regime(s, e) -> str:
        spy = data_by["SPY"]
        w = spy[(spy.index >= s) & (spy.index <= e)]
        eq = buy_and_hold_close(w, run.initial_capital)
        if eq.empty:
            return "Unknown"
        ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        mdd = max_drawdown_from_equity(eq)
        if ret > 0.0 and mdd > -0.15:
            return "Bull"
        if ret < 0.0 and mdd <= -0.20:
            return "Bear"
        return "Sideways"

    windows_df = pd.DataFrame([
        {"window_id": wid, "start": s.date().isoformat(), "end": e.date().isoformat(), "regime": tag_regime(s, e)}
        for s, e, wid in windows
    ])

    # Pre compute per ticker S3 returns and positions
    s3p = S3Params()
    per_ticker_rows = []
    s3_by_window: Dict[Tuple[str, str], Dict[str, pd.Series]] = {}
    for s, e, wid in windows:
        spy_len = len(data_by["SPY"][(data_by["SPY"].index >= s) & (data_by["SPY"].index <= e)])
        for t in tickers + ["SPY", "QQQ"]:
            df = data_by[t]
            df_w = df[(df.index >= s) & (df.index <= e)]
            if len(df_w) < 10 or len(df_w) / spy_len < 0.8:
                continue
            eq, trades = backtest_s3(df_w, s3p, run.initial_capital)
            m = metrics_from_equity(eq["equity"])
            bh = metrics_from_equity(buy_and_hold_close(df_w, run.initial_capital))
            per_ticker_rows.append({
                "window_id": wid, "start": s.date().isoformat(), "end": e.date().isoformat(),
                "ticker": t, "entries": int(((eq["position"] > 0.0) & (eq["position"].shift(1).fillna(0.0) == 0.0)).sum()),
                "exits": int(((eq["position"] == 0.0) & (eq["position"].shift(1).fillna(0.0) > 0.0)).sum()),
                **{f"strat_{k}": v for k, v in m.items()},
                **{f"bh_{k}": v for k, v in bh.items()},
                "beat_bh_sharpe": m.get("sharpe", 0.0) > bh.get("sharpe", 0.0),
                "beat_bh_cagr": m.get("cagr", 0.0) > bh.get("cagr", 0.0),
                "beat_bh_mdd": m.get("max_drawdown", -1.0) > bh.get("max_drawdown", -1.0),
                "mdd_diff_vs_bh": float(m.get("max_drawdown", np.nan)) - float(bh.get("max_drawdown", np.nan)),
            })
            # store for portfolio building
            if (wid, t) not in s3_by_window:
                s3_by_window[(wid, t)] = {}
            s3_by_window[(wid, t)]["ret_close"] = eq["ret_close"]
            s3_by_window[(wid, t)]["position"] = eq["position"]
            s3_by_window[(wid, t)]["close"] = pd.to_numeric(df_w["Close"], errors="coerce")

    per_ticker_df = pd.DataFrame(per_ticker_rows)

    # Build portfolios per strategy
    portfolio_rows = []
    for s, e, wid in windows:
        index_w = data_by["SPY"][(data_by["SPY"].index >= s) & (data_by["SPY"].index <= e)].index
        # gather series dicts
        rets_by = {t: s3_by_window.get((wid, t), {}).get("ret_close", pd.Series(dtype=float)).reindex(index_w).fillna(0.0)
                   for t in tickers + ["SPY", "QQQ"]}
        pos_by  = {t: s3_by_window.get((wid, t), {}).get("position", pd.Series(dtype=float)).reindex(index_w).fillna(0.0)
                   for t in tickers + ["SPY", "QQQ"]}
        close_by= {t: s3_by_window.get((wid, t), {}).get("close", pd.Series(dtype=float)).reindex(index_w).ffill()
                   for t in tickers + ["SPY", "QQQ"]}

        # equal weight active portfolio for baseline S1
        if any(st.portfolio_mode == "equal_weight_active" for st in strategies):
            port_ret = portfolio_equal_weight_active(rets_by, pos_by, index_w)
            eq = run.initial_capital * (1.0 + port_ret).cumprod()
            m = metrics_from_equity(eq)
            # equal weight buy and hold of the same set
            rets_bh = pd.DataFrame({t: close_by[t].pct_change(fill_method=None).fillna(0.0) for t in tickers}).reindex(index_w).fillna(0.0)
            ew_bh_ret = rets_bh.mean(axis=1)
            ew_bh_eq = run.initial_capital * (1.0 + ew_bh_ret).cumprod()
            m_bh = metrics_from_equity(ew_bh_eq)
            portfolio_rows.append({
                "window_id": wid, "start": s.date().isoformat(), "end": e.date().isoformat(),
                "strategy": "S1_S3_Baseline",
                **{f"portfolio_{k}": v for k, v in m.items()},
                **{f"ew_bh_{k}": v for k, v in m_bh.items()},
                "beat_bh_sharpe": m.get("sharpe", 0.0) > m_bh.get("sharpe", 0.0),
                "beat_bh_cagr": m.get("cagr", 0.0) > m_bh.get("cagr", 0.0),
                "beat_bh_mdd": m.get("max_drawdown", -1.0) > m_bh.get("max_drawdown", -1.0),
                "mdd_diff_vs_bh": float(m.get("max_drawdown", np.nan)) - float(m_bh.get("max_drawdown", np.nan)),
            })

        # momentum overlay variants
        for st in strategies:
            if st.portfolio_mode != "momo_overlay":
                continue
            port_ret = portfolio_momentum_overlay(
                close_by_ticker=close_by,
                rets_by_ticker=rets_by,
                pos_by_ticker=pos_by,
                index=index_w,
                reserved=["SPY", "QQQ"],
                top_k=st.momo_top_k,
                freq=st.momo_freq
            )
            eq = run.initial_capital * (1.0 + port_ret).cumprod()
            m = metrics_from_equity(eq)
            # equal weight buy and hold for the whole universe, same as baseline comparison
            rets_bh = pd.DataFrame({t: close_by[t].pct_change(fill_method=None).fillna(0.0) for t in tickers}).reindex(index_w).fillna(0.0)
            ew_bh_ret = rets_bh.mean(axis=1)
            ew_bh_eq = run.initial_capital * (1.0 + ew_bh_ret).cumprod()
            m_bh = metrics_from_equity(ew_bh_eq)
            portfolio_rows.append({
                "window_id": wid, "start": s.date().isoformat(), "end": e.date().isoformat(),
                "strategy": st.name,
                **{f"portfolio_{k}": v for k, v in m.items()},
                **{f"ew_bh_{k}": v for k, v in m_bh.items()},
                "beat_bh_sharpe": m.get("sharpe", 0.0) > m_bh.get("sharpe", 0.0),
                "beat_bh_cagr": m.get("cagr", 0.0) > m_bh.get("cagr", 0.0),
                "beat_bh_mdd": m.get("max_drawdown", -1.0) > m_bh.get("max_drawdown", -1.0),
                "mdd_diff_vs_bh": float(m.get("max_drawdown", np.nan)) - float(m_bh.get("max_drawdown", np.nan)),
            })

    portfolio_df = pd.DataFrame(portfolio_rows)

    # winners and summary
    winners = []
    if not portfolio_df.empty:
        g = portfolio_df.dropna(subset=["portfolio_sharpe"]).copy()
        g["rank_key"] = list(zip(-g["portfolio_sharpe"], -g["portfolio_max_drawdown"], -g["portfolio_cagr"]))
        for wid, grp in g.groupby("window_id"):
            best = grp.sort_values("rank_key").iloc[0]
            winners.append({"window_id": wid, "winner_strategy": best["strategy"]})
    winners_df = pd.DataFrame(winners)

    windows_map = windows_df[["window_id","regime"]]
    port_join = portfolio_df.merge(winners_df, on="window_id", how="left").merge(windows_map, on="window_id", how="left")

    summary_rows = []
    for sname, grp in port_join.groupby("strategy"):
        summary_rows.append({
            "strategy": sname,
            "windows_evaluated": int(len(grp)),
            "wins_total": int((grp["winner_strategy"] == sname).sum()),
            "wins_bull": int(((grp["winner_strategy"] == sname) & (grp["regime"] == "Bull")).sum()),
            "wins_bear": int(((grp["winner_strategy"] == sname) & (grp["regime"] == "Bear")).sum()),
            "wins_sideways": int(((grp["winner_strategy"] == sname) & (grp["regime"] == "Sideways")).sum()),
            "avg_sharpe": float(grp["portfolio_sharpe"].mean()),
            "avg_sortino": float(grp["portfolio_sortino"].mean()),
            "median_max_drawdown": float(grp["portfolio_max_drawdown"].median()),
            "avg_cagr": float(grp["portfolio_cagr"].mean()),
            "beat_ew_bh_sharpe_pct": float(grp["beat_bh_sharpe"].mean()),
            "beat_ew_bh_cagr_pct": float(grp["beat_bh_cagr"].mean()),
            "beat_ew_bh_mdd_pct": float(grp["beat_bh_mdd"].mean()),
            "avg_excess_sharpe_vs_ew_bh": float((grp["portfolio_sharpe"] - grp["ew_bh_sharpe"]).mean()),
            "avg_excess_cagr_vs_ew_bh": float((grp["portfolio_cagr"] - grp["ew_bh_cagr"]).mean()),
            "avg_mdd_diff_vs_ew_bh": float((grp["portfolio_max_drawdown"] - grp["ew_bh_max_drawdown"]).mean()),
            "median_mdd_diff_vs_ew_bh": float((grp["portfolio_max_drawdown"] - grp["ew_bh_max_drawdown"]).median()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["wins_total","avg_sharpe"], ascending=[False, False])

    # head to head
    def head_to_head(df: pd.DataFrame, a: str, b: str) -> dict:
        sub = df.pivot_table(index="window_id", columns="strategy",
                             values=["portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"])
        out = {"A": a, "B": b}
        for metric in ["portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]:
            A = sub[(metric, a)]
            B = sub[(metric, b)]
            valid = A.notna() & B.notna()
            A = A[valid]; B = B[valid]
            win_pct = float((A > B).mean()) if len(A) else np.nan
            avg_diff = float((A - B).mean()) if len(A) else np.nan
            out[f"{metric}_win_pct_A_gt_B"] = win_pct
            out[f"{metric}_avg_diff_A_minus_B"] = avg_diff
        return out

    names = portfolio_df["strategy"].unique().tolist()
    head_rows = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            head_rows.append(head_to_head(portfolio_df, names[i], names[j]))
    head_df = pd.DataFrame(head_rows)

    # regime split win rates
    def regime_pair(df: pd.DataFrame, a: str, b: str) -> List[dict]:
        rows = []
        for reg in ["Bull","Bear","Sideways"]:
            sub = df[df["regime"] == reg].pivot_table(index="window_id", columns="strategy",
                                                      values=["portfolio_sharpe","portfolio_max_drawdown"])
            row = {"regime": reg, "A": a, "B": b}
            for metric in ["portfolio_sharpe","portfolio_max_drawdown"]:
                A = sub[(metric, a)]
                B = sub[(metric, b)]
                valid = A.notna() & B.notna()
                A = A[valid]; B = B[valid]
                win_pct = float((A > B).mean()) if len(A) else np.nan
                row[f"{metric}_win_pct_A_gt_B"] = win_pct
            rows.append(row)
        return rows

    regime_rows = []
    port_reg = portfolio_df.merge(windows_df[["window_id","regime"]], on="window_id", how="left")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            regime_rows.extend(regime_pair(port_reg, names[i], names[j]))
    regime_df = pd.DataFrame(regime_rows)

    # SummaryPlus versus baseline S1
    base = "S1_S3_Baseline"
    base_df = portfolio_df[portfolio_df["strategy"] == base][["window_id","portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]].rename(
        columns={"portfolio_sharpe":"base_sharpe", "portfolio_cagr":"base_cagr", "portfolio_max_drawdown":"base_mdd"}
    )
    plus_rows = []
    for n in names:
        g = portfolio_df[portfolio_df["strategy"] == n][["window_id","portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]].merge(base_df, on="window_id", how="inner")
        if n == base or g.empty:
            plus_rows.append({"strategy": n, "win_sharpe_vs_S1_pct": np.nan, "win_cagr_vs_S1_pct": np.nan, "better_mdd_vs_S1_pct": np.nan,
                              "avg_excess_sharpe_vs_S1": np.nan, "avg_excess_cagr_vs_S1": np.nan, "avg_mdd_diff_vs_S1": np.nan})
        else:
            plus_rows.append({
                "strategy": n,
                "win_sharpe_vs_S1_pct": float((g["portfolio_sharpe"] > g["base_sharpe"]).mean()),
                "win_cagr_vs_S1_pct": float((g["portfolio_cagr"] > g["base_cagr"]).mean()),
                "better_mdd_vs_S1_pct": float((g["portfolio_max_drawdown"] > g["base_mdd"]).mean()),
                "avg_excess_sharpe_vs_S1": float((g["portfolio_sharpe"] - g["base_sharpe"]).mean()),
                "avg_excess_cagr_vs_S1": float((g["portfolio_cagr"] - g["base_cagr"]).mean()),
                "avg_mdd_diff_vs_S1": float((g["portfolio_max_drawdown"] - g["base_mdd"]).mean()),
            })
    summary_plus_df = summary_df.merge(pd.DataFrame(plus_rows), on="strategy", how="left")

    return summary_df, windows_df, per_ticker_df, portfolio_df, head_df, regime_df, summary_plus_df


def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 baseline plus momentum overlay report v2")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
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
    run = RunParams(start_date=start, end_date=end,
                    initial_capital=float(common.get("initial_capital", 100000.0)),
                    min_data_fraction=float(common.get("min_data_fraction", 0.8)))

    strategies_cfg = cfg.get("strategies", [])
    strats: List[StratSpec] = []
    for s in strategies_cfg:
        strats.append(StratSpec(
            name=str(s["name"]),
            portfolio_mode=str(s.get("portfolio_mode", "equal_weight_active")),
            momo_freq=str(s.get("momo_freq", None)),
            momo_top_k=int(s.get("momo_top_k", 5)),
        ))

    summary_df, windows_df, per_ticker_df, portfolio_df, head_df, regime_df, summary_plus_df = compute_all(
        universe_label, tickers, run, strats
    )

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.get("output", {}).get("workbook_path", f"s3_momo_overlay_{universe_label}_{ts}.xlsx")
    build_workbook(out_path, summary_df, windows_df, per_ticker_df, portfolio_df, head_df, regime_df, summary_plus_df)
    print(f"Wrote workbook to {out_path}")


if __name__ == "__main__":
    main()
