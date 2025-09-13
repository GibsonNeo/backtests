
#!/usr/bin/env python3
"""
S3 enhancements report
Baseline S3 moved to S1
All strategies share entry on SMA 200 slope up
Baseline exit is first of price below SMA 200 or SMA 200 slope down
Enhancements add, small percent bands, slope magnitude threshold, weekly signal evaluation, cooldown days, market level gate with SPY, volatility throttle, ensemble slope vote
Signals use Close only, buy and hold uses Close, start flat per window, force close on last bar
Portfolio uses equal weight across active names each day
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


# ---------- backtest core ----------

@dataclass
class StratParams:
    name: str
    ma_type: str = "sma"
    ma_len: int = 200
    ma_len_price: Optional[int] = 200
    ma_len_slope: Optional[int] = 200
    entry_confirm_bars: Optional[int] = 1
    exit_confirm_bars: Optional[int] = 1

    # Variant switches
    percent_band_pct: Optional[float] = None          # e.g. 0.005 for 0.5 percent
    band_on_entry: bool = True                        # require price above MA by band on entry
    slope_mag_days: Optional[int] = None              # e.g. 10
    slope_mag_thresh: Optional[float] = None          # e.g. 0.002 for 0.2 percent
    weekly_signals: bool = False                      # evaluate signals on Friday Close and carry for a week
    cooldown_bars: int = 0                            # require N bars after exit before new entry
    market_gate: Optional[str] = None                 # "spy_above_sma200" or "spy_in_s3"
    vol_lookback: Optional[int] = None                # e.g. 20
    vol_percentile: Optional[float] = None            # e.g. 0.9 blocks top decile of vol for entries
    ensemble_slope_hundred: bool = False              # require SMA 100 slope up as well for entry


@dataclass
class RunParams:
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    slippage_pct: float = 0.0
    fee_per_trade: float = 0.0
    min_data_fraction: float = 0.8


def build_daily_signals(price: pd.Series, p: StratParams) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    # moving averages
    len_price = p.ma_len_price if p.ma_len_price is not None else p.ma_len
    len_slope = p.ma_len_slope if p.ma_len_slope is not None else p.ma_len

    ma_price = _ma(price, len_price, p.ma_type)
    ma_slope = _ma(price, len_slope, p.ma_type)
    ma_100 = _ma(price, 100, p.ma_type) if p.ensemble_slope_hundred else None

    # base relations
    above = price > ma_price
    below = price < ma_price
    slope_up = ma_slope > ma_slope.shift(1)
    slope_down = ma_slope < ma_slope.shift(1)
    slope_up_100 = ma_100 > ma_100.shift(1) if ma_100 is not None else None

    # percent band logic
    if p.percent_band_pct is not None and p.percent_band_pct > 0:
        band = p.percent_band_pct
        above = price > (ma_price * (1.0 + band))
        below = price < (ma_price * (1.0 - band))

    # slope magnitude logic
    mag_up_ok = None
    mag_down_ok = None
    if p.slope_mag_days and p.slope_mag_thresh is not None:
        change = ma_slope.pct_change(p.slope_mag_days)
        mag_up_ok = change >= float(p.slope_mag_thresh)
        mag_down_ok = change <= float(-p.slope_mag_thresh)

    # entry rule base, slope up on 200
    entry_raw = slope_up
    # entry band requirement
    if p.percent_band_pct is not None and p.percent_band_pct > 0 and p.band_on_entry:
        entry_raw = entry_raw & above
    # entry slope magnitude
    if mag_up_ok is not None:
        entry_raw = entry_raw & mag_up_ok
    # ensemble slope vote
    if p.ensemble_slope_hundred and slope_up_100 is not None:
        entry_raw = entry_raw & slope_up_100

    # exit rule, baseline first of price below or slope down
    exit_raw = below | slope_down
    # slope magnitude exit override
    if mag_down_ok is not None and p.slope_mag_days and p.slope_mag_thresh is not None:
        # for the slope magnitude variant we require magnitude on exit as well
        exit_raw = mag_down_ok | below  # allow price break to also exit

    entry_ready = _confirm(entry_raw, p.entry_confirm_bars or 1)
    exit_ready = _confirm(exit_raw, p.exit_confirm_bars or 1)

    ctx = pd.DataFrame(index=price.index)
    ctx["MA_price"] = ma_price
    ctx["MA_slope"] = ma_slope
    if ma_100 is not None:
        ctx["MA_100"] = ma_100
    ctx["price_above"] = above
    ctx["price_below"] = below
    ctx["slope_up"] = slope_up
    ctx["slope_down"] = slope_down
    ctx["entry_ready"] = entry_ready
    ctx["exit_ready"] = exit_ready
    if mag_up_ok is not None:
        ctx["slope_mag_up_ok"] = mag_up_ok
    if mag_down_ok is not None:
        ctx["slope_mag_down_ok"] = mag_down_ok

    return entry_ready, exit_ready, ctx


def apply_weekly_mode(sig: pd.Series) -> pd.Series:
    weekly = sig.resample("W-FRI").last()
    expanded = weekly.reindex(sig.index).ffill()
    return expanded.astype("boolean")


def backtest_one(
    df: pd.DataFrame,
    p: StratParams,
    run: RunParams,
    spy_ctx: Optional[Dict[str, pd.Series]] = None,
    vol_gate_series: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    price = pd.to_numeric(df["Close"], errors="coerce")
    entry_ready, exit_ready, ctx = build_daily_signals(price, p)

    # weekly signal evaluation if asked
    if p.weekly_signals:
        entry_ready = apply_weekly_mode(entry_ready)
        exit_ready = apply_weekly_mode(exit_ready)

    # market gate
    if p.market_gate and spy_ctx is not None:
        if p.market_gate == "spy_above_sma200":
            gate = spy_ctx["spy_above_sma200"].reindex(entry_ready.index).fillna(False)
            entry_ready = (entry_ready & gate).astype("boolean")
        elif p.market_gate == "spy_in_s3":
            gate = spy_ctx["spy_in_s3"].reindex(entry_ready.index).fillna(False)
            entry_ready = (entry_ready & gate).astype("boolean")

    # volatility throttle, block entries on high vol days
    if p.vol_lookback and p.vol_percentile is not None and vol_gate_series is not None:
        gate = vol_gate_series.reindex(entry_ready.index).fillna(False)
        entry_ready = (entry_ready & gate).astype("boolean")

    entry_arr = entry_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)
    exit_arr  = exit_ready.shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(-1)

    open_arr = pd.to_numeric(df["Open"], errors="coerce").to_numpy()
    close_arr = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
    idx = df.index.to_numpy()

    in_pos = False
    cooldown = 0
    trades = []
    entry_idx = None

    for i in range(len(idx)):
        if (not in_pos) and entry_arr[i]:
            if cooldown > 0:
                cooldown -= 1
            else:
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
            cooldown = int(p.cooldown_bars or 0)

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
    exposure = float(rets.ne(0.0).mean())
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


# ---------- windows ----------

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


# ---------- portfolio ----------

def portfolio_equality_active(rets_by_ticker: Dict[str, pd.Series], pos_by_ticker: Dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.Series:
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
    aligned = [s.reindex(index).fillna(0.0) for s in rets_by_ticker.values()]
    if not aligned:
        return pd.Series(dtype=float)
    rets_df = pd.concat(aligned, axis=1)
    ew_ret = rets_df.mean(axis=1).fillna(0.0)
    eq = initial_capital * (1.0 + ew_ret).cumprod()
    eq.name = "ew_buy_hold_equity"
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
    return ("Sideways", ret, mdd)


# ---------- workbook ----------

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
    portfolio_df: pd.DataFrame,
    head_to_head_df: pd.DataFrame,
    regime_winrates_df: pd.DataFrame,
    summary_vs_base_df: pd.DataFrame,
):
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _auto_width_and_formats(writer, "Summary", summary_df)

        summary_vs_base_df.to_excel(writer, sheet_name="SummaryPlus", index=False)
        _auto_width_and_formats(writer, "SummaryPlus", summary_vs_base_df)

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

def compute_all(
    universe_label: str,
    tickers: List[str],
    run_params: RunParams,
    strategies: List[StratParams],
    include_rolling_3y: bool,
    equal_slice_count: int,
    rolling_step_years: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data_by_ticker: Dict[str, pd.DataFrame] = {}
    for t in sorted(set(tickers + ["SPY"])):
        data_by_ticker[t] = load_data(t, run_params.start_date, run_params.end_date)

    spy = data_by_ticker["SPY"]
    shared_index = spy.index

    # pre compute spy context for market gates and S3 state
    spy_ma200 = _ma(pd.to_numeric(spy["Close"], errors="coerce"), 200, "sma")
    spy_above = pd.to_numeric(spy["Close"], errors="coerce") > spy_ma200
    spy_slope_up = spy_ma200 > spy_ma200.shift(1)
    spy_slope_down = spy_ma200 < spy_ma200.shift(1)
    spy_exit = spy_above == False | spy_slope_down
    spy_entry = spy_slope_up
    spy_entry_ready = _confirm(spy_entry, 1)
    spy_exit_ready = _confirm(spy_exit, 1)
    spy_in_s3 = pd.Series(False, index=spy.index)
    state = False
    for i, ts in enumerate(spy.index):
        if (not state) and bool(spy_entry_ready.iloc[i]):
            state = True
        elif state and bool(spy_exit_ready.iloc[i]):
            state = False
        spy_in_s3.iloc[i] = state

    spy_ctx_full = {
        "spy_above_sma200": spy_above.astype("boolean"),
        "spy_in_s3": spy_in_s3.astype("boolean"),
    }

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

    per_ticker_rows = []
    portfolio_rows = []

    for s, e, wid in windows:
        spy_len = len(spy[(spy.index >= s) & (spy.index <= e)])
        if spy_len == 0:
            continue

        spy_ctx = {k: v[(v.index >= s) & (v.index <= e)] for k, v in spy_ctx_full.items()}

        for strat in strategies:
            for t in tickers:
                df_full = data_by_ticker[t]
                df_w = df_full[(df_full.index >= s) & (df_full.index <= e)]
                if len(df_w) < 10:
                    continue
                if len(df_w) / spy_len < run_params.min_data_fraction:
                    continue

                # volatility throttle gate per window
                vol_gate_series = None
                if strat.vol_lookback and strat.vol_percentile is not None:
                    close = pd.to_numeric(df_w["Close"], errors="coerce")
                    rets = close.pct_change()
                    vol = rets.rolling(int(strat.vol_lookback)).std()
                    cutoff = vol.quantile(float(strat.vol_percentile))
                    # allow entries when vol is at or below cutoff
                    vol_gate_series = (vol <= cutoff).astype("boolean")

                eq_w, trades_w, ctx_w = backtest_one(
                    df_w, strat, run_params,
                    spy_ctx=spy_ctx if strat.market_gate else None,
                    vol_gate_series=vol_gate_series
                )

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

                # recompute for portfolio alignment
                vol_gate_series = None
                if strat.vol_lookback and strat.vol_percentile is not None:
                    close = pd.to_numeric(df_w["Close"], errors="coerce")
                    rets = close.pct_change()
                    vol = rets.rolling(int(strat.vol_lookback)).std()
                    cutoff = vol.quantile(float(strat.vol_percentile))
                    vol_gate_series = (vol <= cutoff).astype("boolean")

                eq_w, trades_w, ctx_w = backtest_one(
                    df_w, strat, run_params,
                    spy_ctx=spy_ctx if strat.market_gate else None,
                    vol_gate_series=vol_gate_series
                )
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

    # Winners by window
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

    windows_map = windows_df[["window_id","regime"]]
    port_join = portfolio_df.merge(winners_df, on="window_id", how="left").merge(windows_map, on="window_id", how="left")

    # Summary
    summary_rows = []
    for strat in [s.name for s in strategies]:
        g = port_join[port_join["strategy"] == strat].copy()
        if g.empty:
            continue

        beat_ew_bh_sharpe_pct = float(g["beat_bh_sharpe"].mean())
        beat_ew_bh_cagr_pct   = float(g["beat_bh_cagr"].mean())
        beat_ew_bh_mdd_pct    = float(g["beat_bh_mdd"].mean())
        avg_excess_sharpe_vs_ew_bh = float((g["portfolio_sharpe"] - g["ew_bh_sharpe"]).mean())
        avg_excess_cagr_vs_ew_bh   = float((g["portfolio_cagr"] - g["ew_bh_cagr"]).mean())
        avg_mdd_diff_vs_ew_bh      = float(g["mdd_diff_vs_bh"].mean())
        med_mdd_diff_vs_ew_bh      = float(g["mdd_diff_vs_bh"].median())

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
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("wins_total", ascending=False)

    # Head to head, win rates and average differences for all pairs
    def head_to_head(df: pd.DataFrame, a: str, b: str) -> dict:
        sub = df.pivot_table(index="window_id", columns="strategy",
                             values=["portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"])
        out = {"A": a, "B": b}
        for metric in ["portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]:
            A = sub[(metric, a)]
            B = sub[(metric, b)]
            valid = A.notna() & B.notna()
            A = A[valid]; B = B[valid]
            if len(A) == 0:
                win_pct = np.nan
                avg_diff = np.nan
            else:
                win_pct = float((A > B).mean())
                avg_diff = float((A - B).mean())
            label = metric.replace("portfolio_", "")
            out[f"{label}_win_pct_A_gt_B"] = win_pct
            out[f"{label}_avg_diff_A_minus_B"] = avg_diff
        return out

    head_rows = []
    names = [s.name for s in strategies]
    for a in names:
        for b in names:
            if a == b:
                continue
            head_rows.append(head_to_head(portfolio_df, a, b))
    head_to_head_df = pd.DataFrame(head_rows)

    # Regime split win rates
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
                label = metric.replace("portfolio_", "")
                row[f"{label}_win_pct_A_gt_B"] = win_pct
            rows.append(row)
        return rows

    regime_rows = []
    port_join = portfolio_df.merge(windows_df[["window_id","regime"]], on="window_id", how="left")
    for a in names:
        for b in names:
            if a == b:
                continue
            regime_rows.extend(regime_pair(port_join, a, b))
    regime_winrates_df = pd.DataFrame(regime_rows)

    # SummaryPlus versus baseline
    base = names[0]  # S1 baseline is first in the list
    base_df = portfolio_df[portfolio_df["strategy"] == base][["window_id","portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]]
    base_df = base_df.rename(columns={
        "portfolio_sharpe": "base_sharpe",
        "portfolio_cagr": "base_cagr",
        "portfolio_max_drawdown": "base_mdd"
    })
    plus_rows = []
    for strat in names:
        g = portfolio_df[portfolio_df["strategy"] == strat][["window_id","portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]].merge(base_df, on="window_id", how="inner")
        if strat == base:
            win_sharpe = np.nan; win_cagr = np.nan; better_mdd = np.nan
            d_sh = np.nan; d_cg = np.nan; d_mdd = np.nan
        else:
            win_sharpe = float((g["portfolio_sharpe"] > g["base_sharpe"]).mean())
            win_cagr = float((g["portfolio_cagr"] > g["base_cagr"]).mean())
            better_mdd = float((g["portfolio_max_drawdown"] > g["base_mdd"]).mean())
            d_sh = float((g["portfolio_sharpe"] - g["base_sharpe"]).mean())
            d_cg = float((g["portfolio_cagr"] - g["base_cagr"]).mean())
            d_mdd = float((g["portfolio_max_drawdown"] - g["base_mdd"]).mean())
        plus_rows.append({
            "strategy": strat,
            "win_sharpe_vs_S1_pct": win_sharpe,
            "win_cagr_vs_S1_pct": win_cagr,
            "better_mdd_vs_S1_pct": better_mdd,
            "avg_excess_sharpe_vs_S1": d_sh,
            "avg_excess_cagr_vs_S1": d_cg,
            "avg_mdd_diff_vs_S1": d_mdd,
        })
    summary_vs_base_df = summary_df.merge(pd.DataFrame(plus_rows), on="strategy", how="left")

    return summary_df, windows_df, per_ticker_df, portfolio_df, head_to_head_df, regime_winrates_df, summary_vs_base_df


def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 enhancements report")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config, example s3_enhancements.yml")
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
            ma_len_slope=int(s.get("ma_len_slope", 200)) if s.get("ma_len_slope", None) is not None else None,
            entry_confirm_bars=int(s.get("entry_confirm_bars", 1)) if s.get("entry_confirm_bars", None) is not None else 1,
            exit_confirm_bars=int(s.get("exit_confirm_bars", 1)) if s.get("exit_confirm_bars", None) is not None else 1,
            percent_band_pct=float(s.get("percent_band_pct")) if s.get("percent_band_pct", None) is not None else None,
            band_on_entry=bool(s.get("band_on_entry", True)),
            slope_mag_days=int(s.get("slope_mag_days")) if s.get("slope_mag_days", None) is not None else None,
            slope_mag_thresh=float(s.get("slope_mag_thresh")) if s.get("slope_mag_thresh", None) is not None else None,
            weekly_signals=bool(s.get("weekly_signals", False)),
            cooldown_bars=int(s.get("cooldown_bars", 0)),
            market_gate=str(s.get("market_gate", None)) if s.get("market_gate", None) is not None else None,
            vol_lookback=int(s.get("vol_lookback")) if s.get("vol_lookback", None) is not None else None,
            vol_percentile=float(s.get("vol_percentile")) if s.get("vol_percentile", None) is not None else None,
            ensemble_slope_hundred=bool(s.get("ensemble_slope_hundred", False)),
        ))

    windows_cfg = cfg.get("windows", {})
    equal_slice_count = int(windows_cfg.get("equal_slices", 10))
    include_rolling_3y = bool(windows_cfg.get("include_rolling_3y", True))
    rolling_step_years = int(windows_cfg.get("rolling_step_years", 1))

    summary_df, windows_df, per_ticker_df, portfolio_df, head_to_head_df, regime_winrates_df, summary_vs_base_df = compute_all(
        universe_label=universe_label,
        tickers=tickers,
        run_params=run_params,
        strategies=strategies,
        include_rolling_3y=include_rolling_3y,
        equal_slice_count=equal_slice_count,
        rolling_step_years=rolling_step_years
    )

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.get("output", {}).get("workbook_path", f"s3_enhancements_{universe_label}_{ts}.xlsx")
    build_workbook(out_path, summary_df, windows_df, per_ticker_df, portfolio_df,
                   head_to_head_df, regime_winrates_df, summary_vs_base_df)
    print(f"Wrote workbook to {out_path}")


if __name__ == "__main__":
    main()
