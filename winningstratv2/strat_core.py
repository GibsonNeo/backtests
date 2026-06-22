#!/usr/bin/env python3
"""Pure strategy core for winningstratv2: locked hybrid SMA strat, metrics,
cash chain, combo blend, and Yahoo fetch. No file I/O here."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_sma(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).mean()


def consec_true(mask: pd.Series, n: int) -> pd.Series:
    if n <= 0:
        return pd.Series(False, index=mask.index)
    count = 0
    out = []
    for value in mask.fillna(False).to_numpy():
        count = count + 1 if bool(value) else 0
        out.append(count >= n)
    return pd.Series(out, index=mask.index)


def hybrid_position(
    price_signal: pd.Series,
    long_window: int = 200,
    short_window: int = 20,
    entry_days_long: int = 3,
    entry_days_short: int = 3,
    exit_days_long: int = 2,
    exit_days_short: int = 1,
) -> pd.Series:
    """Regime A (SMA_short >= SMA_long): enter after entry_days_long closes above
    SMA_long, exit after exit_days_long closes at/below it. Regime B
    (SMA_short < SMA_long): enter after entry_days_short days where close > SMA_short
    and SMA_short is rising, exit after exit_days_short closes at/below SMA_short.
    Returns a float 0/1 series, shifted one day, trimmed to first valid SMA date."""
    sma_long = rolling_sma(price_signal, long_window)
    sma_short = rolling_sma(price_signal, short_window)

    above_long = price_signal > sma_long
    above_short = price_signal > sma_short
    slope_up = sma_short > sma_short.shift(1)

    entry_long_ok = consec_true(above_long, entry_days_long)
    entry_short_ok = consec_true(above_short & slope_up, entry_days_short)
    exit_long_ok = consec_true(~above_long, exit_days_long)
    exit_short_ok = consec_true(~above_short, exit_days_short)

    short_below_long = sma_short < sma_long

    idx = price_signal.index
    pos = pd.Series(0.0, index=idx)
    in_pos = 0.0
    for i in range(len(idx)):
        use_short = bool(short_below_long.iat[i])
        entry_ok = entry_short_ok.iat[i] if use_short else entry_long_ok.iat[i]
        exit_ok = exit_short_ok.iat[i] if use_short else exit_long_ok.iat[i]
        if in_pos == 1.0 and exit_ok:
            in_pos = 0.0
        elif in_pos == 0.0 and entry_ok:
            in_pos = 1.0
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0.0)
    fv_long = sma_long.dropna().index.min()
    fv_short = sma_short.dropna().index.min()
    candidates = [x for x in [fv_long, fv_short] if x is not None]
    if candidates:
        first_valid = max(candidates)
        pos = pos[pos.index >= first_valid]
    return pos


def _ulcer_and_maxdd(curve: pd.Series) -> tuple[float, float]:
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(np.sqrt((dd.pow(2)).mean()) * 100.0), float(dd.min())


def metrics_from_returns(daily_rets: pd.Series, sharpe_rf: float = 0.0) -> dict:
    d = daily_rets.dropna()
    if d.empty:
        return dict(CAGR=0.0, AnnReturn=0.0, AnnVol=0.0, Sharpe=0.0, Sharpe_noRF=0.0,
                    MaxDD=0.0, UlcerIndex=0.0, Calmar=0.0, TotalMultiple=1.0, TotalReturn=0.0)

    curve = (1.0 + d).cumprod()
    years = len(d) / TRADING_DAYS
    total_multiple = float(curve.iloc[-1])
    mu = float(d.mean())
    sigma = float(d.std())
    ann_return = mu * TRADING_DAYS
    ann_vol = sigma * np.sqrt(TRADING_DAYS)

    if sigma > 0:
        sharpe = (mu - sharpe_rf / TRADING_DAYS) / sigma * np.sqrt(TRADING_DAYS)
        sharpe_norf = mu / sigma * np.sqrt(TRADING_DAYS)
    else:
        sharpe = sharpe_norf = 0.0

    ui, maxdd = _ulcer_and_maxdd(curve)
    cagr = total_multiple ** (1.0 / max(years, 1e-9)) - 1.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    return dict(CAGR=cagr, AnnReturn=ann_return, AnnVol=ann_vol, Sharpe=sharpe,
                Sharpe_noRF=sharpe_norf, MaxDD=maxdd, UlcerIndex=ui, Calmar=calmar,
                TotalMultiple=total_multiple, TotalReturn=total_multiple - 1.0)


def build_cash_chain(sgov_tr: pd.Series, bil_tr: pd.Series) -> pd.Series:
    """Daily cash returns: BIL before SGOV's first date, SGOV from then on.
    Before BIL's own inception the cash return is 0 (BIL did not exist)."""
    sgov_ret = sgov_tr.pct_change()
    bil_ret = bil_tr.pct_change()
    sgov_start = sgov_tr.dropna().index.min()
    sgov_part = sgov_ret[sgov_ret.index >= sgov_start]
    bil_part = bil_ret[bil_ret.index < sgov_start]
    chain = pd.concat([bil_part, sgov_part]).sort_index()
    chain = chain[~chain.index.duplicated(keep="last")]
    return chain.fillna(0.0)


def daily_from_pos(pos: pd.Series, tr_px: pd.Series, cash_daily: pd.Series) -> pd.Series:
    asset_rets = tr_px.pct_change().fillna(0.0)
    pos = pos.reindex(asset_rets.index).fillna(0.0)
    cash = cash_daily.reindex(asset_rets.index).fillna(0.0)
    return pos * asset_rets + (1.0 - pos) * cash


def combo_blend(sleeve_daily: dict, weights: dict | None = None) -> pd.Series:
    df = pd.DataFrame(sleeve_daily).dropna()
    cols = list(df.columns)
    if weights is None:
        weights = {c: 1.0 / len(cols) for c in cols}
    total = sum(weights[c] for c in cols)
    return sum((weights[c] / total) * df[c] for c in cols)


def combined_rank(df: pd.DataFrame, sharpe_col: str, cagr_col: str,
                  w_sharpe: float = 0.6, w_cagr: float = 0.4) -> pd.Series:
    s_rank = df[sharpe_col].rank(ascending=False, method="min")
    c_rank = df[cagr_col].rank(ascending=False, method="min")
    return w_sharpe * s_rank + w_cagr * c_rank


def _get_col(df, col_name, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_series(ticker: str, start: str, end_inclusive: str) -> tuple[pd.Series, pd.Series]:
    """Return (split-adjusted close for signals, auto-adjusted close for returns)."""
    import yfinance as yf

    end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=False, actions=True)
    if px.empty:
        raise ValueError(f"No data for {ticker}")
    close = _get_col(px, "Close", ticker).astype(float)
    splits = _get_col(px, "Stock Splits", ticker).fillna(0.0).astype(float)
    split_factor = splits.replace(0.0, 1.0)
    split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    price_signal = close * split_adj

    tr = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    price_tr = _get_col(tr, "Close", ticker).astype(float).rename("AdjClose")
    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]
