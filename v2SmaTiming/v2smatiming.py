import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

TRADING_DAYS = 252


def ulcer_index_from_curve(curve: pd.Series) -> tuple[float, float]:
    peak = curve.cummax()
    dd = (curve - peak) / peak
    ui = np.sqrt((dd.pow(2)).mean()) * 100.0
    maxdd = float(dd.min())
    return ui, maxdd


def annualize_mean_std(daily_rets: pd.Series, sharpe_rf_decimal: float = 0.0) -> tuple[float, float, float]:
    ann_return = float(daily_rets.mean() * TRADING_DAYS)
    ann_vol = float(daily_rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = 0.0
    if ann_vol > 0:
        daily_rf = sharpe_rf_decimal / TRADING_DAYS
        sharpe = float((daily_rets.mean() - daily_rf) / daily_rets.std() * np.sqrt(TRADING_DAYS))
    return ann_return, ann_vol, sharpe


def metrics_from_returns(daily_rets: pd.Series, sharpe_rf_decimal: float = 0.0) -> dict:
    curve = (1.0 + daily_rets).cumprod()
    years = len(daily_rets) / TRADING_DAYS
    total_return = float(curve.iloc[-1])
    cagr = total_return ** (1.0 / years) - 1.0
    ann_return, ann_vol, sharpe = annualize_mean_std(daily_rets, sharpe_rf_decimal)
    ui, maxdd = ulcer_index_from_curve(curve)
    return dict(
        CAGR=cagr,
        AnnReturn=ann_return,
        AnnVol=ann_vol,
        Sharpe=sharpe,
        MaxDD=maxdd,
        UlcerIndex=ui,
        TotalReturn=total_return,
    )


def _get_col(df: pd.DataFrame, col_name: str, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_series(ticker: str, start: str, end_inclusive: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns two aligned series
    price_signal, split adjusted close for signals only, dividends excluded
    price_tr, total return close for PnL, splits and dividends included
    """
    end_exclusive_dt = datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)
    end_exclusive = end_exclusive_dt.strftime("%Y-%m-%d")

    px = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=False, actions=True)
    if px.empty:
        raise ValueError(f"No data for {ticker}")

    close = _get_col(px, "Close", ticker).astype(float)
    splits = _get_col(px, "Stock Splits", ticker).fillna(0.0).astype(float)

    split_factor = splits.replace(0.0, 1.0)
    split_adj = (1.0 / split_factor).cumprod().shift(1).fillna(1.0)
    price_signal = close * split_adj

    tr_df = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    price_tr = _get_col(tr_df, "Close", ticker).rename("AdjClose").astype(float)

    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]


def build_daily_strategy_returns(
    price_signal: pd.Series,
    price_tr: pd.Series,
    sma_window: int,
    cash_rate_decimal: float = 0.0,
    proxy_rets: pd.Series | None = None,
) -> pd.Series:
    """
    Compute daily strategy returns using a daily SMA rule
    Never trades before the first date with a fully formed SMA
    """
    asset_rets = price_tr.pct_change()

    sma = price_signal.rolling(window=sma_window, min_periods=sma_window).mean()
    first_valid = sma.dropna().index.min()

    sig = (price_signal > sma).astype(int)
    sig = sig.reindex(price_tr.index).fillna(0)
    sig = sig.shift(1).fillna(0)

    if proxy_rets is None:
        daily_cash = cash_rate_decimal / TRADING_DAYS
        daily = sig * asset_rets + (1 - sig) * daily_cash
    else:
        proxy_rets = proxy_rets.reindex(price_tr.index)
        proxy_rets = proxy_rets.fillna(0.0)
        daily = sig * asset_rets + (1 - sig) * proxy_rets

    daily = daily.dropna()

    if first_valid is not None:
        daily = daily[daily.index >= first_valid]

    if daily.isna().any():
        raise ValueError("NaNs in strategy returns, check index alignment")

    return daily


def to_monthly_signals(price_signal: pd.Series, price_tr: pd.Series, months: int = 10) -> tuple[pd.Series, pd.Series]:
    ms = price_signal.resample("ME").last()
    mt = price_tr.resample("ME").last()

    sma_m = ms.rolling(window=months, min_periods=months).mean()
    first_valid_m = sma_m.dropna().index.min()

    sig_m = (ms > sma_m).astype(int).shift(1).fillna(0)

    sig_daily = sig_m.reindex(mt.index).ffill().reindex(price_tr.index).ffill().fillna(0).astype(int)
    daily_tr = price_tr.pct_change().reindex(sig_daily.index).fillna(0.0)

    if first_valid_m is not None:
        first_daily = daily_tr.index[daily_tr.index.searchsorted(first_valid_m)]
        sig_daily = sig_daily[sig_daily.index >= first_daily]
        daily_tr = daily_tr[daily_tr.index >= first_daily]

    return sig_daily, daily_tr


def apply_max_dd_overlay(daily_rets: pd.Series, cash_daily: float, max_dd_decimal: float) -> pd.Series:
    """
    Overlay, replaces strategy returns with cash when drawdown from peak
    exceeds max_dd_decimal, stays in cash until a new equity peak
    """
    eq = (1.0 + daily_rets).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak

    out = daily_rets.copy()
    in_cash = False

    for i in range(len(out)):
        if not in_cash:
            if dd.iat[i] <= -max_dd_decimal:
                in_cash = True
                out.iat[i] = cash_daily
        else:
            if eq.iat[i] >= peak.iat[i]:
                in_cash = False
                out.iat[i] = daily_rets.iat[i]
            else:
                out.iat[i] = cash_daily
    return out


def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    ticker = cfg["ticker"].strip()
    cash_proxy = str(cfg.get("cash_proxy", "NONE")).strip()

    # new config fields, in percent per year
    cash_rate_percent = float(cfg.get("cash_rate_percent", cfg.get("risk_free_rate", 0.0)))
    sharpe_rf_percent = float(cfg.get("sharpe_rf_percent", 0.0))
    max_dd_percent = float(cfg.get("max_drawdown_percent", 0.0))

    # convert to decimals once
    CASH_RATE = cash_rate_percent / 100.0
    SHARPE_RF = sharpe_rf_percent / 100.0
    MAX_DD = max_dd_percent / 100.0
    DAILY_CASH = CASH_RATE / TRADING_DAYS

    start = cfg.get("start")
    end = cfg.get("end")
    years = int(cfg.get("years", 10))
    sma_windows = list(cfg["sma_windows"])
    outdir = cfg.get("outdir", "outputs")
    include_monthly_10 = bool(cfg.get("include_monthly_10", True))

    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, date):
        end = end.strftime("%Y-%m-%d")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start = (end_dt - timedelta(days=365 * years + 365)).strftime("%Y-%m-%d")

    max_w = max(sma_windows) if sma_windows else 0
    ext_start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)
    ext_start = ext_start_dt.strftime("%Y-%m-%d")

    # fetch full series with warmup preserved
    sig_px_full, tr_px_full = fetch_series(ticker, ext_start, end)

    proxy_rets_full = None
    if cash_proxy != "NONE":
        _, proxy_tr_full = fetch_series(cash_proxy, ext_start, end)
        proxy_rets_full = proxy_tr_full.pct_change().dropna()

    # buy and hold, total return, then slice to the requested window
    asset_rets_full = tr_px_full.pct_change().dropna()
    window_mask = (asset_rets_full.index >= pd.to_datetime(start)) & (asset_rets_full.index <= pd.to_datetime(end))
    asset_rets = asset_rets_full.loc[window_mask]
    results = {"BuyAndHold": metrics_from_returns(asset_rets, SHARPE_RF)}

    # daily SMA variants, build on full series, optional overlay, then slice
    for w in sorted(sma_windows):
        daily_full = build_daily_strategy_returns(
            sig_px_full,
            tr_px_full,
            w,
            cash_rate_decimal=CASH_RATE,
            proxy_rets=proxy_rets_full,
        )
        if MAX_DD > 0.0:
            daily_full = apply_max_dd_overlay(daily_full, DAILY_CASH, MAX_DD)

        daily = daily_full.loc[(daily_full.index >= pd.to_datetime(start)) & (daily_full.index <= pd.to_datetime(end))]
        results[f"SMA{w}"] = metrics_from_returns(daily, SHARPE_RF)

    # monthly ten month, build on full, optional overlay, then slice
    if include_monthly_10:
        sig_daily_full, daily_tr_full = to_monthly_signals(sig_px_full, tr_px_full, months=10)
        if proxy_rets_full is None:
            daily_full = sig_daily_full * daily_tr_full + (1 - sig_daily_full) * DAILY_CASH
        else:
            pr = proxy_rets_full.reindex(daily_tr_full.index).fillna(0.0)
            daily_full = sig_daily_full * daily_tr_full + (1 - sig_daily_full) * pr

        if MAX_DD > 0.0:
            daily_full = apply_max_dd_overlay(daily_full, DAILY_CASH, MAX_DD)

        daily = daily_full.loc[(daily_full.index >= pd.to_datetime(start)) & (daily_full.index <= pd.to_datetime(end))]
        results["M10"] = metrics_from_returns(daily, SHARPE_RF)

    # output
    df = pd.DataFrame(results).T
    df = df[["CAGR", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "UlcerIndex", "TotalReturn"]].T

    cols = [f"SMA{w}" for w in sorted(sma_windows)]
    if include_monthly_10:
        cols = cols + ["M10"]
    cols = cols + ["BuyAndHold"]
    df = df[cols]

    print("Overall stats")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "backtest_results.csv"))


if __name__ == "__main__":
    main()
