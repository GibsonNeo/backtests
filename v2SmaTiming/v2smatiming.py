import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import os

TRADING_DAYS = 252

def ulcer_index_from_curve(curve):
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return np.sqrt((dd.pow(2)).mean()) * 100.0, dd.min()

def annualize_mean_std(daily_rets, rf=0.0):
    ann_return = daily_rets.mean() * TRADING_DAYS
    ann_vol = daily_rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = 0.0
    if ann_vol > 0:
        daily_rf = rf / TRADING_DAYS
        sharpe = (daily_rets.mean() - daily_rf) / daily_rets.std() * np.sqrt(TRADING_DAYS)
    return ann_return, ann_vol, sharpe

def metrics_from_returns(daily_rets, rf=0.0):
    curve = (1.0 + daily_rets).cumprod()
    years = len(daily_rets) / TRADING_DAYS
    total_return = float(curve.iloc[-1])
    cagr = total_return ** (1.0 / years) - 1.0
    ann_return, ann_vol, sharpe = annualize_mean_std(daily_rets, rf)
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

def fetch_series(ticker, start, end):
    # price for signals, split adjusted only
    px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, actions=True)
    if px.empty:
        raise ValueError(f"No data for {ticker}")
    close = px["Close"].copy()
    splits = px["Stock Splits"].fillna(0)
    adj = pd.Series(1.0, index=close.index)
    for dt, ratio in splits[splits != 0].items():
        if float(ratio) != 0:
            adj.loc[adj.index < dt] *= 1.0 / float(ratio)
    price_signal = close * adj

    # total return for PnL, use Adjusted Close which includes splits and dividends
    tr_df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    price_tr = tr_df["Close"].rename("AdjClose")

    return price_signal.dropna(), price_tr.dropna()

def build_daily_strategy_returns(price_signal, price_tr, sma_window, rf=0.0, proxy_rets=None):
    # daily asset returns in total return terms
    asset_rets = price_tr.pct_change().dropna()

    # compute SMA on signal price, require a full window, then align to asset return dates
    sma = price_signal.rolling(window=sma_window, min_periods=sma_window).mean()
    sig = (price_signal > sma).astype(int).reindex(asset_rets.index).fillna(0)
    sig = sig.shift(1).fillna(0)  # trade next day open proxy

    if proxy_rets is None:
        cash_ret = rf / TRADING_DAYS
        daily = sig * asset_rets + (1 - sig) * cash_ret
    else:
        proxy_rets = proxy_rets.reindex(asset_rets.index).fillna(0.0)
        daily = sig * asset_rets + (1 - sig) * proxy_rets
    return daily

def to_monthly_signals(price_signal, price_tr, months=10):
    # month end sampling for the classic 10 month rule
    ms = price_signal.resample("M").last()
    mt = price_tr.resample("M").last()
    sma_m = ms.rolling(window=months, min_periods=months).mean()
    sig_m = (ms > sma_m).astype(int).shift(1).fillna(0)
    # convert monthly signal to daily series by forward fill, then align to daily total return dates
    sig_daily = sig_m.reindex(mt.index).ffill().reindex(price_tr.index).ffill().fillna(0).astype(int)
    daily_tr = price_tr.pct_change().reindex(sig_daily.index).fillna(0.0)
    return sig_daily, daily_tr

def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    ticker = cfg["ticker"].strip()
    cash_proxy = cfg.get("cash_proxy", "NONE").strip()
    rf = float(cfg.get("risk_free_rate", 0.0))
    start = cfg.get("start")
    end = cfg.get("end")
    years = int(cfg.get("years", 10))
    sma_windows = list(cfg["sma_windows"])
    outdir = cfg.get("outdir", "outputs")
    include_monthly_10 = bool(cfg.get("include_monthly_10", True))

    if isinstance(start, date): start = start.strftime("%Y-%m-%d")
    if isinstance(end, date): end = end.strftime("%Y-%m-%d")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start = (end_dt - timedelta(days=365 * years + 365)).strftime("%Y-%m-%d")

    max_w = max(sma_windows) if sma_windows else 0
    # generous warmup so the first valid SMA is truly out of sample
    ext_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)).strftime("%Y-%m-%d")

    sig_px, tr_px = fetch_series(ticker, ext_start, end)

    proxy_rets = None
    if cash_proxy != "NONE":
        p_sig, p_tr = fetch_series(cash_proxy, ext_start, end)
        proxy_rets = p_tr.pct_change().dropna()

    # buy and hold on total return
    asset_rets = tr_px.pct_change().dropna()
    # restrict test window strictly to [start, end]
    asset_rets = asset_rets[asset_rets.index >= pd.to_datetime(start)]
    results = {"BuyAndHold": metrics_from_returns(asset_rets, rf)}

    for w in sorted(sma_windows):
        daily = build_daily_strategy_returns(sig_px, tr_px, w, rf=rf, proxy_rets=proxy_rets)
        daily = daily[daily.index >= pd.to_datetime(start)]
        results[f"SMA{w}"] = metrics_from_returns(daily, rf)

    if include_monthly_10:
        sig_daily, daily_tr = to_monthly_signals(sig_px, tr_px, months=10)
        daily = sig_daily.shift(0) * daily_tr + (1 - sig_daily.shift(0)) * (proxy_rets.reindex(daily_tr.index).fillna(rf / TRADING_DAYS) if proxy_rets is not None else rf / TRADING_DAYS)
        daily = daily[daily.index >= pd.to_datetime(start)]
        results["M10"] = metrics_from_returns(daily, rf)

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
