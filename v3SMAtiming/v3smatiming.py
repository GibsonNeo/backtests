import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

TRADING_DAYS = 252


# ---------- metrics ----------

def ulcer_index_from_curve(curve: pd.Series) -> tuple[float, float]:
    peak = curve.cummax()
    dd = (curve - peak) / peak
    ui = float(np.sqrt((dd.pow(2)).mean()) * 100.0)
    maxdd = float(dd.min())
    return ui, maxdd


def annualize_mean_std(daily_rets: pd.Series, sharpe_rf_decimal: float = 0.0) -> tuple[float, float, float]:
    mu = float(daily_rets.mean())
    sd = float(daily_rets.std())
    ann_return = mu * TRADING_DAYS
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    if ann_vol > 0:
        daily_rf = sharpe_rf_decimal / TRADING_DAYS
        sharpe = (mu - daily_rf) / sd * np.sqrt(TRADING_DAYS)
    else:
        sharpe = 0.0
    return float(ann_return), float(ann_vol), float(sharpe)


def metrics_from_returns(daily_rets: pd.Series, sharpe_rf_decimal: float = 0.0) -> dict:
    daily_rets = daily_rets.dropna()
    curve = (1.0 + daily_rets).cumprod()
    years = len(daily_rets) / TRADING_DAYS
    total_return = float(curve.iloc[-1])
    cagr = total_return ** (1.0 / max(years, 1e-9)) - 1.0
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


# ---------- data fetch ----------

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


# ---------- tiered logic ----------

def _rolling_sma(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=w).mean()


def _consec_true(mask: pd.Series, n: int) -> pd.Series:
    if n <= 0:
        return pd.Series(True, index=mask.index)
    out = []
    c = 0
    for v in mask.values:
        if bool(v):
            c += 1
        else:
            c = 0
        out.append(c >= n)
    return pd.Series(out, index=mask.index)


def sleeve_position_from_sma(price_signal: pd.Series, window: int, entry_days: int, exit_days: int) -> pd.Series:
    """
    Entry when price > SMA for entry_days consecutive sessions
    Exit when price <= SMA for exit_days consecutive sessions
    Trade the next session to avoid lookahead
    """
    sma = _rolling_sma(price_signal, window)
    above = price_signal > sma
    entry_ok = _consec_true(above, entry_days)
    exit_ok = _consec_true(~above, exit_days)

    idx = price_signal.index
    pos = pd.Series(0, index=idx, dtype=int)
    in_pos = 0
    for i in range(len(idx)):
        if in_pos == 1:
            if exit_ok.iat[i]:
                in_pos = 0
        else:
            if entry_ok.iat[i]:
                in_pos = 1
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0).astype(int)
    # never trade before SMA is fully formed
    first_valid = sma.dropna().index.min()
    if first_valid is not None:
        pos = pos[pos.index >= first_valid]
    return pos


def tiered_three_position(price_signal: pd.Series, windows: list[int], entry_days: int, exit_days: int) -> pd.Series:
    """
    Three sleeves, equal weight, windows like [100, 200, 221]
    The same entry and exit delays are applied to all three sleeves
    """
    positions = []
    for w in windows:
        p = sleeve_position_from_sma(price_signal, window=w, entry_days=entry_days, exit_days=exit_days)
        positions.append(p.reindex(price_signal.index).fillna(0))
    pos = sum(positions) / 3.0
    return pos.clip(0.0, 1.0)


def sma200_single_position(price_signal: pd.Series) -> pd.Series:
    return sleeve_position_from_sma(price_signal, window=200, entry_days=0, exit_days=0)


# ---------- overlays ----------

def apply_max_dd_overlay(daily_rets: pd.Series, cash_daily: float, max_dd_decimal: float) -> pd.Series:
    if max_dd_decimal <= 0:
        return daily_rets
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


# ---------- per year stats ----------

def per_year_stats(daily: pd.Series, pos: pd.Series, sharpe_rf_decimal: float, variant: str) -> pd.DataFrame:
    df = pd.DataFrame({"ret": daily, "pos": pos})
    df["year"] = df.index.year
    rows = []
    for y, chunk in df.groupby("year"):
        d = chunk["ret"]
        curve = (1.0 + d).cumprod()
        ann = float(d.mean() * TRADING_DAYS)
        vol = float(d.std() * np.sqrt(TRADING_DAYS))
        if vol > 0:
            sharpe = float((d.mean() - sharpe_rf_decimal / TRADING_DAYS) / d.std() * np.sqrt(TRADING_DAYS))
        else:
            sharpe = 0.0
        ui, mdd = ulcer_index_from_curve(curve)
        exp_days = int(chunk["pos"].sum())
        total_days = int(len(chunk))
        exposure = exp_days / max(total_days, 1)
        pos_shift = chunk["pos"].shift(1).fillna(0)
        entries = int(((chunk["pos"] == 1) & (pos_shift == 0)).sum())
        exits = int(((chunk["pos"] == 0) & (pos_shift == 1)).sum())
        rows.append({
            "variant": variant,
            "year": int(y),
            "cal_year_return": float(curve.iloc[-1] - 1.0),
            "ann_return": ann,
            "ann_vol": vol,
            "sharpe": float(sharpe),
            "max_dd": float(mdd),
            "ulcer_index": float(ui),
            "exposure": float(exposure),
            "entries": entries,
            "exits": exits,
        })
    return pd.DataFrame(rows)


# ---------- main runner ----------

def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["QQQ", "SPY"])
    weights = cfg.get("weights", {"QQQ": 0.70, "SPY": 0.30})
    start = cfg.get("start")
    end = cfg.get("end")
    outdir = cfg.get("outdir", "outputs")

    cash_rate_percent = float(cfg.get("cash_rate_percent", cfg.get("risk_free_rate", 0.0)))
    sharpe_rf_percent = float(cfg.get("sharpe_rf_percent", 0.0))
    max_dd_percent = float(cfg.get("max_drawdown_percent", 0.0))

    tier_windows = list(cfg.get("tier_sma_windows", [100, 200, 221]))
    entry_days_list = list(cfg.get("entry_days_to_test", [0, 1, 2, 3]))
    exit_days_list = list(cfg.get("exit_days_to_test", [0, 1, 2, 3]))

    include_baseline_sma200 = bool(cfg.get("include_baseline_sma200", True))
    include_baseline_buyhold = bool(cfg.get("include_baseline_buyhold", True))

    # normalize possible date objects
    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, date):
        end = end.strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # decimals
    CASH_RATE = cash_rate_percent / 100.0
    SHARPE_RF = sharpe_rf_percent / 100.0
    MAX_DD = max_dd_percent / 100.0
    DAILY_CASH = CASH_RATE / TRADING_DAYS

    # warmup for the longest SMA in the tier
    max_w = max(tier_windows + [200])  # make sure 200 is covered for the baseline
    ext_start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)
    ext_start = ext_start_dt.strftime("%Y-%m-%d")

    # fetch signal and total return series for both tickers
    data = {}
    for tkr in tickers:
        sig_px, tr_px = fetch_series(tkr, ext_start, end)
        # slice to requested window
        sig_px = sig_px[sig_px.index >= pd.to_datetime(start)]
        tr_px = tr_px[tr_px.index >= pd.to_datetime(start)]
        data[tkr] = dict(sig=sig_px, tr=tr_px)

    os.makedirs(outdir, exist_ok=True)

    overall_rows = []
    yearly_rows = []

    # run grid of entry and exit days
    for entry_days in entry_days_list:
        for exit_days in exit_days_list:
            name = f"tier_e{entry_days}_x{exit_days}"

            # build per symbol positions
            pos = {}
            rets = {}
            for tkr in tickers:
                sig_px = data[tkr]["sig"]
                tr_px = data[tkr]["tr"]
                p = tiered_three_position(sig_px, windows=tier_windows, entry_days=entry_days, exit_days=exit_days)
                p = p.reindex(tr_px.index).fillna(0.0)

                asset_rets = tr_px.pct_change().fillna(0.0)
                daily = p * asset_rets + (1 - p) * DAILY_CASH
                pos[tkr] = p
                rets[tkr] = daily

            # blend symbols
            aligned_index = rets[tickers[0]].index
            for tkr in tickers[1:]:
                aligned_index = aligned_index.intersection(rets[tkr].index)
            for tkr in tickers:
                rets[tkr] = rets[tkr].reindex(aligned_index).fillna(0.0)
                pos[tkr] = pos[tkr].reindex(aligned_index).fillna(0.0)

            portfolio_daily = sum(weights[t] * rets[t] for t in tickers)
            if MAX_DD > 0:
                portfolio_daily = apply_max_dd_overlay(portfolio_daily, DAILY_CASH, MAX_DD)

            # overall metrics
            ov = metrics_from_returns(portfolio_daily, SHARPE_RF)
            ov["variant"] = name
            overall_rows.append(ov)

            # per year
            # exposure is blended weight times position, useful to see average risk
            blended_pos = sum(weights[t] * pos[t] for t in tickers)
            yearly_df = per_year_stats(portfolio_daily, blended_pos, SHARPE_RF, name)
            yearly_rows.append(yearly_df)

            # save per variant files
            pd.DataFrame([ov]).to_csv(os.path.join(outdir, f"overall_{name}.csv"), index=False)
            yearly_df.to_csv(os.path.join(outdir, f"per_year_{name}.csv"), index=False)
            print(f"Saved overall_{name}.csv and per_year_{name}.csv")

    # baselines
    if include_baseline_sma200:
        name = "baseline_sma200"
        pos = {}
        rets = {}
        for tkr in tickers:
            p = sma200_single_position(data[tkr]["sig"])
            p = p.reindex(data[tkr]["tr"].index).fillna(0.0)
            pos[tkr] = p
            asset_rets = data[tkr]["tr"].pct_change().fillna(0.0)
            rets[tkr] = p * asset_rets + (1 - p) * DAILY_CASH
        aligned_index = rets[tickers[0]].index
        for tkr in tickers[1:]:
            aligned_index = aligned_index.intersection(rets[tkr].index)
        for tkr in tickers:
            rets[tkr] = rets[tkr].reindex(aligned_index).fillna(0.0)
            pos[tkr] = pos[tkr].reindex(aligned_index).fillna(0.0)
        portfolio_daily = sum(weights[t] * rets[t] for t in tickers)
        if MAX_DD > 0:
            portfolio_daily = apply_max_dd_overlay(portfolio_daily, DAILY_CASH, MAX_DD)
        ov = metrics_from_returns(portfolio_daily, SHARPE_RF)
        ov["variant"] = name
        overall_rows.append(ov)
        blended_pos = sum(weights[t] * pos[t] for t in tickers)
        yearly_df = per_year_stats(portfolio_daily, blended_pos, SHARPE_RF, name)
        yearly_rows.append(yearly_df)
        pd.DataFrame([ov]).to_csv(os.path.join(outdir, f"overall_{name}.csv"), index=False)
        yearly_df.to_csv(os.path.join(outdir, f"per_year_{name}.csv"), index=False)
        print(f"Saved baseline SMA200 files")

    if include_baseline_buyhold:
        name = "baseline_buyhold"
        rets = {}
        aligned_index = None
        for tkr in tickers:
            a = data[tkr]["tr"].pct_change().fillna(0.0)
            rets[tkr] = a
            aligned_index = a.index if aligned_index is None else aligned_index.intersection(a.index)
        for tkr in tickers:
            rets[tkr] = rets[tkr].reindex(aligned_index).fillna(0.0)
        portfolio_daily = sum(weights[t] * rets[t] for t in tickers)
        if MAX_DD > 0:
            portfolio_daily = apply_max_dd_overlay(portfolio_daily, DAILY_CASH, MAX_DD)
        ov = metrics_from_returns(portfolio_daily, SHARPE_RF)
        ov["variant"] = name
        overall_rows.append(ov)
        yearly_df = per_year_stats(portfolio_daily, pd.Series(1.0, index=portfolio_daily.index), SHARPE_RF, name)
        yearly_rows.append(yearly_df)
        pd.DataFrame([ov]).to_csv(os.path.join(outdir, f"overall_{name}.csv"), index=False)
        yearly_df.to_csv(os.path.join(outdir, f"per_year_{name}.csv"), index=False)
        print(f"Saved baseline buy and hold files")

    # combined rollups
    overall_all = pd.DataFrame(overall_rows)
    overall_all = overall_all[["variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "UlcerIndex", "TotalReturn"]]
    overall_all.to_csv(os.path.join(outdir, "overall_all_variants.csv"), index=False)

    yearly_all = pd.concat(yearly_rows, ignore_index=True)
    yearly_all.to_csv(os.path.join(outdir, "per_year_all_variants.csv"), index=False)

    print("Done. Wrote overall_all_variants.csv and per_year_all_variants.csv")


if __name__ == "__main__":
    main()
