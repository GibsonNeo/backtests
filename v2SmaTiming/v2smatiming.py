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


def _compute_position_with_entry_rule(
    price_signal: pd.Series,
    sma: pd.Series,
    entry_mode: str,
    entry_days: int,
    slope_min_per_day: float,
) -> pd.Series:
    """
    Position exits immediately when price <= SMA.
    Entry requires conditions based on mode.
      immediate, enter first day price > SMA
      days_above, require N consecutive days above SMA
      sma_slope, require SMA slope >= slope_min_per_day using the SMA window length
      both, require both days_above and sma_slope
    """
    idx = price_signal.index
    above = price_signal > sma

    # consecutive days above SMA
    consec = pd.Series(0, index=idx, dtype=int)
    c = 0
    for i, is_above in enumerate(above.values):
        if is_above:
            c += 1
        else:
            c = 0
        consec.iat[i] = c

    # infer SMA window length from the first valid index distance
    first_valid = sma.dropna().index.min()
    if first_valid is not None:
        win_est = int(sma.index.get_loc(first_valid))
        win_est = max(win_est, 1)
    else:
        win_est = 1

    window = max(win_est, 1)
    slope_pct_per_day = (sma / sma.shift(window) - 1.0) / window

    # iterative position with delayed entries and immediate exits
    pos = pd.Series(0, index=idx, dtype=int)
    in_pos = 0
    for i in range(len(idx)):
        if in_pos == 1:
            if not above.iat[i]:
                in_pos = 0
        else:
            if above.iat[i]:
                ok_days = consec.iat[i] >= max(1, entry_days)
                ok_slope = False
                val = slope_pct_per_day.iat[i]
                if pd.notna(val):
                    ok_slope = val >= slope_min_per_day

                if entry_mode == "immediate":
                    in_pos = 1
                elif entry_mode == "days_above":
                    in_pos = 1 if ok_days else 0
                elif entry_mode == "sma_slope":
                    in_pos = 1 if ok_slope else 0
                elif entry_mode == "both":
                    in_pos = 1 if (ok_days and ok_slope) else 0
                else:
                    in_pos = 1
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0).astype(int)
    return pos


def build_daily_strategy_returns(
    price_signal: pd.Series,
    price_tr: pd.Series,
    sma_window: int,
    cash_rate_decimal: float = 0.0,
    proxy_rets: pd.Series | None = None,
    entry_cfg: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Daily SMA strategy with configurable entry logic.
    Exit happens the day after price <= SMA.
    Returns daily returns and the daily position series.
    """
    asset_rets = price_tr.pct_change()

    sma = price_signal.rolling(window=sma_window, min_periods=sma_window).mean()
    first_valid = sma.dropna().index.min()

    entry_mode = "immediate"
    entry_days = 1
    slope_min_per_day = 0.0

    if entry_cfg:
        entry_mode = str(entry_cfg.get("mode", entry_mode)).lower()
        entry_days = int(entry_cfg.get("days_above", entry_days))
        slope_min_per_day = float(entry_cfg.get("slope_min_per_day", slope_min_per_day))

    pos = _compute_position_with_entry_rule(
        price_signal=price_signal,
        sma=sma,
        entry_mode=entry_mode,
        entry_days=entry_days,
        slope_min_per_day=slope_min_per_day,
    ).reindex(price_tr.index).fillna(0)

    if proxy_rets is None:
        daily_cash = cash_rate_decimal / TRADING_DAYS
        daily = pos * asset_rets + (1 - pos) * daily_cash
    else:
        proxy_rets = proxy_rets.reindex(price_tr.index).fillna(0.0)
        daily = pos * asset_rets + (1 - pos) * proxy_rets

    daily = daily.dropna()
    pos = pos.loc[daily.index]

    if first_valid is not None:
        daily = daily[daily.index >= first_valid]
        pos = pos.loc[daily.index]

    if daily.isna().any():
        raise ValueError("NaNs in strategy returns, check index alignment")

    return daily, pos


def to_monthly_signals_with_entry(
    price_signal: pd.Series,
    price_tr: pd.Series,
    months: int,
    entry_cfg: dict | None,
    apply_entry: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Monthly variant, can respect entry knobs at the month level.
    Here, days_above counts consecutive months.
    Slope is per month and uses the same months parameter.
    """
    ms = price_signal.resample("ME").last()
    mt = price_tr.resample("ME").last()

    sma_m = ms.rolling(window=months, min_periods=months).mean()
    first_valid_m = sma_m.dropna().index.min()

    if not apply_entry:
        pos_m = (ms > sma_m).astype(int).shift(1).fillna(0)
    else:
        mode = str(entry_cfg.get("mode", "immediate")).lower() if entry_cfg else "immediate"
        months_above = int(entry_cfg.get("days_above", 1)) if entry_cfg else 1
        slope_min = float(entry_cfg.get("slope_min_per_day", 0.0)) if entry_cfg else 0.0

        # reuse the same entry engine at monthly frequency
        # in monthly context, slope_min is per month
        pos_m = _compute_position_with_entry_rule(
            price_signal=ms,
            sma=sma_m,
            entry_mode=mode,
            entry_days=months_above,
            slope_min_per_day=slope_min,
        )

    sig_daily = pos_m.reindex(mt.index).ffill().reindex(price_tr.index).ffill().fillna(0).astype(int)
    daily_tr = price_tr.pct_change().reindex(sig_daily.index).fillna(0.0)

    if first_valid_m is not None:
        first_daily = daily_tr.index[daily_tr.index.searchsorted(first_valid_m)]
        sig_daily = sig_daily[sig_daily.index >= first_daily]
        daily_tr = daily_tr[daily_tr.index >= first_daily]

    daily = sig_daily * daily_tr
    return daily, sig_daily, daily_tr


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


def per_year_stats(daily: pd.Series, pos: pd.Series, sharpe_rf_decimal: float, strat: str, run_name: str) -> pd.DataFrame:
    df = pd.DataFrame({"ret": daily, "pos": pos})
    df["year"] = df.index.year
    rows = []
    for y, chunk in df.groupby("year"):
        d = chunk["ret"]
        curve = (1.0 + d).cumprod()
        ann = d.mean() * TRADING_DAYS
        vol = d.std() * np.sqrt(TRADING_DAYS)
        sharpe = 0.0
        if vol > 0:
            sharpe = (d.mean() - sharpe_rf_decimal / TRADING_DAYS) / d.std() * np.sqrt(TRADING_DAYS)
        ui, mdd = ulcer_index_from_curve(curve)
        exp_days = int(chunk["pos"].sum())
        total_days = int(len(chunk))
        exposure = exp_days / max(total_days, 1)
        pos_shift = chunk["pos"].shift(1).fillna(0)
        entries = ((chunk["pos"] == 1) & (pos_shift == 0)).sum()
        exits = ((chunk["pos"] == 0) & (pos_shift == 1)).sum()
        lengths = []
        run_len = 0
        for v in chunk["pos"].values:
            if v == 1:
                run_len += 1
            elif run_len > 0:
                lengths.append(run_len)
                run_len = 0
        if run_len > 0:
            lengths.append(run_len)
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        med_len = float(np.median(lengths)) if lengths else 0.0

        rows.append({
            "scenario": run_name,
            "strategy": strat,
            "year": int(y),
            "cal_year_return": float(curve.iloc[-1] - 1.0),
            "ann_return": float(ann),
            "ann_vol": float(vol),
            "sharpe": float(sharpe),
            "max_dd": float(mdd),
            "ulcer_index": float(ui),
            "exposure": float(exposure),
            "entries": int(entries),
            "exits": int(exits),
            "avg_trade_days": float(avg_len),
            "median_trade_days": float(med_len),
        })
    return pd.DataFrame(rows)


def run_scenario(
    run_name: str,
    start: str,
    end: str,
    sig_px_full: pd.Series,
    tr_px_full: pd.Series,
    proxy_rets_full: pd.Series | None,
    sma_windows: list[int],
    entry_cfg: dict,
    include_monthly_10: bool,
    apply_entry_to_m10: bool,
    cash_rate_decimal: float,
    sharpe_rf_decimal: float,
    max_dd_decimal: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    DAILY_CASH = cash_rate_decimal / TRADING_DAYS

    asset_rets_full = tr_px_full.pct_change().dropna()
    window_mask = (asset_rets_full.index >= pd.to_datetime(start)) & (asset_rets_full.index <= pd.to_datetime(end))
    asset_rets = asset_rets_full.loc[window_mask]
    overall = {"BuyAndHold": metrics_from_returns(asset_rets, sharpe_rf_decimal)}
    yearly_rows = []

    for w in sorted(sma_windows):
        daily_full, pos_full = build_daily_strategy_returns(
            sig_px_full, tr_px_full, w,
            cash_rate_decimal=cash_rate_decimal,
            proxy_rets=proxy_rets_full,
            entry_cfg=entry_cfg,
        )
        if max_dd_decimal > 0.0:
            daily_full = apply_max_dd_overlay(daily_full, DAILY_CASH, max_dd_decimal)
        daily = daily_full.loc[(daily_full.index >= pd.to_datetime(start)) & (daily_full.index <= pd.to_datetime(end))]
        pos = pos_full.loc[daily.index]
        overall[f"SMA{w}"] = metrics_from_returns(daily, sharpe_rf_decimal)
        yearly_rows.append(per_year_stats(daily, pos, sharpe_rf_decimal, f"SMA{w}", run_name))

    if include_monthly_10:
        daily_full, pos_daily_full, _ = to_monthly_signals_with_entry(
            sig_px_full, tr_px_full, months=10, entry_cfg=entry_cfg, apply_entry=apply_entry_to_m10
        )
        if max_dd_decimal > 0.0:
            daily_full = apply_max_dd_overlay(daily_full, DAILY_CASH, max_dd_decimal)
        daily = daily_full.loc[(daily_full.index >= pd.to_datetime(start)) & (daily_full.index <= pd.to_datetime(end))]
        pos = pos_daily_full.loc[daily.index]
        overall["M10"] = metrics_from_returns(daily, sharpe_rf_decimal)
        yearly_rows.append(per_year_stats(daily, pos, sharpe_rf_decimal, "M10", run_name))

    bh_pos = pd.Series(1, index=asset_rets.index)
    yearly_rows.append(per_year_stats(asset_rets, bh_pos, sharpe_rf_decimal, "BuyAndHold", run_name))

    overall_df = pd.DataFrame(overall).T
    overall_df = overall_df[["CAGR", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "UlcerIndex", "TotalReturn"]]

    yearly_df = pd.concat(yearly_rows, ignore_index=True)
    return overall_df, yearly_df


def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    ticker = cfg["ticker"].strip()
    cash_proxy = str(cfg.get("cash_proxy", "NONE")).strip()

    cash_rate_percent = float(cfg.get("cash_rate_percent", cfg.get("risk_free_rate", 0.0)))
    sharpe_rf_percent = float(cfg.get("sharpe_rf_percent", 0.0))
    max_dd_percent = float(cfg.get("max_drawdown_percent", 0.0))

    entry_cfg = cfg.get("entry", {}) or {}
    apply_entry_to_m10 = bool(cfg.get("apply_entry_to_m10", False))

    CASH_RATE = cash_rate_percent / 100.0
    SHARPE_RF = sharpe_rf_percent / 100.0
    MAX_DD = max_dd_percent / 100.0

    base_start = cfg.get("start")
    base_end = cfg.get("end")
    years = int(cfg.get("years", 10))
    sma_windows = list(cfg["sma_windows"])
    outdir = cfg.get("outdir", "outputs")
    include_monthly_10 = bool(cfg.get("include_monthly_10", True))

    if isinstance(base_start, date):
        base_start = base_start.strftime("%Y-%m-%d")
    if isinstance(base_end, date):
        base_end = base_end.strftime("%Y-%m-%d")

    if base_end is None:
        base_end = datetime.today().strftime("%Y-%m-%d")
    if base_start is None:
        end_dt = datetime.strptime(base_end, "%Y-%m-%d")
        base_start = (end_dt - timedelta(days=365 * years + 365)).strftime("%Y-%m-%d")

    runs = cfg.get("runs", None)
    if not runs:
        runs = [{"name": "default", "start": base_start, "end": base_end}]
    else:
        for r in runs:
            if "name" not in r:
                r["name"] = f"run_{r.get('start','')}_{r.get('end','')}"
            if "start" not in r:
                r["start"] = base_start
            if "end" not in r:
                r["end"] = base_end

    # normalize any date objects in runs to YYYY-MM-DD strings
    def _to_datestr(x):
        if isinstance(x, date):
            return x.strftime("%Y-%m-%d")
        return str(x)

    for r in runs:
        r["start"] = _to_datestr(r["start"])
        r["end"] = _to_datestr(r["end"])

    # compute common warmup fetch window
    max_w = max(sma_windows) if sma_windows else 0
    earliest_start_str = min(r["start"] for r in runs)
    ext_start_dt = datetime.strptime(earliest_start_str, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)
    ext_end = max(r["end"] for r in runs)
    ext_start = ext_start_dt.strftime("%Y-%m-%d")

    sig_px_full, tr_px_full = fetch_series(ticker, ext_start, ext_end)

    proxy_rets_full = None
    if cash_proxy != "NONE":
        _, proxy_tr_full = fetch_series(cash_proxy, ext_start, ext_end)
        proxy_rets_full = proxy_tr_full.pct_change().dropna()

    os.makedirs(outdir, exist_ok=True)

    all_overall = []
    all_yearly = []

    for r in runs:
        name = r["name"]
        start = r["start"]
        end = r["end"]
        overall_df, yearly_df = run_scenario(
            run_name=name,
            start=start,
            end=end,
            sig_px_full=sig_px_full,
            tr_px_full=tr_px_full,
            proxy_rets_full=proxy_rets_full,
            sma_windows=sma_windows,
            entry_cfg=entry_cfg,
            include_monthly_10=include_monthly_10,
            apply_entry_to_m10=apply_entry_to_m10,
            cash_rate_decimal=CASH_RATE,
            sharpe_rf_decimal=SHARPE_RF,
            max_dd_decimal=MAX_DD,
        )
        overall_path = os.path.join(outdir, f"overall_{name}.csv")
        yearly_path = os.path.join(outdir, f"per_year_{name}.csv")
        overall_df.to_csv(overall_path)
        yearly_df.to_csv(yearly_path, index=False)
        print(f"Saved {overall_path} and {yearly_path}")
        all_overall.append(overall_df.assign(scenario=name))
        all_yearly.append(yearly_df)

    combo_overall = pd.concat(all_overall, axis=0)
    combo_overall.to_csv(os.path.join(outdir, "overall_all_scenarios.csv"))

    combo_yearly = pd.concat(all_yearly, axis=0)
    combo_yearly.to_csv(os.path.join(outdir, "per_year_all_scenarios.csv"), index=False)

    print("Overall stats, last scenario printed below")
    print(all_overall[-1].drop(columns=["scenario"]).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
