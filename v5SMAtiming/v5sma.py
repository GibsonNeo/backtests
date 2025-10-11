import os
import itertools
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

TRADING_DAYS = 252

# ---------------- metrics ----------------
def ulcer_index_from_curve(curve: pd.Series):
    peak = curve.cummax()
    dd = (curve - peak) / peak
    ui = float(np.sqrt((dd.pow(2)).mean()) * 100.0)
    maxdd = float(dd.min())
    return ui, maxdd

def metrics_from_returns(daily_rets: pd.Series, sharpe_rf_decimal: float = 0.0):
    d = daily_rets.dropna()
    if d.empty:
        return dict(CAGR=0.0, AnnReturn=0.0, AnnVol=0.0, Sharpe=0.0,
                    MaxDD=0.0, UlcerIndex=0.0, TotalReturn=1.0)
    curve = (1.0 + d).cumprod()
    years = len(d) / TRADING_DAYS
    total_return = float(curve.iloc[-1])
    cagr = total_return ** (1.0 / max(years, 1e-9)) - 1.0
    mu, sigma = float(d.mean()), float(d.std())
    ann_return = mu * TRADING_DAYS
    ann_vol = sigma * np.sqrt(TRADING_DAYS)
    sharpe = (mu - sharpe_rf_decimal / TRADING_DAYS) / sigma * np.sqrt(TRADING_DAYS) if sigma > 0 else 0.0
    ui, maxdd = ulcer_index_from_curve(curve)
    return dict(CAGR=cagr, AnnReturn=ann_return, AnnVol=ann_vol,
                Sharpe=sharpe, MaxDD=maxdd, UlcerIndex=ui,
                TotalReturn=total_return)

# ---------------- data ----------------
def _get_col(df, col_name, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]

def fetch_series(ticker, start, end_inclusive):
    end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
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

# ---------------- sleeve logic ----------------
def _rolling_sma(x, w):
    return x.rolling(window=w, min_periods=w).mean()

def _consec_true(mask, n):
    if n <= 0:
        return pd.Series(True, index=mask.index)
    cnt = 0
    out = []
    for v in mask.values:
        cnt = cnt + 1 if bool(v) else 0
        out.append(cnt >= n)
    return pd.Series(out, index=mask.index)

def sleeve_position(price_signal, window, entry_days, exit_days):
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
    first_valid = sma.dropna().index.min()
    if first_valid is not None:
        pos = pos[pos.index >= first_valid]
    return pos

def hybrid_position(price_signal, tier_windows, entry_days_by_window, exit_days_by_window):
    sleeves = []
    for w in tier_windows:
        e_days = int(entry_days_by_window.get(str(w), 3))
        x_days = int(exit_days_by_window.get(str(w), 0))
        p = sleeve_position(price_signal, window=w, entry_days=e_days, exit_days=x_days)
        sleeves.append(p.reindex(price_signal.index).fillna(0))
    pos = sum(sleeves) / len(sleeves)
    return pos.clip(0.0, 1.0)

# ---------------- per year optional ----------------
def per_year_stats(daily: pd.Series, pos: pd.Series, sharpe_rf_decimal: float, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"ret": daily, "pos": pos})
    df["year"] = df.index.year
    rows = []
    for y, chunk in df.groupby("year"):
        d = chunk["ret"]
        curve = (1.0 + d).cumprod()
        ann = float(d.mean() * TRADING_DAYS)
        vol = float(d.std() * np.sqrt(TRADING_DAYS))
        sharpe = float((d.mean() - sharpe_rf_decimal / TRADING_DAYS) / d.std() * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0
        ui, mdd = ulcer_index_from_curve(curve)
        exp_days = int(chunk["pos"].sum())
        total_days = int(len(chunk))
        exposure = exp_days / max(total_days, 1)
        pos_shift = chunk["pos"].shift(1).fillna(0)
        entries = int(((chunk["pos"] == 1) & (pos_shift == 0)).sum())
        exits = int(((chunk["pos"] == 0) & (pos_shift == 1)).sum())
        rows.append({
            "variant": label,
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

# ---------------- main ----------------
def main():
    with open("config.yml") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["QQQ", "SPY"])
    weights = cfg.get("weights", {"QQQ": 0.70, "SPY": 0.30})
    start, end = cfg.get("start"), cfg.get("end")

    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, date):
        end = end.strftime("%Y-%m-%d")

    outdir = cfg.get("outdir", "outputs_exit_grid")
    cash_rate = cfg.get("cash_rate_percent", 0.0) / 100.0
    sharpe_rf = cfg.get("sharpe_rf_percent", 0.0) / 100.0

    tier_windows = cfg.get("tier_windows", [100, 200, 221])
    entry_days_by_window = cfg.get("entry_days_by_window", {"100": 3, "200": 3, "221": 3})
    auto_binary_exit_grid = bool(cfg.get("auto_binary_exit_grid", True))
    exit_variants = cfg.get("exit_variants")

    write_per_year = bool(cfg.get("write_per_year", False))
    include_baseline_buyhold = bool(cfg.get("include_baseline_buyhold", True))

    daily_cash = cash_rate / TRADING_DAYS
    os.makedirs(outdir, exist_ok=True)

    # Build exit variants dynamically from tier_windows
    if auto_binary_exit_grid:
        vals = [0, 1]
        combos = list(itertools.product(vals, repeat=len(tier_windows)))
        exit_variants = []
        for combo in combos:
            ex_dict = {str(w): combo[i] for i, w in enumerate(tier_windows)}
            exit_variants.append(ex_dict)
    elif not exit_variants:
        raise ValueError("No exit variants provided")

    # Fetch data
    max_w = max(tier_windows + [200])
    ext_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)).strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        sig, tr = fetch_series(t, ext_start, end)
        sig = sig[sig.index >= pd.to_datetime(start)]
        tr = tr[tr.index >= pd.to_datetime(start)]
        data[t] = dict(sig=sig, tr=tr)

    # ---- random sampling windows ----
    rand_cfg = cfg.get("random_sampling", {})
    if rand_cfg.get("enabled", False):
        num_samples = int(rand_cfg.get("num_samples", 20))
        min_years = int(rand_cfg.get("min_years", 5))
        max_years = int(rand_cfg.get("max_years", 10))
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_days = pd.bdate_range(start_dt, end_dt)
        sample_windows = []
        for _ in range(num_samples):
            s = np.random.choice(all_days[:-TRADING_DAYS * min_years])
            if not isinstance(s, pd.Timestamp):
                s = pd.Timestamp(s)
            random_days = int(np.random.randint(min_years * 365, max_years * 365))
            possible_end = s + timedelta(days=random_days)
            e = min(possible_end, end_dt)
            if (e - s).days >= min_years * 365:
                sample_windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    else:
        sample_windows = [(start, end)]

    results = []
    per_year_rows = []

    # ---- main run loop ----
    for (run_start, run_end) in sample_windows:
        print(f"\nRunning window {run_start} → {run_end}")

        for ex in exit_variants:
            # Build a label like: x100_1_x200_0 (or x100_1_x200_0_x221_1 if 221 is present)
            label = "x" + "_".join([f"{w}_{int(ex[str(w)])}" for w in tier_windows])

            hybrid_daily_all, hybrid_pos_all = {}, {}
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = hybrid_position(sig_px, tier_windows, entry_days_by_window, ex)
                asset_rets = tr_px.pct_change().fillna(0.0)
                daily = pos * asset_rets + (1 - pos) * daily_cash
                hybrid_daily_all[t] = daily
                hybrid_pos_all[t] = pos

            common_index = set.intersection(*(set(x.index) for x in hybrid_daily_all.values()))
            common_index = pd.DatetimeIndex(sorted(common_index))
            for t in tickers:
                hybrid_daily_all[t] = hybrid_daily_all[t].reindex(common_index).fillna(0.0)
                hybrid_pos_all[t] = hybrid_pos_all[t].reindex(common_index).fillna(0.0)

            hybrid_daily = sum(weights[t] * hybrid_daily_all[t] for t in tickers)
            m = metrics_from_returns(hybrid_daily, sharpe_rf)
            m["variant"] = label
            m["window"] = f"{run_start}_{run_end}"
            results.append(m)

            if write_per_year:
                blended_pos = sum(weights[t] * hybrid_pos_all[t] for t in tickers)
                per_year_rows.append(per_year_stats(hybrid_daily, blended_pos, sharpe_rf, label))

        # ---- Baselines ----
        if include_baseline_buyhold:
            # Buy & Hold
            bh_daily = []
            for t in tickers:
                rets = data[t]["tr"].loc[run_start:run_end].pct_change().fillna(0.0)
                bh_daily.append(weights[t] * rets)
            bh = sum(bh_daily)
            m = metrics_from_returns(bh, sharpe_rf)
            m["variant"] = "baseline_buyhold"
            m["window"] = f"{run_start}_{run_end}"
            results.append(m)
            if write_per_year:
                per_year_rows.append(per_year_stats(bh, pd.Series(1.0, index=bh.index), sharpe_rf, "baseline_buyhold"))

            # SMA-200 3-in / 1-out
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = sleeve_position(sig_px, window=200, entry_days=3, exit_days=1)
                rets = tr_px.pct_change().fillna(0.0)
                daily = pos * rets + (1 - pos) * daily_cash
                m = metrics_from_returns(daily, sharpe_rf)
                m["variant"] = "baseline_sma200_3in_1out"
                m["window"] = f"{run_start}_{run_end}"
                results.append(m)
                if write_per_year:
                    per_year_rows.append(per_year_stats(daily, pos, sharpe_rf, "baseline_sma200_3in_1out"))

            # SMA-200 3-in / 0-out
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = sleeve_position(sig_px, window=200, entry_days=3, exit_days=0)
                rets = tr_px.pct_change().fillna(0.0)
                daily = pos * rets + (1 - pos) * daily_cash
                m = metrics_from_returns(daily, sharpe_rf)
                m["variant"] = "baseline_sma200_3in_0out"
                m["window"] = f"{run_start}_{run_end}"
                results.append(m)
                if write_per_year:
                    per_year_rows.append(per_year_stats(daily, pos, sharpe_rf, "baseline_sma200_3in_0out"))

            # SMA-100 3-in / 1-out
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = sleeve_position(sig_px, window=100, entry_days=3, exit_days=1)
                rets = tr_px.pct_change().fillna(0.0)
                daily = pos * rets + (1 - pos) * daily_cash
                m = metrics_from_returns(daily, sharpe_rf)
                m["variant"] = "baseline_sma100_3in_1out"
                m["window"] = f"{run_start}_{run_end}"
                results.append(m)
                if write_per_year:
                    per_year_rows.append(per_year_stats(daily, pos, sharpe_rf, "baseline_sma100_3in_1out"))

            # SMA-100 3-in / 0-out
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = sleeve_position(sig_px, window=100, entry_days=3, exit_days=0)
                rets = tr_px.pct_change().fillna(0.0)
                daily = pos * rets + (1 - pos) * daily_cash
                m = metrics_from_returns(daily, sharpe_rf)
                m["variant"] = "baseline_sma100_3in_0out"
                m["window"] = f"{run_start}_{run_end}"
                results.append(m)
                if write_per_year:
                    per_year_rows.append(per_year_stats(daily, pos, sharpe_rf, "baseline_sma100_3in_0out"))

    # ---- write output ----
    overall = pd.DataFrame(results)
    overall = overall[["window", "variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "UlcerIndex", "TotalReturn"]]
    overall.to_csv(os.path.join(outdir, "random_runs_summary.csv"), index=False)

    if write_per_year and per_year_rows:
        yr = pd.concat(per_year_rows, ignore_index=True)
        yr.to_csv(os.path.join(outdir, "per_year_random.csv"), index=False)

    # ---- aggregate summary averages ----
    overall_df = pd.read_csv(os.path.join(outdir, "random_runs_summary.csv"))
    is_baseline = overall_df["variant"].str.contains("baseline", case=False, regex=False)
    strategies = overall_df[~is_baseline].copy()
    baselines = overall_df[is_baseline].copy()

    metrics = ["CAGR","AnnReturn","AnnVol","Sharpe","MaxDD","UlcerIndex","TotalReturn"]

    avg_by_variant = strategies.groupby("variant", as_index=False)[metrics].mean()
    avg_by_variant = avg_by_variant.sort_values(["Sharpe","CAGR"], ascending=[False, False])

    strategies["rank_in_window"] = strategies.groupby("window")["Sharpe"].rank(ascending=False, method="min")
    wins = strategies.loc[strategies["rank_in_window"] == 1.0].groupby("variant").size().rename("wins").reset_index()
    total_windows = max(1, strategies["window"].nunique())
    avg_plus_wins = avg_by_variant.merge(wins, on="variant", how="left").fillna({"wins": 0})
    avg_plus_wins["win_share"] = avg_plus_wins["wins"] / total_windows

    baseline_avg = baselines.groupby("variant", as_index=False)[metrics].mean()

    combined = pd.concat([
        avg_plus_wins.assign(group="strategy"),
        baseline_avg.assign(group="baseline")
    ], ignore_index=True, sort=False)

    avg_path = os.path.join(outdir, "random_runs_summary_averages.csv")
    combined.to_csv(avg_path, index=False)
    print(f"✅ Wrote {avg_path}")

    print(f"\n✅ Wrote {os.path.join(outdir, 'random_runs_summary.csv')}")
    if write_per_year:
        print(f"✅ Wrote {os.path.join(outdir, 'per_year_random.csv')}")

if __name__ == "__main__":
    main()
