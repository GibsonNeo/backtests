#!/usr/bin/env python3
import os
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
        return dict(
            CAGR=0.0,
            AnnReturn=0.0,
            AnnVol=0.0,
            Sharpe=0.0,
            Sharpe_noRF=0.0,
            MaxDD=0.0,
            UlcerIndex=0.0,
            TotalMultiple=1.0,
            TotalReturn=0.0,
        )
    curve = (1.0 + d).cumprod()
    years = len(d) / TRADING_DAYS
    total_multiple = float(curve.iloc[-1])
    total_return = total_multiple - 1.0
    mu, sigma = float(d.mean()), float(d.std())
    ann_return = mu * TRADING_DAYS
    ann_vol = sigma * np.sqrt(TRADING_DAYS)
    if sigma > 0:
        sharpe = (mu - sharpe_rf_decimal / TRADING_DAYS) / sigma * np.sqrt(TRADING_DAYS)
        sharpe_norf = mu / sigma * np.sqrt(TRADING_DAYS)
    else:
        sharpe = 0.0
        sharpe_norf = 0.0
    ui, maxdd = ulcer_index_from_curve(curve)
    return dict(
        CAGR=(total_multiple ** (1.0 / max(years, 1e-9)) - 1.0),
        AnnReturn=ann_return,
        AnnVol=ann_vol,
        Sharpe=sharpe,
        Sharpe_noRF=sharpe_norf,
        MaxDD=maxdd,
        UlcerIndex=ui,
        TotalMultiple=total_multiple,
        TotalReturn=total_return,
    )

# ---------------- data ----------------
def _get_col(df, col_name, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_series(ticker, start, end_inclusive):
    """Fetch price data, create split adjusted signal series and total return series."""
    end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    px = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=False, actions=True)
    if px.empty:
        raise ValueError(f"No data for {ticker}")

    close = _get_col(px, "Close", ticker).astype(float)
    splits = _get_col(px, "Stock Splits", ticker).fillna(0.0).astype(float)
    split_factor = splits.replace(0.0, 1.0)

    # Proper backward adjustment so pre split prices are adjusted downward
    split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    price_signal = close * split_adj  # used for SMA based signals

    # Total return adjusted prices, dividends and splits
    tr_df = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    price_tr = _get_col(tr_df, "Close", ticker).rename("AdjClose").astype(float)

    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]


# ---------------- sleeve logic ----------------
def _rolling_sma(x, w):
    return x.rolling(window=w, min_periods=w).mean()


def _consec_true(mask, n):
    # n equals required consecutive True days
    # 1 means first True triggers, 2 means two consecutive True days, and so on
    if n <= 0:
        # forbid old zero semantics
        return pd.Series(False, index=mask.index)
    cnt = 0
    out = []
    for v in mask.values:
        cnt = cnt + 1 if bool(v) else 0
        out.append(cnt >= n)
    return pd.Series(out, index=mask.index)


def sleeve_position(price_signal, window, entry_days, exit_days):
    """
    Entry, require entry_days consecutive closes above SMA(window)
    Exit, require exit_days consecutive closes at or below SMA(window)
    Signals trade in the next session, positions are shifted by one day
    No trades before SMA forms
    """
    sma = _rolling_sma(price_signal, window)
    above = price_signal > sma
    entry_ok = _consec_true(above, entry_days)
    exit_ok = _consec_true(~above, exit_days)

    idx = price_signal.index
    pos = pd.Series(0, index=idx, dtype=int)
    in_pos = 0
    for i in range(len(idx)):
        if in_pos == 1 and exit_ok.iat[i]:
            in_pos = 0
        elif in_pos == 0 and entry_ok.iat[i]:
            in_pos = 1
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0).astype(int)
    first_valid = sma.dropna().index.min()
    if first_valid is not None:
        pos = pos[pos.index >= first_valid]
    return pos


def hybrid_position(price_signal, tier_windows, entry_days_by_window, exit_days_by_window, default_entry_days=3):
    sleeves = []
    for w in tier_windows:
        e_days = int(entry_days_by_window.get(str(w), default_entry_days))
        x_days = int(exit_days_by_window.get(str(w), 1))  # default 1, first close below exits next session
        p = sleeve_position(price_signal, window=w, entry_days=e_days, exit_days=x_days)
        sleeves.append(p.reindex(price_signal.index).fillna(0))
    pos = sum(sleeves) / len(sleeves)
    return pos.clip(0.0, 1.0)

# ---------------- helpers ----------------
def entries_exits_from_pos(pos: pd.Series, threshold: float = 0.5):
    b = (pos >= threshold).astype(int)
    bs = b.shift(1).fillna(0).astype(int)
    entries = int(((b == 1) & (bs == 0)).sum())
    exits = int(((b == 0) & (bs == 1)).sum())
    return entries, exits


def _daily_from_pos(pos: pd.Series, tr_px: pd.Series, daily_cash: float) -> pd.Series:
    asset_rets = tr_px.pct_change().fillna(0.0)
    pos = pos.reindex(asset_rets.index).fillna(0.0)
    return pos * asset_rets + (1 - pos) * daily_cash


def _align_intersection(series_dict: dict, fill_value: float):
    """
    Align by intersection and fill any residual missing with provided fill value.
    For strategy daily returns, pass daily_cash. For positions, pass 0.0.
    """
    common_index = None
    for s in series_dict.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    common_index = common_index.sort_values()
    out = {}
    for k, s in series_dict.items():
        out[k] = s.reindex(common_index).fillna(fill_value)
    return out, common_index

# ---------------- per year optional ----------------
def per_year_stats(daily: pd.Series, pos: pd.Series, sharpe_rf_decimal: float, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"ret": daily, "pos": pos})
    df["year"] = df.index.year
    rows = []
    for y, chunk in df.groupby("year"):
        d = chunk["ret"].dropna()
        curve = (1.0 + d).cumprod()
        mu, sigma = float(d.mean()), float(d.std())
        ann = float(mu * TRADING_DAYS)
        vol = float(sigma * np.sqrt(TRADING_DAYS))
        if sigma > 0:
            sharpe = float((mu - sharpe_rf_decimal / TRADING_DAYS) / sigma * np.sqrt(TRADING_DAYS))
        else:
            sharpe = 0.0
        ui, mdd = ulcer_index_from_curve(curve)
        exposure = float(chunk["pos"].mean())
        entries, exits = entries_exits_from_pos(chunk["pos"])
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

    # universal default for entry days
    default_entry_days = int(cfg.get("default_entry_days", 3))
    entry_tag = f"entrydays{default_entry_days}"

    tickers = cfg.get("tickers", ["QQQ", "SPY"])
    weights = cfg.get("weights", {"QQQ": 0.70, "SPY": 0.30})
    start, end = cfg.get("start"), cfg.get("end")

    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, date):
        end = end.strftime("%Y-%m-%d")

    # Normalize and validate weights
    missing = [t for t in tickers if t not in weights]
    if missing:
        eq = 1.0 / len(tickers)
        for t in missing:
            weights[t] = eq
    total_w = sum(weights[t] for t in tickers)
    if total_w <= 0:
        raise ValueError("Weights sum must be positive")
    weights = {t: weights[t] / total_w for t in tickers}

    outdir = cfg.get("outdir", "outputs_exit_grid")
    cash_rate = cfg.get("cash_rate_percent", 0.0) / 100.0
    sharpe_rf = cfg.get("sharpe_rf_percent", 0.0) / 100.0

    tier_sets = cfg.get("strategy_tier_sets") or [cfg.get("tier_windows", [100, 200])]
    entry_days_by_window = cfg.get("entry_days_by_window", {"100": 3, "200": 3, "221": 3})

    # require explicit exit variants in the config for hybrid sets
    exit_variants_by_set = cfg.get("exit_variants_by_set")
    if not exit_variants_by_set:
        raise ValueError(
            "exit_variants_by_set is required in config.yml. "
            "Provide exit settings as integers of at least 1. "
            'Example, exit_variants_by_set: {"[100, 200]": [{"100": 1, "200": 2}]}.'
        )

    # single SMA sleeves now come from config
    single_sma_specs = cfg.get("single_sma_strategies", [])
    # validate single SMA specs
    for i, spec in enumerate(single_sma_specs):
        if "window" not in spec:
            raise ValueError(f"single_sma_strategies item {i} missing window")
        if "exit_days" not in spec:
            raise ValueError(f"single_sma_strategies item {i} missing exit_days")
        if int(spec["exit_days"]) < 1:
            raise ValueError(f"single_sma_strategies item {i} exit_days must be at least 1")

    write_per_year = bool(cfg.get("write_per_year", False))
    include_baseline_buyhold = bool(cfg.get("include_baseline_buyhold", True))
    daily_cash = cash_rate / TRADING_DAYS
    os.makedirs(outdir, exist_ok=True)

    # Fetch data, extend back so SMAs warm up
    max_w = max(max(s) for s in tier_sets)
    ext_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w * 3 + 365)).strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        sig, tr = fetch_series(t, ext_start, end)
        sig = sig[sig.index >= pd.to_datetime(start)]
        tr = tr[tr.index >= pd.to_datetime(start)]
        data[t] = dict(sig=sig, tr=tr)

    # Random sampling windows, or single full span when disabled
    rand_cfg = cfg.get("random_sampling", {})
    if rand_cfg.get("enabled", False):
        num_samples = int(rand_cfg.get("num_samples", 20))
        min_years = int(rand_cfg.get("min_years", 5))
        max_years = int(rand_cfg.get("max_years", 10))
        seed_val = rand_cfg.get("random_seed", None)
        rng = np.random.default_rng(int(seed_val)) if seed_val is not None else np.random.default_rng()

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_days = pd.bdate_range(start_dt, end_dt)

        min_span_days = int(min_years * 365)
        if len(all_days) <= TRADING_DAYS * min_years:
            raise ValueError("History is too short for requested random sampling parameters")

        sample_windows = []
        for _ in range(num_samples):
            valid_end = len(all_days) - TRADING_DAYS * min_years
            s = rng.choice(all_days[:valid_end])
            s = pd.Timestamp(s)

            if min_years == max_years:
                window_days = int(min_years * 365)
            else:
                window_days = int(rng.integers(min_years * 365, max_years * 365))

            e = s + timedelta(days=window_days)
            if e > end_dt:
                s = end_dt - timedelta(days=window_days)
                e = end_dt

            if (e - s).days >= min_span_days:
                sample_windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    else:
        # Disabled sampling uses one full range window
        sample_windows = [(start, end)]

    results = []            # blended portfolio rows
    results_per_ticker = [] # per ticker rows
    per_year_rows = []

    # ---- main loop ----
    for (run_start, run_end) in sample_windows:
        print(f"\nRunning window {run_start} to {run_end}")

        # require exit variants per tier set from config
        for tier_windows in tier_sets:
            key = str(sorted(tier_windows))
            if key not in exit_variants_by_set:
                raise ValueError(
                    f"No exit variants provided for tier set {tier_windows}. "
                    f"Add an entry under exit_variants_by_set for key {key}."
                )
            exit_variants = exit_variants_by_set[key]

            # validate exits are integers and at least 1
            for ex in exit_variants:
                for w in tier_windows:
                    val = int(ex.get(str(w), 1))
                    if val < 1:
                        raise ValueError(
                            f"Exit setting for window {w} must be at least 1, got {val}. "
                            "Use 1 for first close below SMA then exit next session, "
                            "use 2 for two consecutive closes below SMA then exit next session."
                        )

            for ex in exit_variants:
                label = "set_" + str(sorted(tier_windows)) + "__x" + "_".join([f"{w}_{int(ex[str(w)])}" for w in tier_windows])
                hybrid_daily_all, hybrid_pos_all = {}, {}

                for t in tickers:
                    sig_px = data[t]["sig"].loc[run_start:run_end]
                    tr_px = data[t]["tr"].loc[run_start:run_end]
                    pos = hybrid_position(sig_px, tier_windows, entry_days_by_window, ex, default_entry_days=default_entry_days)
                    daily = _daily_from_pos(pos, tr_px, daily_cash)
                    hybrid_daily_all[t], hybrid_pos_all[t] = daily, pos

                # align by intersection and treat any residual missing returns as cash
                hybrid_daily_all, common_index = _align_intersection(hybrid_daily_all, daily_cash)
                for t in tickers:
                    hybrid_pos_all[t] = hybrid_pos_all[t].reindex(common_index).fillna(0.0)

                hybrid_daily = sum(weights[t] * hybrid_daily_all[t] for t in tickers)
                m = metrics_from_returns(hybrid_daily, sharpe_rf)
                m["variant"] = label
                m["window"] = f"{run_start}_{run_end}"
                results.append(m)

                # per ticker metrics for the same hybrid variant
                for tt in tickers:
                    m_t = metrics_from_returns(hybrid_daily_all[tt], sharpe_rf)
                    m_t["variant"] = label
                    m_t["window"] = f"{run_start}_{run_end}"
                    m_t["ticker"] = tt
                    results_per_ticker.append(m_t)

                if write_per_year:
                    blended_pos = sum(weights[t] * hybrid_pos_all[t] for t in tickers)
                    per_year_rows.append(per_year_stats(hybrid_daily, blended_pos, sharpe_rf, label))

        # Single SMA sleeves, read from config, now respect universal default
        for spec in single_sma_specs:
            win = int(spec["window"])
            e_in = int(spec.get("entry_days", entry_days_by_window.get(str(win), default_entry_days)))
            x_out = int(spec["exit_days"])
            if x_out < 1:
                raise ValueError(f"Single SMA exit_days must be at least 1 for window {win}")
            name = spec.get("name", f"sma{win}_{e_in}in_{x_out}out")

            daily_mix = []
            per_ticker_daily_cache = {}  # reuse for per ticker rows
            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px = data[t]["tr"].loc[run_start:run_end]
                pos = sleeve_position(sig_px, window=win, entry_days=e_in, exit_days=x_out)
                daily = _daily_from_pos(pos, tr_px, daily_cash)
                per_ticker_daily_cache[t] = daily
                daily_mix.append(weights[t] * daily)

            tmp = {str(i): s for i, s in enumerate(daily_mix)}
            tmp_aligned, _ = _align_intersection(tmp, daily_cash)
            daily_sum = sum(tmp_aligned[k] for k in tmp_aligned.keys())

            m = metrics_from_returns(daily_sum, sharpe_rf)
            m["variant"] = name
            m["window"] = f"{run_start}_{run_end}"
            results.append(m)

            # per ticker metrics for this single SMA variant
            for tt in tickers:
                m_t = metrics_from_returns(per_ticker_daily_cache[tt], sharpe_rf)
                m_t["variant"] = name
                m_t["window"] = f"{run_start}_{run_end}"
                m_t["ticker"] = tt
                results_per_ticker.append(m_t)

        # Baseline buy and hold
        if include_baseline_buyhold:
            bh_daily_all = {}
            for t in tickers:
                rets = data[t]["tr"].loc[run_start:run_end].pct_change().fillna(0.0)
                bh_daily_all[t] = rets

            bh_daily_all, bh_index = _align_intersection(bh_daily_all, 0.0)
            bh = sum(weights[t] * bh_daily_all[t] for t in tickers)

            m = metrics_from_returns(bh, sharpe_rf)
            m["variant"] = "baseline_buyhold"
            m["window"] = f"{run_start}_{run_end}"
            results.append(m)

            # per ticker baseline rows
            for tt in tickers:
                m_b = metrics_from_returns(bh_daily_all[tt], sharpe_rf)
                m_b["variant"] = "baseline_buyhold"
                m_b["window"] = f"{run_start}_{run_end}"
                m_b["ticker"] = tt
                results_per_ticker.append(m_b)

            if write_per_year:
                per_year_rows.append(per_year_stats(bh, pd.Series(1.0, index=bh.index), sharpe_rf, "baseline_buyhold"))

    # Collate results
    overall = pd.DataFrame(results)                 # blended portfolio
    overall_t = pd.DataFrame(results_per_ticker)    # per ticker

    # keep useful columns
    keep_cols = [
        "window", "variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
        "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
    ]
    overall = overall[keep_cols]
    overall.to_csv(os.path.join(outdir, "random_runs_summary.csv"), index=False)

    if write_per_year and per_year_rows:
        yr = pd.concat(per_year_rows, ignore_index=True)
        yr.to_csv(os.path.join(outdir, "per_year_random.csv"), index=False)

    # Aggregation and wins logic for blended portfolio
    is_baseline = overall["variant"].str.contains("baseline", case=False, regex=False)
    strategies = overall[~is_baseline].copy()
    baselines = overall[is_baseline].copy()

    metrics_cols = [
        "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
        "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
    ]

    # Average per variant
    avg_by_variant = (
        strategies.groupby("variant", as_index=False)[metrics_cols]
        .mean()
        .sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    )

    # Baseline metrics by window, for gating
    baseline_sharpe_by_window = baselines.set_index("window")["Sharpe"].to_dict()
    baseline_cagr_by_window = baselines.set_index("window")["CAGR"].to_dict()

    # Award one point whenever a strategy beats the baseline for that window
    total_windows = len(overall["window"].unique())
    strategies = strategies.assign(
        sharpe_beats_baseline = strategies.apply(lambda r: float(r["Sharpe"] > baseline_sharpe_by_window.get(r["window"], np.nan)), axis=1),
        cagr_beats_baseline   = strategies.apply(lambda r: float(r["CAGR"]  > baseline_cagr_by_window.get(r["window"], np.nan)), axis=1),
    )
    sharpe_wins = (
        strategies.groupby("variant")["sharpe_beats_baseline"]
        .sum()
        .rename("sharpe_wins")
        .reset_index()
    )
    cagr_wins = (
        strategies.groupby("variant")["cagr_beats_baseline"]
        .sum()
        .rename("cagr_wins")
        .reset_index()
    )

    # Merge wins into averages
    avg_plus_wins = avg_by_variant.merge(sharpe_wins, on="variant", how="left")
    avg_plus_wins = avg_plus_wins.merge(cagr_wins, on="variant", how="left")
    avg_plus_wins = avg_plus_wins.fillna({"sharpe_wins": 0.0, "cagr_wins": 0.0})
    avg_plus_wins["sharpe_win_share"] = avg_plus_wins["sharpe_wins"] / total_windows
    avg_plus_wins["cagr_win_share"] = avg_plus_wins["cagr_wins"] / total_windows

    # Diagnostics versus baseline, per variant
    sdf = overall.copy()
    is_base_diag = sdf["variant"].str.contains("baseline", case=False, regex=False)
    strat_df = sdf[~is_base_diag].copy()
    base_df = sdf[is_base_diag].copy()

    base_sharpe = base_df.set_index("window")["Sharpe"].to_dict()
    base_cagr = base_df.set_index("window")["CAGR"].to_dict()

    def diag(metric, base_map):
        rows = []
        for v, g in strat_df.groupby("variant"):
            beats = (g[metric] > g["window"].map(base_map))
            rows.append({
                "variant": v,
                f"beats_baseline_{metric}_count": int(beats.sum()),
                f"beats_baseline_{metric}_share": float(beats.mean()),
                f"mean_excess_{metric}_over_baseline": float((g[metric] - g["window"].map(base_map)).mean()),
            })
        return pd.DataFrame(rows)

    diag_df = diag("Sharpe", base_sharpe).merge(diag("CAGR", base_cagr), on="variant")
    avg_plus_wins = avg_plus_wins.merge(diag_df, on="variant", how="left")

    # Baseline averages for reference
    baseline_avg = baselines.groupby("variant", as_index=False)[metrics_cols].mean()

    combined = pd.concat(
        [
            avg_plus_wins.assign(group="strategy"),
            baseline_avg.assign(group="baseline"),
        ],
        ignore_index=True,
        sort=False,
    )

    # Filename prefix, tickers, years label, samples label
    rand_cfg = cfg.get("random_sampling", {})
    ticker_label = "+".join(tickers)
    if rand_cfg.get("enabled", False):
        min_years = int(rand_cfg.get("min_years", 0))
        max_years = int(rand_cfg.get("max_years", 0))
        if min_years > 0 and max_years > 0:
            years_label = f"{min_years}yr" if min_years == max_years else f"{min_years}to{max_years}yr"
        else:
            years_label = "win"
        samples_label = f"{len(sample_windows)}samples"
    else:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        span_years = max(1, int(round((end_dt - start_dt).days / 365.25)))
        years_label = f"{span_years}yr"
        samples_label = f"{len(sample_windows)}samples"

    prefix = f"{ticker_label}-{years_label}-{samples_label}_{entry_tag}_"

    # Save blended outputs with prefix
    avg_path = os.path.join(outdir, f"{prefix}random_runs_summary_averages.csv")
    sum_path = os.path.join(outdir, f"{prefix}random_runs_summary.csv")
    combined.to_csv(avg_path, index=False)
    overall.to_csv(sum_path, index=False)

    # Also write a random like averages file, strategies only, same columns as your random runs summary
    random_like_cols = [
        "variant",
        "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
        "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn",
        "sharpe_wins", "cagr_wins", "sharpe_win_share", "cagr_win_share",
        "beats_baseline_Sharpe_count", "beats_baseline_Sharpe_share", "mean_excess_Sharpe_over_baseline",
        "beats_baseline_CAGR_count", "beats_baseline_CAGR_share", "mean_excess_CAGR_over_baseline",
    ]
    random_like = avg_plus_wins.copy()
    random_like = random_like[[c for c in random_like_cols if c in random_like.columns]]
    random_like_path = os.path.join(outdir, f"{prefix}random_like_averages.csv")
    random_like.to_csv(random_like_path, index=False)

    print(f"Saved {avg_path}")
    print(f"Saved {sum_path}")
    print(f"Saved {random_like_path}")

    # -------- per ticker summaries and rollup --------
    if not overall_t.empty:
        # keep useful columns, add ticker
        keep_cols_t = [
            "window", "ticker", "variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
            "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
        ]
        overall_t = overall_t[keep_cols_t]
        per_ticker_sum_path = os.path.join(outdir, f"{prefix}per_ticker_random_runs_summary.csv")
        overall_t.to_csv(per_ticker_sum_path, index=False)

        # split strategies and baselines per ticker
        is_base_t = overall_t["variant"].str.contains("baseline", case=False, regex=False)
        strategies_t = overall_t[~is_base_t].copy()
        baselines_t = overall_t[is_base_t].copy()

        # baseline per ticker per window maps
        base_sharpe_map = baselines_t.set_index(["ticker", "window"])["Sharpe"].to_dict()
        base_cagr_map   = baselines_t.set_index(["ticker", "window"])["CAGR"].to_dict()

        # wins per ticker
        total_windows_t = len(overall_t["window"].unique())
        strategies_t = strategies_t.assign(
            sharpe_beats_baseline = strategies_t.apply(
                lambda r: float(r["Sharpe"] > base_sharpe_map.get((r["ticker"], r["window"]), np.nan)), axis=1
            ),
            cagr_beats_baseline   = strategies_t.apply(
                lambda r: float(r["CAGR"]  > base_cagr_map.get((r["ticker"], r["window"]), np.nan)), axis=1
            ),
        )

        # average by ticker and variant
        avg_by_ticker = (
            strategies_t.groupby(["ticker", "variant"], as_index=False)[metrics_cols]
            .mean()
        )
        sharpe_wins_t = strategies_t.groupby(["ticker", "variant"])["sharpe_beats_baseline"].sum().rename("sharpe_wins").reset_index()
        cagr_wins_t   = strategies_t.groupby(["ticker", "variant"])["cagr_beats_baseline"].sum().rename("cagr_wins").reset_index()

        per_ticker_avg = avg_by_ticker.merge(sharpe_wins_t, on=["ticker", "variant"], how="left")
        per_ticker_avg = per_ticker_avg.merge(cagr_wins_t, on=["ticker", "variant"], how="left")
        per_ticker_avg = per_ticker_avg.fillna({"sharpe_wins": 0.0, "cagr_wins": 0.0})
        per_ticker_avg["sharpe_win_share"] = per_ticker_avg["sharpe_wins"] / total_windows_t
        per_ticker_avg["cagr_win_share"]   = per_ticker_avg["cagr_wins"]   / total_windows_t

        per_ticker_avg_path = os.path.join(outdir, f"{prefix}per_ticker_random_runs_summary_averages.csv")
        per_ticker_avg.to_csv(per_ticker_avg_path, index=False)

        # rollup across tickers, equal weight across tickers
        rollup_metrics = (
            per_ticker_avg.groupby("variant", as_index=False)[metrics_cols]
            .mean()
        )
        rollup_extra = (
            per_ticker_avg.groupby("variant", as_index=False)[
                ["sharpe_win_share", "cagr_win_share"]
            ].mean()
        )
        rollup = rollup_metrics.merge(rollup_extra, on="variant", how="left")

        # baseline rollup for reference
        base_avg_t = (
            baselines_t.groupby(["ticker", "variant"], as_index=False)[metrics_cols]
            .mean()
        )
        base_rollup = base_avg_t.groupby("variant", as_index=False)[metrics_cols].mean()

        rollup_combined = pd.concat(
            [
                rollup.assign(group="strategy_rollup"),
                base_rollup.assign(group="baseline_rollup"),
            ],
            ignore_index=True,
            sort=False,
        )

        rollup_avg_path = os.path.join(outdir, f"{prefix}rollup_across_tickers_averages.csv")
        rollup_combined.to_csv(rollup_avg_path, index=False)

        print(f"Saved {per_ticker_sum_path}")
        print(f"Saved {per_ticker_avg_path}")
        print(f"Saved {rollup_avg_path}")

    if write_per_year and per_year_rows:
        yr = pd.concat(per_year_rows, ignore_index=True)
        per_year_path = os.path.join(outdir, f"{prefix}per_year_random.csv")
        yr.to_csv(per_year_path, index=False)
        print(f"Saved {per_year_path}")


if __name__ == "__main__":
    main()
