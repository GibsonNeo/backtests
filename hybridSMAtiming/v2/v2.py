#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from itertools import product

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

    # backward adjust, so pre split prices are scaled down
    split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    price_signal = close * split_adj  # for SMA signals

    # total return close with dividends and splits
    tr_df = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    price_tr = _get_col(tr_df, "Close", ticker).rename("AdjClose").astype(float)

    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]

# ---------------- SMA helpers ----------------
def _rolling_sma(x, w):
    return x.rolling(window=w, min_periods=w).mean()

def _consec_true(mask, n):
    if n <= 0:
        return pd.Series(False, index=mask.index)
    cnt = 0
    out = []
    for v in mask.values:
        cnt = cnt + 1 if bool(v) else 0
        out.append(cnt >= n)
    return pd.Series(out, index=mask.index)

# ---------------- strategy, hybrid with cooldown ----------------
def hybrid_position_with_cooldown(price_signal: pd.Series,
                                  w_long: int,
                                  w_short: int,
                                  entry_days: int,
                                  exit_days_long: int,
                                  exit_days_short: int,
                                  cooldown_days: int) -> pd.Series:
    """
    Regime A, short SMA at or above long SMA
      entry, price above long SMA for entry_days consecutive closes
      exit, price at or below long SMA for exit_days_long consecutive closes

    Regime B, short SMA below long SMA
      entry, price above short SMA and short SMA rising, both true for entry_days consecutive days
      exit, price at or below short SMA for exit_days_short consecutive closes

    Cooldown, after any exit, block new entries for cooldown_days sessions
    Fills occur next session by the final shift
    """
    smaL = _rolling_sma(price_signal, w_long)
    smaS = _rolling_sma(price_signal, w_short)

    aboveL = price_signal > smaL
    aboveS = price_signal > smaS
    slopeS_up = smaS > smaS.shift(1)

    entry_long_ok = _consec_true(aboveL, entry_days)
    gate_short = aboveS & slopeS_up
    entry_short_ok = _consec_true(gate_short, entry_days)

    exit_long_ok  = _consec_true(~aboveL, exit_days_long)
    exit_short_ok = _consec_true(~aboveS, exit_days_short)

    short_below_long = smaS < smaL

    idx = price_signal.index
    pos = pd.Series(0, index=idx, dtype=int)
    in_pos = 0
    cd = 0  # cooldown counter in sessions

    for i in range(len(idx)):
        use_entry_ok = entry_short_ok.iat[i] if short_below_long.iat[i] else entry_long_ok.iat[i]
        use_exit_ok  = exit_short_ok.iat[i]  if short_below_long.iat[i] else exit_long_ok.iat[i]

        # evaluate exit first
        if in_pos == 1 and use_exit_ok:
            in_pos = 0
            cd = cooldown_days  # start cooldown after exit

        # evaluate entry with cooldown gate
        elif in_pos == 0 and cd == 0 and use_entry_ok:
            in_pos = 1

        pos.iat[i] = in_pos

        # tick down cooldown when flat
        if in_pos == 0 and cd > 0:
            cd -= 1

    pos = pos.shift(1).fillna(0).astype(int)

    fv_long = smaL.dropna().index.min()
    fv_short = smaS.dropna().index.min()
    first_valid_candidates = [x for x in [fv_long, fv_short] if x is not None]
    first_valid = max(first_valid_candidates) if first_valid_candidates else None
    if first_valid is not None:
        pos = pos[pos.index >= first_valid]
    return pos

# ---------------- helpers ----------------
def _daily_from_pos(pos: pd.Series, tr_px: pd.Series, daily_cash: float) -> pd.Series:
    asset_rets = tr_px.pct_change().fillna(0.0)
    pos = pos.reindex(asset_rets.index).fillna(0.0)
    return pos * asset_rets + (1 - pos) * daily_cash

def _align_intersection(series_dict: dict, fill_value: float):
    common_index = None
    for s in series_dict.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    common_index = common_index.sort_values()
    out = {}
    for k, s in series_dict.items():
        out[k] = s.reindex(common_index).fillna(fill_value)
    return out, common_index

# ---------------- main ----------------
def main():
    with open("config.yml") as f:
        cfg = yaml.safe_load(f)

    # labels
    default_entry_days = int(cfg.get("default_entry_days", 3))
    entry_tag = f"entrydays{default_entry_days}"

    tickers = cfg.get("tickers", ["QQQ", "SPY"])
    weights = cfg.get("weights", None) or {}
    start, end = cfg.get("start"), cfg.get("end")

    if isinstance(start, date):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, date):
        end = end.strftime("%Y-%m-%d")

    # normalize weights or equal weight
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
    include_baseline_buyhold = bool(cfg.get("include_baseline_buyhold", True))
    daily_cash = cash_rate / TRADING_DAYS
    os.makedirs(outdir, exist_ok=True)

    # grids and defaults
    defaults = cfg.get("hybrid20_defaults", {}) or {}
    grids = cfg.get("hybrid20_grids", {}) or {}
    exclusions = cfg.get("hybrid20_exclusions", []) or []

    base_long = int(defaults.get("long_window", 200))
    base_short = int(defaults.get("short_window", 20))
    entry_days = int(defaults.get("entry_days", default_entry_days))
    exit_days_long = int(defaults.get("exit_days_long", 2))
    exit_days_short = int(defaults.get("exit_days_short", 1))

    long_list = [int(x) for x in grids.get("long_windows", [base_long])]
    short_list = [int(x) for x in grids.get("short_windows", [base_short])]
    cd_list = [int(x) for x in grids.get("cooldown_days", [0, 1, 2])]

    def excluded(L, S, C):
        for e in exclusions:
            okL = ("long" not in e) or int(e["long"]) == L
            okS = ("short" not in e) or int(e["short"]) == S
            okC = ("cd" not in e) or int(e["cd"]) == C
            if okL and okS and okC:
                return True
        return False

    # fetch data, extend back for SMA warmup
    max_w_for_warmup = max(max(long_list), max(short_list))
    ext_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=max_w_for_warmup * 3 + 365)).strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        sig, tr = fetch_series(t, ext_start, end)
        sig = sig[sig.index >= pd.to_datetime(start)]
        tr = tr[tr.index >= pd.to_datetime(start)]
        data[t] = dict(sig=sig, tr=tr)

    # sample windows
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

            window_days = int(min_years * 365) if min_years == max_years else int(rng.integers(min_years * 365, max_years * 365))
            e = s + timedelta(days=window_days)
            if e > end_dt:
                s = end_dt - timedelta(days=window_days)
                e = end_dt

            if (e - s).days >= min_span_days:
                sample_windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    else:
        sample_windows = [(start, end)]

    # run
    results = []            # blended portfolio rows
    results_per_ticker = [] # per ticker rows

    # generate variant grid
    grid = [(L, S, C) for L, S, C in product(long_list, short_list, cd_list) if not excluded(L, S, C)]

    for (run_start, run_end) in sample_windows:
        print(f"\nRunning window {run_start} to {run_end}")

        # baseline buy and hold, compute once per window
        if include_baseline_buyhold:
            bh_daily_all = {}
            for t in tickers:
                rets = data[t]["tr"].loc[run_start:run_end].pct_change().fillna(0.0)
                bh_daily_all[t] = rets
            bh_daily_all, _ = _align_intersection(bh_daily_all, 0.0)
            bh_blend = sum(weights[t] * bh_daily_all[t] for t in tickers)

            m_bh = metrics_from_returns(bh_blend, sharpe_rf)
            m_bh["variant"] = "baseline_buyhold"
            m_bh["window"]  = f"{run_start}_{run_end}"
            results.append(m_bh)

            for tt in tickers:
                m_tb = metrics_from_returns(bh_daily_all[tt], sharpe_rf)
                m_tb["variant"] = "baseline_buyhold"
                m_tb["window"]  = f"{run_start}_{run_end}"
                m_tb["ticker"]  = tt
                results_per_ticker.append(m_tb)

        # run each variant
        for L, S, C in grid:
            name = f"hybridL{L}_S{S}_cd{C}"

            daily_mix = []
            per_ticker_daily_cache = {}

            for t in tickers:
                sig_px = data[t]["sig"].loc[run_start:run_end]
                tr_px  = data[t]["tr"].loc[run_start:run_end]
                pos = hybrid_position_with_cooldown(
                    sig_px,
                    w_long=L,
                    w_short=S,
                    entry_days=entry_days,
                    exit_days_long=exit_days_long,
                    exit_days_short=exit_days_short,
                    cooldown_days=C
                )
                daily = _daily_from_pos(pos, tr_px, daily_cash)
                per_ticker_daily_cache[t] = daily
                daily_mix.append(weights[t] * daily)

            tmp = {str(i): s for i, s in enumerate(daily_mix)}
            tmp_aligned, _ = _align_intersection(tmp, daily_cash)
            daily_sum = sum(tmp_aligned[k] for k in tmp_aligned.keys())

            m = metrics_from_returns(daily_sum, sharpe_rf)
            m["variant"] = name
            m["window"]  = f"{run_start}_{run_end}"
            results.append(m)

            for tt in tickers:
                m_t = metrics_from_returns(per_ticker_daily_cache[tt], sharpe_rf)
                m_t["variant"] = name
                m_t["window"]  = f"{run_start}_{run_end}"
                m_t["ticker"]  = tt
                results_per_ticker.append(m_t)

    # Collate
    overall = pd.DataFrame(results)
    overall_t = pd.DataFrame(results_per_ticker)

    keep_cols = [
        "window", "variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
        "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
    ]
    overall = overall[keep_cols]

    # split strategies vs baseline
    is_baseline = overall["variant"].str.contains("baseline", case=False, regex=False)
    strategies = overall[~is_baseline].copy()
    baselines = overall[is_baseline].copy()

    metrics_cols = [
        "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
        "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
    ]

    # strategy averages, plus win shares versus baseline by window
    avg_by_variant = (
        strategies.groupby("variant", as_index=False)[metrics_cols]
        .mean()
        .sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    )

    baseline_sharpe_by_window = baselines.set_index("window")["Sharpe"].to_dict()
    baseline_cagr_by_window = baselines.set_index("window")["CAGR"].to_dict()

    total_windows = len(overall["window"].unique())
    strategies = strategies.assign(
        sharpe_beats_baseline = strategies.apply(lambda r: float(r["Sharpe"] > baseline_sharpe_by_window.get(r["window"], np.nan)), axis=1),
        cagr_beats_baseline   = strategies.apply(lambda r: float(r["CAGR"]  > baseline_cagr_by_window.get(r["window"], np.nan)), axis=1),
    )
    sharpe_wins = strategies.groupby("variant")["sharpe_beats_baseline"].sum().rename("sharpe_wins").reset_index()
    cagr_wins   = strategies.groupby("variant")["cagr_beats_baseline"].sum().rename("cagr_wins").reset_index()

    avg_plus_wins = avg_by_variant.merge(sharpe_wins, on="variant", how="left")
    avg_plus_wins = avg_plus_wins.merge(cagr_wins, on="variant", how="left")
    avg_plus_wins = avg_plus_wins.fillna({"sharpe_wins": 0.0, "cagr_wins": 0.0})
    avg_plus_wins["sharpe_win_share"] = avg_plus_wins["sharpe_wins"] / max(1, total_windows)
    avg_plus_wins["cagr_win_share"]   = avg_plus_wins["cagr_wins"] / max(1, total_windows)

    # add baseline averages into the same summary file
    baseline_avg = baselines.groupby("variant", as_index=False)[metrics_cols].mean()
    avg_plus_wins["is_baseline"] = False
    baseline_avg["sharpe_wins"] = np.nan
    baseline_avg["cagr_wins"] = np.nan
    baseline_avg["sharpe_win_share"] = np.nan
    baseline_avg["cagr_win_share"] = np.nan
    baseline_avg["is_baseline"] = True

    combined_avg = pd.concat([avg_plus_wins, baseline_avg], ignore_index=True, sort=False)

    # filename labels
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

    # write reports
    avg_path = os.path.join(outdir, f"{prefix}random_runs_summary_averages.csv")
    combined_avg.to_csv(avg_path, index=False)

    if not overall_t.empty:
        keep_cols_t = [
            "window", "ticker", "variant", "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
            "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
        ]
        overall_t = overall_t[keep_cols_t]

        is_base_t = overall_t["variant"].str.contains("baseline", case=False, regex=False)
        strategies_t = overall_t[~is_base_t].copy()
        baselines_t = overall_t[is_base_t].copy()

        base_sharpe_map = baselines_t.set_index(["ticker", "window"])["Sharpe"].to_dict()
        base_cagr_map   = baselines_t.set_index(["ticker", "window"])["CAGR"].to_dict()

        total_windows_t = len(overall_t["window"].unique())
        strategies_t = strategies_t.assign(
            sharpe_beats_baseline = strategies_t.apply(
                lambda r: float(r["Sharpe"] > base_sharpe_map.get((r["ticker"], r["window"]), np.nan)), axis=1
            ),
            cagr_beats_baseline   = strategies_t.apply(
                lambda r: float(r["CAGR"]  > base_cagr_map.get((r["ticker"], r["window"]), np.nan)), axis=1
            ),
        )

        metrics_cols = [
            "CAGR", "AnnReturn", "AnnVol", "Sharpe", "Sharpe_noRF",
            "MaxDD", "UlcerIndex", "TotalMultiple", "TotalReturn"
        ]

        avg_by_ticker = strategies_t.groupby(["ticker", "variant"], as_index=False)[metrics_cols].mean()
        sharpe_wins_t = strategies_t.groupby(["ticker", "variant"])["sharpe_beats_baseline"].sum().rename("sharpe_wins").reset_index()
        cagr_wins_t   = strategies_t.groupby(["ticker", "variant"])["cagr_beats_baseline"].sum().rename("cagr_wins").reset_index()

        per_ticker_avg = avg_by_ticker.merge(sharpe_wins_t, on=["ticker", "variant"], how="left")
        per_ticker_avg = per_ticker_avg.merge(cagr_wins_t, on=["ticker", "variant"], how="left")
        per_ticker_avg = per_ticker_avg.fillna({"sharpe_wins": 0.0, "cagr_wins": 0.0})
        per_ticker_avg["sharpe_win_share"] = per_ticker_avg["sharpe_wins"] / max(1, total_windows_t)
        per_ticker_avg["cagr_win_share"]   = per_ticker_avg["cagr_wins"] / max(1, total_windows_t)
        per_ticker_avg["is_baseline"] = False

        base_avg_t = baselines_t.groupby(["ticker", "variant"], as_index=False)[metrics_cols].mean()
        base_avg_t["sharpe_wins"] = np.nan
        base_avg_t["cagr_wins"] = np.nan
        base_avg_t["sharpe_win_share"] = np.nan
        base_avg_t["cagr_win_share"] = np.nan
        base_avg_t["is_baseline"] = True

        per_ticker_with_baseline = pd.concat([per_ticker_avg, base_avg_t], ignore_index=True, sort=False)

        per_ticker_avg_path = os.path.join(outdir, f"{prefix}per_ticker_random_runs_summary_averages.csv")
        per_ticker_with_baseline.to_csv(per_ticker_avg_path, index=False)
        print(f"Saved {avg_path}")
        print(f"Saved {per_ticker_avg_path}")
    else:
        print(f"Saved {avg_path}")
        print("Per ticker averages not written, overall_t is empty")

if __name__ == "__main__":
    main()