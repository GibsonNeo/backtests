#!/usr/bin/env python3
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

TRADING_DAYS = 252


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
    mu = float(d.mean())
    sigma = float(d.std())
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


def _get_col(df, col_name, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_series(ticker, start, end_inclusive):
    end_exclusive = (
        datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    px = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        progress=False,
        auto_adjust=False,
        actions=True,
    )
    if px.empty:
        raise ValueError(f"No data for {ticker}")

    close = _get_col(px, "Close", ticker).astype(float)
    splits = _get_col(px, "Stock Splits", ticker).fillna(0.0).astype(float)
    split_factor = splits.replace(0.0, 1.0)

    # Signals use the split-adjusted close, not the total return series.
    split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    price_signal = close * split_adj

    tr_df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        progress=False,
        auto_adjust=True,
    )
    price_tr = _get_col(tr_df, "Close", ticker).rename("AdjClose").astype(float)

    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]


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


def single_sma_position(price_signal: pd.Series, window: int, entry_days: int, exit_days: int):
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


def hybrid_position_no_cooldown(
    price_signal: pd.Series,
    w_long: int,
    w_short: int,
    entry_days_long: int,
    entry_days_short: int,
    exit_days_long: int,
    exit_days_short: int,
):
    """
    Regime A: if SMA(short) >= SMA(long)
      entry after entry_days_long closes above SMA(long)
      exit after exit_days_long closes at or below SMA(long)

    Regime B: if SMA(short) < SMA(long)
      entry after entry_days_short days where close > SMA(short) and SMA(short) is rising
      exit after exit_days_short closes at or below SMA(short)

    The slope gate is a simple day-over-day rise in the short SMA. There is no
    percentage threshold beyond "greater than yesterday".
    """
    smaL = _rolling_sma(price_signal, w_long)
    smaS = _rolling_sma(price_signal, w_short)

    aboveL = price_signal > smaL
    aboveS = price_signal > smaS
    slopeS_up = smaS > smaS.shift(1)

    entry_long_ok = _consec_true(aboveL, entry_days_long)
    gate_short = aboveS & slopeS_up
    entry_short_ok = _consec_true(gate_short, entry_days_short)

    exit_long_ok = _consec_true(~aboveL, exit_days_long)
    exit_short_ok = _consec_true(~aboveS, exit_days_short)

    short_below_long = smaS < smaL

    idx = price_signal.index
    pos = pd.Series(0, index=idx, dtype=int)
    in_pos = 0

    for i in range(len(idx)):
        use_entry_ok = entry_short_ok.iat[i] if short_below_long.iat[i] else entry_long_ok.iat[i]
        use_exit_ok = exit_short_ok.iat[i] if short_below_long.iat[i] else exit_long_ok.iat[i]

        if in_pos == 1 and use_exit_ok:
            in_pos = 0
        elif in_pos == 0 and use_entry_ok:
            in_pos = 1

        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0).astype(int)

    fv_long = smaL.dropna().index.min()
    fv_short = smaS.dropna().index.min()
    first_valid_candidates = [x for x in [fv_long, fv_short] if x is not None]
    first_valid = max(first_valid_candidates) if first_valid_candidates else None
    if first_valid is not None:
        pos = pos[pos.index >= first_valid]
    return pos


def _daily_from_pos(pos: pd.Series, tr_px: pd.Series, daily_cash: float):
    asset_rets = tr_px.pct_change().fillna(0.0)
    pos = pos.reindex(asset_rets.index).fillna(0.0)
    return pos * asset_rets + (1 - pos) * daily_cash


def _align_intersection(series_dict: dict, fill_value: float):
    common_index = None
    for s in series_dict.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)

    if common_index is None:
        return {}, pd.Index([])

    common_index = common_index.sort_values()
    out = {}
    for k, s in series_dict.items():
        out[k] = s.reindex(common_index).fillna(fill_value)
    return out, common_index


def _to_datestr(value):
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _resolve_weights(tickers, weights_cfg):
    weights = (weights_cfg or {}).copy()
    missing = [t for t in tickers if t not in weights]
    if missing:
        eq = 1.0 / len(tickers)
        for t in missing:
            weights[t] = eq

    total_w = sum(float(weights[t]) for t in tickers)
    if total_w <= 0:
        raise ValueError("Weights sum must be positive")
    return {t: float(weights[t]) / total_w for t in tickers}


def _variant_position(price_signal: pd.Series, variant: dict):
    kind = variant.get("kind", "hybrid")
    if kind == "hybrid":
        return hybrid_position_no_cooldown(
            price_signal=price_signal,
            w_long=int(variant["long_window"]),
            w_short=int(variant["short_window"]),
            entry_days_long=int(variant["entry_days_long"]),
            entry_days_short=int(variant["entry_days_short"]),
            exit_days_long=int(variant["exit_days_long"]),
            exit_days_short=int(variant["exit_days_short"]),
        )
    if kind == "single_sma":
        return single_sma_position(
            price_signal=price_signal,
            window=int(variant["window"]),
            entry_days=int(variant["entry_days"]),
            exit_days=int(variant["exit_days"]),
        )
    raise ValueError(f"Unsupported variant kind: {kind}")


def _warmup_days(variants):
    max_window = 0
    for variant in variants:
        if variant.get("kind", "hybrid") == "hybrid":
            max_window = max(
                max_window,
                int(variant["long_window"]),
                int(variant["short_window"]),
            )
        else:
            max_window = max(max_window, int(variant["window"]))
    return max_window * 3 + 365


def _load_variant_specs(cfg):
    variants = cfg.get("selected_variants", [])
    if not variants:
        raise ValueError("config.yml requires selected_variants")

    normalized = []
    for variant in variants:
        spec = dict(variant)
        spec["kind"] = spec.get("kind", "hybrid")
        if spec["kind"] == "hybrid":
            for key in [
                "name",
                "long_window",
                "short_window",
                "entry_days_long",
                "entry_days_short",
                "exit_days_long",
                "exit_days_short",
            ]:
                if key not in spec:
                    raise ValueError(f"Hybrid variant missing {key}: {spec}")
        elif spec["kind"] == "single_sma":
            for key in ["name", "window", "entry_days", "exit_days"]:
                if key not in spec:
                    raise ValueError(f"single_sma variant missing {key}: {spec}")
        normalized.append(spec)
    return normalized


def _load_data(tickers, ticker_inceptions, end, warmup_days):
    data = {}
    for ticker in tickers:
        start = pd.Timestamp(ticker_inceptions[ticker])
        ext_start = (start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        sig, tr = fetch_series(ticker, ext_start, end)

        sig = sig[sig.index >= start]
        tr = tr[tr.index >= start]
        if sig.empty or tr.empty:
            raise ValueError(f"No post-inception data for {ticker}")

        actual_start = max(sig.index.min(), tr.index.min())
        sig = sig[sig.index >= actual_start]
        tr = tr[tr.index >= actual_start]
        data[ticker] = {"sig": sig, "tr": tr}
    return data


def _build_row(
    ticker,
    variant,
    metrics,
    daily,
    sig,
    tr,
    note="",
):
    row = {
        "ticker": ticker,
        "variant": variant["name"],
        "kind": variant.get("kind", "hybrid"),
        "note": note or variant.get("note", ""),
        "start": sig.index.min().strftime("%Y-%m-%d"),
        "end": tr.index.max().strftime("%Y-%m-%d"),
        "observations": int(daily.dropna().shape[0]),
    }

    for key in [
        "long_window",
        "short_window",
        "entry_days_long",
        "entry_days_short",
        "exit_days_long",
        "exit_days_short",
        "window",
        "entry_days",
        "exit_days",
    ]:
        row[key] = variant.get(key)

    row.update(metrics)
    return row


def _baseline_variant():
    return {"name": "baseline_buyhold", "kind": "baseline", "note": "total return buy and hold"}


def _sort_metrics(df):
    return df.sort_values(["ticker", "Sharpe", "CAGR"], ascending=[True, False, False])


def main():
    cfg_path = "config.yml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", [])
    if not tickers:
        raise ValueError("config.yml requires tickers")

    ticker_inceptions = cfg.get("ticker_inceptions", {})
    missing_inceptions = [t for t in tickers if t not in ticker_inceptions]
    if missing_inceptions:
        raise ValueError(f"Missing ticker_inceptions for: {', '.join(missing_inceptions)}")

    ticker_inceptions = {t: _to_datestr(ticker_inceptions[t]) for t in tickers}
    end = _to_datestr(cfg.get("end", date.today()))
    outdir = cfg.get("outdir", "outputs")
    os.makedirs(outdir, exist_ok=True)

    cash_rate = float(cfg.get("cash_rate_percent", 0.0)) / 100.0
    sharpe_rf = float(cfg.get("sharpe_rf_percent", 0.0)) / 100.0
    daily_cash = cash_rate / TRADING_DAYS
    include_baseline = bool(cfg.get("include_baseline_buyhold", True))
    run_common_overlap = bool(cfg.get("run_common_overlap", True))

    variants = _load_variant_specs(cfg)
    warmup_days = _warmup_days(variants)
    weights = _resolve_weights(tickers, cfg.get("weights"))

    print("Fetching price history...")
    data = _load_data(tickers, ticker_inceptions, end, warmup_days)

    per_ticker_rows = []
    daily_cache = {}

    for ticker in tickers:
        sig = data[ticker]["sig"]
        tr = data[ticker]["tr"]
        daily_cache[ticker] = {}

        if include_baseline:
            baseline_daily = tr.pct_change().fillna(0.0)
            baseline_metrics = metrics_from_returns(baseline_daily, sharpe_rf)
            per_ticker_rows.append(
                _build_row(
                    ticker=ticker,
                    variant=_baseline_variant(),
                    metrics=baseline_metrics,
                    daily=baseline_daily,
                    sig=sig,
                    tr=tr,
                )
            )
            daily_cache[ticker]["baseline_buyhold"] = baseline_daily

        for variant in variants:
            pos = _variant_position(sig, variant)
            daily = _daily_from_pos(pos, tr, daily_cash)
            metrics = metrics_from_returns(daily, sharpe_rf)
            per_ticker_rows.append(
                _build_row(
                    ticker=ticker,
                    variant=variant,
                    metrics=metrics,
                    daily=daily,
                    sig=sig,
                    tr=tr,
                )
            )
            daily_cache[ticker][variant["name"]] = daily

    per_ticker = pd.DataFrame(per_ticker_rows)
    per_ticker = _sort_metrics(per_ticker)
    per_ticker_path = os.path.join(outdir, "per_ticker_full_span.csv")
    per_ticker.to_csv(per_ticker_path, index=False)

    if include_baseline:
        base = per_ticker[per_ticker["variant"] == "baseline_buyhold"].set_index("ticker")
        strategies = per_ticker[per_ticker["variant"] != "baseline_buyhold"].copy()
        strategies["beats_baseline_sharpe"] = strategies.apply(
            lambda r: float(r["Sharpe"] > base.at[r["ticker"], "Sharpe"]), axis=1
        )
        strategies["beats_baseline_cagr"] = strategies.apply(
            lambda r: float(r["CAGR"] > base.at[r["ticker"], "CAGR"]), axis=1
        )
        strategies["better_than_baseline_maxdd"] = strategies.apply(
            lambda r: float(r["MaxDD"] > base.at[r["ticker"], "MaxDD"]), axis=1
        )
    else:
        strategies = per_ticker.copy()
        strategies["beats_baseline_sharpe"] = np.nan
        strategies["beats_baseline_cagr"] = np.nan
        strategies["better_than_baseline_maxdd"] = np.nan

    metric_cols = [
        "CAGR",
        "AnnReturn",
        "AnnVol",
        "Sharpe",
        "Sharpe_noRF",
        "MaxDD",
        "UlcerIndex",
        "TotalMultiple",
        "TotalReturn",
    ]

    variant_rollup = (
        strategies.groupby("variant", as_index=False)[metric_cols]
        .mean()
        .sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    )
    meta_cols = [
        "variant",
        "kind",
        "note",
        "long_window",
        "short_window",
        "entry_days_long",
        "entry_days_short",
        "exit_days_long",
        "exit_days_short",
        "window",
        "entry_days",
        "exit_days",
    ]
    variant_meta = strategies[meta_cols].drop_duplicates(subset=["variant"])
    variant_rollup = variant_rollup.merge(variant_meta, on="variant", how="left")
    variant_rollup["beat_sharpe_count"] = (
        strategies.groupby("variant")["beats_baseline_sharpe"].sum().reindex(variant_rollup["variant"]).to_numpy()
    )
    variant_rollup["beat_cagr_count"] = (
        strategies.groupby("variant")["beats_baseline_cagr"].sum().reindex(variant_rollup["variant"]).to_numpy()
    )
    variant_rollup["better_maxdd_count"] = (
        strategies.groupby("variant")["better_than_baseline_maxdd"].sum().reindex(variant_rollup["variant"]).to_numpy()
    )
    variant_rollup["beat_sharpe_share"] = variant_rollup["beat_sharpe_count"] / max(1, len(tickers))
    variant_rollup["beat_cagr_share"] = variant_rollup["beat_cagr_count"] / max(1, len(tickers))
    variant_rollup["better_maxdd_share"] = variant_rollup["better_maxdd_count"] / max(1, len(tickers))
    variant_rollup_path = os.path.join(outdir, "variant_rollup_full_span.csv")
    variant_rollup.to_csv(variant_rollup_path, index=False)

    best_by_ticker = (
        per_ticker[per_ticker["variant"] != "baseline_buyhold"]
        .sort_values(["ticker", "Sharpe", "CAGR"], ascending=[True, False, False])
        .groupby("ticker", as_index=False)
        .head(1)
    )
    best_by_ticker_path = os.path.join(outdir, "best_variant_by_ticker.csv")
    best_by_ticker.to_csv(best_by_ticker_path, index=False)

    overlap_rows = []
    if run_common_overlap:
        overlap_variants = [v["name"] for v in variants]
        if include_baseline:
            overlap_variants = ["baseline_buyhold"] + overlap_variants

        for variant_name in overlap_variants:
            series_map = {ticker: daily_cache[ticker][variant_name] for ticker in tickers}
            aligned, common_index = _align_intersection(series_map, daily_cash if variant_name != "baseline_buyhold" else 0.0)
            if common_index.empty:
                continue

            blended = sum(weights[ticker] * aligned[ticker] for ticker in tickers)
            metrics = metrics_from_returns(blended, sharpe_rf)
            overlap_rows.append(
                {
                    "variant": variant_name,
                    "start": common_index.min().strftime("%Y-%m-%d"),
                    "end": common_index.max().strftime("%Y-%m-%d"),
                    "observations": int(blended.dropna().shape[0]),
                    **metrics,
                }
            )

        common_overlap = pd.DataFrame(overlap_rows).sort_values(
            ["Sharpe", "CAGR"], ascending=[False, False]
        )
        common_overlap_path = os.path.join(outdir, "common_overlap_portfolio_full_span.csv")
        common_overlap.to_csv(common_overlap_path, index=False)
    else:
        common_overlap_path = None

    inception_rows = [
        {
            "ticker": ticker,
            "configured_inception": ticker_inceptions[ticker],
            "first_signal_date_used": data[ticker]["sig"].index.min().strftime("%Y-%m-%d"),
            "last_date_used": data[ticker]["tr"].index.max().strftime("%Y-%m-%d"),
            "source_url": cfg.get("ticker_inception_source_urls", {}).get(ticker, ""),
        }
        for ticker in tickers
    ]
    inception_path = os.path.join(outdir, "ticker_inceptions_used.csv")
    pd.DataFrame(inception_rows).to_csv(inception_path, index=False)

    print(f"Saved {per_ticker_path}")
    print(f"Saved {variant_rollup_path}")
    print(f"Saved {best_by_ticker_path}")
    print(f"Saved {inception_path}")
    if common_overlap_path:
        print(f"Saved {common_overlap_path}")


if __name__ == "__main__":
    main()
