
#!/usr/bin/env python3
"""
MA S3 Momentum Overlay, variable basket breadth

Features
- Baseline S3 per ticker, entry on SMA200 slope up, exit on SMA200 slope down or price below SMA200
- Overlay strategies with 10 percent SPY and 10 percent QQQ reserved slices, each gated by S3
  Remaining weight is allocated to a momentum basket from the rest of the universe,
  size between breadth_min and breadth_max inclusive, selected at each rebalance date,
  only among names that pass S3 on that date, cash holds unassigned weight
- New baseline S2_EQ_SPY_QQQ_5050, equal weight SPY and QQQ, each gated by S3, otherwise cash
- Windows, equal slices and rolling 3 year with step in years
- Uses Close only, all pnl is close to close, cash earns zero out of market
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

@dataclass
class Config:
    universe_label: str
    tickers: List[str]
    overall_start: str
    overall_end: str
    equal_slices: int
    include_rolling_3y: bool
    rolling_step_years: int
    initial_capital: float
    slippage_pct: float
    fee_per_trade: float
    min_data_fraction: float
    rebalances: List[str]           # e.g. ["weekly","monthly","quarterly","semiannual"]
    breadth_min: int                # e.g. 3
    breadth_max: int                # e.g. 7
    workbook_path: str

# ---------- Utility ----------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _ensure_series(x) -> pd.Series:
    """Coerce a possible one column frame into a Series, preserve index and name."""
    if isinstance(x, pd.DataFrame):
        x = x.squeeze("columns")
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index)
    return x

def _s3_position(close: pd.Series) -> pd.Series:
    """State machine, entry when SMA200 slope up, exit on slope down or price below SMA200"""
    close = _ensure_series(close).sort_index().astype(float)
    if close.name is None:
        close.name = "close"
    ma = _sma(close, 200)
    slope_up = (ma > ma.shift(1)).fillna(False).astype(bool)
    slope_down = (ma < ma.shift(1)).fillna(False).astype(bool)
    price_below = (close < ma).fillna(False).astype(bool)
    pos = np.zeros(len(close), dtype=int)
    in_pos = False
    # Use .iat with integer position, after coercing to numpy bools
    su = slope_up.to_numpy(dtype=bool, copy=False)
    sd = slope_down.to_numpy(dtype=bool, copy=False)
    pb = price_below.to_numpy(dtype=bool, copy=False)
    for i in range(len(close)):
        if not in_pos:
            if su[i]:
                in_pos = True
                pos[i] = 1
            else:
                pos[i] = 0
        else:
            if sd[i] or pb[i]:
                in_pos = False
                pos[i] = 0
            else:
                pos[i] = 1
    return pd.Series(pos, index=close.index, name=close.name)

def _momentum_12_1(close: pd.Series) -> pd.Series:
    """Total return 252 day minus last 21 day return, classic 12 minus 1 momentum approximation"""
    close = _ensure_series(close).sort_index().astype(float)
    ret_12 = close.pct_change(252, fill_method=None)
    ret_1 = close.pct_change(21, fill_method=None)
    score = ret_12.fillna(0.0) - ret_1.fillna(0.0)
    score.name = close.name
    return score

def _rebalance_dates(index: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    s = index.to_series()
    cadence = cadence.lower()
    if cadence == "weekly":
        return s.resample("W-FRI").last().dropna().index
    if cadence == "monthly":
        return s.resample("ME").last().dropna().index
    if cadence == "quarterly":
        return s.resample("QE").last().dropna().index
    if cadence in ("semiannual","6m","six_month"):
        months = s.resample("M").last().dropna().index
        keep = months[::6]
        return pd.DatetimeIndex(keep)
    raise ValueError("cadence must be weekly, monthly, quarterly, or semiannual")

def _metrics_from_equity(eq: pd.Series) -> Dict[str, float]:
    eq = _ensure_series(eq).dropna()
    if eq.empty:
        return dict(cagr=0.0, sharpe=0.0, mdd=0.0)
    rets = eq.pct_change(fill_method=None).fillna(0.0)
    start_val = float(eq.iloc[0])
    end_val = float(eq.iloc[-1])
    days = max(1, (eq.index[-1] - eq.index[0]).days)
    cagr = (end_val / start_val) ** (365.25 / days) - 1.0 if start_val > 0 else 0.0
    vol = float(rets.std()) * math.sqrt(252.0)
    sharpe = float(rets.mean()) * 252.0 / vol if vol > 0 else 0.0
    roll_max = eq.cummax()
    draw = (eq / roll_max) - 1.0
    mdd = float(draw.min())
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd)

def _buyhold_equity(close: pd.Series, initial: float) -> pd.Series:
    close = _ensure_series(close).sort_index().astype(float)
    rets = close.pct_change(fill_method=None).fillna(0.0)
    eq = initial * (1.0 + rets).cumprod()
    eq.iloc[0] = initial
    eq.name = "buyhold"
    return eq

def _load_close(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance not available")
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker}")
    close = data["Close"]
    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close

# ---------- Core computations ----------

def build_universe(tickers: List[str], start: str, end: str) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
    closes = {}
    for t in tickers:
        s = _load_close(t, start, end)
        closes[t] = _ensure_series(s).astype(float)
    all_index = pd.DatetimeIndex(sorted(set().union(*[s.index for s in closes.values()])))
    close_by = {t: closes[t].reindex(all_index).ffill() for t in tickers}
    s3_by = {t: _s3_position(close_by[t]) for t in tickers}
    mom_by = {t: _momentum_12_1(close_by[t]) for t in tickers}
    return all_index, close_by, s3_by, mom_by

def make_equal_slices(index: pd.DatetimeIndex, k: int):
    if k <= 1:
        return [(index[0], index[-1], "FULL")]
    n = len(index)
    edges = [int(round(i * n / k)) for i in range(k + 1)]
    spans = []
    for i in range(k):
        a = max(0, edges[i])
        b = min(n - 1, edges[i + 1] - 1)
        if b <= a:
            continue
        start = index[a]
        end = index[b]
        spans.append((start, end, f"S{i+1:02d}"))
    return spans

def make_rolling_3y(index: pd.DatetimeIndex, step_years: int):
    spans = []
    if len(index) == 0:
        return spans
    start_date = index[0]
    final_date = index[-1]
    years = pd.date_range(start=start_date, end=final_date, freq=f"{step_years}YS")
    for s in years:
        e = s + pd.DateOffset(years=3) - pd.DateOffset(days=1)
        if e > final_date:
            e = final_date
        if s >= e:
            continue
        s2 = index[index.get_indexer([s], method="nearest")[0]]
        e2 = index[index.get_indexer([e], method="nearest")[0]]
        spans.append((s2, e2, f"R_{s2.date()}_{e2.date()}"))
    return spans

def build_windows(index: pd.DatetimeIndex, equal_slices: int, include_rolling_3y: bool, rolling_step_years: int):
    out = make_equal_slices(index, equal_slices)
    if include_rolling_3y:
        out += make_rolling_3y(index, rolling_step_years)
    return out

def portfolio_s3_baseline(index: pd.DatetimeIndex, close_by: Dict[str, pd.Series], s3_by: Dict[str, pd.Series], tickers: List[str], initial: float) -> pd.Series:
    rets = pd.DataFrame({t: _ensure_series(close_by[t]).pct_change(fill_method=None).fillna(0.0) for t in tickers}).reindex(index).fillna(0.0)
    pos = pd.DataFrame({t: _ensure_series(s3_by[t]).reindex(index).fillna(0).astype(int) for t in tickers})
    w = 1.0 / len(tickers) if len(tickers) > 0 else 0.0
    port_rets = (rets * pos * w).sum(axis=1)
    eq = initial * (1.0 + port_rets).cumprod()
    eq.iloc[0] = initial
    eq.name = "S3_Baseline"
    return eq

def portfolio_spy_qqq_5050(index: pd.DatetimeIndex, close_by: Dict[str, pd.Series], s3_by: Dict[str, pd.Series], initial: float) -> pd.Series:
    needed = [t for t in ["SPY","QQQ"] if t in close_by]
    if not needed:
        return pd.Series(initial, index=index, name="EQ_SPY_QQQ_5050")
    rets = pd.DataFrame({t: _ensure_series(close_by[t]).pct_change(fill_method=None).fillna(0.0) for t in needed}).reindex(index).fillna(0.0)
    pos = pd.DataFrame({t: _ensure_series(s3_by[t]).reindex(index).fillna(0).astype(int) for t in needed})
    w = 0.5
    port = pd.Series(0.0, index=index)
    if "QQQ" in needed:
        port = port.add(w * rets["QQQ"] * pos["QQQ"], fill_value=0.0)
    if "SPY" in needed:
        port = port.add(w * rets["SPY"] * pos["SPY"], fill_value=0.0)
    eq = initial * (1.0 + port).cumprod()
    eq.iloc[0] = initial
    eq.name = "EQ_SPY_QQQ_5050"
    return eq

def portfolio_momo_topk_overlay(index: pd.DatetimeIndex,
                                close_by: Dict[str, pd.Series],
                                s3_by: Dict[str, pd.Series],
                                mom_by: Dict[str, pd.Series],
                                universe: List[str],
                                breadth_min: int,
                                breadth_max: int,
                                cadence: str,
                                initial: float) -> pd.Series:
    tickers = [t for t in universe]
    index = pd.DatetimeIndex(index)
    rets = pd.DataFrame({t: _ensure_series(close_by[t]).pct_change(fill_method=None).fillna(0.0) for t in tickers}).reindex(index).fillna(0.0)
    pos = pd.DataFrame({t: _ensure_series(s3_by[t]).reindex(index).fillna(0).astype(int) for t in tickers})
    mom = pd.DataFrame({t: _ensure_series(mom_by[t]).reindex(index).fillna(-1e9) for t in tickers})
    weights = pd.DataFrame(0.0, index=index, columns=tickers)
    rdates = _rebalance_dates(index, cadence)
    core = [t for t in ["SPY","QQQ"] if t in tickers]
    rest = [t for t in tickers if t not in core]
    for i, rd in enumerate(rdates):
        start = rd
        end = rdates[i+1] - pd.Timedelta(days=1) if i + 1 < len(rdates) else index[-1]
        seg = (index >= start) & (index <= end)
        # Core slices, each 10 percent if eligible
        for t in core:
            eligible = bool(pos.loc[rd, t]) if rd in pos.index else False
            weights.loc[seg, t] = 0.10 if eligible else 0.0
        remaining = 0.80
        # Eligible pool in rest
        elig = [t for t in rest if (rd in pos.index and bool(pos.loc[rd, t]))]
        if len(elig) >= max(0, breadth_min):
            scores = pd.Series({t: float(mom.loc[rd, t]) if rd in mom.index else -1e9 for t in elig})
            scores = scores.sort_values(ascending=False)
            k = min(len(scores), max(0, breadth_max))
            chosen = list(scores.index[:k])
            if k > 0:
                w_each = remaining / k
                for t in chosen:
                    weights.loc[seg, t] = w_each
        # else, no eligible basket of at least breadth_min, remaining stays in cash
    port_rets = (weights * pos * rets).sum(axis=1)
    eq = initial * (1.0 + port_rets).cumprod()
    eq.iloc[0] = initial
    eq.name = f"MomoTopK_{breadth_min}to{breadth_max}_{cadence}"
    return eq

# ---------- Reporting ----------

def compute_all(cfg: Config):
    index, close_by, s3_by, mom_by = build_universe(cfg.tickers, cfg.overall_start, cfg.overall_end)
    windows = build_windows(index, cfg.equal_slices, cfg.include_rolling_3y, cfg.rolling_step_years)

    ew_bh_rets = pd.DataFrame({t: _ensure_series(close_by[t]).pct_change(fill_method=None).fillna(0.0) for t in cfg.tickers}).reindex(index).fillna(0.0)
    ew_w = 1.0 / len(cfg.tickers) if len(cfg.tickers) > 0 else 0.0
    ew_bh_eq = cfg.initial_capital * (1.0 + (ew_bh_rets * ew_w).sum(axis=1)).cumprod()
    ew_bh_eq.iloc[0] = cfg.initial_capital

    strategies = []
    eq_S1 = portfolio_s3_baseline(index, close_by, s3_by, cfg.tickers, cfg.initial_capital)
    strategies.append(("S1_S3_Baseline", eq_S1))
    eq_S2 = portfolio_spy_qqq_5050(index, close_by, s3_by, cfg.initial_capital)
    strategies.append(("S2_EQ_SPY_QQQ_5050", eq_S2))
    for cad in cfg.rebalances:
        tag = cad.lower()
        eq = portfolio_momo_topk_overlay(index, close_by, s3_by, mom_by, cfg.tickers, cfg.breadth_min, cfg.breadth_max, tag, cfg.initial_capital)
        strategies.append((f"S3_MomoTopK_{cfg.breadth_min}to{cfg.breadth_max}_with_SPY10_QQQ10_{tag}", eq))

    rows_port = []
    windows_list = windows
    for name, eq in strategies:
        m_full = _metrics_from_equity(eq)
        m_bh_full = _metrics_from_equity(ew_bh_eq)
        rows_port.append(dict(window_id="FULL", strategy=name,
                              portfolio_sharpe=m_full["sharpe"], portfolio_cagr=m_full["cagr"], portfolio_max_drawdown=m_full["mdd"],
                              ew_bh_sharpe=m_bh_full["sharpe"], ew_bh_cagr=m_bh_full["cagr"], ew_bh_max_drawdown=m_bh_full["mdd"],
                              beat_bh_sharpe=float(m_full["sharpe"] > m_bh_full["sharpe"]),
                              beat_bh_cagr=float(m_full["cagr"] > m_bh_full["cagr"]),
                              beat_bh_mdd=float(m_full["mdd"] > m_bh_full["mdd"]),
                              mdd_diff_vs_bh=m_bh_full["mdd"] - m_full["mdd"]
                              ))
        for (ws, we, wid) in windows_list:
            eq_w = eq.loc[(eq.index >= ws) & (eq.index <= we)]
            bh_w = ew_bh_eq.loc[(ew_bh_eq.index >= ws) & (ew_bh_eq.index <= we)]
            m_w = _metrics_from_equity(eq_w)
            m_bh_w = _metrics_from_equity(bh_w)
            rows_port.append(dict(window_id=wid, strategy=name,
                                  portfolio_sharpe=m_w["sharpe"], portfolio_cagr=m_w["cagr"], portfolio_max_drawdown=m_w["mdd"],
                                  ew_bh_sharpe=m_bh_w["sharpe"], ew_bh_cagr=m_bh_w["cagr"], ew_bh_max_drawdown=m_bh_w["mdd"],
                                  beat_bh_sharpe=float(m_w["sharpe"] > m_bh_w["sharpe"]),
                                  beat_bh_cagr=float(m_w["cagr"] > m_bh_w["cagr"]),
                                  beat_bh_mdd=float(m_w["mdd"] > m_bh_w["mdd"]),
                                  mdd_diff_vs_bh=m_bh_w["mdd"] - m_w["mdd"]
                                  ))
    portfolio_df = pd.DataFrame(rows_port)

    mask_sub = portfolio_df["window_id"] != "FULL"
    agg = portfolio_df[mask_sub].groupby("strategy").agg(
        windows_evaluated=("window_id","nunique"),
        avg_sharpe=("portfolio_sharpe","mean"),
        avg_cagr=("portfolio_cagr","mean"),
        median_max_drawdown=("portfolio_max_drawdown","median"),
        beat_ew_bh_sharpe_pct=("beat_bh_sharpe","mean"),
        beat_ew_bh_cagr_pct=("beat_bh_cagr","mean"),
        beat_ew_bh_mdd_pct=("beat_bh_mdd","mean"),
        avg_excess_sharpe_vs_ew_bh=("portfolio_sharpe","mean"),
        avg_excess_cagr_vs_ew_bh=("portfolio_cagr","mean"),
        avg_mdd_diff_vs_ew_bh=("mdd_diff_vs_bh","mean"),
        median_mdd_diff_vs_ew_bh=("mdd_diff_vs_bh","median"),
    ).reset_index()
    agg["rank_key"] = list(zip(-agg["avg_sharpe"].fillna(-9e9),
                               agg["median_max_drawdown"].fillna(0.0),
                               -agg["avg_cagr"].fillna(-9e9)))
    agg = agg.sort_values("rank_key").drop(columns=["rank_key"])

    winners = portfolio_df[mask_sub].loc[portfolio_df[mask_sub].groupby("window_id")["portfolio_sharpe"].idxmax()]
    windows_df = winners[["window_id","strategy","portfolio_sharpe","portfolio_cagr","portfolio_max_drawdown"]].reset_index(drop=True)

    base = "S1_S3_Baseline"
    h2h_rows = []
    for s in agg["strategy"]:
        if s == base:
            continue
        subA = portfolio_df[(portfolio_df["strategy"] == s) & mask_sub]
        subB = portfolio_df[(portfolio_df["strategy"] == base) & mask_sub]
        merged = subA.merge(subB, on="window_id", suffixes=("_A","_B"))
        if merged.empty:
            continue
        h2h_rows.append(dict(
            A=s, B=base,
            sharpe_win_pct_A_gt_B=float(np.mean(merged["portfolio_sharpe_A"] > merged["portfolio_sharpe_B"])),
            mdd_win_pct_A_gt_B=float(np.mean(merged["portfolio_max_drawdown_A"] > merged["portfolio_max_drawdown_B"])),
            avg_delta_sharpe=(merged["portfolio_sharpe_A"] - merged["portfolio_sharpe_B"]).mean(),
            avg_mdd_improvement=(merged["portfolio_max_drawdown_B"] - merged["portfolio_max_drawdown_A"]).mean(),
        ))
    head_df = pd.DataFrame(h2h_rows)

    spy = close_by.get("SPY")
    if spy is not None:
        spy_eq = _buyhold_equity(spy.reindex(index), 100.0)
        s_max = spy_eq.cummax()
        dd = spy_eq / s_max - 1.0
        regime = pd.Series(np.where(dd <= -0.20, "Bear", np.where(dd >= -0.05, "Bull", "Sideways")), index=index)
    else:
        regime = pd.Series("Sideways", index=index)

    reg_rows = []
    for s in agg["strategy"]:
        if s == base:
            continue
        subA = portfolio_df[(portfolio_df["strategy"] == s) & mask_sub]
        subB = portfolio_df[(portfolio_df["strategy"] == base) & mask_sub]
        merged = subA.merge(subB, on="window_id", suffixes=("_A","_B"))
        if merged.empty:
            continue
        w_lookup = {}
        for (ws, we, wid) in build_windows(index, cfg.equal_slices, cfg.include_rolling_3y, cfg.rolling_step_years):
            w_lookup[wid] = (ws, we)
        rows = []
        for _, r in merged.iterrows():
            wid = r["window_id"]
            if wid not in w_lookup:
                continue
            ws, we = w_lookup[wid]
            reg_slice = regime[(regime.index >= ws) & (regime.index <= we)]
            if reg_slice.empty:
                continue
            reg = reg_slice.mode().iloc[0]
            rows.append((reg, r["portfolio_sharpe_A"] > r["portfolio_sharpe_B"],
                              r["portfolio_max_drawdown_A"] > r["portfolio_max_drawdown_B"]))
        if rows:
            df = pd.DataFrame(rows, columns=["regime","sharpe_win_A","mdd_win_A"])
            for reg_name, grp in df.groupby("regime"):
                reg_rows.append(dict(A=s, B=base, regime=reg_name,
                                    portfolio_sharpe_win_pct_A_gt_B=float(grp["sharpe_win_A"].mean()),
                                    portfolio_max_drawdown_win_pct_A_gt_B=float(grp["mdd_win_A"].mean())))
    regime_df = pd.DataFrame(reg_rows)

    plus_rows = []
    for s in agg["strategy"]:
        if s == base:
            continue
        A = portfolio_df[(portfolio_df["strategy"] == s) & mask_sub]
        B = portfolio_df[(portfolio_df["strategy"] == base) & mask_sub]
        M = A.merge(B, on="window_id", suffixes=("_A","_B"))
        if M.empty:
            continue
        plus_rows.append(dict(
            strategy=s,
            win_sharpe_vs_S1_pct=float(np.mean(M["portfolio_sharpe_A"] > M["portfolio_sharpe_B"])),
            win_cagr_vs_S1_pct=float(np.mean(M["portfolio_cagr_A"] > M["portfolio_cagr_B"])),
            better_mdd_vs_S1_pct=float(np.mean(M["portfolio_max_drawdown_A"] > M["portfolio_max_drawdown_B"])),
            avg_excess_sharpe_vs_S1=float((M["portfolio_sharpe_A"] - M["portfolio_sharpe_B"]).mean()),
            avg_excess_cagr_vs_S1=float((M["portfolio_cagr_A"] - M["portfolio_cagr_B"]).mean()),
            avg_mdd_diff_vs_S1=float((M["portfolio_max_drawdown_B"] - M["portfolio_max_drawdown_A"]).mean()),
        ))
    summary_plus = pd.DataFrame(plus_rows)

    summary = agg.copy()

    return summary, portfolio_df, windows_df, head_df, regime_df, summary_plus

def write_workbook(cfg: Config,
                   summary: pd.DataFrame,
                   portfolio: pd.DataFrame,
                   windows: pd.DataFrame,
                   head: pd.DataFrame,
                   regime: pd.DataFrame,
                   summary_plus: pd.DataFrame):
    path = cfg.workbook_path
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        summary_plus.to_excel(writer, sheet_name="SummaryPlus", index=False)
        windows.to_excel(writer, sheet_name="Windows", index=False)
        portfolio.to_excel(writer, sheet_name="Portfolio", index=False)
        head.to_excel(writer, sheet_name="HeadToHead", index=False)
        regime.to_excel(writer, sheet_name="Regime_WinRates", index=False)
        wb = writer.book
        pct = wb.add_format({"num_format": "0.0%"})
        ratio = wb.add_format({"num_format": "0.0000"})
        # freeze panes, auto width, number formats
        sheets = {
            "Summary": summary,
            "SummaryPlus": summary_plus,
            "Windows": windows,
            "Portfolio": portfolio,
            "HeadToHead": head,
            "Regime_WinRates": regime,
        }
        for name, df in sheets.items():
            ws = writer.sheets[name]
            ws.freeze_panes(1, 1)
            for col_idx, col in enumerate(df.columns):
                ws.set_column(col_idx, col_idx, max(12, len(str(col)) + 2))
            for col_idx, col in enumerate(df.columns):
                label = str(col).lower()
                if "sharpe" in label:
                    ws.set_column(col_idx, col_idx, 14, ratio)
                if "drawdown" in label or "mdd" in label or "cagr" in label or "pct" in label:
                    ws.set_column(col_idx, col_idx, 14, pct)

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="MA S3 Momentum Overlay with variable breadth and EQ SPY QQQ baseline")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args(argv)

    y = load_yaml(args.config)
    cfg = Config(
        universe_label=y.get("universe_label","Universe"),
        tickers=y["tickers"],
        overall_start=y["overall"]["start"],
        overall_end=y["overall"]["end"],
        equal_slices=int(y["windows"]["equal_slices"]),
        include_rolling_3y=bool(y["windows"].get("include_rolling_3y", True)),
        rolling_step_years=int(y["windows"].get("rolling_step_years", 1)),
        initial_capital=float(y["common"]["initial_capital"]),
        slippage_pct=float(y["common"].get("slippage_pct", 0.0)),
        fee_per_trade=float(y["common"].get("fee_per_trade", 0.0)),
        min_data_fraction=float(y["common"].get("min_data_fraction", 0.8)),
        rebalances=y.get("rebalances", ["weekly","monthly","quarterly","semiannual"]),
        breadth_min=int(y.get("breadth_min", 3)),
        breadth_max=int(y.get("breadth_max", 7)),
        workbook_path=y["output"]["workbook_path"],
    )

    summary, portfolio, windows, head, regime, summary_plus = compute_all(cfg)
    write_workbook(cfg, summary, portfolio, windows, head, regime, summary_plus)
    print(f"Wrote {cfg.workbook_path}")

if __name__ == "__main__":
    main()
