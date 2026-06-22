import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

# v1 reference implementation for equivalence testing.
V1 = HERE.parent / "winningstrat" / "winningstrat.py"
v1spec = importlib.util.spec_from_file_location("winningstrat_v1", V1)
v1 = importlib.util.module_from_spec(v1spec)
v1spec.loader.exec_module(v1)


def _synthetic_prices(n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    trend = np.linspace(0.0, 0.6, n)
    wobble = np.sin(np.linspace(0, 25, n)) * 0.15
    noise = rng.normal(0, 0.01, n).cumsum()
    return pd.Series(100.0 * np.exp(trend + wobble + noise), index=idx)


def test_consec_true_counts_runs():
    mask = pd.Series([True, True, False, True, True, True])
    out = core.consec_true(mask, 3).tolist()
    assert out == [False, False, False, False, False, True]


def test_hybrid_position_matches_v1():
    px = _synthetic_prices()
    got = core.hybrid_position(
        px, long_window=200, short_window=20,
        entry_days_long=3, entry_days_short=3, exit_days_long=2, exit_days_short=1,
    )
    ref = v1.hybrid_position_no_cooldown(
        price_signal=px, w_long=200, w_short=20,
        entry_days_long=3, entry_days_short=3, exit_days_long=2, exit_days_short=1,
    )
    aligned = ref.reindex(got.index)
    assert (got.astype(float).to_numpy() == aligned.astype(float).to_numpy()).all()
    assert set(got.unique()).issubset({0.0, 1.0})


def test_metrics_positive_drift_series():
    idx = pd.bdate_range("2018-01-01", periods=252)
    daily = pd.Series(0.0004, index=idx)  # steady positive return, no drawdown
    m = core.metrics_from_returns(daily)
    assert m["CAGR"] > 0
    assert m["Sharpe"] > 0
    assert m["MaxDD"] == 0.0 or abs(m["MaxDD"]) < 1e-9
    assert m["Calmar"] == 0.0  # no drawdown -> Calmar defined as 0 to avoid div/0


def test_metrics_calmar_with_drawdown():
    up = pd.Series(0.01, index=pd.bdate_range("2018-01-01", periods=50))
    down = pd.Series(-0.02, index=pd.bdate_range(up.index[-1] + pd.offsets.BDay(1), periods=20))
    rec = pd.Series(0.01, index=pd.bdate_range(down.index[-1] + pd.offsets.BDay(1), periods=80))
    daily = pd.concat([up, down, rec])
    m = core.metrics_from_returns(daily)
    assert m["MaxDD"] < 0
    assert abs(m["Calmar"] - (m["CAGR"] / abs(m["MaxDD"]))) < 1e-9


def test_build_cash_chain_splices_bil_then_sgov():
    bil_idx = pd.bdate_range("2010-01-01", periods=200)
    bil = pd.Series(np.linspace(50, 51, 200), index=bil_idx)            # gentle BIL drift
    sgov_idx = pd.bdate_range("2010-04-01", periods=160)                 # SGOV starts later
    sgov = pd.Series(np.linspace(100, 103, 160), index=sgov_idx)
    chain = core.build_cash_chain(sgov, sgov_tr_placeholder := sgov) if False else core.build_cash_chain(sgov, bil)
    sgov_start = sgov.index.min()
    # Before SGOV exists, chain equals BIL returns; on/after, equals SGOV returns.
    bil_ret = bil.pct_change()
    sgov_ret = sgov.pct_change()
    pre = chain[chain.index < sgov_start]
    post = chain[chain.index >= sgov_start]
    assert np.allclose(pre.to_numpy(), bil_ret[bil_ret.index < sgov_start].fillna(0).to_numpy())
    assert np.allclose(post.to_numpy(), sgov_ret[sgov_ret.index >= sgov_start].fillna(0).to_numpy())


def test_daily_from_pos_pure_states():
    idx = pd.bdate_range("2020-01-01", periods=10)
    tr = pd.Series(np.linspace(100, 110, 10), index=idx)
    cash = pd.Series(0.0001, index=idx)
    all_in = core.daily_from_pos(pd.Series(1.0, index=idx), tr, cash)
    all_out = core.daily_from_pos(pd.Series(0.0, index=idx), tr, cash)
    assert np.allclose(all_in.to_numpy(), tr.pct_change().fillna(0).to_numpy())
    assert np.allclose(all_out.to_numpy(), cash.to_numpy())


def test_combo_blend_equal_weight_on_overlap():
    idx = pd.bdate_range("2020-01-01", periods=30)
    a = pd.Series(0.01, index=idx)
    b = pd.Series(0.03, index=idx[5:])  # shorter -> overlap is idx[5:]
    blend = core.combo_blend({"A": a, "B": b})
    assert blend.index.min() == idx[5]
    assert np.allclose(blend.to_numpy(), 0.02)  # mean of 0.01 and 0.03


def test_combined_rank_lower_is_better():
    df = pd.DataFrame({"t": ["x", "y"], "s": [1.0, 2.0], "c": [0.10, 0.20]})
    r = core.combined_rank(df, "s", "c")
    # y has higher sharpe and cagr -> rank 1 each -> lower combined score
    assert r.iloc[1] < r.iloc[0]
