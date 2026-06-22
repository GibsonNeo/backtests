# winningstratv2 Ticker-Swap Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `winningstratv2/` study that holds the "best winning strat" fixed and screens a ~70-ticker universe through a robustness-gated funnel (per-ticker screen → cross-window robustness → 4-sleeve combo search) to find sleeves that beat the incumbents `SPMO/IWY/QQQ/DGRW`.

**Architecture:** A tested pure-logic core (`strat_core.py`) carries the locked hybrid SMA strategy, metrics, the SGOV→BIL cash chain, and the equal-weight combo blend. Three thin tier scripts (`screen.py`, `robustness.py`, `combo_search.py`) orchestrate fetch + evaluation around that core; `run_all.py` chains them and writes a report. Bulky, unchanged window-generation helpers are copied verbatim from the proven v1 samplers in `winningstrat/`.

**Tech Stack:** Python 3.12, pandas, numpy, yfinance, pyyaml, matplotlib, tabulate, pytest. Per-folder `.venv` (the base interpreter has no scientific stack).

> **GIT POLICY (project override):** The user's global policy is *never commit or push*. Every task ends with a **Checkpoint** the user runs manually — the executing agent MUST NOT run `git commit`/`git push`. Run tests and report; leave git to the user.

---

## File Structure

| File | Responsibility |
|---|---|
| `winningstratv2/requirements.txt` | Pinned-by-name dependency list (mirrors `rotation-sleeve/requirements.txt`). |
| `winningstratv2/config.yml` | Universe (theme→tickers), locked variant, cash chain, gate, window + tier settings. |
| `winningstratv2/strat_core.py` | Pure core: SMA helpers, `hybrid_position`, `metrics_from_returns` (+Calmar), `build_cash_chain`, `daily_from_pos`, `combo_blend`, `fetch_series`, `combined_rank`. |
| `winningstratv2/screen.py` | Tier 1: per-ticker full-span screen → `outputs/tier1_screen.csv`. |
| `winningstratv2/robustness.py` | Tier 2: random + structured window sampling on survivors → `outputs/tier2_robustness.csv`. |
| `winningstratv2/combo_search.py` | Tier 3: 4-sleeve combo search among finalists → `outputs/tier3_combos.csv`. |
| `winningstratv2/run_all.py` | Orchestrate Tier 1→2→3 with a shared price cache; write `outputs/report.md` + charts. |
| `winningstratv2/tests/test_strat_core.py` | Offline synthetic-data tests for the core (no Yahoo calls). |
| `winningstratv2/README.md` | How to set up, run, and interpret outputs. |

**Interface contract (defined in Task 2–6, used everywhere after):**

```python
TRADING_DAYS = 252
rolling_sma(x: pd.Series, window: int) -> pd.Series
consec_true(mask: pd.Series, n: int) -> pd.Series
hybrid_position(price_signal, long_window=200, short_window=20,
                entry_days_long=3, entry_days_short=3,
                exit_days_long=2, exit_days_short=1) -> pd.Series  # float 0/1, shifted, trimmed
metrics_from_returns(daily_rets: pd.Series, sharpe_rf: float = 0.0) -> dict
build_cash_chain(sgov_tr: pd.Series, bil_tr: pd.Series) -> pd.Series  # daily cash returns
daily_from_pos(pos: pd.Series, tr_px: pd.Series, cash_daily: pd.Series) -> pd.Series
combo_blend(sleeve_daily: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series
fetch_series(ticker: str, start: str, end_inclusive: str) -> tuple[pd.Series, pd.Series]  # (signal, total_return)
combined_rank(df: pd.DataFrame, sharpe_col: str, cagr_col: str,
              w_sharpe: float = 0.6, w_cagr: float = 0.4) -> pd.Series  # lower is better
```

`metrics_from_returns` returns keys: `CAGR, AnnReturn, AnnVol, Sharpe, Sharpe_noRF, MaxDD, UlcerIndex, Calmar, TotalMultiple, TotalReturn`.

---

## Task 1: Scaffold folder, venv, requirements, config

**Files:**
- Create: `winningstratv2/requirements.txt`
- Create: `winningstratv2/config.yml`
- Create: `winningstratv2/.gitignore`
- Create: `winningstratv2/tests/__init__.py` (empty)

- [ ] **Step 1: Write `requirements.txt`**

```
numpy
pandas
yfinance
pyyaml
matplotlib
tabulate
pytest
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
.pytest_cache/
outputs/
```

- [ ] **Step 3: Write `config.yml`**

```yaml
# Locked "best winning strat" variant (matches winningstrat/readme.md top section).
fixed_variant:
  name: "200SMA_3in_2out_hybrid20SMA"
  kind: hybrid
  long_window: 200
  short_window: 20
  entry_days_long: 3
  entry_days_short: 3
  exit_days_long: 2
  exit_days_short: 1

# Out-of-market holding: SGOV when it exists, BIL before it. Same for every ticker.
cash_chain: [SGOV, BIL]

end: today
outdir: outputs

# Tier 1 -> Tier 2 gate.
min_history_days: 1500        # ~6 years of trading days
gate_require_positive_sharpe_delta: true

# Tier 2 (cross-window robustness).
tier2:
  shared_start: auto          # auto = max inception across survivors; or an explicit YYYY-MM-DD
  random_samples: 100
  random_min_years: 3
  random_max_years: auto
  random_seed: 49
  structured_buckets: [3, 5, 7, 10]
  structured_step_fraction: 0.5
  structured_tail_overlap: 0.6
  rank_sharpe_weight: 0.6
  rank_cagr_weight: 0.4

# Tier 2 -> Tier 3.
n_finalists: 12

# Tier 3 (4-sleeve combo search).
tier3:
  combo_size: 4
  score_weights: {sharpe: 0.5, cagr: 0.3, maxdd: 0.2}
  top_n_report: 10

incumbents: [SPMO, IWY, QQQ, DGRW]

universe:
  incumbent:       [SPMO, IWY, QQQ, DGRW]
  momentum:        [MTUM, PDP, FDMO, XMMO, QMOM, VFMO, JMOM]
  large_growth:    [SCHG, VUG, IWF, MGK, SPYG, VONG, IUSG, XLG, MGC]
  tech:            [VGT, XLK, IYW, FTEC, IGM, SOXX, SMH, IGV]
  quality:         [QUAL, SPHQ, JQUA, DGRO, GSLC]
  dividend_growth: [SCHD, VIG, NOBL, RDVY]
  multifactor:     [OMFL, LRGF, DSTL, SPGP]
  broad_control:   [RSP, EQWL, SPY, OEF]
  midcap_growth:   [IWP, VOT, ARKK]
  sector:          [XLY, XLV, XLF, XLI, XLP, XLU, XLE, XLB, XLRE, XLC, XBI, ITA, GLD]
  international:   [EFA, VEA, EFG, IMTM, IQLT, VWO, VEU, INDA, EWT]
```

- [ ] **Step 4: Create the venv and install deps**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Expected: pip installs numpy/pandas/yfinance/pyyaml/matplotlib/tabulate/pytest with no errors.

- [ ] **Step 5: Create `tests/__init__.py`** (empty file).

- [ ] **Step 6: Verify the universe count is exactly 70**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python -c "import yaml; u=yaml.safe_load(open('config.yml'))['universe']; a=[t for v in u.values() for t in v]; print('total', len(a), 'unique', len(set(a)))"
```
Expected: `total 70 unique 70`.

- [ ] **Step 7: Checkpoint (user commits)** — report files created and the count output.

---

## Task 2: Core SMA helpers + locked hybrid position (TDD)

**Files:**
- Create: `winningstratv2/strat_core.py`
- Test: `winningstratv2/tests/test_strat_core.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strat_core.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/wes/github/backtests/winningstratv2 && .venv/bin/python -m pytest tests/test_strat_core.py -q`
Expected: FAIL — `strat_core.py` has no `consec_true`/`hybrid_position` (ModuleNotFoundError or AttributeError).

- [ ] **Step 3: Write the implementation**

Create `strat_core.py` with the header and these functions:
```python
#!/usr/bin/env python3
"""Pure strategy core for winningstratv2: locked hybrid SMA strat, metrics,
cash chain, combo blend, and Yahoo fetch. No file I/O here."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_sma(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).mean()


def consec_true(mask: pd.Series, n: int) -> pd.Series:
    if n <= 0:
        return pd.Series(False, index=mask.index)
    count = 0
    out = []
    for value in mask.fillna(False).to_numpy():
        count = count + 1 if bool(value) else 0
        out.append(count >= n)
    return pd.Series(out, index=mask.index)


def hybrid_position(
    price_signal: pd.Series,
    long_window: int = 200,
    short_window: int = 20,
    entry_days_long: int = 3,
    entry_days_short: int = 3,
    exit_days_long: int = 2,
    exit_days_short: int = 1,
) -> pd.Series:
    """Regime A (SMA_short >= SMA_long): enter after entry_days_long closes above
    SMA_long, exit after exit_days_long closes at/below it. Regime B
    (SMA_short < SMA_long): enter after entry_days_short days where close > SMA_short
    and SMA_short is rising, exit after exit_days_short closes at/below SMA_short.
    Returns a float 0/1 series, shifted one day, trimmed to first valid SMA date."""
    sma_long = rolling_sma(price_signal, long_window)
    sma_short = rolling_sma(price_signal, short_window)

    above_long = price_signal > sma_long
    above_short = price_signal > sma_short
    slope_up = sma_short > sma_short.shift(1)

    entry_long_ok = consec_true(above_long, entry_days_long)
    entry_short_ok = consec_true(above_short & slope_up, entry_days_short)
    exit_long_ok = consec_true(~above_long, exit_days_long)
    exit_short_ok = consec_true(~above_short, exit_days_short)

    short_below_long = sma_short < sma_long

    idx = price_signal.index
    pos = pd.Series(0.0, index=idx)
    in_pos = 0.0
    for i in range(len(idx)):
        use_short = bool(short_below_long.iat[i])
        entry_ok = entry_short_ok.iat[i] if use_short else entry_long_ok.iat[i]
        exit_ok = exit_short_ok.iat[i] if use_short else exit_long_ok.iat[i]
        if in_pos == 1.0 and exit_ok:
            in_pos = 0.0
        elif in_pos == 0.0 and entry_ok:
            in_pos = 1.0
        pos.iat[i] = in_pos

    pos = pos.shift(1).fillna(0.0)
    fv_long = sma_long.dropna().index.min()
    fv_short = sma_short.dropna().index.min()
    candidates = [x for x in [fv_long, fv_short] if x is not None]
    if candidates:
        first_valid = max(candidates)
        pos = pos[pos.index >= first_valid]
    return pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/wes/github/backtests/winningstratv2 && .venv/bin/python -m pytest tests/test_strat_core.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Checkpoint (user commits).**

---

## Task 3: Metrics with Calmar (TDD)

**Files:**
- Modify: `winningstratv2/strat_core.py`
- Test: `winningstratv2/tests/test_strat_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strat_core.py`:
```python
def test_metrics_positive_drift_series():
    idx = pd.bdate_range("2018-01-01", periods=TRADING_DAYS := 252)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py::test_metrics_positive_drift_series -q`
Expected: FAIL — `metrics_from_returns` not defined.

- [ ] **Step 3: Add the implementation to `strat_core.py`**

```python
def _ulcer_and_maxdd(curve: pd.Series) -> tuple[float, float]:
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(np.sqrt((dd.pow(2)).mean()) * 100.0), float(dd.min())


def metrics_from_returns(daily_rets: pd.Series, sharpe_rf: float = 0.0) -> dict:
    d = daily_rets.dropna()
    if d.empty:
        return dict(CAGR=0.0, AnnReturn=0.0, AnnVol=0.0, Sharpe=0.0, Sharpe_noRF=0.0,
                    MaxDD=0.0, UlcerIndex=0.0, Calmar=0.0, TotalMultiple=1.0, TotalReturn=0.0)

    curve = (1.0 + d).cumprod()
    years = len(d) / TRADING_DAYS
    total_multiple = float(curve.iloc[-1])
    mu = float(d.mean())
    sigma = float(d.std())
    ann_return = mu * TRADING_DAYS
    ann_vol = sigma * np.sqrt(TRADING_DAYS)

    if sigma > 0:
        sharpe = (mu - sharpe_rf / TRADING_DAYS) / sigma * np.sqrt(TRADING_DAYS)
        sharpe_norf = mu / sigma * np.sqrt(TRADING_DAYS)
    else:
        sharpe = sharpe_norf = 0.0

    ui, maxdd = _ulcer_and_maxdd(curve)
    cagr = total_multiple ** (1.0 / max(years, 1e-9)) - 1.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    return dict(CAGR=cagr, AnnReturn=ann_return, AnnVol=ann_vol, Sharpe=sharpe,
                Sharpe_noRF=sharpe_norf, MaxDD=maxdd, UlcerIndex=ui, Calmar=calmar,
                TotalMultiple=total_multiple, TotalReturn=total_multiple - 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Checkpoint (user commits).**

---

## Task 4: Cash chain + daily-from-position (TDD)

**Files:**
- Modify: `winningstratv2/strat_core.py`
- Test: `winningstratv2/tests/test_strat_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strat_core.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py::test_daily_from_pos_pure_states -q`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Add the implementation to `strat_core.py`**

```python
def build_cash_chain(sgov_tr: pd.Series, bil_tr: pd.Series) -> pd.Series:
    """Daily cash returns: BIL before SGOV's first date, SGOV from then on.
    Before BIL's own inception the cash return is 0 (BIL did not exist)."""
    sgov_ret = sgov_tr.pct_change()
    bil_ret = bil_tr.pct_change()
    sgov_start = sgov_tr.dropna().index.min()
    sgov_part = sgov_ret[sgov_ret.index >= sgov_start]
    bil_part = bil_ret[bil_ret.index < sgov_start]
    chain = pd.concat([bil_part, sgov_part]).sort_index()
    chain = chain[~chain.index.duplicated(keep="last")]
    return chain.fillna(0.0)


def daily_from_pos(pos: pd.Series, tr_px: pd.Series, cash_daily: pd.Series) -> pd.Series:
    asset_rets = tr_px.pct_change().fillna(0.0)
    pos = pos.reindex(asset_rets.index).fillna(0.0)
    cash = cash_daily.reindex(asset_rets.index).fillna(0.0)
    return pos * asset_rets + (1.0 - pos) * cash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Checkpoint (user commits).**

---

## Task 5: Equal-weight combo blend + combined rank (TDD)

**Files:**
- Modify: `winningstratv2/strat_core.py`
- Test: `winningstratv2/tests/test_strat_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strat_core.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py::test_combo_blend_equal_weight_on_overlap -q`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Add the implementation to `strat_core.py`**

```python
def combo_blend(sleeve_daily: dict, weights: dict | None = None) -> pd.Series:
    df = pd.DataFrame(sleeve_daily).dropna()
    cols = list(df.columns)
    if weights is None:
        weights = {c: 1.0 / len(cols) for c in cols}
    total = sum(weights[c] for c in cols)
    return sum((weights[c] / total) * df[c] for c in cols)


def combined_rank(df: pd.DataFrame, sharpe_col: str, cagr_col: str,
                  w_sharpe: float = 0.6, w_cagr: float = 0.4) -> pd.Series:
    s_rank = df[sharpe_col].rank(ascending=False, method="min")
    c_rank = df[cagr_col].rank(ascending=False, method="min")
    return w_sharpe * s_rank + w_cagr * c_rank
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_strat_core.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Checkpoint (user commits).**

---

## Task 6: Yahoo fetch + auto-inception helper

**Files:**
- Modify: `winningstratv2/strat_core.py`

This calls the network, so it is verified by a live smoke run rather than an offline unit test.

- [ ] **Step 1: Add `fetch_series` and `load_universe_prices` to `strat_core.py`**

```python
def _get_col(df, col_name, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_name][ticker]
    return df[col_name]


def fetch_series(ticker: str, start: str, end_inclusive: str) -> tuple[pd.Series, pd.Series]:
    """Return (split-adjusted close for signals, auto-adjusted close for returns)."""
    import yfinance as yf

    end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=False, actions=True)
    if px.empty:
        raise ValueError(f"No data for {ticker}")
    close = _get_col(px, "Close", ticker).astype(float)
    splits = _get_col(px, "Stock Splits", ticker).fillna(0.0).astype(float)
    split_factor = splits.replace(0.0, 1.0)
    split_adj = (1.0 / split_factor)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    price_signal = close * split_adj

    tr = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=True)
    price_tr = _get_col(tr, "Close", ticker).astype(float).rename("AdjClose")
    idx = price_signal.index.intersection(price_tr.index)
    return price_signal.loc[idx], price_tr.loc[idx]
```

- [ ] **Step 2: Smoke-test the fetch live**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python -c "
import importlib.util; from pathlib import Path
s=importlib.util.spec_from_file_location('c','strat_core.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c)
sig,tr=c.fetch_series('SPY','2015-01-01','2016-01-01')
print('rows', len(sig), 'first', sig.index.min().date(), 'last', tr.index.max().date())
assert len(sig) > 200
"
```
Expected: ~250 rows, first date early 2015, last late 2015. (If Yahoo rate-limits, retry once.)

- [ ] **Step 3: Checkpoint (user commits).**

---

## Task 7: Tier 1 — per-ticker full-span screen

**Files:**
- Create: `winningstratv2/screen.py`

`screen.py` exposes `run_screen(cfg, cache=None) -> pd.DataFrame` (importable by `run_all.py`) and a `main()` that loads `config.yml`, runs, and writes `outputs/tier1_screen.csv`.

- [ ] **Step 1: Write `screen.py`**

```python
#!/usr/bin/env python3
"""Tier 1: run the locked strat on every candidate over its own full history."""
from __future__ import annotations

import importlib.util
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)


def _resolve_end(value) -> str:
    if value is None or str(value).strip().lower() == "today":
        return date.today().strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_config(path: str = None) -> dict:
    path = path or str(HERE / "config.yml")
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_universe(cfg: dict) -> dict:
    """Return {ticker: theme}. First theme wins if a ticker appears twice."""
    out = {}
    for theme, tickers in cfg["universe"].items():
        for t in tickers:
            out.setdefault(t, theme)
    return out


def build_cache(tickers, end: str, warmup_days: int = 965) -> dict:
    """Fetch each ticker from its earliest available date (auto-inception)."""
    cache = {}
    early = "1990-01-01"
    for t in tickers:
        try:
            sig, tr = core.fetch_series(t, early, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN skipping {t}: {exc}")
            continue
        if sig.empty or tr.empty:
            print(f"  WARN skipping {t}: empty series")
            continue
        cache[t] = {"sig": sig, "tr": tr}
    return cache


def cash_chain_from_cache(cfg: dict, cache: dict) -> pd.Series:
    sgov_name, bil_name = cfg["cash_chain"]
    return core.build_cash_chain(cache[sgov_name]["tr"], cache[bil_name]["tr"])


def run_screen(cfg: dict, cache: dict = None) -> tuple[pd.DataFrame, dict]:
    end = _resolve_end(cfg.get("end"))
    theme_of = flatten_universe(cfg)
    variant = cfg["fixed_variant"]
    needed = sorted(set(theme_of) | set(cfg["cash_chain"]))
    if cache is None:
        print("Fetching price history...")
        cache = build_cache(needed, end)
    for name in cfg["cash_chain"]:
        if name not in cache:
            raise ValueError(f"Cash chain ticker {name} failed to download")
    cash_daily = cash_chain_from_cache(cfg, cache)

    rows = []
    for ticker, theme in theme_of.items():
        if ticker not in cache:
            continue
        sig = cache[ticker]["sig"]
        tr = cache[ticker]["tr"]
        pos = core.hybrid_position(
            sig,
            long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"],
        )
        strat_daily = core.daily_from_pos(pos, tr, cash_daily)
        bh_daily = tr.pct_change().fillna(0.0)
        sm = core.metrics_from_returns(strat_daily)
        bm = core.metrics_from_returns(bh_daily)
        rows.append({
            "ticker": ticker, "theme": theme,
            "start": tr.index.min().strftime("%Y-%m-%d"), "end": tr.index.max().strftime("%Y-%m-%d"),
            "observations": int(strat_daily.dropna().shape[0]),
            "strat_sharpe": sm["Sharpe"], "strat_cagr": sm["CAGR"], "strat_maxdd": sm["MaxDD"],
            "strat_calmar": sm["Calmar"], "strat_ulcer": sm["UlcerIndex"],
            "bh_sharpe": bm["Sharpe"], "bh_cagr": bm["CAGR"], "bh_maxdd": bm["MaxDD"],
            "sharpe_delta": sm["Sharpe"] - bm["Sharpe"],
            "cagr_delta": sm["CAGR"] - bm["CAGR"],
            "maxdd_delta": sm["MaxDD"] - bm["MaxDD"],
        })

    df = pd.DataFrame(rows)
    df["combined_rank_score"] = core.combined_rank(df, "strat_sharpe", "strat_cagr")
    df = df.sort_values(["combined_rank_score", "strat_sharpe"], ascending=[True, False]).reset_index(drop=True)
    df["combined_rank"] = df.index + 1

    # Survivor gate.
    min_days = int(cfg.get("min_history_days", 1500))
    incumbents = set(cfg.get("incumbents", []))
    gate = (df["observations"] >= min_days)
    if cfg.get("gate_require_positive_sharpe_delta", True):
        gate = gate & (df["sharpe_delta"] > 0)
    df["survivor"] = (gate | df["ticker"].isin(incumbents)).astype(int)
    return df, cache


def main():
    cfg = load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    df, _ = run_screen(cfg)
    path = outdir / "tier1_screen.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}  ({df['survivor'].sum()} survivors of {len(df)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run Tier 1 on a 3-ticker temp config**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python -c "
import importlib.util; from pathlib import Path
s=importlib.util.spec_from_file_location('screen','screen.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
cfg=m.load_config()
cfg['universe']={'incumbent':['SPMO','QQQ'],'broad_control':['SPY']}
cfg['cash_chain']=['SGOV','BIL']
df,_=m.run_screen(cfg)
print(df[['ticker','theme','observations','strat_sharpe','bh_sharpe','sharpe_delta','survivor']].to_string(index=False))
assert {'ticker','theme','sharpe_delta','survivor'}.issubset(df.columns)
assert len(df)==3
"
```
Expected: 3 rows with sane Sharpe values (~0.6–1.1), `sharpe_delta` populated, SPMO/QQQ marked survivor.

- [ ] **Step 3: Checkpoint (user commits).**

---

## Task 8: Tier 2 — cross-window robustness on survivors

**Files:**
- Create: `winningstratv2/robustness.py`

This module ports the proven window generators from v1 and replaces the scalar-cash evaluation with the cash chain.

- [ ] **Step 1: Copy the two window generators verbatim**

Into `robustness.py`, copy these functions **unchanged** (they are pure index math):
- From `winningstrat/common_random_sampling.py` lines 79–126: `_generate_windows` (rename to `generate_random_windows`).
- From `winningstrat/structured_window_sampling.py` lines 88–209: `_align_next`, `_align_prev`, `_window_overlap_fraction`, `_make_bucket_windows`, `_generate_windows` (rename the last to `generate_structured_windows`).

Keep their reliance on a module-level `TRADING_DAYS` (set `TRADING_DAYS = core.TRADING_DAYS`).

- [ ] **Step 2: Write the evaluation + ranking body of `robustness.py`**

```python
#!/usr/bin/env python3
"""Tier 2: random + structured window sampling on Tier-1 survivors, ranked by
cross-window consistency (avg delta + beat-rate) vs buy-hold."""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)
TRADING_DAYS = core.TRADING_DAYS

# --- (window generators copied verbatim in Step 1 go here) ---


def _shared_index(cache: dict, tickers, shared_start, end) -> pd.DatetimeIndex:
    common = None
    for t in tickers:
        idx = cache[t]["tr"].loc[shared_start:end].index
        common = idx if common is None else common.intersection(idx)
    return common.sort_values()


def _eval_window(cache, tickers, variant, cash_daily, run_start, run_end):
    out = []
    for t in tickers:
        sig = cache[t]["sig"].loc[run_start:run_end]
        tr = cache[t]["tr"].loc[run_start:run_end]
        pos = core.hybrid_position(
            sig, long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"],
        )
        strat = core.metrics_from_returns(core.daily_from_pos(pos, tr, cash_daily))
        bh = core.metrics_from_returns(tr.pct_change().fillna(0.0))
        out.append({"ticker": t,
                    "sharpe_delta": strat["Sharpe"] - bh["Sharpe"],
                    "cagr_delta": strat["CAGR"] - bh["CAGR"],
                    "beat_sharpe": float(strat["Sharpe"] > bh["Sharpe"]),
                    "beat_cagr": float(strat["CAGR"] > bh["CAGR"]),
                    "strat_sharpe": strat["Sharpe"], "strat_cagr": strat["CAGR"]})
    return out


def run_robustness(cfg: dict, cache: dict, survivors: list, theme_of: dict) -> pd.DataFrame:
    t2 = cfg["tier2"]
    end = pd.Timestamp(date.today() if str(cfg.get("end", "today")).lower() == "today" else cfg["end"])
    if str(t2.get("shared_start", "auto")).lower() == "auto":
        shared_start = max(cache[t]["tr"].index.min() for t in survivors)
    else:
        shared_start = pd.Timestamp(t2["shared_start"])
    cash_daily = core.build_cash_chain(cache[cfg["cash_chain"][0]]["tr"], cache[cfg["cash_chain"][1]]["tr"])

    common = _shared_index(cache, survivors, shared_start, end)
    rand_windows = generate_random_windows(common, t2["random_samples"], t2["random_min_years"],
                                           t2["random_max_years"], t2.get("random_seed"))
    struct_windows = generate_structured_windows(common, t2["structured_buckets"],
                                                 t2["structured_step_fraction"], True, t2["structured_tail_overlap"])

    records = []
    for w in pd.concat([rand_windows.assign(scheme="random"),
                        struct_windows.assign(scheme="structured")], ignore_index=True).itertuples(index=False):
        for r in _eval_window(cache, survivors, cfg["fixed_variant"], cash_daily,
                              pd.Timestamp(w.start), pd.Timestamp(w.end)):
            r["scheme"] = w.scheme
            records.append(r)

    per = pd.DataFrame(records)
    summary = per.groupby("ticker", as_index=False).agg(
        avg_sharpe_delta=("sharpe_delta", "mean"), avg_cagr_delta=("cagr_delta", "mean"),
        sharpe_beat_rate=("beat_sharpe", "mean"), cagr_beat_rate=("beat_cagr", "mean"),
        avg_strat_sharpe=("strat_sharpe", "mean"), avg_strat_cagr=("strat_cagr", "mean"))
    summary["theme"] = summary["ticker"].map(theme_of)
    summary["delta_sharpe_rank"] = summary["avg_sharpe_delta"].rank(ascending=False, method="min")
    summary["delta_cagr_rank"] = summary["avg_cagr_delta"].rank(ascending=False, method="min")
    summary["combined_robust_score"] = (t2["rank_sharpe_weight"] * summary["delta_sharpe_rank"]
                                        + t2["rank_cagr_weight"] * summary["delta_cagr_rank"])
    summary = summary.sort_values(["combined_robust_score", "avg_sharpe_delta"],
                                  ascending=[True, False]).reset_index(drop=True)
    summary["robust_rank"] = summary.index + 1
    summary.attrs["shared_start"] = common.min().strftime("%Y-%m-%d")
    summary.attrs["shared_end"] = common.max().strftime("%Y-%m-%d")
    return summary


def finalists(summary: pd.DataFrame, cfg: dict) -> list:
    n = int(cfg.get("n_finalists", 12))
    picks = summary.head(n)["ticker"].tolist()
    for inc in cfg.get("incumbents", []):
        if inc not in picks and inc in set(summary["ticker"]):
            picks.append(inc)
    return picks


def main():
    import screen as scr
    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(outdir / "tier1_screen.csv")
    survivors = t1[t1["survivor"] == 1]["ticker"].tolist()
    theme_of = scr.flatten_universe(cfg)
    cache = scr.build_cache(sorted(set(survivors) | set(cfg["cash_chain"])), scr._resolve_end(cfg.get("end")))
    summary = run_robustness(cfg, cache, [t for t in survivors if t in cache], theme_of)
    summary.to_csv(outdir / "tier2_robustness.csv", index=False)
    print(f"Saved {outdir/'tier2_robustness.csv'}  finalists: {finalists(summary, cfg)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run Tier 2 on 4 survivors**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python -c "
import importlib.util; from pathlib import Path
def load(n,f):
    s=importlib.util.spec_from_file_location(n,f); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
scr=load('screen','screen.py'); rob=load('robustness','robustness.py')
cfg=scr.load_config(); cfg['tier2']['random_samples']=10
survivors=['SPMO','IWY','QQQ','DGRW']; theme=scr.flatten_universe(cfg)
cache=scr.build_cache(survivors+cfg['cash_chain'], scr._resolve_end(cfg.get('end')))
summ=rob.run_robustness(cfg, cache, survivors, theme)
print(summ[['ticker','theme','avg_sharpe_delta','sharpe_beat_rate','robust_rank']].to_string(index=False))
assert 'sharpe_beat_rate' in summ.columns and len(summ)==4
"
```
Expected: 4 rows ranked, beat-rates in [0,1], `avg_sharpe_delta` populated.

- [ ] **Step 4: Checkpoint (user commits).**

---

## Task 9: Tier 3 — 4-sleeve combo search among finalists

**Files:**
- Create: `winningstratv2/combo_search.py`

- [ ] **Step 1: Write `combo_search.py`**

```python
#!/usr/bin/env python3
"""Tier 3: exhaustive equal-weight 4-sleeve combos among Tier-2 finalists,
scored on a blended robust metric, with a correlation guard and the incumbent
combo printed as the bar to beat."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("strat_core", HERE / "strat_core.py")
core = importlib.util.module_from_spec(_s)
_s.loader.exec_module(core)


def sleeve_daily_returns(cache: dict, tickers, variant, cash_daily) -> dict:
    out = {}
    for t in tickers:
        pos = core.hybrid_position(
            cache[t]["sig"], long_window=variant["long_window"], short_window=variant["short_window"],
            entry_days_long=variant["entry_days_long"], entry_days_short=variant["entry_days_short"],
            exit_days_long=variant["exit_days_long"], exit_days_short=variant["exit_days_short"])
        out[t] = core.daily_from_pos(pos, cache[t]["tr"], cash_daily)
    return out


def _avg_pairwise_corr(cache: dict, tickers) -> float:
    rets = pd.DataFrame({t: cache[t]["tr"].pct_change() for t in tickers}).dropna()
    c = rets.corr().to_numpy()
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(c[iu]))


def run_combo_search(cfg: dict, cache: dict, finalists: list, cash_daily) -> pd.DataFrame:
    variant = cfg["fixed_variant"]
    sw = cfg["tier3"]["score_weights"]
    size = int(cfg["tier3"].get("combo_size", 4))
    sleeves = sleeve_daily_returns(cache, finalists, variant, cash_daily)

    rows = []
    for combo in itertools.combinations(sorted(finalists), size):
        blend = core.combo_blend({t: sleeves[t] for t in combo})
        m = core.metrics_from_returns(blend)
        rows.append({"combo": ",".join(combo), "n_obs": int(blend.dropna().shape[0]),
                     "start": blend.index.min().strftime("%Y-%m-%d"), "end": blend.index.max().strftime("%Y-%m-%d"),
                     "sharpe": m["Sharpe"], "cagr": m["CAGR"], "maxdd": m["MaxDD"],
                     "calmar": m["Calmar"], "avg_corr": _avg_pairwise_corr(cache, combo)})
    df = pd.DataFrame(rows)

    # Blended robust score: z-score each metric (maxdd higher=better since less negative).
    for col, sign in [("sharpe", 1), ("cagr", 1), ("maxdd", 1)]:
        z = (df[col] - df[col].mean()) / (df[col].std(ddof=0) or 1.0)
        df[f"z_{col}"] = sign * z
    df["blended_score"] = sw["sharpe"] * df["z_sharpe"] + sw["cagr"] * df["z_cagr"] + sw["maxdd"] * df["z_maxdd"]
    df = df.sort_values("blended_score", ascending=False).reset_index(drop=True)
    df["combo_rank"] = df.index + 1
    return df


def main():
    import screen as scr
    import robustness as rob
    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    t2 = pd.read_csv(outdir / "tier2_robustness.csv")
    fin = t2.sort_values("robust_rank").head(int(cfg["n_finalists"]))["ticker"].tolist()
    for inc in cfg["incumbents"]:
        if inc not in fin:
            fin.append(inc)
    cache = scr.build_cache(sorted(set(fin) | set(cfg["cash_chain"])), scr._resolve_end(cfg.get("end")))
    cash_daily = scr.cash_chain_from_cache(cfg, cache)
    fin = [t for t in fin if t in cache]
    df = run_combo_search(cfg, cache, fin, cash_daily)
    df.to_csv(outdir / "tier3_combos.csv", index=False)
    inc_combo = ",".join(sorted(cfg["incumbents"]))
    inc_row = df[df["combo"] == inc_combo]
    print(df.head(int(cfg["tier3"]["top_n_report"]))[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "avg_corr", "blended_score"]].to_string(index=False))
    if not inc_row.empty:
        print("\nIncumbents (bar to beat):")
        print(inc_row[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "avg_corr", "blended_score"]].to_string(index=False))
    print(f"\nSaved {outdir/'tier3_combos.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run Tier 3 on 5 finalists**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python -c "
import importlib.util; from pathlib import Path
def load(n,f):
    s=importlib.util.spec_from_file_location(n,f); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
scr=load('screen','screen.py'); cs=load('combo_search','combo_search.py')
cfg=scr.load_config()
fin=['SPMO','IWY','QQQ','DGRW','SCHG']
cache=scr.build_cache(fin+cfg['cash_chain'], scr._resolve_end(cfg.get('end')))
cash=scr.cash_chain_from_cache(cfg,cache)
df=cs.run_combo_search(cfg, cache, fin, cash)
print(df[['combo_rank','combo','sharpe','cagr','maxdd','avg_corr','blended_score']].to_string(index=False))
assert len(df)==5  # C(5,4)=5
assert {'avg_corr','blended_score'}.issubset(df.columns)
"
```
Expected: 5 combos ranked, `avg_corr` ~0.7–0.95 for these correlated growth names, `blended_score` populated.

- [ ] **Step 3: Checkpoint (user commits).**

---

## Task 10: Orchestrator + report + charts

**Files:**
- Create: `winningstratv2/run_all.py`

- [ ] **Step 1: Write `run_all.py`**

```python
#!/usr/bin/env python3
"""Run Tier 1 -> 2 -> 3 with one shared price cache; write report.md + charts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

HERE = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, str(HERE / f))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    core = _load("strat_core", "strat_core.py")
    scr = _load("screen", "screen.py")
    rob = _load("robustness", "robustness.py")
    cs = _load("combo_search", "combo_search.py")

    cfg = scr.load_config()
    outdir = HERE / cfg.get("outdir", "outputs")
    (outdir / "charts").mkdir(parents=True, exist_ok=True)
    theme_of = scr.flatten_universe(cfg)

    # One shared cache for the entire universe + cash chain.
    needed = sorted(set(theme_of) | set(cfg["cash_chain"]))
    cache = scr.build_cache(needed, scr._resolve_end(cfg.get("end")))
    cash_daily = scr.cash_chain_from_cache(cfg, cache)

    t1, cache = scr.run_screen(cfg, cache=cache)
    t1.to_csv(outdir / "tier1_screen.csv", index=False)
    survivors = [t for t in t1[t1["survivor"] == 1]["ticker"] if t in cache]

    t2 = rob.run_robustness(cfg, cache, survivors, theme_of)
    t2.to_csv(outdir / "tier2_robustness.csv", index=False)
    fin = rob.finalists(t2, cfg)
    fin = [t for t in fin if t in cache]

    t3 = cs.run_combo_search(cfg, cache, fin, cash_daily)
    t3.to_csv(outdir / "tier3_combos.csv", index=False)

    # Charts: incumbents vs best combo.
    best = t3.iloc[0]["combo"].split(",")
    inc = sorted(cfg["incumbents"])
    sleeves = cs.sleeve_daily_returns(cache, sorted(set(best) | set(inc)), cfg["fixed_variant"], cash_daily)
    best_curve = (1 + core.combo_blend({t: sleeves[t] for t in best})).cumprod()
    inc_curve = (1 + core.combo_blend({t: sleeves[t] for t in inc})).cumprod()
    if plt is not None:
        eq = pd.DataFrame({"best_combo": best_curve, "incumbents": inc_curve}).dropna()
        ax = eq.plot(figsize=(12, 7), title="Best robust combo vs incumbents (growth of $1)")
        ax.set_ylabel("Growth of $1"); ax.grid(True, alpha=0.25); ax.figure.tight_layout()
        ax.figure.savefig(outdir / "charts" / "equity_incumbents_vs_best.png", dpi=160)
        dd = eq / eq.cummax() - 1.0
        ax2 = dd.plot(figsize=(12, 7), title="Drawdown: best robust combo vs incumbents")
        ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.25); ax2.figure.tight_layout()
        ax2.figure.savefig(outdir / "charts" / "drawdown_incumbents_vs_best.png", dpi=160)

    # Report.
    top = t3.head(int(cfg["tier3"]["top_n_report"]))
    inc_combo = ",".join(inc)
    inc_row = t3[t3["combo"] == inc_combo]
    lines = ["# winningstratv2 — Ticker-Swap Study Report", "",
             f"Universe screened: {len(t1)} tickers. Survivors to Tier 2: {len(survivors)}. "
             f"Finalists to Tier 3: {len(fin)}.",
             f"Tier-2 shared window: {t2.attrs.get('shared_start')} to {t2.attrs.get('shared_end')}.", "",
             "## Top individual candidates (Tier 1, by combined rank)", "",
             t1.head(15)[["combined_rank", "ticker", "theme", "strat_sharpe", "strat_cagr",
                          "sharpe_delta", "cagr_delta"]].to_markdown(index=False), "",
             "## Robustness finalists (Tier 2)", "",
             t2.head(int(cfg["n_finalists"]))[["robust_rank", "ticker", "theme", "avg_sharpe_delta",
                          "sharpe_beat_rate", "cagr_beat_rate"]].to_markdown(index=False), "",
             "## Best 4-sleeve combos (Tier 3)", "",
             top[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "calmar", "avg_corr",
                  "blended_score"]].to_markdown(index=False), ""]
    if not inc_row.empty:
        lines += ["## Incumbents (bar to beat)", "",
                  inc_row[["combo_rank", "combo", "sharpe", "cagr", "maxdd", "calmar", "avg_corr",
                           "blended_score"]].to_markdown(index=False), ""]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outdir/'report.md'} and charts/. Best combo: {best}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline end to end**

Run:
```bash
cd /home/wes/github/backtests/winningstratv2
.venv/bin/python run_all.py
```
Expected: prints survivor/finalist counts and "Best combo: [...]"; creates `outputs/tier1_screen.csv`, `outputs/tier2_robustness.csv`, `outputs/tier3_combos.csv`, `outputs/report.md`, and two PNGs under `outputs/charts/`. (Full ~70-ticker fetch may take several minutes; retry once if Yahoo rate-limits.)

- [ ] **Step 3: Sanity-check the report**

Run: `sed -n '1,40p' outputs/report.md`
Expected: header, Tier-1 top table, Tier-2 finalists, Tier-3 top combos, and an incumbents row. Confirm the incumbent combo's `blended_score` is present so "did we beat it?" is answerable.

- [ ] **Step 4: Checkpoint (user commits).**

---

## Task 11: README + final verification

**Files:**
- Create: `winningstratv2/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# winningstratv2 — Ticker-Swap Study

Holds the "best winning strat" fixed (`200SMA_3in_2out_hybrid20SMA`) and screens a
~70-ticker universe through a robustness-gated funnel to find sleeves that beat the
incumbents `SPMO/IWY/QQQ/DGRW`. See `DESIGN.md` for the rationale and `PLAN.md` for
the build steps.

## Setup
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## Run
    .venv/bin/python run_all.py            # full Tier 1 -> 2 -> 3 + report
    .venv/bin/python screen.py             # Tier 1 only
    .venv/bin/python robustness.py         # Tier 2 (reads outputs/tier1_screen.csv)
    .venv/bin/python combo_search.py       # Tier 3 (reads outputs/tier2_robustness.csv)

## Strategy (locked)
SMA20 >= SMA200: enter after 3 closes above SMA200, exit after 2 at/below.
SMA20 <  SMA200: enter after 3 days close>SMA20 & SMA20 rising, exit after 1 at/below SMA20.
Out of market: hold SGOV (BIL before SGOV's 2023 inception), identical for every ticker.

## Outputs (in `outputs/`)
- `tier1_screen.csv` — per-ticker full-span strat vs buy-hold, themed, ranked, survivor flag.
- `tier2_robustness.csv` — cross-window avg deltas + beat-rates; finalists.
- `tier3_combos.csv` — every 4-sleeve combo ranked by blended robust score, with avg pairwise correlation.
- `report.md` — narrative with the incumbents as the explicit bar to beat.
- `charts/` — best combo vs incumbents equity + drawdown.

## Anti-overfitting
Strategy is fixed; only tickers vary. Finalists are chosen by cross-window beat-rate
(not a single full-span estimate); newer tickers are judged on shared windows; combos
are reported top-N (not one) with a correlation guard; incumbents are the benchmark.

## Tests
    .venv/bin/python -m pytest tests/ -q
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/wes/github/backtests/winningstratv2 && .venv/bin/python -m pytest tests/ -q`
Expected: 8 passed.

- [ ] **Step 3: Confirm output files exist**

Run: `ls -1 outputs outputs/charts`
Expected: the four CSV/MD files and two PNGs from Task 10.

- [ ] **Step 4: Checkpoint (user commits).**

---

## Self-Review Notes

- **Spec coverage:** locked variant (Task 1 config, Task 2), SGOV→BIL cash chain (Task 4), auto-inception (Task 6/7 `build_cache` from 1990), ~70 themed universe (Task 1 + count check), Tier 1 screen with gate (Task 7), Tier 2 random+structured robustness with beat-rate ranking (Task 8), Tier 3 combo search with blended score + correlation guard + incumbents benchmark (Task 9), report + charts (Task 10), tests + README (Task 2–5, 11). All spec sections map to a task.
- **Type consistency:** `hybrid_position`, `metrics_from_returns` (keys incl. `Calmar`), `build_cash_chain`, `daily_from_pos`, `combo_blend`, `combined_rank`, `fetch_series` signatures are fixed in the contract and used identically across `screen.py`, `robustness.py`, `combo_search.py`, `run_all.py`. `run_screen` returns `(df, cache)`; `run_all` consumes that tuple.
- **Cash pre-2007 caveat:** before BIL's inception the cash return is 0 (documented in `build_cash_chain`); identical across same-era tickers so it does not bias ranking.
- **Placeholders:** none — every code step contains full code; the only "copy verbatim" instruction (Task 8 Step 1) points to exact in-repo files and line ranges for pure, unchanged index math.
```
