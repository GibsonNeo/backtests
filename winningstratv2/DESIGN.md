# winningstratv2 — Ticker-Swap Study Design

## Goal

Hold the "best winning strat" fixed and screen a wide candidate universe (~70 ETFs)
to find tickers that perform better than the current 4 sleeves
(`SPMO, IWY, QQQ, DGRW`) — while controlling overfitting through a tiered,
robustness-gated funnel rather than crowning a single in-sample maximum.

The deliverable is a ranking of candidates and a short list of 4-sleeve
combinations that beat the incumbents on *robust* (cross-period) criteria, not on
a single lucky historical path.

## Fixed strategy (locked)

Variant `200SMA_3in_2out_hybrid20SMA`, identical to the readme's "BEST WINNING STRAT":

```
If SMA20 >= SMA200:
    Enter after 3 closes above SMA200
    Exit  after 2 closes at/below SMA200
If SMA20 < SMA200:
    Enter after 3 days where close > SMA20 and SMA20 is rising
    Exit  after 1 close at/below SMA20
When out: hold the cash chain (below)
```

Parameters: `long_window=200, short_window=20, entry_days_long=3,
entry_days_short=3, exit_days_long=2, exit_days_short=1`. The slope gate is a
plain day-over-day rise in SMA20 (no percentage threshold). Signals use the
split-adjusted close; returns use the total-return (auto-adjusted) series.
Positions are shifted one day to avoid look-ahead. This logic is ported verbatim
from `winningstrat/winningstrat.py::hybrid_position_no_cooldown`.

## Out-of-market holding (cash chain)

When flat, the sleeve earns a real T-bill return: **SGOV** when it exists
(2023-05+), **BIL** before that. The same chain is applied identically to every
candidate so it never affects relative ranking, only absolute levels. Built once
and reused across all tiers.

## Candidate universe (~70, tagged by theme)

Auto-inception: each ticker's start date is the earliest date `yfinance`
returns — no hand-maintained inception table. A ticker is dropped with a logged
warning if it has no data or fewer than a configurable minimum of trading days.

| Theme | Tickers |
|---|---|
| `incumbent` | SPMO, IWY, QQQ, DGRW |
| `momentum` | MTUM, PDP, FDMO, XMMO, QMOM, VFMO, JMOM |
| `large_growth` | SCHG, VUG, IWF, MGK, SPYG, VONG, IUSG, XLG, MGC |
| `tech` | VGT, XLK, IYW, FTEC, IGM, SOXX, SMH, IGV |
| `quality` | QUAL, SPHQ, JQUA, DGRO, GSLC |
| `dividend_growth` | SCHD, VIG, NOBL, RDVY |
| `multifactor` | OMFL, LRGF, DSTL, SPGP |
| `broad_control` | RSP, EQWL, SPY, OEF |
| `midcap_growth` | IWP, VOT, ARKK |
| `sector` | XLY, XLV, XLF, XLI, XLP, XLU, XLE, XLB, XLRE, XLC, XBI, ITA, GLD |
| `international` | EFA, VEA, EFG, IMTM, IQLT, VWO, VEU, INDA, EWT |

Total: 70. The `broad_control` tickers (SPY/RSP/OEF) and equal-weight EQWL are
deliberate controls: if the strat doesn't beat buy-hold on them, the edge is
concentrated in growth/momentum, not the timing rule itself. Sector,
international, and gold are included for genuine diversification value in the
Tier-3 combo search, not because they are expected to win individually.

## Architecture

Self-contained folder; reuses proven logic but does not import from v1.

```
winningstratv2/
  config.yml              # universe, fixed variant, cash chain, window + tier settings
  strat_core.py           # hybrid position, metrics, fetch, cash chain (shared)
  screen.py               # Tier 1: per-ticker full-span screen
  robustness.py           # Tier 2: random + structured window sampling
  combo_search.py         # Tier 3: 4-sleeve combo search among finalists
  run_all.py              # orchestrates Tier 1 -> 2 -> 3, writes report.md
  requirements.txt        # numpy, pandas, yfinance, pyyaml, matplotlib, tabulate
  outputs/                # CSVs, report.md, charts/
  tests/test_strat_core.py
```

`strat_core.py` is the single source of truth for: split-adjusted + total-return
fetch, `hybrid_position_no_cooldown`, `metrics_from_returns` (CAGR, AnnVol,
Sharpe, MaxDD, Ulcer, Calmar, TotalReturn), the SGOV/BIL cash chain, and the
daily-return-from-position helper. Each tier module is independently runnable and
also callable by `run_all.py`.

## Tiered pipeline

### Tier 1 — individual full-span screen (`screen.py`)

Run the fixed strat on every candidate over its own full history. For each
ticker compute strat metrics, buy-hold metrics, and the deltas.

- Output: `outputs/tier1_screen.csv` — columns: ticker, theme, start, end,
  observations, strat Sharpe/CAGR/MaxDD/Ulcer/Calmar, buyhold Sharpe/CAGR/MaxDD,
  Sharpe delta, CAGR delta, MaxDD delta, combined rank (`0.6*Sharpe_rank +
  0.4*CAGR_rank`, lower better — same formula as the v1 rankings file).
- Gate to Tier 2: `Sharpe_delta > 0` (strat improves risk-adjusted return vs
  buy-hold) AND `observations >= min_history_days` (default ≈ 6 years). The gate
  thresholds are config-driven.

### Tier 2 — cross-window robustness (`robustness.py`)

For Tier-1 survivors, run two samplers on a **shared date range** (so newer and
older tickers are judged on identical periods):

1. Random shared-date windows (ported from
   `winningstrat/common_random_sampling.py`).
2. Structured rolling windows — fixed 3y/5y/7y/10y buckets (ported from
   `winningstrat/structured_window_sampling.py`).

Rank by **consistency**: average Sharpe/CAGR delta vs buy-hold AND beat-rate
(share of windows where the strat beats buy-hold). This is the primary
anti-overfit signal — it rewards tickers that work across regimes, not those
flattered by one bull run.

- Output: `outputs/tier2_robustness.csv` — per ticker: avg Sharpe delta, avg
  CAGR delta, Sharpe beat-rate, CAGR beat-rate, combined robust rank, plus theme.
- Finalists: top ~10–12 by combined robust rank (config-driven `n_finalists`),
  always including the 4 incumbents for reference.

### Tier 3 — 4-sleeve combo search among finalists (`combo_search.py`)

Exhaustive equal-weight 4-ticker combinations of the Tier-2 finalists only
(`C(12,4) = 495` combos at the default size — small and fast). Each combo is run
as an equal-weight daily blend of the per-sleeve strat returns over the combo's
own common-overlap window.

Selection is robustness-gated, never the single in-sample max:

- Score each combo on a **blended robust metric** (Sharpe + CAGR + drawdown,
  weights config-driven), not raw full-span Sharpe.
- Report the **top ~10 combos**, not one.
- For each reported combo, show **average pairwise correlation** of the four
  underlying buy-hold return series (diversification guard — flags "four clones
  of large-cap growth").
- Print the **incumbent `SPMO/IWY/QQQ/DGRW`** combo inline as the explicit bar
  to beat; a new combo only "wins" if it beats incumbents on the blended robust
  score.

- Output: `outputs/tier3_combos.csv` (all combos ranked) + the curated top-10 in
  `report.md`.

### Orchestration (`run_all.py`)

Runs Tier 1 → 2 → 3 in sequence, reusing one fetched price cache, and writes:

- `outputs/report.md` — narrative: top individual candidates by theme, the
  robustness finalists, the top-10 combos with correlation, and a head-to-head
  vs the incumbents.
- `outputs/charts/` — equity + drawdown for the incumbents vs the best
  robust combo.

## Anti-overfitting principles (the spine of this study)

1. Strategy is fixed — only tickers vary. No parameter tuning per ticker.
2. Selection is driven by **cross-window beat-rate**, not a single full-span
   point estimate.
3. Newer tickers are judged on **shared windows** so missing bear markets give
   no artificial edge.
4. We report **top-N**, not a single winner, and inspect **membership
   stability** across metrics/windows.
5. **Correlation guard** prevents picking four near-identical sleeves.
6. **Incumbents are the benchmark** — a swap must beat them on robust criteria,
   not just on raw history.

## Config schema (`config.yml`)

```yaml
universe:            # theme -> [tickers]
fixed_variant:       # the locked hybrid params
cash_chain: [SGOV, BIL]
end: <date or 'today'>
min_history_days: 1500          # ~6 years; Tier-1 -> Tier-2 gate
tier2:
  shared_start: auto            # or explicit date
  random_samples: 100
  structured_buckets: [3, 5, 7, 10]
n_finalists: 12
tier3:
  combo_size: 4
  score_weights: {sharpe: 0.5, cagr: 0.3, maxdd: 0.2}
  top_n_report: 10
outdir: outputs
```

## Outputs

- `outputs/tier1_screen.csv`
- `outputs/tier2_robustness.csv`
- `outputs/tier3_combos.csv`
- `outputs/report.md`
- `outputs/charts/equity_incumbents_vs_best.png`
- `outputs/charts/drawdown_incumbents_vs_best.png`

## Testing

`tests/test_strat_core.py` uses synthetic series only (no Yahoo calls):

- Hybrid position matches v1 logic on a known constructed series.
- Cash chain splices SGOV after its start and BIL before, with no gaps/overlap.
- Metrics math (CAGR, Sharpe, MaxDD, Calmar) correct on a hand-checkable series.
- Combo blend equals the mean of per-sleeve returns on the overlap window.

## Decisions locked

- Variant: `200SMA_3in_2out_hybrid20SMA` (eL3/eS3/xL2/xS1).
- Out-of-market: SGOV→BIL real T-bill chain, identical across tickers.
- Universe: ~70 tickers, all retained, tagged by theme.
- Selection: tiered funnel, robustness-gated; incumbents as the bar to beat.

## Assumptions / open items

- Data source is Yahoo via `yfinance`; auto-adjusted close for returns,
  split-adjusted close for signals (same convention as v1).
- Equal weight within each 4-sleeve combo (matches the v1 common-overlap model).
- Spec lives in `winningstratv2/DESIGN.md` (not the skill default `docs/` path)
  to keep the study self-contained. Per project policy, this file is **not**
  committed by the assistant.
