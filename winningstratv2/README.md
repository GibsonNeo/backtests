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
