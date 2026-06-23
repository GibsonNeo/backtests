# Missed Re-entry Fallback Experiment

Portfolio sleeves: IWF, SPMO, OEF, DGRO, XLI. Equal-weight blend. Window: 2015-10-12 to 2026-06-23.

Method: every historical baseline cash-to-long transition is intentionally missed. The fallback variant stays in the SGOV/BIL cash chain until its SMA-based rule fires while the original signal remains long; after joining, it follows the original exit.

This is a research backtest, not investment advice.

## Portfolio Results

| variant                                       | CAGR   |   Sharpe | MaxDD   |   Calmar |   UlcerIndex | TotalMultiple   | dCAGR_pp   |   dSharpe | dMaxDD_pp   |
|:----------------------------------------------|:-------|---------:|:--------|---------:|-------------:|:----------------|:-----------|----------:|:------------|
| baseline: follow original signal              | 15.39% |    1.261 | -12.80% |     1.2  |         4.46 | 4.60x           | +0.00pp    |     0     | +0.00pp     |
| above rising SMA50                            | 14.42% |    1.249 | -12.61% |     1.14 |         4.33 | 4.21x           | -0.96pp    |    -0.012 | +0.19pp     |
| first reclaim among SMA20/50/100              | 14.12% |    1.263 | -12.85% |     1.1  |         3.95 | 4.09x           | -1.26pp    |     0.002 | -0.05pp     |
| above rising SMA20                            | 16.00% |    1.317 | -12.61% |     1.27 |         4.05 | 4.87x           | +0.61pp    |     0.056 | +0.19pp     |
| reclaim SMA100 after pullback/touch           | 12.91% |    1.288 | -11.10% |     1.16 |         4.18 | 3.65x           | -2.48pp    |     0.027 | +1.70pp     |
| above rising SMA100                           | 13.19% |    1.194 | -12.61% |     1.05 |         4.84 | 3.75x           | -2.19pp    |    -0.067 | +0.19pp     |
| touch/undercut SMA100                         | 12.03% |    1.123 | -12.94% |     0.93 |         5.16 | 3.36x           | -3.35pp    |    -0.138 | -0.14pp     |
| reclaim SMA200 after pullback/touch           | 9.76%  |    1.228 | -9.31%  |     1.05 |         3.54 | 2.70x           | -5.63pp    |    -0.033 | +3.49pp     |
| reclaim SMA20 after pullback/touch            | 11.38% |    1.132 | -12.61% |     0.9  |         4.5  | 3.16x           | -4.00pp    |    -0.129 | +0.19pp     |
| reclaim SMA50 after pullback/touch            | 10.89% |    1.127 | -12.85% |     0.85 |         3.88 | 3.01x           | -4.50pp    |    -0.134 | -0.05pp     |
| first touch among SMA20/50/100                | 12.31% |    1.066 | -14.76% |     0.83 |         5.47 | 3.45x           | -3.07pp    |    -0.195 | -1.96pp     |
| touch/undercut SMA200                         | 9.30%  |    1.047 | -12.07% |     0.77 |         3.86 | 2.58x           | -6.09pp    |    -0.214 | +0.73pp     |
| touch/undercut SMA50                          | 8.55%  |    0.866 | -14.13% |     0.61 |         6.34 | 2.40x           | -6.84pp    |    -0.395 | -1.33pp     |
| touch/undercut SMA20                          | 7.02%  |    0.692 | -20.76% |     0.34 |         8.26 | 2.06x           | -8.37pp    |    -0.569 | -7.96pp     |
| no fallback: stay bonds until next full cycle | 2.14%  |    8.454 | -0.09%  |    24.43 |         0.02 | 1.25x           | -13.25pp   |     7.193 | +12.71pp    |

## Current Sleeve State

| ticker   | latest_date   | in_signal_now   | last_entry   | last_exit   |   days_since_entry |   entry_adj_price |   last_adj_price | pct_from_entry   |   signal_price |   sma20 | pct_vs_sma20   |   sma50 | pct_vs_sma50   |   sma100 | pct_vs_sma100   |   sma200 | pct_vs_sma200   |
|:---------|:--------------|:----------------|:-------------|:------------|-------------------:|------------------:|-----------------:|:-----------------|---------------:|--------:|:---------------|--------:|:---------------|---------:|:----------------|---------:|:----------------|
| IWF      | 2026-06-23    | True            | 2026-04-15   | 2026-03-03  |                 47 |            117.12 |           119.8  | 2.28%            |         119.8  |  124.07 | -3.44%         |  100.91 | 18.72%         |    64.37 | 86.11%          |    46.91 | 155.37%         |
| SPMO     | 2026-06-23    | True            | 2026-04-13   | 2026-03-20  |                 49 |            124.4  |           154.3  | 24.04%           |         154.3  |  152.25 | 1.35%          |  142.46 | 8.31%          |   130.13 | 18.57%          |   124.99 | 23.45%          |
| OEF      | 2026-06-23    | True            | 2026-04-15   | 2026-03-20  |                 47 |            344.43 |           360.37 | 4.63%            |         360.37 |  369.32 | -2.42%         |  362.48 | -0.58%         |   346.75 | 3.93%           |   342.59 | 5.19%           |
| DGRO     | 2026-06-23    | True            | 2025-05-07   | 2025-04-04  |                282 |             58.88 |            75.13 | 27.61%           |          75.13 |   75.13 | 0.01%          |   73.96 | 1.58%          |    72.92 | 3.04%           |    70.92 | 5.93%           |
| XLI      | 2026-06-23    | True            | 2025-05-07   | 2025-03-31  |                282 |            132.04 |           178.13 | 34.90%           |         178.13 |  175.79 | 1.33%          |  173.79 | 2.50%          |   171.61 | 3.80%           |   163.43 | 9.00%           |

## Baseline Entry Episode Count

| ticker   |   baseline_entry_episodes |
|:---------|--------------------------:|
| IWF      |                        59 |
| SPMO     |                        19 |
| OEF      |                        72 |
| DGRO     |                        27 |
| XLI      |                        81 |
