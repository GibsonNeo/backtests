# Ticker Rankings For `hybridL200_S20_eL3_eS2_xL2_xS1`

These rankings use the original zero-cash winning strategy:

- `hybridL200_S20_eL3_eS2_xL2_xS1`
- Data source: `outputs/per_ticker_full_span.csv`
- Combined rank formula: `60% Sharpe rank + 40% CAGR rank`
- Lower combined score is better

Note: each ticker uses its own full available history from its configured inception date through `2026-03-26`, so the test spans are not identical across tickers.

## Combined Rank

| Rank | Ticker | Strategy Sharpe | Strategy CAGR | Buy/Hold Sharpe | Buy/Hold CAGR | Sharpe Delta | CAGR Delta | Sharpe Rank | CAGR Rank | Combined Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | SPMO | 1.074 | 15.45% | 0.878 | 16.65% | +0.196 | -1.19% | 1 | 1 | 1.0 |
| 2 | IWY | 0.987 | 14.19% | 0.892 | 16.27% | +0.095 | -2.08% | 2 | 3 | 2.4 |
| 3 | QQQM | 0.977 | 14.68% | 0.692 | 13.79% | +0.285 | +0.89% | 3 | 2 | 2.6 |
| 4 | DGRW | 0.890 | 9.55% | 0.829 | 12.38% | +0.061 | -2.82% | 4 | 7 | 5.2 |
| 5 | AUSF | 0.884 | 10.62% | 0.701 | 12.33% | +0.182 | -1.72% | 6 | 5 | 5.6 |
| 6 | SCHG | 0.859 | 13.43% | 0.818 | 15.29% | +0.041 | -1.86% | 7 | 4 | 5.8 |
| 7 | DGRO | 0.888 | 9.53% | 0.766 | 11.74% | +0.122 | -2.22% | 5 | 8 | 6.2 |
| 8 | EQWL | 0.847 | 9.72% | 0.625 | 10.19% | +0.222 | -0.47% | 8 | 6 | 7.2 |
| 9 | SPHQ | 0.659 | 7.98% | 0.565 | 9.45% | +0.093 | -1.47% | 9 | 9 | 9.0 |

## Strategy Vs Buy/Hold

| Ticker | Strategy Sharpe | Strategy CAGR | Buy/Hold Sharpe | Buy/Hold CAGR | Sharpe Delta | CAGR Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AUSF | 0.884 | 10.62% | 0.701 | 12.33% | +0.182 | -1.72% |
| DGRO | 0.888 | 9.53% | 0.766 | 11.74% | +0.122 | -2.22% |
| DGRW | 0.890 | 9.55% | 0.829 | 12.38% | +0.061 | -2.82% |
| EQWL | 0.847 | 9.72% | 0.625 | 10.19% | +0.222 | -0.47% |
| IWY | 0.987 | 14.19% | 0.892 | 16.27% | +0.095 | -2.08% |
| QQQM | 0.977 | 14.68% | 0.692 | 13.79% | +0.285 | +0.89% |
| SCHG | 0.859 | 13.43% | 0.818 | 15.29% | +0.041 | -1.86% |
| SPHQ | 0.659 | 7.98% | 0.565 | 9.45% | +0.093 | -1.47% |
| SPMO | 1.074 | 15.45% | 0.878 | 16.65% | +0.196 | -1.19% |

## Sharpe Order

| Rank | Ticker | Sharpe | CAGR |
| --- | --- | ---: | ---: |
| 1 | SPMO | 1.074 | 15.45% |
| 2 | IWY | 0.987 | 14.19% |
| 3 | QQQM | 0.977 | 14.68% |
| 4 | DGRW | 0.890 | 9.55% |
| 5 | DGRO | 0.888 | 9.53% |
| 6 | AUSF | 0.884 | 10.62% |
| 7 | SCHG | 0.859 | 13.43% |
| 8 | EQWL | 0.847 | 9.72% |
| 9 | SPHQ | 0.659 | 7.98% |

## CAGR Order

| Rank | Ticker | CAGR | Sharpe |
| --- | --- | ---: | ---: |
| 1 | SPMO | 15.45% | 1.074 |
| 2 | QQQM | 14.68% | 0.977 |
| 3 | IWY | 14.19% | 0.987 |
| 4 | SCHG | 13.43% | 0.859 |
| 5 | AUSF | 10.62% | 0.884 |
| 6 | EQWL | 9.72% | 0.847 |
| 7 | DGRW | 9.55% | 0.890 |
| 8 | DGRO | 9.53% | 0.888 |
| 9 | SPHQ | 7.98% | 0.659 |

## Shared Random Sampling Rank

This section uses the same strategy on a revised ticker set with shared-date random windows:

- Tickers: `DGRO, DGRW, EQWL, SPHQ, QQQ, SCHG, IWY, SPMO`
- Removed: `AUSF`
- Replaced: `QQQM` with `QQQ`
- Configured shared start: `2015-10-09`
- Actual first shared trading date used: `2015-10-12`
- Dynamic end date used for this run: `2026-03-26`
- Sample count: `100`
- Sample length range: about `3.06` to `10.43` years
- Average sample length: about `6.77` years
- Data sources: `outputs_common_random_sampling/sample_windows.csv` and `outputs_common_random_sampling/ticker_ranking_vs_buyhold.csv`
- Combined rank formula: `60% delta-Sharpe rank + 40% delta-CAGR rank`
- Lower combined score is better

Note: these rankings are based on average performance across shared random windows, not per-ticker full-history since inception.

| Rank | Ticker | Strategy Sharpe | Strategy CAGR | Buy/Hold Sharpe | Buy/Hold CAGR | Avg Sharpe Delta | Avg CAGR Delta | Sharpe Beat Rate | CAGR Beat Rate | Combined Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | SPHQ | 0.937 | 11.62% | 0.786 | 14.03% | +0.151 | -2.41% | 88% | 4% | 1.6 |
| 2 | SPMO | 1.007 | 14.38% | 0.855 | 17.01% | +0.152 | -2.63% | 92% | 4% | 1.8 |
| 3 | IWY | 1.030 | 16.47% | 0.905 | 18.96% | +0.125 | -2.49% | 83% | 7% | 2.6 |
| 4 | EQWL | 0.903 | 10.64% | 0.799 | 13.51% | +0.105 | -2.87% | 86% | 4% | 4.4 |
| 5 | QQQ | 0.992 | 16.97% | 0.893 | 19.83% | +0.099 | -2.86% | 82% | 12% | 4.6 |
| 6 | DGRO | 0.820 | 9.10% | 0.757 | 12.65% | +0.063 | -3.55% | 79% | 2% | 6.4 |
| 7 | SCHG | 0.865 | 14.96% | 0.855 | 18.13% | +0.010 | -3.18% | 53% | 4% | 6.6 |
| 8 | DGRW | 0.828 | 9.14% | 0.828 | 13.68% | -0.000 | -4.54% | 59% | 0% | 8.0 |

## Shared Random Sampling Vs Buy/Hold

| Ticker | Strategy Sharpe | Strategy CAGR | Buy/Hold Sharpe | Buy/Hold CAGR | Avg Sharpe Delta | Avg CAGR Delta | Sharpe Beat Rate | CAGR Beat Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DGRO | 0.820 | 9.10% | 0.757 | 12.65% | +0.063 | -3.55% | 79% | 2% |
| DGRW | 0.828 | 9.14% | 0.828 | 13.68% | -0.000 | -4.54% | 59% | 0% |
| EQWL | 0.903 | 10.64% | 0.799 | 13.51% | +0.105 | -2.87% | 86% | 4% |
| IWY | 1.030 | 16.47% | 0.905 | 18.96% | +0.125 | -2.49% | 83% | 7% |
| QQQ | 0.992 | 16.97% | 0.893 | 19.83% | +0.099 | -2.86% | 82% | 12% |
| SCHG | 0.865 | 14.96% | 0.855 | 18.13% | +0.010 | -3.18% | 53% | 4% |
| SPHQ | 0.937 | 11.62% | 0.786 | 14.03% | +0.151 | -2.41% | 88% | 4% |
| SPMO | 1.007 | 14.38% | 0.855 | 17.01% | +0.152 | -2.63% | 92% | 4% |
