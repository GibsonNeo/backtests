# Winningstrat vs Rotation Comparison

Top winningstrat tickers used: SPMO, IWY, QQQ, DGRW

`QQQ` is used instead of `QQQM` to avoid limiting the test to QQQM's 2020 inception.

## Summary

| strategy                                    | start      | end        |   months |      cagr |   annualized_volatility |   sharpe |   sortino |   max_drawdown |   calmar |   best_year |   worst_year |   monthly_win_rate |   total_return |   switches |
|:--------------------------------------------|:-----------|:-----------|---------:|----------:|------------------------:|---------:|----------:|---------------:|---------:|------------:|-------------:|-------------------:|---------------:|-----------:|
| winningstrat_top4_bil_cash                  | 2017-03-31 | 2026-05-31 |      111 | 0.185537  |                0.130403 | 1.2013   |   1.97288 |      -0.144936 | 1.28013  |    0.305584 |   -0.0909914 |           0.684685 |        3.82743 |          0 |
| winningstrat_top4_zero_cash                 | 2017-03-31 | 2026-05-31 |      111 | 0.182155  |                0.130809 | 1.17584  |   1.91717 |      -0.146031 | 1.24737  |    0.305584 |   -0.0973034 |           0.657658 |        3.70151 |          0 |
| hybrid_rotation_gate_or_winning_daily_bil   | 2017-03-31 | 2026-05-31 |      111 | 0.195368  |                0.142908 | 1.16669  |   1.87633 |      -0.154005 | 1.26858  |    0.307806 |   -0.075393  |           0.693694 |        4.21062 |          0 |
| rotation_top4_bil                           | 2017-03-31 | 2026-05-31 |      111 | 0.137412  |                0.118358 | 0.956471 |   1.41144 |      -0.109966 | 1.24959  |    0.305585 |   -0.0895716 |           0.693694 |        2.29034 |          0 |
| hybrid_rotation_gate_plus_winning_daily_bil | 2017-03-31 | 2026-05-31 |      111 | 0.126676  |                0.118237 | 0.876381 |   1.27583 |      -0.13574  | 0.93323  |    0.305584 |   -0.104574  |           0.675676 |        2.01399 |          0 |
| rotation_original_cefs_bil                  | 2017-03-31 | 2026-05-31 |      111 | 0.0880645 |                0.106785 | 0.627883 |   1.09125 |      -0.124219 | 0.708947 |    0.298628 |   -0.124219  |           0.630631 |        1.18301 |          0 |

## Strategy Definitions

- `rotation_original_cefs_bil`: monthly 3+7 momentum rotation on `ADX, RQI, KYN, NMZ`, fallback `BIL`.
- `rotation_top4_bil`: same monthly 3+7 momentum rotation on `SPMO, IWY, QQQ, DGRW`, fallback `BIL`.
- `winningstrat_top4_zero_cash`: daily `hybridL200_S20_eL3_eS2_xL2_xS1` on `SPMO, IWY, QQQ, DGRW`, zero-return cash when out.
- `winningstrat_top4_bil_cash`: same daily rule, but out-of-market sleeves hold `BIL`.
- `hybrid_rotation_gate_plus_winning_daily_bil`: top-four sleeves hold a ticker only when the prior month-end 3+7 momentum gate is long and the daily winningstrat state is long; otherwise the sleeve holds `BIL`.
- `hybrid_rotation_gate_or_winning_daily_bil`: top-four sleeves hold a ticker when either the prior month-end 3+7 momentum gate or the daily winningstrat state is long; otherwise the sleeve holds `BIL`.

## Files

- `comparison_summary.csv`
- `comparison_monthly_returns.csv`
- `comparison_daily_returns.csv`
- `comparison_equity_curves.csv`
- `comparison_drawdowns.csv`
- `winningstrat_variant_grid_bil.csv`
- `charts/comparison_equity.png`
- `charts/comparison_drawdowns.png`
