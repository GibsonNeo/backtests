This directory isolates the winning regime-switch strategy and a few close variants.

Signal semantics:
- The trigger compares the ETF's split-adjusted close against the SMA line.
- The regime switch itself is based on whether `SMA20` is above or below `SMA200`.
- The short-slope gate is a simple `SMA20[t] > SMA20[t-1]`. There is no percentage threshold.

Primary winning rule:
- If `SMA20 >= SMA200`: enter after 2 closes above `SMA200`, exit after 2 closes at or below `SMA200`.
- If `SMA20 < SMA200`: enter after 2 days where `close > SMA20` and `SMA20` is rising, exit after 1 close at or below `SMA20`.

Included close variants:
- `hybridL200_S20_eL2_eS2_xL1_xS1`
- `hybridL200_S20_eL3_eS2_xL2_xS1`
- `200SMA_3in_2out_hybrid20SMA`
- `sma200_3in_2out`

Outputs:
- `outputs/per_ticker_full_span.csv`: full-span results for each ETF from its actual inception date.
- `outputs/variant_rollup_full_span.csv`: mean metrics across the requested ETFs plus beat-buy-hold counts.
- `outputs/best_variant_by_ticker.csv`: top Sharpe variant per ETF.
- `outputs/common_overlap_portfolio_full_span.csv`: equal-weight portfolio across all tickers using the common overlap window.
- `outputs/ticker_inceptions_used.csv`: configured inception dates and source URLs.
