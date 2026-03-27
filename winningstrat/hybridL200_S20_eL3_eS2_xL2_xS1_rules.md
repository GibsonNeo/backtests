# `hybridL200_S20_eL3_eS2_xL2_xS1` Rules

## Inputs
- Use the ETF's daily split-adjusted close.
- Calculate:
  - `SMA200` from the split-adjusted close
  - `SMA20` from the split-adjusted close

## Regime Selection
- If `SMA20 >= SMA200`, use the `200-SMA regime`.
- If `SMA20 < SMA200`, use the `20-SMA re-entry regime`.

## 200-SMA Regime
- Enter long after `3 consecutive daily closes above SMA200`.
- Exit long after `2 consecutive daily closes at or below SMA200`.

## 20-SMA Re-entry Regime
- Enter long after `2 consecutive days` where both are true:
  - `close > SMA20`
  - `SMA20 today > SMA20 yesterday`
- Exit long after `1 daily close at or below SMA20`.

## Execution
- All entries and exits happen on the `next trading day`.
- There is `no cooldown`.
- The `SMA20 rising` rule has `no percentage threshold`; it only needs to be higher than the prior day.

## Short Version
- If `20 >= 200`: `3 closes above 200` to enter, `2 closes at/below 200` to exit.
- If `20 < 200`: `2 days of close > 20 and 20 rising` to enter, `1 close at/below 20` to exit.
