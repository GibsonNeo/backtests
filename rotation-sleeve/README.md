# 4 CEF Rotation Sleeve Backtest

This folder reproduces and stress-tests the Seeking Alpha "4 CEF Rotation Model" from Financially Free Investor's June 2026 article, "How We Would Invest $100,000 Today: Our Two Best Strategies, One Yields 7.75%."

The goal is not just to match the article numbers. The runner also tests whether the result is mostly explained by the momentum rule, the specific CEF choices, or using `IEF` as the defensive asset during a long bond bull market.

## Strategy Being Tested

Original sleeve universe:

- `ADX`: Adams Diversified Equity Fund
- `RQI`: Cohen & Steers Quality Income Realty Fund
- `KYN`: Kayne Anderson Energy Infrastructure Fund
- `NMZ`: Nuveen Municipal High Income Opportunity Fund

Defensive asset:

- `IEF`: iShares 7-10 Year Treasury Bond ETF

Each of the four sleeves has a 25% target weight. At each month end, the script calculates total return momentum over trailing 3 months and trailing 7 months using Yahoo adjusted close data:

```text
avg_momentum = (return_3m + return_7m) / 2
```

It compares that value with the same average lookback return for the cash proxy, defaulting to `BIL`. If the fund's average momentum is greater than the cash proxy, that sleeve holds the fund for the next month. Otherwise it holds the defensive asset. The portfolio rebalances monthly.

Signals are calculated with month-end data and traded on the next monthly return, which avoids look-ahead bias.

## What Gets Run

Primary tests:

- `A_original_cefs_ief`: original CEFs, fallback `IEF`
- `B_original_cefs_bil`: original CEFs, fallback `BIL`
- `B2_original_cefs_sgov`: original CEFs, fallback `SGOV`
- `C_etfs_ief`: broad ETFs `SPY`, `VNQ`, `XLE`, `MUB`, fallback `IEF`
- `D_etfs_bil`: broad ETFs, fallback `BIL`
- `D2_etfs_sgov`: broad ETFs, fallback `SGOV`
- `E_original_cefs_6m_ief`: original CEFs, 6-month momentum only, fallback `IEF`
- `F_original_cefs_12m_ief`: original CEFs, 12-month momentum only, fallback `IEF`
- `G_original_cefs_3_6_12m_ief`: original CEFs, average of 3, 6, and 12 months, fallback `IEF`

Benchmarks:

- `SPY_buy_hold`
- Equal-weight buy-and-hold original CEF basket
- Equal-weight buy-and-hold broad ETF basket

Sensitivity grid:

- Lookbacks: `3`, `6`, `7`, `9`, `12`, `3+7`, `3+12`, `6+12`
- Defensive assets: `IEF`, `TLT`, `SHY`, `BIL`, `SGOV`, `Cash`

Regime analysis:

- 2006 to 2012
- 2013 to 2019
- 2020 to 2021
- 2022 to latest

## Setup

Use a local virtual environment from this folder:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python cef_rotation.py --start 2006-01-01 --outdir outputs
```

Optional:

```bash
python cef_rotation.py --start 2006-01-01 --end 2026-06-22 --bootstrap-runs 5000 --outdir outputs
```

Yahoo's `end` date is passed through to `yfinance`; leaving it blank uses the latest available data.

## Outputs

The script writes these files into `outputs/`:

- `monthly_returns.csv`: monthly returns for every strategy and benchmark
- `equity_curves.csv`: growth of $1
- `drawdowns.csv`: drawdown series
- `summary_comparison.csv`: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, best/worst year, win rate, switches, and exposures
- `sensitivity_analysis.csv`: parameter grid across lookbacks and defensive assets
- `regime_analysis.csv`: period-specific CAGR, drawdown, volatility, and defensive exposure
- `bootstrap_analysis.csv`: median, 5th percentile, and 95th percentile bootstrap CAGR plus median max drawdown
- `attribution_analysis.csv`: proxy estimates for trend filter, IEF duration exposure, CEF selection, and distribution capture limits
- `data_integrity.csv`: first/last month, observed months, and missing values inside each ticker's observed history
- `report.md`: written summary answering the key research questions
- `charts/equity_curves.png`
- `charts/drawdowns.png`
- `charts/exposure_*.png`

## Interpretation Notes

The analysis uses Yahoo adjusted close data, so distributions are included in returns. It does not separately decompose market-price return versus distribution return because that would require a second clean source for price-only and distribution history across all CEFs.

`BIL` and `SGOV` did not exist for all of 2006. The script documents ticker inception dates and missing values in `data_integrity.csv`; this matters when comparing against an article result that may have used a Treasury bill series instead of live ETF history.

The most important comparison is not the original strategy in isolation. Check:

- `A_original_cefs_ief` versus `B_original_cefs_bil` for the duration effect.
- `B_original_cefs_bil` versus `B2_original_cefs_sgov` for Treasury bill ETF choice.
- `A_original_cefs_ief` versus `C_etfs_ief` for CEF selection versus broad ETF sleeves.
- `B_original_cefs_bil` versus `D_etfs_bil` for CEF selection after removing the IEF duration tailwind.
- The sensitivity grid for whether `(3m + 7m) / 2` is unusually privileged.
- The 2022-to-latest regime for deterioration after the bond bear market began.

## Tests

```bash
python -m pytest tests/test_cef_rotation.py
```

The tests use synthetic data and do not call Yahoo Finance.
