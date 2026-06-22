import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cef_rotation import (
    StrategySpec,
    StrategyResult,
    build_monthly_prices_from_daily,
    drawdown_series,
    max_drawdown,
    regime_table,
    run_rotation_strategy,
    summarize_returns,
)


def test_monthly_prices_use_last_available_trading_day():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"])
    daily = pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0]}, index=idx)

    monthly = build_monthly_prices_from_daily(daily)

    assert list(monthly.index.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-29"]
    assert monthly.loc[pd.Timestamp("2020-01-31"), "AAA"] == 11.0
    assert monthly.loc[pd.Timestamp("2020-02-29"), "AAA"] == 13.0


def test_rotation_uses_average_lookback_signal_and_shifts_allocation_one_month():
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    prices = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171],
            "IEF": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "BIL": [100, 100, 100, 101, 102.01, 103.0301, 104.060401, 105.101005, 106.152015, 107.213535],
        },
        index=idx,
    )
    spec = StrategySpec(name="test", assets=("AAA",), fallback="IEF", lookbacks=(3, 7))

    result = run_rotation_strategy(spec, prices, cash_proxy="BIL")

    # First valid signal is available at 2020-08-31, so the return/exposure
    # series starts with the 2020-09-30 monthly return.
    assert list(result.monthly_returns.index.strftime("%Y-%m-%d")) == ["2020-09-30", "2020-10-31"]
    assert result.allocations.loc[pd.Timestamp("2020-09-30"), "AAA"] == 1.0
    assert result.monthly_returns.loc[pd.Timestamp("2020-09-30")] == np.round(177.1561 / 161.051 - 1.0, 10)


def test_summary_metrics_include_drawdown_best_worst_and_monthly_win_rate():
    returns = pd.Series([0.10, -0.20, 0.05, 0.04], index=pd.date_range("2020-01-31", periods=4, freq="ME"))

    summary = summarize_returns("sample", returns)

    assert summary["strategy"] == "sample"
    assert round(summary["total_return"], 6) == round((1.10 * 0.80 * 1.05 * 1.04) - 1.0, 6)
    assert round(summary["max_drawdown"], 6) == round(max_drawdown((1.0 + returns).cumprod()), 6)
    assert summary["best_year"] == summary["worst_year"]
    assert summary["monthly_win_rate"] == 0.75


def test_regime_table_counts_sgov_as_defensive_exposure():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.Series([0.01, 0.02, -0.01], index=idx)
    equity = (1.0 + returns).cumprod()
    allocations = pd.DataFrame({"SGOV": [1.0, 0.5, 0.0], "ADX": [0.0, 0.5, 1.0]}, index=idx)
    result = StrategyResult(
        StrategySpec("sgov_test", ("ADX",), "SGOV", (3, 7)),
        returns,
        equity,
        drawdown_series(equity),
        allocations,
        allocations,
        0,
        summarize_returns("sgov_test", returns, exposure=allocations.mean()),
    )

    regimes = regime_table({"sgov_test": result}, {"sample": ("2024-01-01", "2024-12-31")})

    assert regimes.loc[0, "defensive_exposure"] == 0.5
