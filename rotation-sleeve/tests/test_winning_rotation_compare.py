import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from winning_rotation_compare import (
    build_monthly_gate,
    build_winning_variant_grid,
    combine_monthly_gate_or_daily_trend,
    combine_monthly_gate_with_daily_trend,
)


def test_monthly_gate_uses_prior_month_signal_for_current_month():
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    monthly = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171],
            "BIL": [100, 100, 100, 101, 102.01, 103.0301, 104.060401, 105.101005, 106.152015, 107.213535],
        },
        index=idx,
    )

    gate = build_monthly_gate(monthly, ("AAA",), cash_proxy="BIL", lookbacks=(3, 7))

    assert gate.loc[pd.Timestamp("2020-08-31"), "AAA"] == 0.0
    assert gate.loc[pd.Timestamp("2020-09-30"), "AAA"] == 1.0


def test_combined_gate_holds_asset_only_when_monthly_and_daily_rules_are_long():
    idx = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"])
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.10, 0.10, 0.10]}, index=idx)
    bil_returns = pd.Series([0.0, 0.01, 0.01, 0.01], index=idx, name="BIL")
    monthly_gate = pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2020-01-31"]))
    daily_trend = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 1.0]}, index=idx)

    result = combine_monthly_gate_with_daily_trend(asset_returns, bil_returns, monthly_gate, daily_trend)

    assert result.loc[pd.Timestamp("2020-02-03")] == 0.10
    assert result.loc[pd.Timestamp("2020-02-04")] == 0.01
    assert result.loc[pd.Timestamp("2020-02-05")] == 0.10


def test_or_gate_holds_asset_when_either_monthly_or_daily_rule_is_long():
    idx = pd.to_datetime(["2020-02-03", "2020-02-04", "2020-02-05"])
    asset_returns = pd.DataFrame({"AAA": [0.10, 0.10, 0.10]}, index=idx)
    bil_returns = pd.Series([0.01, 0.01, 0.01], index=idx, name="BIL")
    monthly_gate = pd.DataFrame({"AAA": [0.0]}, index=pd.to_datetime(["2020-01-31"]))
    daily_trend = pd.DataFrame({"AAA": [1.0, 0.0, 1.0]}, index=idx)

    result = combine_monthly_gate_or_daily_trend(asset_returns, bil_returns, monthly_gate, daily_trend)

    assert result.loc[pd.Timestamp("2020-02-03")] == 0.10
    assert result.loc[pd.Timestamp("2020-02-04")] == 0.01
    assert result.loc[pd.Timestamp("2020-02-05")] == 0.10


def test_winning_variant_grid_contains_compact_approved_grid():
    variants = build_winning_variant_grid()

    assert len(variants) == 8
    assert variants[0]["name"] == "hybridL200_S20_eL2_eS2_xL1_xS1"
    assert variants[-1]["name"] == "hybridL200_S20_eL3_eS3_xL2_xS1"
