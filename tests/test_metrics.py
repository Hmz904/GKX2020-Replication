import numpy as np
import pandas as pd

from gkx.evaluation import (
    dm_matrix,
    dm_test,
    panel_r2,
    portfolio_returns,
    r2_table,
    turnover_series,
)


def test_panel_r2_nontrivial():
    y = np.array([1.0, -1.0, 2.0])
    p = np.array([0.5, -0.5, 1.0])
    expected = 1.0 - np.sum((y - p) ** 2) / np.sum(y**2)
    assert np.isclose(panel_r2(y, p), expected)


def test_r2_table_reports_observation_counts():
    z = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")] * 6,
            "id": range(6),
            "ret_excess": np.linspace(-0.03, 0.03, 6),
            "mve": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "model": "m",
            "prediction": 0.0,
        }
    )
    out = r2_table(z, n=2)
    assert set(out.columns) == {"model", "sample", "n_obs", "r2_oos"}
    assert int(out.loc[out["sample"] == "all", "n_obs"].iloc[0]) == 6
    assert int(out.loc[out["sample"] == "top_2", "n_obs"].iloc[0]) == 2
    assert int(out.loc[out["sample"] == "bottom_2", "n_obs"].iloc[0]) == 2


def test_portfolios():
    z = pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-31"),
            "id": range(20),
            "ret_excess": np.arange(20) / 100,
            "mve": 1.0,
            "model": "m",
            "prediction": np.arange(20),
        }
    )
    out = portfolio_returns(z, bins=10)
    assert "H-L" in set(out.bin)


def test_turnover_uses_total_return_and_includes_drift():
    rows = []
    for date, total_returns in [
        (pd.Timestamp("2020-01-31"), [0.10, 0.00, 0.00, 0.00]),
        (pd.Timestamp("2020-02-29"), [0.00, 0.00, 0.00, 0.00]),
    ]:
        for i in range(4):
            rows.append(
                {
                    "date": date,
                    "id": i,
                    # Make excess returns deliberately different so the test detects the column used.
                    "ret_excess": 0.0,
                    "ret_total": total_returns[i],
                    "mve": 1.0,
                    "model": "m",
                    "prediction": float(i),
                }
            )
    pred = pd.DataFrame(rows)
    monthly = turnover_series(pred, bins=2)
    ew = monthly.loc[monthly.weighting == "ew", "turnover"].iloc[0]
    # Same membership, but one short-leg stock drifts after its +10% total return.
    assert np.isclose(ew, 3 / 19)


def test_turnover_requires_total_return():
    pred = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")] * 4,
            "id": range(4),
            "ret_excess": 0.0,
            "mve": 1.0,
            "model": "m",
            "prediction": np.arange(4),
        }
    )
    try:
        turnover_series(pred, bins=2)
    except ValueError as exc:
        assert "ret_total" in str(exc)
    else:
        raise AssertionError("turnover_series should reject excess-return-only input")


def test_dm_identical_predictions():
    base = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "id": [1, 1],
            "ret_excess": [0.1, -0.1],
            "prediction": [0.0, 0.0],
        }
    )
    other = base.copy()
    other["prediction"] = [0.01, -0.02]
    t, p = dm_test(base, other, lags=0)
    assert np.isfinite(t)
    assert np.isfinite(p)


def test_dm_matrix_is_full_and_antisymmetric():
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    rows = []
    for model, shift in [("a", 0.0), ("b", 0.01), ("c", -0.01)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "id": 1,
                    "ret_excess": 0.02 * np.sin(i),
                    "model": model,
                    "prediction": 0.005 * np.cos(i) + shift,
                }
            )
    out = dm_matrix(pd.DataFrame(rows), lags=1)
    assert len(out) == 9
    for model in ["a", "b", "c"]:
        diag = out[(out.row_model == model) & (out.column_model == model)].iloc[0]
        assert diag.dm_t == 0.0
        assert diag.p_value == 1.0
    ab = out[(out.row_model == "a") & (out.column_model == "b")].iloc[0]
    ba = out[(out.row_model == "b") & (out.column_model == "a")].iloc[0]
    assert np.isclose(ab.dm_t, -ba.dm_t)
    assert np.isclose(ab.p_value, ba.p_value)


def test_value_weighted_evaluation_excludes_invalid_market_cap():
    z = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")] * 10,
            "id": range(10),
            "ret_excess": np.linspace(-0.05, 0.05, 10),
            "mve": [1.0, 0.0, np.nan, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "model": "m",
            "prediction": np.arange(10),
        }
    )
    out = portfolio_returns(z, bins=2)
    assert np.isfinite(out.loc[out.bin == 2, "vw_ret"]).all()
