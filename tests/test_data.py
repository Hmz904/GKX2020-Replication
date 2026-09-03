from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gkx.data import (
    build_design_matrix,
    load_panel,
    rank_characteristics,
    rank_characteristics_inplace,
    resolve_feature_schema,
    validate_panel,
)


def test_rank_transform_and_monthly_median_fill():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4),
            "x": [10.0, 20.0, 30.0, np.nan],
        }
    )
    ranked = rank_characteristics(df, "date", ["x"])[:, 0]
    assert np.allclose(ranked[:3], [-1.0, 0.0, 1.0])
    assert ranked[3] == 0.0


def test_rank_characteristics_inplace_matches_direct_transform():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-29"] * 3),
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    expected = rank_characteristics(df, "date", ["x"])[:, 0]
    rank_characteristics_inplace(df, "date", ["x"])
    assert np.allclose(df["x"], expected)


def test_prebuilt_features_must_be_explicit(tmp_path: Path):
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": [202001],
            "id": [1],
            "ret_excess": [0.1],
            "mve": [1.0],
            "x": [2.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="features.columns"):
        load_panel(
            {
                "path": str(path),
                "date_col": "date",
                "id_col": "id",
                "target_col": "ret_excess",
                "market_cap_col": "mve",
            },
            {"mode": "prebuilt", "columns": None},
        )


def test_market_cap_can_also_be_explicit_feature():
    df = pd.DataFrame({"mve": [1.0], "x": [2.0]})
    schema = resolve_feature_schema(df, {"mode": "prebuilt", "columns": ["mve", "x"]})
    assert list(schema.feature_names) == ["mve", "x"]


def test_interactions_build_industry_dummies():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3),
            "c": [1.0, 2.0, 3.0],
            "macro": [2.0, 2.0, 2.0],
            "sic2": [10, 20, 10],
        }
    )
    cfg = {
        "mode": "interact",
        "characteristic_columns": ["c"],
        "macro_columns": ["macro"],
        "industry_code_col": "sic2",
        "industry_values": [10, 20],
        "expected_num_features": 4,
        "fill_missing_monthly_median": True,
    }
    schema = resolve_feature_schema(df, cfg)
    x = build_design_matrix(df, cfg, "date", schema)
    assert list(x.columns) == ["c", "c__x__macro", "sic2__10", "sic2__20"]
    assert np.allclose(x["c"], [-1.0, 0.0, 1.0])
    assert np.allclose(x["c__x__macro"], [-2.0, 0.0, 2.0])


def test_pre_ranked_interactions_do_not_rerank():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3),
            "c": [-1.0, 0.0, 1.0],
            "macro": [2.0, 2.0, 2.0],
            "sic2": [10, 20, 10],
        }
    )
    cfg = {
        "mode": "interact",
        "characteristic_columns": ["c"],
        "macro_columns": ["macro"],
        "industry_code_col": "sic2",
        "industry_values": [10, 20],
    }
    schema = resolve_feature_schema(df, cfg)
    x = build_design_matrix(df, cfg, "date", schema, characteristics_ranked=True)
    assert np.allclose(x["c"], [-1.0, 0.0, 1.0])


def test_numeric_yyyymm_date_parsing(tmp_path: Path):
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": [202001, 202002],
            "id": [1, 1],
            "ret_excess": [0.1, 0.2],
            "mve": [10.0, 11.0],
            "x": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    df = load_panel(
        {
            "path": str(path),
            "date_col": "date",
            "id_col": "id",
            "target_col": "ret_excess",
            "market_cap_col": "mve",
        },
        {"mode": "prebuilt", "columns": ["x"]},
    )
    assert df.date.dt.strftime("%Y-%m").tolist() == ["2020-01", "2020-02"]


def test_validate_rejects_target_or_total_return_as_feature_but_allows_market_cap():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "id": [1, 2],
            "ret_excess": [0.1, 0.2],
            "ret_total": [0.11, 0.21],
            "mve": [10.0, 20.0],
            "x": [1.0, 2.0],
        }
    )
    data_cfg = {
        "date_col": "date",
        "id_col": "id",
        "target_col": "ret_excess",
        "return_col": "ret_total",
        "market_cap_col": "mve",
    }
    okay_cfg = {"mode": "prebuilt", "columns": ["mve", "x"]}
    okay_schema = resolve_feature_schema(df, okay_cfg)
    validate_panel(df, data_cfg, okay_cfg, okay_schema)

    for leaky in ["ret_excess", "ret_total"]:
        bad_cfg = {"mode": "prebuilt", "columns": [leaky, "x"]}
        bad_schema = resolve_feature_schema(df, bad_cfg)
        with pytest.raises(ValueError, match="leaky"):
            validate_panel(df, data_cfg, bad_cfg, bad_schema)


def test_interact_dimension_invariants():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"]),
            "c": [1.0],
            "macro": [2.0],
            "sic2": [10],
        }
    )
    cfg = {
        "mode": "interact",
        "characteristic_columns": ["c"],
        "macro_columns": ["macro"],
        "industry_code_col": "sic2",
        "industry_values": [10],
        "expected_num_characteristics": 2,
    }
    with pytest.raises(ValueError, match="characteristics"):
        resolve_feature_schema(df, cfg)


def test_industry_inference_uses_pretest_cutoff():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["1986-12-31", "1987-01-31"]),
            "c": [1.0, 2.0],
            "macro": [1.0, 1.0],
            "sic2": [10, 99],
        }
    )
    cfg = {
        "mode": "interact",
        "characteristic_columns": ["c"],
        "macro_columns": ["macro"],
        "industry_code_col": "sic2",
    }
    schema = resolve_feature_schema(
        df,
        cfg,
        date_col="date",
        industry_cutoff="1986-12-31",
    )
    assert schema.industry_values == (10,)


def test_validate_warns_but_does_not_reject_some_invalid_market_cap():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4),
            "id": [1, 2, 3, 4],
            "ret_excess": [0.1, 0.2, -0.1, 0.0],
            "mve": [10.0, np.nan, 0.0, -2.0],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    data_cfg = {
        "date_col": "date",
        "id_col": "id",
        "target_col": "ret_excess",
        "market_cap_col": "mve",
        "min_valid_market_cap_fraction": 0.01,
    }
    feat_cfg = {"mode": "prebuilt", "columns": ["x"]}
    schema = resolve_feature_schema(df, feat_cfg)
    with pytest.warns(UserWarning, match="excluded"):
        validate_panel(df, data_cfg, feat_cfg, schema)


def test_validate_rejects_all_invalid_market_cap():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3),
            "id": [1, 2, 3],
            "ret_excess": [0.1, 0.2, -0.1],
            "mve": [np.nan, 0.0, -2.0],
            "x": [1.0, 2.0, 3.0],
        }
    )
    data_cfg = {
        "date_col": "date",
        "id_col": "id",
        "target_col": "ret_excess",
        "market_cap_col": "mve",
    }
    feat_cfg = {"mode": "prebuilt", "columns": ["x"]}
    schema = resolve_feature_schema(df, feat_cfg)
    with pytest.raises(ValueError, match="market_cap_col"):
        validate_panel(df, data_cfg, feat_cfg, schema)
