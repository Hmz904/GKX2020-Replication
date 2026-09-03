import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from gkx.config import load_config
from gkx.runner import _prediction_hash, _tuning_frame, run
from gkx.synthetic import make_synthetic


def tiny_config():
    return {
        "run_name": "tiny",
        "seed": 1,
        "tuning_n_jobs": 1,
        "checkpoint": {"resume": True},
        "data": {
            "path": "data/processed/synthetic.csv",
            "date_col": "date",
            "id_col": "id",
            "target_col": "ret_excess",
            "return_col": "ret_total",
            "market_cap_col": "mve",
            "min_valid_market_cap_fraction": 0.01,
        },
        "features": {
            "mode": "prebuilt",
            "columns": ["x0", "x1", "x2", "x3", "x4"],
            "expected_num_features": 5,
        },
        "split": {
            "train_start": "2000-01-01",
            "initial_train_end": "2001-12-31",
            "initial_validation_start": "2002-01-01",
            "initial_validation_end": "2002-12-31",
            "test_start": "2003-01-01",
            "test_end": "2003-12-31",
            "refit_frequency": "yearly",
        },
        "models": {
            "names": ["ols"],
            "ols_huber_quantile": 0.999,
            "ols3_features": ["x0", "x1", "x2"],
        },
        "evaluation": {
            "top_bottom_n": 5,
            "annualization": 12,
            "portfolio_bins": 2,
            "newey_west_lags": 1,
            "save_predictions": True,
        },
    }



def test_run_requires_total_return_before_writing_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tiny_config()
    cfg["data"].pop("return_col")
    with pytest.raises(ValueError, match="data.return_col"):
        run(cfg)
    assert not Path("outputs").exists()


def test_default_model_config_is_fully_classified():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    for model in cfg["models"]["names"]:
        assert len(_prediction_hash(cfg, model)) == 16

def test_prediction_hash_ignores_nonprediction_settings():
    cfg = tiny_config()
    expected = _prediction_hash(cfg, "ols")

    variants = []
    z = copy.deepcopy(cfg)
    z["tuning_n_jobs"] = 8
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["evaluation"]["newey_west_lags"] = 12
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["evaluation"]["top_bottom_n"] = 500
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["models"]["names"].append("nn5")
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["run_name"] = "another-output-directory"
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["data"]["market_cap_col"] = "another_market_cap"
    variants.append(z)
    z = copy.deepcopy(cfg)
    z["data"]["return_col"] = "another_total_return"
    variants.append(z)

    assert all(_prediction_hash(z, "ols") == expected for z in variants)


def test_prediction_hash_excludes_known_execution_only_nested_settings():
    cfg = tiny_config()
    cfg["models"].update(
        {
            "rf": {
                "n_estimators": 10,
                "max_depth": [2],
                "max_features": [3],
                "forest_n_jobs": 1,
            },
            "neural_net": {
                "epochs": 2,
                "batch_size": 32,
                "inference_batch_size": 64,
                "learning_rate": [0.001],
                "l1_lambda": [0.0001],
                "patience": 1,
                "seeds": [1],
                "device": "cpu",
            },
        }
    )
    rf_hash = _prediction_hash(cfg, "rf")
    nn_hash = _prediction_hash(cfg, "nn1")
    changed = copy.deepcopy(cfg)
    changed["models"]["rf"]["forest_n_jobs"] = 8
    changed["models"]["neural_net"]["inference_batch_size"] = 1024
    assert _prediction_hash(changed, "rf") == rf_hash
    assert _prediction_hash(changed, "nn1") == nn_hash


def test_prediction_hash_changes_for_prediction_inputs():
    cfg = tiny_config()
    expected = _prediction_hash(cfg, "ols")

    z = copy.deepcopy(cfg)
    z["seed"] = 2
    assert _prediction_hash(z, "ols") != expected

    z = copy.deepcopy(cfg)
    z["split"]["initial_train_end"] = "2001-11-30"
    assert _prediction_hash(z, "ols") != expected

    z = copy.deepcopy(cfg)
    z["models"]["ols_huber_quantile"] = 0.95
    assert _prediction_hash(z, "ols") != expected


def test_unknown_top_level_model_option_fails_loudly():
    cfg = tiny_config()
    cfg["models"]["ols_tol"] = 1e-9
    with pytest.raises(ValueError, match="ols_tol"):
        _prediction_hash(cfg, "ols")


def test_new_nested_model_option_is_hashed_by_default():
    cfg = tiny_config()
    cfg["models"]["rf"] = {
        "n_estimators": 10,
        "max_depth": [2],
        "max_features": [3],
        "forest_n_jobs": 1,
    }
    expected = _prediction_hash(cfg, "rf")
    cfg["models"]["rf"]["future_predictive_option"] = 123
    assert _prediction_hash(cfg, "rf") != expected


def test_tuning_frame_has_stable_json_columns():
    records = [
        {
            "year": 2003,
            "model": "ols",
            "prediction_hash": "abc",
            "val_mse": 0.1,
            "params": {"b": 2, "a": 1},
            "diagnostics": {"sgd_n_iter": 5},
        },
        {
            "year": 2003,
            "model": "gbrt_huber",
            "prediction_hash": "def",
            "val_mse": 0.2,
            "params": {"max_depth": 1},
            "diagnostics": {"huber_cutoff": 0.3},
        },
    ]
    frame = _tuning_frame(records)
    assert list(frame.columns) == [
        "year",
        "model",
        "prediction_hash",
        "val_mse",
        "params_json",
        "diagnostics_json",
    ]
    assert json.loads(frame.loc[frame.model == "ols", "diagnostics_json"].iloc[0]) == {
        "sgd_n_iter": 5
    }


def test_checkpoint_and_resume(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_synthetic("data/processed/synthetic.csv", n_ids=20, periods=48, p=5, seed=1)
    cfg = tiny_config()
    out = run(cfg)
    checkpoint = out / "checkpoints" / "2003" / "ols.csv.gz"
    assert checkpoint.exists()
    before = checkpoint.stat().st_mtime_ns

    # Prediction checkpoints contain predictions/keys only. Evaluation fields are reattached from
    # the current panel, so changing size/turnover mappings never requires retraining.
    checkpoint_columns = set(pd.read_csv(checkpoint, nrows=1).columns)
    assert checkpoint_columns == {"date", "id", "model", "prediction", "refit_year"}

    out2 = run(cfg)
    assert out2 == out
    assert checkpoint.stat().st_mtime_ns == before
    tuning = pd.read_csv(out / "tuning.csv")
    assert list(tuning.model) == ["ols"]
    assert {"params_json", "diagnostics_json"}.issubset(tuning.columns)

    meta_path = out / "checkpoints" / "2003" / "ols.json"
    first_hash = json.loads(meta_path.read_text())["prediction_hash"]

    evaluation_only = copy.deepcopy(cfg)
    evaluation_only["evaluation"]["top_bottom_n"] = 10
    run(evaluation_only)
    assert checkpoint.stat().st_mtime_ns == before
    assert json.loads(meta_path.read_text())["prediction_hash"] == first_hash

    changed = copy.deepcopy(cfg)
    changed["seed"] = 2
    run(changed)
    second_hash = json.loads(meta_path.read_text())["prediction_hash"]
    assert second_hash != first_hash
