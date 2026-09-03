from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import (
    build_design_matrix,
    load_panel,
    rank_characteristics_inplace,
    resolve_feature_schema,
    validate_panel,
)
from .evaluation import (
    dm_matrix,
    portfolio_returns,
    portfolio_stats,
    r2_table,
    turnover_from_holdings,
    turnover_series,
)
from .models import fit_model
from .splits import recursive_windows

LOG = logging.getLogger("gkx")

# Top-level model keys are intentionally explicit. Nested model dictionaries are hashed in full
# except for the small set of explicitly non-predictive execution settings below. If a developer
# adds a new top-level model option (for example models.ols_tol) without registering it here, the
# run fails loudly instead of silently reusing stale checkpoints.
_MODEL_TOP_LEVEL_KEYS = {
    "names",
    "ols_huber_quantile",
    "ols_eta0",
    "ols_max_iter",
    "ols3_features",
    "ols3_huber_quantile",
    "ols3_eta0",
    "ols3_max_iter",
    "enet_huber",
    "pcr",
    "pls",
    "glm",
    "rf",
    "gbrt_huber",
    "neural_net",
}
_MODEL_NONPREDICTIVE_NESTED_KEYS = {
    "rf": {"forest_n_jobs"},
    "neural_net": {"inference_batch_size"},
}
_DATA_NONPREDICTIVE_KEYS = {
    "market_cap_col",
    "return_col",
    "min_valid_market_cap_fraction",
}


def _validate_models_config(models_cfg: dict) -> None:
    unknown = sorted(set(models_cfg) - _MODEL_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            "Unknown models configuration key(s): "
            f"{unknown}. Register each new top-level model option in runner._MODEL_TOP_LEVEL_KEYS "
            "so checkpoint invalidation cannot silently ignore it."
        )


def _without_keys(mapping: dict, excluded: set[str]) -> dict:
    return {key: value for key, value in mapping.items() if key not in excluded}


def _prediction_config(cfg: dict, model: str) -> dict:
    """Return configuration fields that can change one model's predictions."""
    models_cfg = cfg["models"]
    _validate_models_config(models_cfg)

    if model == "zero":
        model_cfg = {}
    elif model == "ols":
        model_cfg = {
            "huber_quantile": models_cfg.get("ols_huber_quantile", 0.999),
            "eta0": models_cfg.get("ols_eta0", 1e-3),
            "max_iter": models_cfg.get("ols_max_iter", 5000),
        }
    elif model == "ols3":
        model_cfg = {
            "features": models_cfg.get("ols3_features", []),
            "huber_quantile": models_cfg.get("ols3_huber_quantile", 0.999),
            "eta0": models_cfg.get("ols3_eta0", 1e-3),
            "max_iter": models_cfg.get("ols3_max_iter", 5000),
        }
    elif model in {"enet_huber", "pcr", "pls", "glm", "rf", "gbrt_huber"}:
        if model not in models_cfg:
            raise ValueError(f"Missing models.{model} configuration")
        model_cfg = _without_keys(
            models_cfg[model],
            _MODEL_NONPREDICTIVE_NESTED_KEYS.get(model, set()),
        )
    elif model.startswith("nn"):
        if "neural_net" not in models_cfg:
            raise ValueError("Missing models.neural_net configuration")
        model_cfg = _without_keys(
            models_cfg["neural_net"],
            _MODEL_NONPREDICTIVE_NESTED_KEYS["neural_net"],
        )
    else:
        raise ValueError(f"Unknown model {model!r}")

    # Hash all data options by default except columns/settings used only after prediction.
    # This makes newly introduced data-loader semantics fail safe: they invalidate predictions
    # unless they are deliberately classified as evaluation-only above.
    data_cfg = _without_keys(cfg["data"], _DATA_NONPREDICTIVE_KEYS)
    return {
        "implementation_version": __version__,
        "data": data_cfg,
        "features": cfg["features"],
        "split": cfg["split"],
        "seed": cfg.get("seed", 42),
        "model": model,
        "model_config": model_cfg,
    }


def _prediction_hash(cfg: dict, model: str) -> str:
    payload = json.dumps(
        _prediction_config(cfg, model),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _checkpoint_paths(root: Path, year: int, model: str):
    directory = root / "checkpoints" / str(year)
    return directory / f"{model}.csv.gz", directory / f"{model}.json"


def _save_checkpoint(pred_path: Path, meta_path: Path, pred: pd.DataFrame, metadata: dict):
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_tmp = pred_path.with_name(pred_path.name + ".tmp")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    pred.to_csv(pred_tmp, index=False, compression="gzip")
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    pred_tmp.replace(pred_path)
    meta_tmp.replace(meta_path)


def _load_checkpoints(out: Path, models: list[str], cfg: dict):
    predictions = []
    tuning = []
    for pred_path in sorted((out / "checkpoints").glob("*/*.csv.gz")):
        model = pred_path.stem.replace(".csv", "")
        if model not in models:
            continue
        meta_path = pred_path.with_suffix("").with_suffix(".json")
        if not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("prediction_hash") != _prediction_hash(cfg, model):
            continue
        predictions.append(pd.read_csv(pred_path, parse_dates=["date"]))
        tuning.append(metadata)
    if not predictions:
        raise RuntimeError("No test predictions produced; check dates and checkpoint state")
    return pd.concat(predictions, ignore_index=True), tuning


def _tuning_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "year": record["year"],
                "model": record["model"],
                "prediction_hash": record["prediction_hash"],
                "val_mse": record["val_mse"],
                "params_json": json.dumps(
                    record.get("params", {}), sort_keys=True, separators=(",", ":")
                ),
                "diagnostics_json": json.dumps(
                    record.get("diagnostics", {}), sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "model"]).reset_index(drop=True)


def _attach_evaluation_columns(pred: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]["date_col"]
    idc = cfg["data"]["id_col"]
    yc = cfg["data"]["target_col"]
    cap = cfg["data"]["market_cap_col"]
    return_col = cfg["data"].get("return_col")

    columns = [dc, idc, yc, cap]
    rename = {dc: "date", idc: "id", yc: "ret_excess", cap: "mve"}
    if return_col:
        columns.append(return_col)
        rename[return_col] = "ret_total"
    evaluation = df.loc[:, list(dict.fromkeys(columns))].rename(columns=rename)
    return pred.merge(evaluation, on=["date", "id"], how="left", validate="many_to_one")


def run(cfg, selected=None):
    _validate_models_config(cfg["models"])
    if not cfg["data"].get("return_col"):
        raise ValueError(
            "data.return_col is required before fitting because GKX turnover uses stock total "
            "returns over the same holding period as the prediction target."
        )
    out = Path("outputs") / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    df = load_panel(cfg["data"], cfg["features"])

    dc = cfg["data"]["date_col"]
    idc = cfg["data"]["id_col"]
    yc = cfg["data"]["target_col"]
    schema = resolve_feature_schema(
        df,
        cfg["features"],
        date_col=dc,
        industry_cutoff=cfg["split"].get("initial_validation_end"),
    )
    validate_panel(df, cfg["data"], cfg["features"], schema)

    # Preserve raw evaluation fields before in-place characteristic ranking. This matters when
    # market_cap_col is itself one of the predictive characteristics.
    eval_columns = [dc, idc, yc, cfg["data"]["market_cap_col"]]
    if cfg["data"].get("return_col"):
        eval_columns.append(cfg["data"]["return_col"])
    evaluation_source = df.loc[:, list(dict.fromkeys(eval_columns))].copy()

    characteristics_ranked = cfg["features"].get("mode") == "interact"
    if characteristics_ranked:
        LOG.info("precomputing monthly characteristic ranks once for the full panel")
        rank_characteristics_inplace(
            df,
            dc,
            list(cfg["features"]["characteristic_columns"]),
            fill_monthly_median=cfg["features"].get("fill_missing_monthly_median", True),
        )

    models = selected or cfg["models"]["names"]
    resume = cfg.get("checkpoint", {}).get("resume", True)
    tuning_n_jobs = int(cfg.get("tuning_n_jobs", 1))

    for window in recursive_windows(cfg["split"]):
        pending = []
        for name in models:
            pred_path, meta_path = _checkpoint_paths(out, window.refit_year, name)
            reusable = False
            if resume and pred_path.exists() and meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    reusable = metadata.get("prediction_hash") == _prediction_hash(cfg, name)
                except (OSError, json.JSONDecodeError):
                    reusable = False
            if reusable:
                LOG.info("resume year=%s model=%s", window.refit_year, name)
            else:
                pending.append(name)
        if not pending:
            continue

        train_mask = (df[dc] >= pd.Timestamp(cfg["split"]["train_start"])) & (
            df[dc] <= window.train_end
        )
        val_mask = (df[dc] >= window.val_start) & (df[dc] <= window.val_end)
        test_mask = (df[dc] >= window.test_start) & (df[dc] <= window.test_end)
        tr = df.loc[train_mask]
        va = df.loc[val_mask]
        te = df.loc[test_mask]
        if min(len(tr), len(va), len(te)) == 0:
            LOG.warning("Skipping empty year %s", window.refit_year)
            continue

        LOG.info(
            "building design year=%s train=%s val=%s test=%s features=%s",
            window.refit_year,
            len(tr),
            len(va),
            len(te),
            len(schema.feature_names),
        )
        Xtr = build_design_matrix(
            tr,
            cfg["features"],
            dc,
            schema,
            characteristics_ranked=characteristics_ranked,
        )
        Xv = build_design_matrix(
            va,
            cfg["features"],
            dc,
            schema,
            characteristics_ranked=characteristics_ranked,
        )
        Xte = build_design_matrix(
            te,
            cfg["features"],
            dc,
            schema,
            characteristics_ranked=characteristics_ranked,
        )
        ytr = tr[yc].to_numpy(dtype=np.float32, copy=False)
        yv = va[yc].to_numpy(dtype=np.float32, copy=False)

        for name in pending:
            cols = (
                cfg["models"].get("ols3_features", [])
                if name == "ols3"
                else list(schema.feature_names)
            )
            missing = [c for c in cols if c not in Xtr.columns]
            if missing:
                raise ValueError(f"{name}: missing features {missing}")
            LOG.info(
                "fit year=%s model=%s train=%s val=%s test=%s",
                window.refit_year,
                name,
                len(tr),
                len(va),
                len(te),
            )
            result = fit_model(
                name,
                cfg["models"],
                Xtr[cols],
                ytr,
                Xv[cols],
                yv,
                cfg.get("seed", 42),
                tuning_n_jobs,
            )
            prediction = np.asarray(result.model.predict(Xte[cols])).ravel()
            z = te[[dc, idc]].copy()
            z.columns = ["date", "id"]
            z["model"] = name
            z["prediction"] = prediction
            z["refit_year"] = window.refit_year
            metadata = {
                "year": window.refit_year,
                "model": name,
                "prediction_hash": _prediction_hash(cfg, name),
                "val_mse": result.val_mse,
                "params": result.params,
                "diagnostics": result.diagnostics,
            }
            pred_path, meta_path = _checkpoint_paths(out, window.refit_year, name)
            _save_checkpoint(pred_path, meta_path, z, metadata)

        del Xtr, Xv, Xte, tr, va, te
        gc.collect()

    pred, tuning = _load_checkpoints(out, models, cfg)
    pred = _attach_evaluation_columns(pred, evaluation_source, cfg)
    if cfg["evaluation"].get("save_predictions", True):
        pred.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    _tuning_frame(tuning).to_csv(out / "tuning.csv", index=False)
    r2_table(pred, n=cfg["evaluation"].get("top_bottom_n", 1000)).to_csv(
        out / "r2_oos.csv", index=False
    )
    dm_matrix(pred, lags=cfg["evaluation"].get("newey_west_lags", 6)).to_csv(
        out / "dm_tests.csv", index=False
    )
    returns = portfolio_returns(pred, cfg["evaluation"].get("portfolio_bins", 10))
    returns.to_csv(out / "portfolio_returns.csv", index=False)
    portfolio_stats(returns, cfg["evaluation"].get("annualization", 12)).to_csv(
        out / "portfolio_stats.csv", index=False
    )

    if "ret_total" not in pred.columns:
        raise ValueError(
            "data.return_col is required for GKX turnover because the turnover drift formula uses "
            "stock total returns, not excess returns."
        )
    monthly_turnover = turnover_series(pred, cfg["evaluation"].get("portfolio_bins", 10))
    monthly_turnover.to_csv(out / "turnover_monthly.csv", index=False)
    turnover_from_holdings(pred, cfg["evaluation"].get("portfolio_bins", 10)).to_csv(
        out / "turnover.csv", index=False
    )
    return out
