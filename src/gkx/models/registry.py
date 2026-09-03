import numpy as np

from .base import FitResult, ZeroModel
from .neural import fit_nn
from .sklearn_models import (
    fit_enet_huber,
    fit_gbrt,
    fit_glm,
    fit_ols,
    fit_pcr,
    fit_pls,
    fit_rf,
)


def fit_model(name, cfg, Xtr, ytr, Xv, yv, seed=42, tuning_n_jobs=1):
    if name == "zero":
        return FitResult(ZeroModel().fit(Xtr, ytr), {}, float(np.mean(yv**2)))
    if name == "ols":
        return fit_ols(
            Xtr,
            ytr,
            Xv,
            yv,
            robust=True,
            seed=seed,
            huber_quantile=cfg.get("ols_huber_quantile", 0.999),
            eta0=cfg.get("ols_eta0", 1e-3),
            max_iter=cfg.get("ols_max_iter", 5000),
        )
    if name == "ols3":
        return fit_ols(
            Xtr,
            ytr,
            Xv,
            yv,
            robust=True,
            seed=seed,
            huber_quantile=cfg.get("ols3_huber_quantile", 0.999),
            eta0=cfg.get("ols3_eta0", 1e-3),
            max_iter=cfg.get("ols3_max_iter", 5000),
        )
    if name == "enet_huber":
        return fit_enet_huber(cfg[name], Xtr, ytr, Xv, yv, seed, tuning_n_jobs)
    if name == "pcr":
        return fit_pcr(cfg[name], Xtr, ytr, Xv, yv, seed, tuning_n_jobs)
    if name == "pls":
        return fit_pls(cfg[name], Xtr, ytr, Xv, yv)
    if name == "glm":
        return fit_glm(cfg[name], Xtr, ytr, Xv, yv, seed, tuning_n_jobs)
    if name == "rf":
        return fit_rf(cfg[name], Xtr, ytr, Xv, yv, seed, tuning_n_jobs)
    if name == "gbrt_huber":
        return fit_gbrt(cfg[name], Xtr, ytr, Xv, yv, seed, tuning_n_jobs)
    if name.startswith("nn"):
        return fit_nn(name, cfg["neural_net"], Xtr, ytr, Xv, yv)
    raise KeyError(name)
