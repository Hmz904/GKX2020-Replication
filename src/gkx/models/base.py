from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone


@dataclass
class FitResult:
    model: object
    params: dict
    val_mse: float
    diagnostics: dict = field(default_factory=dict)


def grid(params):
    keys = list(params)
    values = (params[k] if isinstance(params[k], list) else [params[k]] for k in keys)
    for vals in itertools.product(*values):
        yield dict(zip(keys, vals))


def _fit_candidate(estimator, params, Xtr, ytr, Xv, yv):
    model = clone(estimator).set_params(**params)
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xv)).ravel()
    mse = float(np.mean((yv - pred) ** 2))
    return FitResult(model, params, mse)


def tune(estimator, param_grid, Xtr, ytr, Xv, yv, n_jobs=1):
    candidates = list(grid(param_grid))
    if n_jobs == 1:
        results = [_fit_candidate(estimator, p, Xtr, ytr, Xv, yv) for p in candidates]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_candidate)(estimator, p, Xtr, ytr, Xv, yv) for p in candidates
        )
    return min(results, key=lambda result: result.val_mse)


class ZeroModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))
