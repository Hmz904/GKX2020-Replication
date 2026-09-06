from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .base import FitResult, tune


def _huber_cutoff(y, pred, q=0.999):
    residual = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    finite = np.abs(residual[np.isfinite(residual)])
    if finite.size == 0:
        raise ValueError("Cannot estimate Huber cutoff from empty residuals")
    return max(float(np.quantile(finite, q)), 1e-8)


def _tail_fraction(y, pred, cutoff):
    residual = np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
    return float(np.mean(residual > cutoff))


def _fit_capture_convergence(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, warned


def _sgd_diagnostics(model, warned, prefix="sgd"):
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return {
        f"{prefix}_n_iter": int(estimator.n_iter_),
        f"{prefix}_convergence_warning": bool(warned),
    }


def _scaled_sgd(**kwargs):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SGDRegressor(**kwargs)),
        ]
    )


def _divergence_reference(yv):
    """MSE of the zero predictor, i.e. the denominator of the GKX out-of-sample R^2."""
    return max(float(np.mean(np.asarray(yv, dtype=float) ** 2)), 1e-30)


def _is_diverged(result, yv, factor):
    """True when a fitted candidate is numerically broken rather than merely poor.

    Fixed-step SGD on real characteristic distributions can diverge at small alpha: the
    coefficients blow up and validation MSE exceeds the zero-predictor benchmark by many
    orders of magnitude. Taking argmin over such a grid still returns a model, so the
    failure propagates silently into the results table as an absurd negative R^2 rather
    than as an error. Grid points that fail this check are excluded from selection.
    """
    if not np.isfinite(result.val_mse):
        return True
    return result.val_mse > factor * _divergence_reference(yv)


def fit_ols(
    Xtr,
    ytr,
    Xv,
    yv,
    robust=False,
    seed=42,
    huber_quantile=0.999,
    eta0=1e-3,
    max_iter=5000,
):
    if not robust:
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xv)
        return FitResult(model, {}, float(np.mean((yv - pred) ** 2)))

    pilot = LinearRegression().fit(Xtr, ytr)
    cutoff = _huber_cutoff(ytr, pilot.predict(Xtr), huber_quantile)
    model = _scaled_sgd(
        loss="huber",
        epsilon=cutoff,
        penalty=None,
        max_iter=max_iter,
        tol=1e-6,
        random_state=seed,
        learning_rate="adaptive",
        eta0=eta0,
    )
    model, warned = _fit_capture_convergence(model, Xtr, ytr)
    pred = model.predict(Xv)
    diagnostics = {
        "huber_cutoff": cutoff,
        "huber_tail_fraction_train": _tail_fraction(ytr, model.predict(Xtr), cutoff),
        **_sgd_diagnostics(model, warned),
    }
    params = {"huber_quantile": huber_quantile, "eta0": eta0, "max_iter": max_iter}
    return FitResult(model, params, float(np.mean((yv - pred) ** 2)), diagnostics)


def _fit_enet_candidate(cfg, alpha, Xtr, ytr, Xv, yv, seed):
    paper_rho = float(cfg.get("paper_rho", 0.5))
    sklearn_l1_ratio = 1.0 - paper_rho
    common = dict(
        penalty="elasticnet",
        alpha=float(alpha),
        l1_ratio=sklearn_l1_ratio,
        max_iter=int(cfg.get("max_iter", 5000)),
        tol=1e-6,
        random_state=seed,
        learning_rate="adaptive",
        eta0=cfg.get("eta0", 1e-3),
    )
    pilot = _scaled_sgd(loss="squared_error", **common)
    pilot, pilot_warned = _fit_capture_convergence(pilot, Xtr, ytr)
    cutoff = _huber_cutoff(ytr, pilot.predict(Xtr), cfg.get("huber_quantile", 0.999))
    model = _scaled_sgd(loss="huber", epsilon=cutoff, **common)
    model, warned = _fit_capture_convergence(model, Xtr, ytr)
    pred = model.predict(Xv)
    diagnostics = {
        "huber_cutoff": cutoff,
        "huber_tail_fraction_train": _tail_fraction(ytr, model.predict(Xtr), cutoff),
        **_sgd_diagnostics(pilot, pilot_warned, prefix="pilot_sgd"),
        **_sgd_diagnostics(model, warned),
    }
    params = {
        "sklearn_alpha": float(alpha),
        "paper_rho": paper_rho,
        "sklearn_l1_ratio": sklearn_l1_ratio,
    }
    return FitResult(model, params, float(np.mean((yv - pred) ** 2)), diagnostics)


def fit_enet_huber(cfg, Xtr, ytr, Xv, yv, seed=42, n_jobs=1):
    alphas = list(cfg["sklearn_alpha"])
    if n_jobs == 1:
        results = [_fit_enet_candidate(cfg, alpha, Xtr, ytr, Xv, yv, seed) for alpha in alphas]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_enet_candidate)(cfg, alpha, Xtr, ytr, Xv, yv, seed)
            for alpha in alphas
        )
    factor = float(cfg.get("divergence_factor", 100.0))
    usable = [r for r in results if not _is_diverged(r, yv, factor)]
    n_diverged = len(results) - len(usable)
    if not usable:
        raise RuntimeError(
            f"All {len(results)} elastic-net grid points diverged: validation MSE exceeded "
            f"{factor}x the zero-predictor benchmark ({_divergence_reference(yv):.6g}). "
            "Lower models.enet_huber.eta0 or drop the smallest sklearn_alpha values."
        )
    best = min(usable, key=lambda result: result.val_mse)
    if n_diverged:
        warnings.warn(
            f"{n_diverged} of {len(results)} elastic-net grid points diverged and were "
            "excluded from selection.",
            UserWarning,
            stacklevel=2,
        )
    best.diagnostics["n_diverged_grid_points"] = n_diverged
    best.diagnostics["divergence_reference_mse"] = _divergence_reference(yv)
    return best


def fit_pcr(cfg, Xtr, ytr, Xv, yv, seed=42, n_jobs=1):
    estimator = Pipeline([("pca", PCA(random_state=seed)), ("model", LinearRegression())])
    values = [k for k in cfg["n_components"] if k <= min(Xtr.shape)]
    if not values:
        raise ValueError("PCR has no valid n_components for this training matrix")
    return tune(estimator, {"pca__n_components": values}, Xtr, ytr, Xv, yv, n_jobs=n_jobs)


def fit_pls(cfg, Xtr, ytr, Xv, yv):
    best = None
    for k in [x for x in cfg["n_components"] if x <= min(Xtr.shape)]:
        model = PLSRegression(n_components=k, scale=False, max_iter=1000).fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xv)).ravel()
        result = FitResult(model, {"n_components": k}, float(np.mean((yv - pred) ** 2)))
        if best is None or result.val_mse < best.val_mse:
            best = result
    if best is None:
        raise ValueError("PLS has no valid n_components for this training matrix")
    return best


def _take_rows(X, index):
    if hasattr(X, "iloc"):
        return X.iloc[index]
    return X[index]


class StreamingSplineFeatures:
    """Fit/transform a dense spline basis in bounded row chunks."""

    def __init__(self, n_knots=3, batch_size=4096):
        self.spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
        self.scaler = StandardScaler(copy=False)
        self.batch_size = int(batch_size)

    def fit(self, X):
        self.spline.fit(X)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            transformed = self.spline.transform(_take_rows(X, slice(start, stop)))
            transformed = np.asarray(transformed, dtype=np.float32)
            self.scaler.partial_fit(transformed)
        return self

    @property
    def n_features_out_(self):
        return int(self.spline.n_features_out_)

    def transform_rows(self, X, index):
        transformed = self.spline.transform(_take_rows(X, index))
        transformed = np.asarray(transformed, dtype=np.float32)
        return self.scaler.transform(transformed).astype(np.float32, copy=False)


class StreamingSplineSGDModel:
    """Prediction wrapper that never materializes the full spline-expanded matrix."""

    def __init__(self, features, model):
        self.features = features
        self.model = model

    def predict(self, X):
        out = np.empty(len(X), dtype=np.float32)
        batch_size = self.features.batch_size
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            z = self.features.transform_rows(X, slice(start, stop))
            out[start:stop] = self.model.predict(z)
        return out


def _stream_fit_sgd(model, features, X, y, *, epochs, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, features.batch_size):
            index = order[start : start + features.batch_size]
            z = features.transform_rows(X, index)
            model.partial_fit(z, np.asarray(y)[index])
    return model


def _glm_sgd(cfg, alpha, seed, *, loss, epsilon=None):
    paper_rho = float(cfg.get("paper_rho", 0.5))
    kwargs = {
        "loss": loss,
        "penalty": "elasticnet",
        "alpha": float(alpha),
        "l1_ratio": 1.0 - paper_rho,
        # partial_fit performs the externally controlled streaming epochs below.
        "max_iter": 1,
        "tol": None,
        "shuffle": False,
        "random_state": seed,
        "learning_rate": "invscaling",
        "eta0": float(cfg.get("eta0", 1e-3)),
    }
    if epsilon is not None:
        kwargs["epsilon"] = float(epsilon)
    return SGDRegressor(**kwargs)


def _fit_glm_candidate(cfg, alpha, features, Xtr, ytr, Xv, yv, seed):
    epochs = int(cfg.get("streaming_epochs", 5))
    paper_rho = float(cfg.get("paper_rho", 0.5))
    sklearn_l1_ratio = 1.0 - paper_rho

    pilot_estimator = _glm_sgd(cfg, alpha, seed, loss="squared_error")
    _stream_fit_sgd(pilot_estimator, features, Xtr, ytr, epochs=epochs, seed=seed)
    pilot = StreamingSplineSGDModel(features, pilot_estimator)
    cutoff = _huber_cutoff(ytr, pilot.predict(Xtr), cfg.get("huber_quantile", 0.999))

    robust_estimator = _glm_sgd(cfg, alpha, seed, loss="huber", epsilon=cutoff)
    _stream_fit_sgd(robust_estimator, features, Xtr, ytr, epochs=epochs, seed=seed)
    model = StreamingSplineSGDModel(features, robust_estimator)
    pred = model.predict(Xv)
    diagnostics = {
        "huber_cutoff": cutoff,
        "huber_tail_fraction_train": _tail_fraction(ytr, model.predict(Xtr), cutoff),
        "optimizer": "streaming_partial_fit",
        "streaming_epochs": epochs,
        "transform_batch_size": features.batch_size,
        "spline_features": features.n_features_out_,
    }
    params = {
        "sklearn_alpha": float(alpha),
        "paper_rho": paper_rho,
        "sklearn_l1_ratio": sklearn_l1_ratio,
        "n_knots": cfg.get("n_knots", 3),
    }
    return FitResult(model, params, float(np.mean((yv - pred) ** 2)), diagnostics)


def fit_glm(cfg, Xtr, ytr, Xv, yv, seed=42, n_jobs=1):
    """Memory-bounded spline + Huber elastic-net approximation to GKX's GLM."""
    # GLM candidates are intentionally serial: parallel basis expansion would multiply peak RAM.
    _ = n_jobs
    features = StreamingSplineFeatures(
        n_knots=int(cfg.get("n_knots", 3)),
        batch_size=int(cfg.get("transform_batch_size", 4096)),
    ).fit(Xtr)
    results = [
        _fit_glm_candidate(cfg, alpha, features, Xtr, ytr, Xv, yv, seed)
        for alpha in cfg["sklearn_alpha"]
    ]
    return min(results, key=lambda result: result.val_mse)


def fit_rf(cfg, Xtr, ytr, Xv, yv, seed=42, tuning_n_jobs=1):
    estimator = RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 300),
        bootstrap=True,
        random_state=seed,
        n_jobs=1 if tuning_n_jobs != 1 else cfg.get("forest_n_jobs", -1),
    )
    return tune(
        estimator,
        {
            "max_depth": cfg["max_depth"],
            "max_features": cfg["max_features"],
            "min_samples_leaf": [cfg.get("min_samples_leaf", 1)],
        },
        Xtr,
        ytr,
        Xv,
        yv,
        n_jobs=tuning_n_jobs,
    )


def fit_gbrt(cfg, Xtr, ytr, Xv, yv, seed=42, tuning_n_jobs=1):
    """Tune GBRT over the n_estimators grid without refitting for each grid point.

    A gradient boosting model with N trees contains every model with fewer trees as
    a prefix, so `staged_predict` recovers the validation prediction at each grid
    point from a single fit at max(n_estimators). Predictions are bit-identical to
    fitting each grid point separately; only the wasted refits are removed.
    """
    n_estimators_grid = sorted(set(int(n) for n in cfg["n_estimators"]))
    max_trees = n_estimators_grid[-1]
    learning_rates = cfg["learning_rate"]
    max_depths = cfg["max_depth"]
    min_samples_leaf = cfg.get("min_samples_leaf", 1)
    quantile = cfg.get("huber_quantile", 0.999)

    yv_arr = np.asarray(yv, dtype=float).ravel()
    wanted = set(n_estimators_grid)

    def run_combo(learning_rate, max_depth):
        model = GradientBoostingRegressor(
            loss="huber",
            alpha=quantile,
            random_state=seed,
            n_estimators=max_trees,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        model.fit(Xtr, ytr)
        # staged_predict yields the prediction after 1, 2, ... max_trees stages.
        local = None
        for stage, pred in enumerate(model.staged_predict(Xv), start=1):
            if stage not in wanted:
                continue
            mse = float(np.mean((yv_arr - np.asarray(pred).ravel()) ** 2))
            if local is None or mse < local[0]:
                local = (
                    mse,
                    stage,
                    {
                        "n_estimators": stage,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                    model,
                )
        return local

    combos = [(lr, d) for lr in learning_rates for d in max_depths]
    fit_n_jobs = int(cfg.get("fit_n_jobs", 1))
    if fit_n_jobs == 1:
        found = [run_combo(lr, d) for lr, d in combos]
    else:
        # Threads only: sklearn's tree builder releases the GIL, and threads avoid
        # the process-spawn path that hangs on Windows.
        found = Parallel(n_jobs=fit_n_jobs, prefer="threads")(
            delayed(run_combo)(lr, d) for lr, d in combos
        )

    val_mse, n_selected, params, full_model = min(found, key=lambda item: item[0])

    # Truncate the retained model to the selected number of stages so that later
    # .predict() calls use exactly the tuned configuration.
    selected = clone(full_model).set_params(n_estimators=n_selected)
    selected.__dict__.update(full_model.__dict__)
    selected.estimators_ = full_model.estimators_[:n_selected]
    selected.n_estimators = n_selected
    selected.n_estimators_ = n_selected
    if getattr(full_model, "train_score_", None) is not None:
        selected.train_score_ = full_model.train_score_[:n_selected]

    result = FitResult(selected, params, val_mse)
    residual = np.abs(np.asarray(ytr) - result.model.predict(Xtr))
    cutoff = float(np.quantile(residual, quantile))
    result.diagnostics.update(
        {
            "huber_cutoff": cutoff,
            "huber_tail_fraction_train": float(np.mean(residual > cutoff)),
        }
    )
    return result
