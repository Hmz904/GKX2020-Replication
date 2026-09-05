"""Guard against silent numerical divergence in the SGD-based elastic net.

Motivation, from an actual run: on real datashare characteristics with the shipped
``eta0=1e-3``, every point of the ``sklearn_alpha`` grid diverged. ``min(results,
key=val_mse)`` still returned a model, and the run completed with an out-of-sample R^2 of
-2.1e12 percent sitting in ``r2_oos.csv`` next to the honest numbers. The same code scored
+2.287% -- best of all models -- on the synthetic stress panel, whose AR(1) latents are far
better conditioned. Synthetic testing cannot catch this class of failure, so it is asserted
here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gkx.models.base import FitResult
from gkx.models.sklearn_models import _divergence_reference, _is_diverged, fit_enet_huber


def _panel(n=400, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype("float32")
    y = (0.02 * X[:, 0] - 0.015 * X[:, 1] + rng.normal(scale=0.05, size=n)).astype("float64")
    return X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :]


def test_divergence_reference_is_the_zero_predictor_mse():
    yv = np.array([0.1, -0.2, 0.3])
    assert _divergence_reference(yv) == pytest.approx(float(np.mean(yv**2)))
    # Never zero, so the comparison in _is_diverged cannot divide-by-zero downstream.
    assert _divergence_reference(np.zeros(5)) > 0.0


def test_is_diverged_flags_blowups_and_nonfinite_but_not_merely_poor_fits():
    yv = np.full(100, 0.1)
    ref = _divergence_reference(yv)
    poor = FitResult(None, {}, ref * 2.0)
    blown = FitResult(None, {}, ref * 1e6)
    nan = FitResult(None, {}, float("nan"))
    # A model twice as bad as predicting zero is bad, not broken: it must survive.
    assert not _is_diverged(poor, yv, factor=100.0)
    assert _is_diverged(blown, yv, factor=100.0)
    assert _is_diverged(nan, yv, factor=100.0)


def test_diverged_grid_points_are_excluded_and_reported():
    Xtr, ytr, Xv, yv = _panel()
    cfg = {
        "sklearn_alpha": [1e-6, 1e-3],
        "paper_rho": 0.5,
        "huber_quantile": 0.999,
        "eta0": 1e-4,
        "max_iter": 200,
    }
    result = fit_enet_huber(cfg, Xtr, ytr, Xv, yv, seed=0)
    # The selected model must never be worse than the guard threshold, and the count of
    # rejected points must be recorded rather than lost.
    assert result.val_mse <= 100.0 * _divergence_reference(yv)
    assert "n_diverged_grid_points" in result.diagnostics
    assert result.diagnostics["divergence_reference_mse"] == pytest.approx(
        _divergence_reference(yv)
    )


def test_all_diverged_raises_instead_of_returning_a_broken_model():
    Xtr, ytr, Xv, yv = _panel()
    # A step size this large diverges on every grid point. The old code returned the
    # least-bad divergent fit; the contract now is a loud failure.
    cfg = {
        "sklearn_alpha": [1e-6, 1e-5],
        "paper_rho": 0.5,
        "huber_quantile": 0.999,
        "eta0": 1e12,
        "max_iter": 200,
    }
    with pytest.raises(RuntimeError, match="diverged"):
        fit_enet_huber(cfg, Xtr, ytr, Xv, yv, seed=0)


def test_guard_is_configurable():
    Xtr, ytr, Xv, yv = _panel()
    cfg = {
        "sklearn_alpha": [1e-3],
        "paper_rho": 0.5,
        "huber_quantile": 0.999,
        "eta0": 1e-4,
        "max_iter": 200,
        "divergence_factor": 1e-9,  # absurdly strict: nothing can pass
    }
    with pytest.raises(RuntimeError, match="diverged"):
        fit_enet_huber(cfg, Xtr, ytr, Xv, yv, seed=0)
