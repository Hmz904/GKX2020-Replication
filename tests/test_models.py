import numpy as np
import pandas as pd

from gkx.evaluation import panel_r2
from gkx.models import fit_model
from gkx.models.neural import Net


def toy(n=180, p=8, seed=1):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = 0.02 * x.x0 - 0.01 * x.x1 + rng.normal(scale=0.05, size=n)
    return x.iloc[:120], y.iloc[:120].to_numpy(), x.iloc[120:], y.iloc[120:].to_numpy()


def test_classical_model_families_smoke():
    Xtr, ytr, Xv, yv = toy()
    cfg = {
        "ols_huber_quantile": 0.95,
        "ols3_features": ["x0", "x1", "x2"],
        "ols3_huber_quantile": 0.95,
        "enet_huber": {
            "sklearn_alpha": [1e-4],
            "paper_rho": 0.5,
            "huber_quantile": 0.95,
        },
        "pcr": {"n_components": [2]},
        "pls": {"n_components": [1]},
        "glm": {
            "n_knots": 3,
            "transform_batch_size": 32,
            "streaming_epochs": 2,
            "sklearn_alpha": [1e-4],
            "paper_rho": 0.5,
            "huber_quantile": 0.95,
        },
        "rf": {
            "n_estimators": 10,
            "max_depth": [2],
            "max_features": [3],
            "min_samples_leaf": 1,
            "forest_n_jobs": 1,
        },
        "gbrt_huber": {
            "n_estimators": [10],
            "learning_rate": [0.1],
            "max_depth": [1],
            "min_samples_leaf": 1,
            "huber_quantile": 0.95,
        },
    }
    for name in ["ols", "ols3", "enet_huber", "pcr", "pls", "glm", "rf", "gbrt_huber"]:
        result = fit_model(name, cfg, Xtr, ytr, Xv, yv, seed=1, tuning_n_jobs=1)
        pred = np.asarray(result.model.predict(Xv)).ravel()
        assert pred.shape == (len(Xv),)
        assert np.isfinite(pred).all()
        assert np.isfinite(result.val_mse)
        if name in {"ols", "ols3", "enet_huber", "glm"}:
            assert result.diagnostics["huber_cutoff"] > 0
            assert 0 <= result.diagnostics["huber_tail_fraction_train"] <= 1
        if name in {"ols", "ols3", "enet_huber"}:
            assert "scale" in result.model.named_steps
            assert result.diagnostics["sgd_n_iter"] >= 1
            assert isinstance(result.diagnostics["sgd_convergence_warning"], bool)
        if name == "glm":
            assert result.diagnostics["optimizer"] == "streaming_partial_fit"
            assert result.diagnostics["transform_batch_size"] == 32
            assert result.diagnostics["spline_features"] == Xtr.shape[1] * 3


def test_nn_handles_singleton_remainder_and_batched_prediction():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(70, 5)), columns=list("abcde"))
    y = rng.normal(size=70).astype(np.float32)
    cfg = {
        "neural_net": {
            "epochs": 2,
            "batch_size": 32,  # 65 training rows below -> remainder 1
            "inference_batch_size": 7,
            "learning_rate": [0.001],
            "l1_lambda": [0.0001],
            "patience": 1,
            "seeds": [1],
            "device": "cpu",
        }
    }
    result = fit_model("nn1", cfg, x.iloc[:65], y[:65], x.iloc[65:], y[65:], seed=1)
    pred = result.model.predict(x.iloc[65:])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


def test_nn_output_initialization_is_small():
    import torch

    torch.manual_seed(1)
    net = Net(20, [32], output_init_scale=0.01)
    x = torch.randn(512, 20)
    pred = net(x).detach().numpy()
    assert abs(float(pred.mean())) < 0.02
    assert float(pred.std()) < 0.02


def test_nn_signal_smoke_has_nonnegative_oos_r2():
    rng = np.random.default_rng(7)
    n, p = 1200, 12
    x = rng.normal(size=(n, p)).astype(np.float32)
    y = (
        0.04 * x[:, 0]
        - 0.03 * x[:, 1]
        + 0.02 * np.maximum(x[:, 2], 0)
        + rng.normal(scale=0.06, size=n)
    ).astype(np.float32)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(p)])
    cfg = {
        "neural_net": {
            "epochs": 10,
            "batch_size": 256,
            "inference_batch_size": 256,
            "learning_rate": [0.001],
            "l1_lambda": [0.0001],
            "patience": 3,
            "seeds": [1],
            "device": "cpu",
            "output_init_scale": 0.01,
        }
    }
    result = fit_model(
        "nn1",
        cfg,
        frame.iloc[:800],
        y[:800],
        frame.iloc[800:1000],
        y[800:1000],
    )
    pred = result.model.predict(frame.iloc[1000:])
    assert panel_r2(y[1000:], pred) >= 0.0


def test_glm_spline_transform_is_row_bounded(monkeypatch):
    from sklearn.preprocessing import SplineTransformer

    Xtr, ytr, Xv, yv = toy(n=220, p=6, seed=9)
    original = SplineTransformer.transform
    seen = []

    def guarded(self, X):
        seen.append(len(X))
        assert len(X) <= 17
        return original(self, X)

    monkeypatch.setattr(SplineTransformer, "transform", guarded)
    cfg = {
        "glm": {
            "n_knots": 3,
            "transform_batch_size": 17,
            "streaming_epochs": 2,
            "sklearn_alpha": [1e-4],
            "paper_rho": 0.5,
            "huber_quantile": 0.95,
            "eta0": 1e-3,
        }
    }
    result = fit_model("glm", cfg, Xtr, ytr, Xv, yv, seed=1, tuning_n_jobs=1)
    assert seen
    assert max(seen) <= 17
    assert np.isfinite(result.val_mse)
