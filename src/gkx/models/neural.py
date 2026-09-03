from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import FitResult

ARCH = {
    "nn1": [32],
    "nn2": [32, 16],
    "nn3": [32, 16, 8],
    "nn4": [32, 16, 8, 4],
    "nn5": [32, 16, 8, 4, 2],
}


class Net(nn.Module):
    def __init__(self, p, widths, output_init_scale=0.01):
        super().__init__()
        layers = []
        for q in widths:
            # GKX Internet Appendix B.3 states that BN is applied after each ReLU activation.
            layers.extend([nn.Linear(p, q), nn.ReLU(), nn.BatchNorm1d(q)])
            p = q
        output = nn.Linear(p, 1)
        with torch.no_grad():
            output.weight.mul_(float(output_init_scale))
            output.bias.zero_()
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _linear_weight_l1(net):
    return sum(layer.weight.abs().sum() for layer in net.modules() if isinstance(layer, nn.Linear))


def _set_deterministic(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _predict_one(net, array, device, batch_size):
    out = np.empty(len(array), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            stop = min(start + batch_size, len(array))
            xb = torch.from_numpy(array[start:stop]).to(device)
            out[start:stop] = net(xb).detach().cpu().numpy()
    return out


def _batched_mse(net, array, target, device, batch_size):
    squared_error = 0.0
    count = 0
    net.eval()
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            stop = min(start + batch_size, len(array))
            xb = torch.from_numpy(array[start:stop]).to(device)
            yb = torch.from_numpy(target[start:stop]).to(device)
            diff = net(xb) - yb
            squared_error += float(torch.sum(diff * diff).cpu())
            count += stop - start
    return squared_error / count


class NNEnsemble:
    def __init__(self, nets, device, inference_batch_size):
        self.nets = nets
        self.device = device
        self.inference_batch_size = inference_batch_size

    def predict(self, X):
        array = np.asarray(X, dtype=np.float32)
        total = np.zeros(len(array), dtype=np.float64)
        for net in self.nets:
            total += _predict_one(net, array, self.device, self.inference_batch_size)
        return (total / len(self.nets)).astype(np.float32)


def fit_nn(name, cfg, Xtr, ytr, Xv, yv):
    train = np.asarray(Xtr, dtype=np.float32)
    valid = np.asarray(Xv, dtype=np.float32)
    y_train = np.asarray(ytr, dtype=np.float32)
    y_valid = np.asarray(yv, dtype=np.float32)
    if len(train) < 2:
        raise ValueError("Neural-network training requires at least two observations")

    requested = cfg.get("device", "auto")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    inference_batch_size = int(cfg.get("inference_batch_size", 65536))
    best_ensemble = None

    l1_grid = cfg.get("l1_lambda", [1e-5, 1e-4, 1e-3])
    lr_grid = cfg.get("learning_rate", [1e-3, 1e-2])
    if not isinstance(lr_grid, list):
        lr_grid = [lr_grid]

    for l1 in l1_grid:
        for lr in lr_grid:
            nets = []
            for seed in cfg["seeds"]:
                _set_deterministic(seed)
                net = Net(
                    train.shape[1],
                    ARCH[name],
                    output_init_scale=cfg.get("output_init_scale", 0.01),
                ).to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                mse_loss = nn.MSELoss()
                dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(y_train))
                batch_size = min(int(cfg["batch_size"]), len(dataset))
                drop_last = len(dataset) % batch_size == 1 and len(dataset) > batch_size
                generator = torch.Generator().manual_seed(seed)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                    generator=generator,
                    pin_memory=device == "cuda",
                )
                state = None
                best = np.inf
                stale = 0
                for _ in range(int(cfg["epochs"])):
                    net.train()
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        optimizer.zero_grad()
                        pred = net(xb)
                        loss = mse_loss(pred, yb) + float(l1) * _linear_weight_l1(net)
                        loss.backward()
                        optimizer.step()
                    val_mse = _batched_mse(
                        net,
                        valid,
                        y_valid,
                        device,
                        inference_batch_size,
                    )
                    if val_mse < best - 1e-8:
                        best = val_mse
                        state = copy.deepcopy(net.state_dict())
                        stale = 0
                    else:
                        stale += 1
                    if stale >= int(cfg["patience"]):
                        break
                if state is None:
                    state = copy.deepcopy(net.state_dict())
                net.load_state_dict(state)
                net.eval()
                nets.append(net)

            ensemble = NNEnsemble(nets, device, inference_batch_size)
            mse = float(np.mean((y_valid - ensemble.predict(valid)) ** 2))
            result = FitResult(
                ensemble,
                {
                    "l1_lambda": l1,
                    "learning_rate": lr,
                    "architecture": ARCH[name],
                    "seeds": cfg["seeds"],
                    "output_init_scale": cfg.get("output_init_scale", 0.01),
                },
                mse,
            )
            if best_ensemble is None or result.val_mse < best_ensemble.val_mse:
                best_ensemble = result
    return best_ensemble
