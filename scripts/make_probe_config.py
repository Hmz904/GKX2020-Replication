"""Derive a single-point cost probe from a full GKX2020 run config.

The point of a probe is to measure the unit cost of each model on a given panel before
committing to the full grid. Every grid axis is cut to one value and the test window to one
year; nothing else changes, so the per-model wall times it produces are the real ones.

Do not hand-write the probe. The config carries 94 characteristic names and an industry count
that was measured when the panel was built, and a transcription error in either produces a run
that either fails late or silently measures something else.

Extrapolating afterwards: multiply each model's probe time by its own grid size and by the
number of test years. The multipliers are NOT the same across models, so a single total-time
figure multiplied by one number is wrong. With the stock grid:

    ols, ols3        1  x years          (no grid)
    enet_huber       6  x years          (6 alphas)
    pcr              6  x years
    pls              5  x years
    glm              6  x years
    rf              36  x years          (6 max_depth x 6 max_features)
    gbrt_huber       4  x years          (2 lr x 2 max_depth; n_estimators is free
                                          via staged_predict, which is why that
                                          optimization mattered)
    nn1..nn5        18  x years each     (2 lr x 3 l1 x 3 seeds)

Usage
  python scripts/make_probe_config.py --config configs/min189.yaml \
                                      --out configs/min189_probe.yaml \
                                      --test-year 2011
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="full run config to derive from")
parser.add_argument("--out", required=True)
parser.add_argument("--test-year", type=int, default=2011)
parser.add_argument("--suffix", default="_probe")
parser.add_argument("--keep-models", nargs="*", default=None,
                    help="restrict models.names; default keeps all so the cost split is visible")
args = parser.parse_args()

# utf-8-sig: two configs in the sibling repo carried a BOM that yaml.safe_load does not strip,
# which turned the first key into "\ufeffprotocol". Read defensively, write clean.
cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))

cfg["run_name"] = f"{cfg.get('run_name', 'gkx')}{args.suffix}"

split = cfg["split"]
split["test_start"] = f"{args.test_year}-01-01"
split["test_end"] = f"{args.test_year}-12-31"

models = cfg["models"]
if args.keep_models:
    models["names"] = list(args.keep_models)

# One value per grid axis. Pick from the middle of each list rather than an endpoint: the cheap
# end of a grid is not representative of its average cost, and for tree depth or component count
# the endpoints differ from the middle by more than a constant factor.
def middle(values):
    seq = list(values)
    return [seq[len(seq) // 2]]

grids = {
    "enet_huber": ["sklearn_alpha"],
    "pcr": ["n_components"],
    "pls": ["n_components"],
    "glm": ["sklearn_alpha"],
    "rf": ["max_depth", "max_features"],
    "gbrt_huber": ["n_estimators", "learning_rate", "max_depth"],
}
cut = []
for name, keys in grids.items():
    block = models.get(name)
    if not isinstance(block, dict):
        continue
    for key in keys:
        if isinstance(block.get(key), list) and len(block[key]) > 1:
            before = len(block[key])
            block[key] = middle(block[key])
            cut.append(f"{name}.{key}: {before} -> 1 ({block[key][0]})")

nn = models.get("neural_net")
if isinstance(nn, dict):
    for key in ("learning_rate", "l1_lambda"):
        if isinstance(nn.get(key), list) and len(nn[key]) > 1:
            before = len(nn[key])
            nn[key] = middle(nn[key])
            cut.append(f"neural_net.{key}: {before} -> 1 ({nn[key][0]})")
    if isinstance(nn.get("seeds"), list) and len(nn["seeds"]) > 1:
        before = len(nn["seeds"])
        nn["seeds"] = [nn["seeds"][0]]
        cut.append(f"neural_net.seeds: {before} -> 1")

# epochs, patience and n_estimators-as-a-fixed-value stay untouched. How many epochs a model
# actually needs is part of what the probe is measuring; capping it would measure the cap.

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

print(f"derived {out} from {args.config}")
print(f"  run_name   : {cfg['run_name']}")
print(f"  test window: {split['test_start']} .. {split['test_end']}")
print(f"  models     : {len(models['names'])} ({', '.join(models['names'])})")
print("  grid cuts  :")
for line in cut:
    print(f"    {line}")
print()
print("  tuning_n_jobs:", cfg.get("tuning_n_jobs"),
      "(must be 1 on Windows; joblib deadlocks silently above that)")
print("  features.mode:", cfg["features"]["mode"],
      f"-> {cfg['features']['expected_num_features']} columns at load time")
