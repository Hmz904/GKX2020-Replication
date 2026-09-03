# GKX (2020) — Empirical Asset Pricing via Machine Learning

Independent Python replication framework for Gu, Kelly, and Xiu (2020), *The Review of Financial Studies*, 33(5), 2223–2273.

> This is not the authors' official code. The goal is a transparent, testable replication framework. Where this repository does not implement the authors' exact numerical algorithm, that difference is stated explicitly below rather than hidden behind a model label.

## Scope

Implemented model families:

- robust OLS+H and robust OLS-3+H
- robust elastic net (library-level approximation; see Fidelity notes)
- PCR and PLS
- spline-additive GLM (approximation; see Fidelity notes)
- random forest
- Huber gradient boosted regression trees
- NN1–NN5

The recursive design uses the paper's initial training / validation / testing chronology and annual refits. Each `(test year, model)` is checkpointed independently so a long run can resume after interruption.

## Data preprocessing: important

GKX do **not** feed the raw stock-characteristic table directly into pooled scikit-learn preprocessing. Their empirical design cross-sectionally ranks every stock characteristic each month and maps ranks to `[-1, 1]`; missing characteristics are replaced with the monthly cross-sectional median. This repository implements those operations in `features.mode: interact` before constructing the predictive design.

The raw/interact path then constructs:

- 94 ranked stock characteristics;
- each characteristic interacted with 8 aggregate macro predictors (the characteristic itself is the constant interaction); and
- SIC2 industry dummy columns.

The intended total is `94 × (1 + 8) + 74 = 920` predictors. Set `features.expected_num_features: 920` to make this an enforced invariant.

`features.characteristic_columns` must be supplied explicitly. This is deliberate: automatic selection of every numeric column can silently include returns, prices, identifiers, industry codes, or other metadata and create leakage. For an already prepared 920-column matrix, use `features.mode: prebuilt` and provide `features.columns` explicitly.

The monthly characteristic ranks are computed once for the loaded panel and then reused across annual refits; because each rank uses only the contemporaneous cross-section, this cache does not introduce time-series look-ahead. If `features.industry_values` is omitted, the SIC2 dummy schema is inferred only from the initial pre-test sample (through `initial_validation_end`), not from future test years. For the strictest reproducibility, supply the intended industry values explicitly.

The raw market-cap column used for size sorts and value weighting is preserved separately from the transformed design. If the same source variable is also listed as a predictive characteristic, its ranked version remains a predictor while the untransformed raw column is retained for portfolio weights. Some invalid market caps are tolerated and excluded only from size/value-weighted evaluation, but an obviously broken mapping is not silent: `data.min_valid_market_cap_fraction` defaults to `0.01`, and validation fails if fewer than that fraction of rows have finite positive market cap. `r2_oos.csv` reports `n_obs` for every all/top/bottom R² row so exclusions are visible.

## Fidelity notes

### Sample design and evaluation

The repository implements the paper-style chronological training/validation/test recursion, pooled zero-benchmark OOS R², top/bottom-size OOS R², monthly-cross-section modified Diebold–Mariano comparisons, prediction-sorted decile portfolios, and 10–1 spreads.

Turnover uses the paper's pre-rebalance drift correction:

`sum_i | w[i,t+1] - w[i,t] * (1 + r[i,t+1]) / (1 + sum_j w[j,t] r[j,t+1]) |`

Here `r[i,t+1]` is the stock's **total return**, supplied through `data.return_col`; it is not the excess-return prediction target. The implementation follows GKX's reported sum of absolute trades and therefore does **not** divide the long-short spread turnover by two. See the turnover definition in Gu, Kelly, and Xiu (2020): https://doi.org/10.1093/rfs/hhaa009. `turnover_monthly.csv` stores each transition and `turnover.csv` stores the time average. Portfolio-return files contain **gross** returns; this repository does not subtract a fixed number of basis points unrelated to actual trading.

### Huber loss

The registry key `ols` denotes the paper's **OLS+H** reporting variant, not plain squared-loss OLS. GKX explicitly state in their results discussion that the reported OLS, ENet, GLM, and GBRT versions use Huber loss; the paper source is https://doi.org/10.1093/rfs/hhaa009. `ols3` analogously denotes OLS-3+H.

For OLS+H, OLS-3+H, and the SGD-based linear/GLM approximations, the Huber cutoff is obtained from **pilot-model residuals**, not from the unconditional magnitude of the return target. The chosen cutoff and the fraction of final training residuals beyond it are stored in the JSON diagnostics field of `tuning.csv`. The robust linear fits use scikit-learn SGD rather than the authors' exact numerical routine, so they target the intended Huber objective but are not claimed to be bit-for-bit reproductions. sklearn's `GradientBoostingRegressor(loss="huber", alpha=...)` applies its Huber quantile to residuals stage by stage.

### Elastic net

The paper defines

`lambda * (1-rho) * L1 + 0.5 * lambda * rho * L2`,

so `rho=0` is Lasso and `rho=1` is Ridge. sklearn's `l1_ratio` uses the opposite convention; therefore `sklearn_l1_ratio = 1 - paper_rho`. At the paper's `rho=0.5` the numerical value happens to be unchanged.

The remaining limitation is optimization: GKX use accelerated/proximal numerical routines for the penalized models, whereas this repository currently uses `SGDRegressor` for the robust linear and elastic-net approximations. The SGD-based OLS/ENet paths standardize their design internally with `StandardScaler` so the optimizer is not driven by raw scale differences among characteristic-macro interactions. This scaling reparameterizes the penalized problem, and `SGDRegressor.alpha` is **not** asserted to equal GKX's lambda. The config consequently calls this field `sklearn_alpha` and tunes it on the validation sample. Selected OLS/ENet metadata records SGD iteration counts and whether scikit-learn emitted a `ConvergenceWarning`.

### GLM

GKX define `z` as the full `P`-dimensional predictor vector and their OLS benchmark uses all 920 predictors; Equation (13) then applies the spline basis to every `z_j`, `j=1,...,P`. Therefore this repository does **not** reduce the GLM expansion to only the 94 base characteristics as a memory shortcut. GKX use an order-two spline expansion with group lasso and accelerated proximal gradient; see https://doi.org/10.1093/rfs/hhaa009.

The current `glm` remains a **documented approximation**: it uses scikit-learn's order-two `SplineTransformer` plus Huber elastic-net shrinkage rather than the paper's truncated-power/group-lasso APG solver. To prevent the 920→approximately 2,760-column expansion from requiring a 50–100+ GB dense intermediate, spline transformation, standardization, SGD training, and prediction are performed in row chunks (`transform_batch_size`). `streaming_epochs` controls the explicit partial-fit passes over the training sample. GLM grid points are intentionally evaluated serially to bound peak memory. No `groupyr` dependency is declared because the code does not use it.

### Neural networks

This implementation chooses `Linear → ReLU → BatchNorm` for hidden layers, consistent with one direct reading of the appendix's wording that normalization is applied to activations after ReLU. Because this ordering can materially affect optimization, the repository treats it as an implementation choice rather than claiming that alternative placements are impossible. The L1 penalty is applied to linear-layer weight matrices only, excluding biases and BatchNorm gamma/beta parameters.

Validation and inference are batched (`inference_batch_size`) so the full 12-year validation block or test block is not moved to GPU at once. Training also protects against the BatchNorm singleton-last-batch failure. The final prediction layer is initialized at a small scale (`output_init_scale`, default `0.01`) with zero bias so an untrained network starts near the zero-return benchmark rather than with predictions an order of magnitude larger than monthly returns. Synthetic CI tests require the NN smoke model to achieve nonnegative OOS R² on a seeded signal problem. PyTorch deterministic behavior is requested where the backend supports it.

## Computational reality

The full GKX problem is large: millions of stock-months and 920 predictors. Exact sklearn GBRT is particularly expensive because its trees are fit serially. This repository retains it for closer algorithmic correspondence rather than silently replacing it with a different boosting implementation.

The current loader still materializes the **projected raw panel** in one pandas DataFrame. Column projection, float32 downcasting, and per-window interaction construction substantially reduce peak memory relative to building the full 920-column design up front, but this is not yet an out-of-core engine. Final reporting also concatenates matching prediction checkpoints before computing summary tables.

Practical recommendations:

- use Parquet rather than CSV for the full panel;
- keep `downcast_float32: true`;
- store raw characteristics/macros rather than a full-sample dense 920-column interaction file;
- use checkpoints and test one or two models/years before launching the horse race;
- set `tuning_n_jobs` conservatively to avoid memory pressure;
- expect the full 30-year × 13-model experiment to require substantial CPU/GPU time and tens of GB of RAM. A 32–64 GB machine is a more realistic floor than a typical laptop, and the exact full grid can require far more compute.

Grid-point tuning can run through joblib threads for ENet, PCR, RF, and GBRT. Random forests avoid nested oversubscription by using one internal job per forest when the outer grid is parallelized. The GLM path is deliberately serial because each candidate streams the spline-expanded design in bounded chunks; parallel candidates would multiply peak memory. Years are also processed serially to keep peak memory bounded.

With the default full grids, one annual refit involves 89 classical-model fits plus 300 neural-network seed trainings (five architectures × six hyperparameter pairs × ten seeds). Across 30 test years that is roughly **2,670 classical fits + 9,000 neural-network trainings** before any reruns, so wall-clock time is highly hardware dependent and can be very large.

## Checkpoints and resume

Each successful fit writes immediately to:

```text
outputs/<run_name>/checkpoints/<year>/<model>.csv.gz
outputs/<run_name>/checkpoints/<year>/<model>.json
```

The prediction file and tuning/diagnostic metadata are written atomically. Prediction checkpoints contain only keys plus predictions; current target/market-cap/total-return fields are reattached from the panel when summary tables are rebuilt. Each checkpoint stores a **model-specific prediction hash** covering the implementation version, prediction-relevant data settings, feature construction, recursive split, seed, and that model's predictive settings. Reporting-only settings (`evaluation.*`), `tuning_n_jobs`, `run_name`, `models.names`, RF worker count, and NN inference batch size are intentionally excluded.

Nested model dictionaries are fail-safe: every key is hashed unless explicitly classified as execution-only. The few top-level model options (`ols_*`, `ols3_*`, etc.) are registered explicitly; an unknown top-level key such as a newly added `models.ols_tol` raises an error until the developer classifies it, rather than silently reusing stale predictions. Therefore changing top/bottom cutoffs or Newey-West lags rebuilds summary tables without retraining, and adding NN5 after other models have finished trains only NN5. `run_config.json` still snapshots the full configuration used for the current invocation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"

python scripts/inspect_data.py data/processed/gkx_panel.parquet
gkx validate-data --config configs/default.yaml
gkx run --config configs/default.yaml
ruff check .
pytest -q
```

For a cheap end-to-end test:

```bash
gkx make-synthetic --output data/processed/synthetic.csv
gkx run --config configs/synthetic.yaml --models ols,ols3,rf,nn1
```

## Expected input

The runner expects one stock-month per row, with explicit configuration for:

| field | purpose |
|---|---|
| date | panel month |
| security ID | unique stock identifier |
| next-month excess return | prediction target |
| stock total return | turnover weight-drift calculation (`data.return_col`) |
| lagged raw market equity | size subsets and value weights |
| 94 characteristic columns | raw/interact mode |
| 8 macro columns | raw/interact mode |
| SIC2 | raw/interact mode, converted to dummies |

CRSP/Compustat/IBES-derived data are not redistributed. `configs/default.yaml` therefore points to a **prepared** stock-month panel rather than pretending the public characteristic file is directly runnable. The paper predicts excess returns, so construct the appropriate next-month excess-return target, but also retain the stock **total return over that same next-month holding period** separately for the turnover formula. The public Xiu characteristic data do not provide every field needed for portfolio evaluation in the exact form expected here; retain an appropriate lagged raw market-cap measure for value weighting and merge the eight macro predictors before validation.

## Outputs

```text
run_config.json
predictions.csv.gz
r2_oos.csv
dm_tests.csv
portfolio_returns.csv
portfolio_stats.csv
turnover_monthly.csv
turnover.csv
tuning.csv                  # stable columns + params_json / diagnostics_json
checkpoints/<year>/<model>.*
```

## Repository layout

```text
configs/                 experiment definitions
scripts/                 inspection utilities
src/gkx/data.py          loading, validation, GKX preprocessing and feature construction
src/gkx/splits.py        recursive train/validation/test windows
src/gkx/models/          model implementations and registry
src/gkx/evaluation.py    R², DM, portfolio and turnover metrics
src/gkx/runner.py        annual estimation, checkpoint/resume, final reporting
src/gkx/cli.py           command-line interface
tests/                   unit and smoke tests
.github/workflows/       CI
CHANGELOG.md              factual implementation changes and known limitations
```

## Citation

```bibtex
@article{gu2020empirical,
  title={Empirical Asset Pricing via Machine Learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2223--2273},
  year={2020}
}
```
