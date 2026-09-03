# Changelog

## 0.2.2 — 2026-09-03

Third public-release hardening pass focused on checkpoint safety, GLM memory, and evaluation transparency.

- Hardened checkpoint fingerprints. Nested model dictionaries are hashed by default except explicitly execution-only keys (`rf.forest_n_jobs`, `neural_net.inference_batch_size`); unknown top-level `models.*` options now raise instead of being silently omitted from checkpoint invalidation. The package implementation version is also included in each prediction hash.
- Prediction checkpoints now contain only identifiers and forecasts. Targets, raw market caps, and total returns are reattached from the current panel during reporting, so evaluation-only column mappings can change without retraining or leaving stale evaluation data inside checkpoints.
- Reworked the approximate GLM into a bounded-memory streaming path. The full predictor set is still spline-expanded, but spline transformation, `StandardScaler`, SGD `partial_fit`, and prediction operate in row chunks; the complete dense 920→~2,760-column matrix is never materialized. GLM grid points are intentionally serial to bound peak RAM.
- Added a hard minimum valid-market-cap coverage check (`data.min_valid_market_cap_fraction`, default 1%) while retaining warning-and-exclude behavior for a minority of invalid observations. `r2_oos.csv` now reports `n_obs` for every sample row.
- Normalized `tuning.csv` to stable columns with `params_json` and `diagnostics_json` rather than model-specific sparse columns. GBRT Huber diagnostics use the same diagnostic names as the robust linear paths.
- Monthly cross-sectional characteristic ranks are now computed once per loaded panel and reused across annual refits instead of being recomputed for every train/validation/test window.
- When `features.industry_values` is omitted, SIC2 categories are inferred only from the initial pre-test sample rather than the full panel, removing a small schema look-ahead.
- Turnover now requires a separate stock total-return field (`data.return_col`) for weight drift. The field is validated before any output directory or model fit is started, so a missing mapping cannot fail only after an expensive run. The excess-return prediction target is no longer reused in the turnover formula, and the README documents GKX's no-`1/2` sum-of-absolute-trades convention.
- Added a full DM-matrix regression test and computes each unordered model pair once before mirroring the statistic/sign.
- Removed the dead `feature_columns()` helper and its obsolete tests.
- Removed the pandas `astype(copy=False)` deprecation path that surfaced under newer pandas versions.
- Expanded regression coverage for unknown hash keys, execution-only hash exclusions, total-return turnover, market-cap coverage, `n_obs`, pre-test industry inference, one-time rank preprocessing, streaming GLM diagnostics, and stable tuning JSON output.

## 0.2.1 — 2026-09-03

Follow-up hardening after a second public-release audit.

- Fixed the remaining Ruff import-order failure in `config.py`.
- Replaced the full-config checkpoint hash with a model-specific prediction hash. Evaluation settings, `tuning_n_jobs`, `run_name`, and `models.names` no longer invalidate expensive predictions; data/features/split/seed and the model's own predictive settings still do.
- Added `StandardScaler` inside all SGD-based robust OLS, elastic-net, and approximate GLM pipelines; selected-fit metadata now records SGD iteration counts and captured `ConvergenceWarning` status.
- Added small-scale, zero-bias initialization for the NN output layer and a seeded signal test requiring nonnegative NN OOS R².
- Relaxed full-panel market-cap validation: invalid market caps generate a warning and remain usable for prediction, while size-sorted/value-weighted evaluation excludes them.
- Expanded checkpoint-hash, market-cap, scaling, convergence-diagnostic, and neural-network regression tests.

## 0.2.0 — 2026-09-03

This release focuses on correctness and failure recovery before public release.

### Correctness

- Full OLS and OLS-3 now both use the robust Huber path, matching the paper's reported OLS+H / OLS-3+H variants.
- Fixed portfolio-weight construction in turnover calculations. The old `Series.rpow(-1)` expression computed `(-1) ** n`, not `1 / n`.
- Reimplemented turnover for the 10-1 spread using GKX's return-drift adjustment before rebalancing; turnover is no longer divided by two.
- Removed the fixed-per-month transaction-cost subtraction. Gross portfolio returns are reported; a cost model should be applied to actual trades/turnover rather than as a constant return haircut.
- Added GKX monthly cross-sectional characteristic ranks mapped to `[-1, 1]`, followed by monthly cross-sectional median filling (remaining all-missing monthly cells fall back to zero).
- Disabled implicit numeric-column feature discovery. Prebuilt designs require an explicit feature list, preventing accidental inclusion of returns, prices, identifiers, or other metadata.
- `market_cap_col` is no longer implicitly excluded from an explicitly requested feature design. Raw market equity used for portfolio weights remains untouched when a corresponding characteristic is rank transformed for prediction.
- Added construction of SIC2 dummy columns in raw/interact mode rather than requiring users to pre-build them silently.
- Huber cutoffs for OLS-3 and the SGD-based ENet/GLM approximations are now estimated from pilot-model residuals rather than directly from `|y|`.
- Huber diagnostics are written to tuning metadata so users can inspect the fraction of training residuals outside the cutoff.

### Neural networks

- Validation and prediction are batched instead of moving the entire validation/test matrix to the accelerator at once.
- Avoided the BatchNorm singleton-batch crash by dropping a final training batch only when the remainder would be exactly one observation.
- L1 regularization now applies only to linear-layer weight matrices, not biases or BatchNorm gamma/beta parameters.
- Removed NumPy global RNG mutation from NN fitting and enabled deterministic PyTorch behavior where supported.

### Reliability and scale

- Added one prediction checkpoint and one metadata checkpoint per `(test year, model)`.
- Checkpoints are written atomically; resume requires a matching configuration hash, preventing stale results from being silently reused after configuration changes.
- Each run writes `run_config.json` as a reproducibility snapshot.
- Input readers project only the explicitly required columns when the file format supports it.
- Floating-point panel columns are downcast to float32 by default.
- Feature interactions are constructed separately for each recursive window instead of duplicating the entire full-sample panel up front.
- Grid tuning for ENet, GLM, PCR, RF, and GBRT can use joblib threads through `tuning_n_jobs`; random-forest internal parallelism is disabled during parallel grid search to avoid nested oversubscription.

### Reproducibility / repository hygiene

- Added GitHub Actions for `pytest` and `ruff`.
- Expanded tests for turnover, preprocessing, date parsing, leakage protection, interaction construction, Huber diagnostics, neural singleton-batch handling, and checkpoint/resume behavior.
- Removed unused runtime dependencies and unused factor-regression code.
- `initial_validation_start`, `initial_validation_end`, and `refit_frequency` are now enforced by the split code rather than being no-op configuration fields.
- Replaced the self-certifying `AUDIT.md` with this factual change log plus README fidelity notes.
- Completed the MIT copyright line with a project-contributor designation.

### Remaining fidelity limitations

- The full projected raw panel is still loaded into pandas memory and final reporting concatenates prediction checkpoints; the implementation is more memory-conscious but not out-of-core.
- Robust OLS/OLS-3 use scikit-learn SGD to optimize a Huber objective rather than the authors' exact numerical routine.
- The ENet implementation still uses scikit-learn SGD rather than GKX's accelerated proximal-gradient solver. `SGDRegressor.alpha` is therefore explicitly treated as an implementation tuning parameter, not the paper's lambda.
- The GLM remains an approximation: scikit-learn spline expansion plus Huber elastic-net shrinkage is not GKX's truncated-power spline plus Huber group lasso/APG implementation.
- Exact sklearn GBRT is intentionally retained for methodological closeness; the full GKX grid is computationally expensive. Faster XGBoost/HistGBRT backends would change the algorithm and should be offered as separate, clearly labelled alternatives rather than silently substituted.
