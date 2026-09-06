# Findings

Defects and results found by actually executing this pipeline on real data. Every number below
was measured. Where an earlier conclusion in this file turned out to be wrong, it is corrected
in place rather than deleted.

**Reference run.** 298 permnos sampled at random from the Xiu characteristic datashare,
1995-01 to 2015-12, 26,410 firm-months, 892-column interact design (94 characteristics x
(constant + 8 macro) + 46 industry dummies), expanding window, test years 2011-2015
(n = 5,521, roughly 88 names per monthly cross-section). All 13 models. The target is
reconstructed from public data and is **not** the CRSP return used by Gu-Kelly-Xiu; see §8.

---

## 1. A linear model that passed synthetic stress testing diverged on real data — fixed

`enet_huber` scored **+2.287% OOS R², the best of all models**, on the 920-predictor synthetic
stress panel. The same code, same protocol, on real datashare characteristics produced:

```
enet_huber   -2,147,246,851,270.789 %
```

That is numerical divergence, not poor performance: predictions grow until they overwhelm the
denominator of the R² statistic. The cause was the fixed-step `SGDRegressor` (`eta0=1e-3`)
diverging at small `sklearn_alpha` on real characteristic distributions. The synthetic
generator draws AR(1) latents with far better conditioning, which hid the failure completely.

**This is the central methodological result of this repository.** Replication infrastructure
that passes its own synthetic tests can still be broken, and broken specifically in ways only
real data exposes. A synthetic stress test proves the pipeline can recover known structure; it
does not certify any individual estimator.

**Status: resolved.** Two changes:

- a divergence guard that rejects any grid point whose validation MSE exceeds the target
  variance by a set factor, asserted in `tests/test_enet_divergence.py`;
- `eta0` lowered from `1e-3` to `1e-4` for `ols`, `ols3`, `enet_huber` and `glm`.

Measured on the reference run, unchanged in every other respect:

| | OOS R² |
|---|---|
| `eta0 = 1e-3` | −2,147,246,851,270.789 % |
| `eta0 = 1e-4` | **+0.0497 %** |

`enet_huber` goes from the worst model in the table to the best. The guard is what makes the
first row an error rather than a result.

---

## 2. Full horse race, and the cost of an unregularised design

Reference run, all 13 models, test years 2011-2015 (n = 5,521):

| model | OOS R² (all) | top 30 | bottom 30 |
|---|---|---|---|
| enet_huber | **+0.050 %** | +0.287 % | −0.076 % |
| rf | **+0.042 %** | +0.419 % | −0.238 % |
| ols3 | −0.055 % | +0.034 % | +0.142 % |
| gbrt_huber | −0.131 % | −0.356 % | +0.007 % |
| pcr | −0.140 % | −0.215 % | −0.198 % |
| nn1 | −0.185 % | +0.001 % | −0.276 % |
| nn3 | −0.248 % | −0.080 % | −0.445 % |
| glm | −0.262 % | −0.258 % | −0.269 % |
| nn2 | −0.290 % | −0.054 % | −0.532 % |
| nn5 | −0.397 % | −0.309 % | −0.607 % |
| nn4 | −0.466 % | −0.343 % | −0.489 % |
| pls | −0.641 % | −0.924 % | −0.467 % |
| **ols** | **−24.96 %** | −28.52 % | −19.67 % |

### What is statistically defensible

Diebold-Mariano tests across all 78 model pairs give exactly two families of significant
results:

- **`ols` is worse than every one of the other twelve models**, t between 5.85 and 6.82,
  p between 3.6e-9 and 9.3e-12.
- **`pls` is worse than `enet_huber` (t = −3.28, p = 0.0010), `ols3` (t = −3.21, p = 0.0013)
  and `rf` (t = −2.43, p = 0.015)**; marginal against `pcr` (p = 0.062).

Everything else is indistinguishable. In particular the top two models, `enet_huber` and
`rf`, differ by **t = −0.054, p = 0.957**. On this sample the claim "trees beat linear models"
is **not supported**; they tie. The size split points the right way — `rf` earns +0.419% on the
largest 30 names against −0.238% on the smallest, and `enet_huber` +0.287% against −0.076% —
which is the direction GKX report, but 1,770 observations cannot carry that claim.

### The one result the sample size does support

Comparing against the earlier degraded run in this repository, holding the pipeline fixed and
varying only the width of the design:

| design | `ols` OOS R² |
|---|---|
| 152 columns (500 permnos, 2000-2015, total returns) | −0.767 % |
| **892 columns (this run)** | **−24.96 %** |

Widening the predictor set from 152 to 892 columns degrades unregularised OLS by a factor of
roughly 32. This is a monotone dose-response between predictor dimension and overfitting under
one fixed pipeline, and it is the sharpest evidence here for the paper's central argument that
regularisation is not optional at this width. It is also, unlike any individual R², large
enough to survive the sample size.

---

## 3. Target construction: the lag convention, pinned three independent ways

The datashare README states that for `DATE=19570329` the response variable is the return in
`195703` — the **same** month, because the authors have already lagged the characteristics.

Measured independently from the data: `retvol(t)` correlates with `|mom1m(t)|` at **0.573**,
against 0.443 and 0.414 for neighbouring months; `maxret(t)` correlates with `mom1m(t)` at
**+0.201**, against −0.129 and −0.025. The daily-frequency characteristics and `mom1m` are
measured over the same month, and `mom1m(t)` is the month `t-1` return.

A third check, added when the panel builder was written and now asserted at build time: the
equal-weighted cross-section of the reconstructed return is correlated against the CRSP
value-weighted market return at several lags. On the reference panel:

| lag | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|
| corr | −0.006 | +0.038 | **+0.759** | +0.222 | −0.067 |

All three routes agree: **`target(t) = mom1m(t+1)`, a shift of −1.**

A shift of −2 loses a month of information. A shift of 0 makes same-month volatility
characteristics (`retvol`, `maxret`, `baspread`, `std_dolvol`) predict their own month's
return — severe look-ahead that produces an *attractively high* R². This is the failure mode
most likely to be accepted as a good result, so the lag-0 peak is a hard assertion in
`scripts/build_public_panel.py`; deliberately inverting the shift makes the correlation peak
move to lag +1 and aborts the build.

The residual +0.222 at lag +1 is not an error. An 88-name equal-weighted cross-section is
noisy, and short-horizon reversal and non-synchronous trading put real content at that lag. It
is reported rather than suppressed because it quantifies how well the reconstructed target
aligns, which "the shift was checked" does not.

### One check that looked like validation and was not

Comparing `mom6m(t)` against the sum of `mom1m(t-0..t-5)` gives correlation 0.88, versus 0.70
for the shifted alternative. This **cannot identify the absolute lag**: if `mom1m` and `mom6m`
are lagged by the same amount, their relative alignment is unchanged and the correlation is
identical. It establishes only that the two columns share a convention. A check that appears
to confirm a hypothesis without being able to discriminate it is worse than no check.

---

## 4. Market capitalisation was double-exponentiated (fixed)

`stress.py` built market cap as `exp(9.0 + 1.4 * mvel1)`, but `mvel1` is in `_SKEWED` and is
already `exp(latent)`. Consequences measured on the generated panel:

- caps spanning **27 orders of magnitude** (median 3.66e4, max 5.98e30);
- a single stock holding a **median 94.4%** of total market value, above 50% in **241 of 263
  months** — value-weighted portfolio statistics were effectively a one-stock time series;
- one row overflowing float32 to `+inf`, which is worse than `NaN` because a single `inf`
  silently destroys an entire month's value-weighted sum.

Fixed by constructing in log space with log-cap sd 1.5 and asserting finiteness. After the fix:
top-1 weight median 7.6%, max 45.3%, no `inf`.

---

## 5. Rank transform overwrites the market cap column

`rank_characteristics_inplace` mutates the panel. When `mvel1` serves as both a predictive
characteristic and `market_cap_col`, value weighting silently reads rank values in [-1, 1]
instead of dollars. A separate raw copy is mandatory before ranking. `configs/stress.yaml`
carries `mvel1_raw` for this reason, and `scripts/build_public_panel.py` writes a separate
`mve` column for the same reason.

---

## 6. `tuning_n_jobs > 1` hangs silently on Windows

The first attempt at the reference run was launched with the shipped default
`tuning_n_jobs: 4`. It logged three models, reached `pcr`, and then stopped. Nine hours later:

- no traceback, no timeout, no error of any kind — the log simply ended mid-run;
- the parent process had accumulated **0.016 seconds** of CPU time;
- the second process had accumulated **248 seconds over nine hours**, i.e. about 0.008 cores;
- two `python.exe` processes existed with **identical full command lines**, created in the
  same second, one launched from the virtualenv interpreter and one from the *global*
  interpreter that the virtualenv shadows.

The child had re-executed the entire original command line rather than a worker entry point,
which is consistent with `loky` spawning through the `gkx.exe` console script under Windows
`spawn` semantics; the `if __name__ == "__main__"` guard in `cli.py` does not protect that
path. Setting `tuning_n_jobs: 1` resolved it and the same run completed in 16 minutes.

**Linux CI never reproduces this**, because `fork` does not re-import the entry module. It is
the second Windows-only defect found in this codebase in one session — the first was
`Path.read_text()` without an explicit encoding, which raises `UnicodeDecodeError` on any
non-ASCII byte under a GBK locale and is likewise invisible to a UTF-8 CI runner.

Two lessons worth stating plainly. First, a green CI on one platform certifies that platform
only. Second, this failure mode is the dangerous kind: it does not crash, it waits, and nine
hours of wall clock produced no output and no diagnostic.

---

## 7. Both tree models select the *minimum* complexity in the grid — correcting an earlier claim

**Earlier claim in this file, now superseded.** On the synthetic stress panel every selected
random forest configuration sat at the grid's upper boundary (`max_depth=6`), and this was
recorded as evidence of under-tuning and an argument for widening the grid upward.

On the real subsample the selection is **reversed**:

| year | `rf` max_depth | `rf` max_features | `gbrt_huber` n_estimators |
|---|---|---|---|
| 2011 | 1 | 3 | **1** |
| 2012 | 1 | 3 | **1** |
| 2013 | 1 | 3 | **1** |
| 2014 | 1 | 3 | **1** |
| 2015 | 5 | 3 | 300 |

`max_features=3` is the grid's lower bound in all five years. `max_depth=1` is a forest of
decision stumps. And gradient boosting, offered up to 1,000 trees, selects **a single tree** in
four years out of five.

Two independent tree ensembles, tuned by independent validation splits, both land on the least
flexible configuration available. The correct reading is not that the grid is misplaced but
that at roughly 88 names per cross-section the signal is weak enough that **model capacity is
almost purely a liability**. This is consistent with `rf` and `enet_huber` tying (§2): a
depth-1 forest sampling 3 of 892 columns per split is close to an additive univariate screen,
so it is unsurprising that it matches a linear model.

It also retires a planned deviation. Capping the `n_estimators` grid at 500 to save compute
was under consideration; the data show 1,000 was never selected, so the cap would have cost
nothing — and would have been an unnecessary entry in the deviations list.

The earlier boundary observation is not withdrawn; both are real. Grid selection here is
driven by sample size, not by grid placement, and a single panel cannot tell you which.

---

## 8. What the reference run is not

1. The target is a monthly return reconstructed by shifting `mom1m`, with no CRSP delisting
   adjustment. Rows without a `t+1` observation are dropped (1.16% overall), which removes
   delisting months and biases realised returns **upward**. The 2015 rate of 8.83% is a
   truncation artefact of ending the sample at 2015-12, not a wave of delistings.
2. The risk-free rate is Goyal-Welch `Rfree`, not the Ibbotson one-month Treasury bill. Excess
   returns are therefore self-consistent within one data source but not the paper's series.
3. 298 permnos, roughly **88 names per monthly cross-section**, against the full CRSP universe.
4. Five test years (2011-2015, n = 5,521) against thirty in the paper.
5. 46 industry dummies, not 74. Industry categories are fixed from the pre-test sample to
   avoid schema look-ahead; four SIC2 codes (14, 15, 54, 64) appear only after 2011 and are
   dropped. 133 rows (0.50%) carry a missing `sic2`.
6. Neural network ensembles use 3 seeds, not the paper's 10.
7. Macro predictors are lagged one month relative to the row date, since a row dated T carries
   information known at the end of T-1. Merging on the same month would give eight macro
   series — and, through the interact design, all 846 interaction columns — sight of the month
   in which the target is realised.

The pipeline mechanics are validated. **No R² in this document is comparable to the paper's
0.40%,** and the reported ±0.05% magnitudes are, by the Diebold-Mariano tests in §2,
indistinguishable from zero for every model except `ols` and `pls`.

---

## 9. Gradient boosting: a 5x speedup with bit-identical output

`gbrt_huber` is the binding compute constraint at 892 columns and does not use a GPU. Two
changes, neither of which alters a single prediction:

**Grid evaluation via `staged_predict`.** The shipped implementation fits the model once per
`n_estimators` grid point: `1 + 10 + 50 + 100 + 300 + 500 + 1000 = 1,961` trees per
`(learning_rate, max_depth)` combination. But a boosted model with N trees contains every
shorter model as a prefix, so a single fit at 1,000 trees plus `staged_predict` recovers the
validation prediction at all seven cut points. Tree count falls to 1,000, a ratio of **1.96x**.
Verified against the original implementation: identical selected hyperparameters, identical
validation MSE to twelve decimal places, and a maximum holdout prediction difference of
**exactly 0.0**.

**Thread-parallel combinations.** scikit-learn's tree builder releases the GIL, so the four
`(learning_rate, max_depth)` combinations parallelise across threads without touching the
process-spawn path of §6.

Measured on test year 2011 (15,243 training rows, 892 columns):

| implementation | time |
|---|---|
| original | did not finish in 80 minutes |
| `staged_predict`, serial | 69.5 min |
| `staged_predict` + 4 threads | **26.6 min** |

Threads contribute 2.61x on top of 1.96x, for roughly **5x overall**, with predictions
unchanged. Five test years drop from an unmeasured multi-hour run to about 2.2 hours.

`HistGradientBoostingRegressor` would likely be faster still, but it does not offer the Huber
loss the paper specifies, so substituting it is a modelling deviation rather than an
optimisation and has not been made.

---

## 10. Operational notes

- The documented `--n-firms 1400` stress panel exceeds 4 GB of RAM at the RF stage and is
  killed **with no traceback** — the log simply stops.
- `checkpoint: resume` was exercised by five unplanned container restarts, one deliberate
  interrupt and one Windows forced reboot, and recovered correctly every time. Evaluation
  settings can be changed without invalidating prediction checkpoints; `top_bottom_n` was
  raised from a stale value in exactly this way.
- `top_bottom_n` defaults to 1,000, which silently exceeds an 88-name cross-section: the
  `all`, `top` and `bottom` rows of `r2_oos.csv` came back byte-identical because all three
  selected the entire sample. A size-split statistic that quietly degenerates to the full
  sample is worse than an absent one.

---

## 11. Open

The stress generator's docstring claims ~85% of signal variance is unrepresentable by the
920-column linear design, and therefore that trees and networks should win. **They do not**: on
the stress panel `enet_huber` (+2.287%) beat `rf` (+1.781%) and `nn1` (+1.023%), with a
Diebold-Mariano t of −2.41 (p = 0.016) against `nn1`.

Two causes, both real:

- the `1{r[retvol] > 0.5}` threshold term is largely linearly capturable — the indicator
  correlates with its own argument at roughly 0.75 — so the claimed variance split is
  overstated;
- RF searches for characteristic-by-characteristic interactions among 920 columns with
  `max_features=30`, giving any specific pair a low chance of meeting in one path.

Until the generator is corrected, **do not assert "nonlinear beats linear" in CI**. The
assertions that hold today are: all models beat zero on the planted signal, OLS is clearly
worst, and RF/NN beat PCR/PLS.

The reference run does not settle this either: `rf` and `enet_huber` are statistically tied
(§2). Distinguishing them needs a larger cross-section, which is the next planned run.
