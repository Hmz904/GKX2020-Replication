# Findings

Results and defects found by actually executing this pipeline. Everything below is measured,
not inferred.

---

## 1. Synthetic stress testing gave a linear model a false clean bill of health

`enet_huber` scored **+2.287% OOS R², the best of all models**, on the 920-predictor synthetic
stress panel. On real datashare characteristics, the same code with the same protocol produced:

```
enet_huber   -2,147,246,851,270.789 %
```

That is numerical divergence, not poor performance: predictions grow until they overwhelm the
denominator of the R² statistic. The likely cause is the fixed-step `SGDRegressor`
(`eta0=1e-3`) diverging at small `sklearn_alpha` on real characteristic distributions. The
synthetic generator draws AR(1) latents with far better conditioning, which hid the failure
completely.

**This is the central methodological result of this repository.** Replication infrastructure
that passes its own synthetic tests can still be broken, and broken specifically in ways only
real data exposes. A synthetic stress test proves the pipeline can recover known structure; it
does not certify any individual estimator.

Required fix: a divergence guard on `enet_huber` — reject a grid point whose validation MSE
exceeds the target variance by some factor — asserted in tests.

## 2. First real-data result (degraded design)

500 permnos, 2000-2015, 41,212 firm-months, 152-column design, test period 2012-2015
(n = 9,877).

| model | OOS R² |
|---|---|
| rf | **+0.858%** |
| pls | +0.214% |
| pcr | +0.195% |
| nn1 | +0.186% |
| nn2 | +0.144% |
| ols3 | +0.130% |
| ols | −2.767% |
| enet_huber | diverged (see §1) |

The ordering reproduces GKX's qualitative claims: trees beat linear models; unregularized OLS
on the full predictor set is **negative**; the three-predictor OLS-3 benchmark is positive.

NN1/NN2 are **not comparable** here — their grid was cut to one learning rate, one L1 value,
two seeds and 50 epochs to fit inside a single-core budget, against the paper's 10-seed
ensembles.

### What this number is not

1. The target is a raw monthly return reconstructed from `mom1m`, with no CRSP delisting
   adjustment.
2. No risk-free rate was available, so it is a **total** return, not an excess return.
3. The design is 152 columns; GKX use 920. All 846 characteristic-by-macro interaction
   columns are absent.
4. The monthly cross-section is ~213 names, not the full CRSP universe.

It validates pipeline mechanics. It is not comparable to the paper's 0.40%.

## 3. Target construction: the lag convention, pinned two independent ways

The datashare README states that for `DATE=19570329` the response variable is the return in
`195703` — the **same** month, because the authors have already lagged the characteristics.

Measured independently from the data: `retvol(t)` correlates with `|mom1m(t)|` at **0.573**,
against 0.443 and 0.414 for neighbouring months; `maxret(t)` correlates with `mom1m(t)` at
**+0.201**, against −0.129 and −0.025. The daily-frequency characteristics and `mom1m` are
measured over the same month, and `mom1m(t)` is the month `t-1` return.

Both routes agree: **`target(t) = mom1m(t+1)`, a shift of −1.**

A shift of −2 loses a month of information. A shift of 0 makes same-month volatility
characteristics (`retvol`, `maxret`, `baspread`, `std_dolvol`) predict their own month's
return — severe look-ahead that produces an *attractively high* R². This is the failure mode
most likely to be accepted as a good result, and it must be locked by an assertion.

### One check that looked like validation and was not

Comparing `mom6m(t)` against the sum of `mom1m(t-0..t-5)` gives correlation 0.88, versus 0.70
for the shifted alternative. This **cannot identify the absolute lag**: if `mom1m` and `mom6m`
are lagged by the same amount, their relative alignment is unchanged and the correlation is
identical. It establishes only that the two columns share a convention. A check that appears
to confirm a hypothesis without being able to discriminate it is worse than no check.

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

## 5. Rank transform overwrites the market cap column

`rank_characteristics_inplace` mutates the panel. When `mvel1` serves as both a predictive
characteristic and `market_cap_col`, value weighting silently reads rank values in [-1, 1]
instead of dollars. A separate raw copy (`mvel1_raw`) is mandatory before ranking. This is why
`configs/stress.yaml` carries `mvel1_raw`, and it needs an assertion, not a comment.

## 6. Operational notes

- The documented `--n-firms 1400` stress panel exceeds 4 GB of RAM at the RF stage and is
  killed **with no traceback** — the log simply stops. Worth stating in the config comment.
- `gbrt_huber` is the binding compute constraint at 920 columns and does not use a GPU.
  `HistGradientBoostingRegressor` is the obvious substitute.
- `checkpoint: resume` was exercised by five unplanned container restarts across two runs and
  recovered correctly every time.

## 7. Open

The stress generator's docstring claims ~85% of signal variance is unrepresentable by the
920-column linear design, and therefore that trees and networks should win. **They do not**: on
the stress panel `enet_huber` (+2.287%) beat `rf` (+1.781%) and `nn1` (+1.023%), with a
Diebold-Mariano t of −2.41 (p = 0.016) against `nn1`.

Two causes, both real:

- the `1{r[retvol] > 0.5}` threshold term is largely linearly capturable — the indicator
  correlates with its own argument at roughly 0.75 — so the claimed variance split is
  overstated;
- RF searches for characteristic-by-characteristic interactions among 920 columns with
  `max_features=30`, giving any specific pair a low chance of meeting in one path. Every
  selected RF configuration sat at the grid boundary (`max_depth=6`), which is under-tuning,
  not a fair comparison.

Until the generator is corrected, **do not assert "nonlinear beats linear" in CI**. The
assertions that hold today are: all models beat zero on the planted signal, OLS is clearly
worst, and RF/NN beat PCR/PLS.
