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

Reference run, all 13 models, test years 2011-2015 (n = 5,521). **This ranking is not stable
— see §12**, which reruns the same panel over thirteen test years and reorders almost all of it:

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
is **not supported**.

**Corrected by §12.4.** An independent test on 14,653 observations gives t = −0.89, p = 0.372 —
still insignificant, but with the two models now 1.66 percentage points apart rather than 0.008.
Read together, the two tests show **insufficient power, not equivalence**. The word "tie" was
wrong: p = 0.957 measures the test's inability to separate them, not their similarity. The size split points the right way — `rf` earns +0.419% on the
largest 30 names against −0.238% on the smallest, and `enet_huber` +0.287% against −0.076% —
which is the direction GKX report, but 1,770 observations cannot carry that claim.

§12.3 revisits this with 2.65x the data: on 2003-2015 excluding 2008, `rf` earns **+1.046 %** on
the largest 30 and **+0.837 %** on the smallest, both positive, direction unchanged. That is a
stronger version of the same finding, subject to the ex-post exclusion documented there.

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

## 12. Thirteen test years: what more data changed, and what it did not

The reference run tests on 2011-2015 (n = 5,521), and §2 concludes that almost nothing in it is
statistically distinguishable. The obvious remedy is more test observations. Widening the
cross-section was measured and rejected (§12.4); lengthening the test window was run instead.

### 12.1 The run

`configs/subsample_long.yaml`, run name `gkx_subsample_long`. The same 298-name panel, with
the rolling window moved back as a block:

| | reference run | long run |
|---|---|---|
| train | 1995-2005 | 1995-2000 |
| validation | 2006-2010 (5 years) | 2001-2002 (2 years) |
| test | 2011-2015 | 2003-2015 |
| test observations | 5,521 | **14,653** |
| wall clock | — | 8 h 22 m |

**This changes two things at once**, and that has to be said before any result is read: the test
window lengthened *and* the validation window shrank from five years to two. Starting the test in
2001 instead was rejected — it would leave the first refit with roughly three years of training
data against 892 columns, and §2's dose-response result says unregularised fits at that width and
that sample size produce noise, which would lower the power the run exists to raise.

Full-sample OOS R², both runs:

| model | 2011-2015 (n=5,521) | 2003-2015 (n=14,653) |
|---|---:|---:|
| nn5 | −0.397 % | **+0.264 %** |
| glm | −0.262 % | **+0.254 %** |
| ols3 | −0.055 % | +0.015 % |
| nn2 | −0.290 % | −0.086 % |
| nn3 | −0.248 % | −0.118 % |
| nn4 | −0.466 % | −0.166 % |
| enet_huber | **+0.050 %** | −0.022 % |
| pcr | −0.140 % | −0.279 % |
| gbrt_huber | −0.131 % | −0.399 % |
| pls | −0.641 % | −0.581 % |
| nn1 | −0.185 % | −0.997 % |
| rf | **+0.042 %** | **−1.679 %** |
| ols | −24.96 % | −44.07 % |

Almost no ordering survives. That is the first result: **the reference run's ranking was not
stable.** The rest of this section separates why.

### 12.2 Validation length, isolated

Both runs predict the same 2011-2015 observations. Restricting the long run to those months gives
a controlled comparison in which the test data, the target and the features are identical and
**only the validation window differs**:

| model | 5-year validation | 2-year validation | delta |
|---|---:|---:|---:|
| glm | −0.262 % | **+0.301 %** | +0.563 |
| enet_huber | +0.050 % | **+0.282 %** | +0.232 |
| ols3 | −0.055 % | +0.130 % | +0.185 |
| pcr | −0.140 % | −0.041 % | +0.099 |
| pls | −0.641 % | −0.400 % | +0.241 |
| ols | −24.96 % | −12.62 % | +12.34 |
| nn4 | −0.466 % | −0.511 % | −0.045 |
| nn3 | −0.248 % | −0.561 % | −0.313 |
| nn5 | −0.397 % | −0.611 % | −0.214 |
| nn2 | −0.290 % | −0.786 % | −0.496 |
| **rf** | **+0.042 %** | **−0.063 %** | −0.105 |
| **gbrt_huber** | −0.131 % | **−0.794 %** | −0.663 |
| nn1 | −0.185 % | −1.074 % | −0.889 |

n = 5,521 in both columns, confirmed identical.

The sign of the change **sorts by model family**. Every linear and regularised-linear model
improves; both tree models and the networks get worse. Two mechanisms move in opposite
directions here and the split is consistent with both acting: shortening validation hands three
extra years (2008-2010, the crisis included) to training, which helps models that want data, and
it removes three years from the sample used to pick hyperparameters, which hurts models whose
selection is already unstable — §7 shows both tree models pinned to the minimum of their grids.

Whatever the mechanism, the operational conclusion stands on its own: **the length of the
hyperparameter-selection window is a first-order design choice, not a detail**, and its effect is
large enough to reverse the sign of several models' R². GKX fix this window by convention; this
repository now has a measured reason to report it as a deviation whenever it is changed.

### 12.3 The pooled R² is a crisis-year statistic

`rf` falls from +0.042 % to −1.679 %, the worst move in the table. Year by year it is not a
decline at all:

| year | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rf | +1.68 | +1.05 | −0.21 | −0.14 | −2.47 | **−17.89** | +2.56 | +1.65 | +0.43 | −0.23 | +1.65 | −0.71 | −1.17 |
| enet_huber | +0.51 | +1.03 | −0.25 | −0.10 | −2.75 | −3.36 | +1.41 | +1.50 | +0.03 | +0.51 | +2.31 | −0.33 | −0.68 |
| glm | +5.53 | −1.44 | +0.14 | −0.28 | −1.57 | −5.13 | +1.50 | +1.73 | −0.08 | +0.49 | +3.37 | −1.05 | −0.67 |
| ols | −43.62 | −52.09 | −35.14 | −48.77 | −77.18 | **−148.91** | −18.09 | −8.96 | −19.93 | −15.94 | −9.69 | −13.54 | −4.79 |

2008 contributes **13.45 %** of the denominator (the largest year is 2009 at 20.24 %), and it
carries the entire negative result:

| `rf` sample | OOS R² |
|---|---:|
| 2003-2015, all | −1.679 % |
| **2003-2015 excluding 2008** | **+0.841 %** |
| — excluding 2008, largest 30 by market cap | **+1.046 %** |
| — excluding 2008, smallest 30 | **+0.837 %** |

Excluding one year of thirteen moves `rf` from twelfth place to first, and its size split turns
positive on both halves while keeping the direction GKX report. **Dropping 2008 is an ex-post
choice and the full-sample number remains the headline**; the point is not that `rf` is really
the best model, it is that a single pooled R² over a window containing a crisis compresses two
strong and opposite signals — best-in-class in twelve normal years, catastrophic in one — into a
number that describes neither. The per-year table is not a robustness appendix here; it is the
result.

This also explains the low power against `rf` noted in §12.4: its loss series is dominated by
two tail years pointing opposite ways (2008 at −17.89 %, 2009 at +2.56 %), which inflates the
variance the Diebold-Mariano statistic divides by.

### 12.4 The comparison the extra data was supposed to settle, and did not

`enet_huber` vs `rf` was the motivating pair. Two independent tests:

| test sample | observations | t | p | R² gap |
|---|---:|---:|---:|---:|
| 2011-2015 | 5,521 | −0.054 | 0.957 | 0.008 pp |
| 2003-2015 | 14,653 | −0.89 | 0.372 | 1.66 pp |

Still not significant with 2.65x the data — but the point estimates are no longer close. §2 reads
the first row as "they tie"; with the second row in hand, the defensible reading of both is
**insufficient power**, not equivalence. §2 has been corrected accordingly.

What the extra observations did buy is a family of results invisible at n = 5,521: **network depth
now separates**. `nn1` is significantly worse than `nn2` (p = 0.0033), `nn4` (p = 0.034),
`nn5` (p = 0.0008) and `ols3` (p = 0.028), and `nn2` is worse than `nn5` (p = 0.0049). None
of these pairs was distinguishable in the reference run. Significant pairs rise from 15 to 19 of
78; `ols` remains worse than all twelve but its t-statistics fall from 5.85-6.82 to 2.70-3.05.

### 12.5 Compute cost is anisotropic

The two ways of adding rows do not cost the same. Doubling the cross-section was calibrated
directly (599 names, gbrt only, 2015 only, `configs/subsample_calib600.yaml`); lengthening the
series was measured from the long run's own per-year timestamps:

| direction | rows | time | exponent |
|---|---|---|---:|
| wider cross-section | 26,543 → 54,909 (2.07x) | 35.9 → 169.1 min (4.71x) | **2.13** |
| longer series | 9,120 → 12,932 training rows (1.42x) | 20.2 → 30.1 min (1.49x) | **1.14** |

Both are `gbrt_huber`, the model that accounts for roughly 76 % of a year's compute. Adding
months is close to linear; adding names is close to quadratic. The practical consequence is that
the originally planned 1000-name run was projected at 60-80 hours on this hardware and abandoned,
while the thirteen-year run cost 8 h 22 m — and the exponent measured in one direction must not be
used to budget the other, which is how the 1000-name plan survived as long as it did.

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

The reference run does not settle this either: `rf` and `enet_huber` are indistinguishable
in it (§2). The plan recorded here was to distinguish them with a larger cross-section. That plan
was measured and abandoned: a calibration run put a 1000-name panel at 60-80 hours on this
hardware (§12.5). Thirteen test years were run instead, at 2.65x the observations for 8 h 22 m,
and the pair is **still** not separated (t = −0.89, p = 0.372, §12.4) — though the two are no
longer close in point estimate, and `rf`'s loss series turns out to be dominated by two opposing
crisis-era years (§12.3), which is a concrete reason the test has so little power against it.
Separating them is therefore still open, and more test observations alone are not obviously the
route.

## The 2003–2015 reordering is a test-window result

`gkx_subsample_long` changed three things at once relative to the baseline — the train/validation
split, the training-window length, and the test window — so its dramatic reordering (nn5 and glm
to the top, rf collapsing from +0.042% to −1.679%) was not attributable to any of them.

Two control runs turn this into a chain in which each step moves exactly one:

| step | what changes | rf R² | Δ | share |
|---|---|---:|---:|---:|
| `gkx_subsample` | baseline, train 11y / valid 5y / test 2011–15 | +0.042% | — | — |
| → `gkx_subsample_valid2y` | train/validation boundary, pre-test span fixed at 16y | −0.063% | −0.106pp | 6.1% |
| → `gkx_subsample_span8y` | training window 14y → 6y, validation fixed at 2y | −0.257% | −0.194pp | 11.3% |
| → `gkx_subsample_long` | test window 2011–15 → 2003–15 | −1.679% | −1.422pp | **82.6%** |

**The test window accounts for roughly five sixths of the collapse.** `gkx_subsample_long` is a
statement about the 2003–2010 out-of-sample period, not about the tuning protocol. The steps
telescope, so the shares summing to 100% is arithmetic rather than a check, and they are
path-dependent: a different ordering of the same three edits would apportion them differently.

Step 1 also produces a clean split across all thirteen models. Everything that leans on the
validation set for early stopping or a non-degenerate grid got worse (nn1 −0.889pp, gbrt
−0.662pp, nn2 −0.495pp); everything else got better (glm +0.563pp, enet_huber +0.233pp, ols3
+0.185pp). No exceptions.

Only glm and enet_huber isolate the training-length channel cleanly — both selected identical
hyperparameters in all five test years of both runs, ten fits each with zero variation, so glm's
+0.965pp on `top_30` owes nothing to the validation window. gbrt flipped qualitatively
(n_estimators 1 → 300–500 in four of five years, lr 0.1 → 0.01): a shorter validation set pushed
it to train harder and overfit, not to be more cautious. rf's selections move across all three
runs, with only the first test year stable, so its −0.106pp mixes both channels and no step
isolates a pure training-length effect for it.

Full decomposition, per-year hyperparameter tables and caveats:
[docs/gkx_findings_valid2y.md](docs/gkx_findings_valid2y.md).
