# Attributing the 2003–2015 reordering: a three-run decomposition

## Why these runs exist

`gkx_subsample_long` (train 1995–2000, validation 2001–2002, test 2003–2015) produced a ranking
almost unrelated to the `gkx_subsample` baseline: nn5 +0.264% and glm +0.254% at the top,
enet_huber down to −0.022%, and rf collapsing from +0.042% to −1.679%.

That run changed three things at once relative to the baseline:

1. the split of the pre-test span between training and validation (11y/5y → 6y/2y),
2. the length of the training window (11y → 6y, and with it the pre-test span, 16y → 8y),
3. the test window (2011–2015 → 2003–2015).

With three simultaneous changes, none of the reordering was attributable. Two control runs turn
this into a chain in which each step moves exactly one of them.

## Design

| run | train | validation | test | pre-test span |
|---|---|---|---|---|
| `gkx_subsample` (baseline) | 1995–2005 (11y) | 2006–2010 (5y) | 2011–2015 | 16y |
| `gkx_subsample_valid2y` | 1995–2008 (14y) | 2009–2010 (2y) | 2011–2015 | 16y |
| `gkx_subsample_span8y` | 1995–2000 (6y) | 2001–2002 (2y) | 2011–2015 | 8y |
| `gkx_subsample_long` | 1995–2000 (6y) | 2001–2002 (2y) | 2003–2015 | 8y |

Each consecutive pair differs in one respect:

- **baseline → valid2y**: the train/validation boundary moves; the pre-test span is fixed at 16
  years. Three years are reallocated from validation into training, so this step simultaneously
  lengthens training and shortens validation. The two effects are separated below using the
  selected hyperparameters rather than by assumption.
- **valid2y → span8y**: the training window shortens from 14 to 6 years with validation held at
  2 years. Test window unchanged.
- **span8y → long**: the test window extends from 2011–2015 to 2003–2015. Training and validation
  windows are identical.

## Out-of-sample R² (%), `sample = all`

| model | baseline 11y/5y | valid2y 14y/2y | span8y 6y/2y |
|---|---:|---:|---:|
| enet_huber | +0.0497 | +0.2824 | −0.1653 |
| rf | +0.0422 | −0.0633 | −0.2574 |
| ols3 | −0.0553 | +0.1298 | −0.2817 |
| gbrt_huber | −0.1311 | −0.7935 | −0.5493 |
| pcr | −0.1397 | −0.0405 | −0.5178 |
| nn1 | −0.1849 | −1.0735 | −0.9421 |
| nn3 | −0.2482 | −0.5611 | **+0.5371** |
| glm | −0.2620 | +0.3009 | −1.8021 |
| nn2 | −0.2904 | −0.7856 | **+0.3666** |
| nn5 | −0.3970 | −0.6113 | **+0.2567** |
| nn4 | −0.4659 | −0.5112 | **+0.5282** |
| pls | −0.6407 | −0.4000 | −1.5419 |
| ols | −24.9596 | −12.6209 | −96.2817 |

`gkx_subsample_long` is not a column here because it is scored on a different, 2.65x larger test
sample (n_obs 14,653 against 5,521), so its values are not comparable row-by-row against the
other three. Its reported figures are quoted individually where used.

## Step 1: reallocating three years from validation to training

Thirteen models split cleanly by whether they lean on the validation set.

| | models | Δ range |
|---|---|---|
| improved | ols, ols3, pcr, pls, glm, enet_huber | +0.099 to +12.339pp |
| worsened | rf, gbrt_huber, nn1–nn5 | −0.045 to −0.889pp |

No exceptions. The largest single move anywhere in the run is glm on `top_30`: −0.2583% →
+0.7070%, i.e. +0.965pp. The `top_30` and `bottom_30` splits carry the same sign pattern as
`all`.

**glm and enet_huber isolate the training-length channel exactly.** Both selected identical
hyperparameters in all five test years of both runs:

```
glm          n_knots 3, alpha 0.01, l1_ratio 0.5     identical, 5/5 years, both runs
enet_huber   alpha 0.01, l1_ratio 0.5                identical, 5/5 years, both runs
```

Ten selections each, zero variation. glm's +0.563pp on `all` and +0.965pp on `top_30` therefore
owe nothing to the validation window; three extra years of training data is the whole story. The
same holds for enet_huber's +0.233pp.

**gbrt_huber's selections change qualitatively.** Per test year 2011–2015:

```
baseline 11y/5y   n_estimators   1,   1,   1,   1, 300    depth 2, 1, 2, 1, 1   lr .1, .01, .1, .1, .01
valid2y  14y/2y   n_estimators   1, 500, 300, 300, 300    depth 1, 1, 1, 1, 1   lr .1, .01, .01, .01, .01
```

Under 5-year validation gbrt selects a single tree in four of five years — effectively declining
to learn. Under 2-year validation it selects 300–500 trees at lr 0.01 in four of five years, a
conventional boosting configuration. Out-of-sample R² falls 0.662pp. The shorter validation set
did not push the model toward caution; it pushed it toward training much harder. The runtime
confirms the switch is real: gbrt took 77–78% of each test year in the valid2y run (31m57s of
41m18s in 2011, rising to roughly 41m of 53m by 2015), for a total run time of 4h02m.

**nn1 concentrates on one corner of its grid.**

```
baseline 11y/5y   l1  1e-3, 1e-4, 1e-5, 1e-5, 1e-3    lr .01, .01, .001, .001, .001
valid2y  14y/2y   l1  1e-3, 1e-4, 1e-3, 1e-3, 1e-3    lr .01 x5
```

Four of five years at the strongest L1 and the largest learning rate, where the baseline was
dispersed across three L1 values and two learning rates. nn1 is the worst-hit model in this step
at −0.889pp.

**rf's selections also move**, so its −0.106pp mixes both channels rather than isolating the
training window:

```
baseline 11y/5y   max_depth 1, 1, 1, 1, 5    max_features 3,  3,  3,  3,  3
valid2y  14y/2y   max_depth 1, 4, 4, 5, 6    max_features 3, 20, 50,  3,  3
span8y   6y/2y    max_depth 1, 3, 6, 1, 1    max_features 3,  3,  3, 30, 50
```

Only the first test year is stable across all three runs. rf's grid selection is sensitive to both
the validation window and the training window, which is worth recording in its own right: the
earlier observation that rf "selects minimum grid complexity" describes the baseline's first four
years and does not generalize across configurations.

## Step 2: shortening the training window from 14 to 6 years

With validation fixed at two years, cutting training from 14 years to 6 moves rf a further
−0.194pp (−0.063% → −0.257%) and glm −2.103pp (+0.301% → −1.802%). ols falls to −96.28%, which
confirms directly that six years of training against 892 features is too thin for an unregularized
linear model — the reason this configuration was rejected as a starting point when the 2003–2015
run was designed.

**Four of the five neural nets turn positive and occupy the top four places of the table**: nn3
+0.537%, nn4 +0.528%, nn2 +0.367%, nn5 +0.257%, against −0.248%, −0.466%, −0.290% and −0.397% in
the baseline. nn1 does not join them (−0.942%).

Diebold–Mariano tests give the neural nets nominal support: nn2–nn5 beat glm, nn1 and pls at
p < 0.05, and nn3/nn4 beat rf at p < 0.05, with signs matching the R² ordering. The support does
not survive correction for multiple comparisons — 78 pairs are tested, the Bonferroni threshold is
6.4e-4, and only the ols-involving pairs clear it. nn2–nn5 are also not significantly better than
the middle of the table (enet_huber, ols3, gbrt_huber). The reading is that the four nets are
distinguishable from the *worst* models in this configuration, not that they are good.

## Step 3: extending the test window

Holding training and validation at 1995–2000 and 2001–2002, extending the test window from
2011–2015 to 2003–2015 takes rf from −0.257% to −1.679%, a further −1.422pp.

## The decomposition

For rf on `sample = all`:

| step | what changes | rf R² | Δ | share of total |
|---|---|---:|---:|---:|
| baseline | — | +0.042% | — | — |
| → valid2y | train/validation boundary, span fixed | −0.063% | −0.106pp | 6.1% |
| → span8y | training window 14y → 6y | −0.257% | −0.194pp | 11.3% |
| → long | test window 2011–15 → 2003–15 | −1.679% | −1.422pp | **82.6%** |
| total | | | −1.721pp | 100% |

**The test window accounts for roughly five sixths of rf's collapse.** The headline of
`gkx_subsample_long` is a statement about the 2003–2010 out-of-sample period, not about the tuning
protocol.

Two limits on how far this table can be pushed:

- The steps telescope, so the shares summing to 100% is arithmetic, not a validation of the
  decomposition.
- The shares are path-dependent. A different ordering of the same three changes would attribute
  different amounts to each, because the effects need not be additive. The ordering here was
  chosen so that each step is a single, nameable edit; it is not the only defensible one.

## Caveats

- Tree models use a single seed per configuration; the neural nets average three seeds internally.
  Differences below roughly 0.1pp should not be read as real, which covers nn4's −0.045pp in step
  1 and arguably rf's −0.106pp.
- The return target is reconstructed (`mom1m` shifted −1), so the absolute R² levels are not
  comparable to the published table. Only differences between runs, which share the target, are
  interpreted here.
- Steps 1 and 3 each isolate a single edit. In step 2 the training window and the pre-test span
  change together, since validation is held fixed — in this design they are the same edit and
  cannot be separated without a further run.
- `r2_oos.csv` is keyed by `(model, sample)` with `sample` in `{all, top_30, bottom_30}`;
  comparisons across runs must merge on both keys. Merging on `model` alone silently aligns rows
  by position across the three samples.
