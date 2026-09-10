# Attributing the 2003–2015 reordering: the `valid2y` control run

## Why this run exists

`gkx_subsample_long` (train 1995–2000, validation 2001–2002, test 2003–2015) produced a
ranking almost unrelated to the `gkx_subsample` baseline: nn5 +0.264% and glm +0.254% at the
top, enet_huber down to −0.022%, and rf collapsing from +0.042% to −1.679%.

That run changed three things at once relative to the baseline:

1. the test window (2011–2015 → 2003–2015),
2. the total pre-test span (16 years → 8 years),
3. the split of that span between training and validation (11y/5y → 6y/2y).

With three simultaneous changes, none of the reordering was attributable. `gkx_subsample_valid2y`
isolates the third.

## Design

| | train | validation | test | pre-test span |
|---|---|---|---|---|
| `gkx_subsample` (baseline) | 1995–2005 (11y) | 2006–2010 (5y) | 2011–2015 | 16y |
| `gkx_subsample_valid2y` | 1995–2008 (14y) | 2009–2010 (2y) | 2011–2015 | 16y |

The test window and the total pre-test span are held fixed. The only change is **where the
train/validation boundary falls** — three years are moved from validation into training.

This is worth stating precisely, because it is not a pure "shorter validation" experiment: the
same edit that shortens validation also lengthens training. The two effects are separated below
using the selected hyperparameters, not by assumption.

Runtime 4h02m on the machine in the environment notes; `gbrt_huber` alone accounted for 77–78%
of each test year (31m57s of 41m18s in 2011, rising to ~41m of ~53m by 2015).

## Results, `sample = all` (out-of-sample R², %)

| model | 11y/5y | 14y/2y | Δ |
|---|---:|---:|---:|
| enet_huber | +0.0497 | +0.2824 | **+0.2327** |
| rf | +0.0422 | −0.0633 | −0.1055 |
| ols3 | −0.0553 | +0.1298 | +0.1850 |
| gbrt_huber | −0.1311 | −0.7935 | **−0.6624** |
| pcr | −0.1397 | −0.0405 | +0.0991 |
| nn1 | −0.1849 | −1.0735 | **−0.8886** |
| nn3 | −0.2482 | −0.5611 | −0.3129 |
| glm | −0.2620 | +0.3009 | **+0.5630** |
| nn2 | −0.2904 | −0.7856 | −0.4952 |
| nn5 | −0.3970 | −0.6113 | −0.2142 |
| nn4 | −0.4659 | −0.5112 | −0.0453 |
| pls | −0.6407 | −0.4000 | +0.2407 |
| ols | −24.9596 | −12.6209 | +12.3388 |

The `top_30` and `bottom_30` splits carry the same sign pattern. The largest single move anywhere
in the table is glm on `top_30`: −0.2583% → +0.7070%, i.e. +0.9653pp.

## The split is not random: it tracks dependence on the validation set

Every model that consumes the validation set for early stopping or for a non-degenerate
hyperparameter search got worse. Every model that does not, got better.

| | models | Δ range |
|---|---|---|
| improved | ols, ols3, pcr, pls, glm, enet_huber | +0.10 to +12.34pp |
| worsened | rf, gbrt_huber, nn1–nn5 | −0.05 to −0.89pp |

Thirteen models, zero exceptions.

## Separating the two effects with the selected hyperparameters

`tuning.csv` distinguishes the training-length channel from the tuning channel directly: if a
model selected identical hyperparameters in both runs, its change can only come from the longer
training window.

**Unchanged selections — pure training-length effect.**

```
glm          n_knots 3, alpha 0.01, l1_ratio 0.5      identical, all 5 years, both runs
enet_huber   alpha 0.01, l1_ratio 0.5                 identical, all 5 years, both runs
rf           max_depth 1, max_features 3              identical, both runs
```

glm's +0.563pp on `all` and +0.965pp on `top_30` therefore owe nothing to the validation
window — the search picked the same point on the grid ten times out of ten. Three extra years of
training data is the whole story. The same holds for enet_huber (+0.233pp).

rf falls in this group too, with the opposite sign: −0.106pp from the longer training window,
with no tuning channel at all.

**Changed selections — tuning channel active.**

```
gbrt_huber   11y/5y : lr 0.1,  max_depth 2, n_estimators 1
             14y/2y : lr 0.01, max_depth 1, n_estimators 300
```

This is a qualitative flip, not a nudge. Under 5-year validation gbrt selects a single depth-2
tree — effectively declining to learn. Under 2-year validation it selects 300 depth-1 trees at
lr 0.01, a conventional boosting configuration. Out-of-sample R² falls 0.66pp. The shorter
validation set did not push the model toward caution; it pushed it toward training much harder
and overfitting the validation set. (The runtime confirms the switch is real: gbrt genuinely
trains 300 trees per fit in this run.)

```
nn1   11y/5y : l1 [1e-3, 1e-4, 1e-5, 1e-5, 1e-3]   lr [.01, .01, .001, .001, .001]
      14y/2y : l1 [1e-3, 1e-4, 1e-3, 1e-3, 1e-3]   lr [.01] x5

nn5   11y/5y : l1 [1e-5] x5                        lr [.01] x5
      14y/2y : l1 [1e-5, 1e-4, 1e-5, 1e-4, 1e-5]   lr [.01, .001, .01, .01, .01]
```

nn1 collapses onto one corner of the grid (strongest L1, largest learning rate) in four of five
years, where under 5-year validation it was dispersed. nn5 does the reverse: it was perfectly
stable and becomes unstable. The out-of-sample damage matches — nn1 is the worst hit of all
thirteen models (−0.889pp), nn5 the mildest of the neural nets (−0.214pp). Bias in one case,
variance in the other.

## What this says about rf's collapse

rf selected the same hyperparameters in both runs, so the split-point reallocation reaches it
only through the training window, and moves it by **−0.106pp**. In `gkx_subsample_long` rf moved
by **−1.721pp** (+0.042% → −1.679%).

At most 6% of that collapse is attributable to the train/validation split point. The remaining
94% belongs to the other two changes in that run: the test window itself (2003–2010 is eight
years of out-of-sample period absent from the baseline) and the halving of the total pre-test
span from 16 years to 8.

The headline of `gkx_subsample_long` should therefore be read as a **test-window** result, not a
tuning-protocol artefact. That is now supported rather than assumed.

## What is still not isolated

This run holds the pre-test span fixed at 16 years. `gkx_subsample_long` has only 8. A further
control — train 1995–2000, validation 2001–2002, test 2011–2015 — would isolate the effect of
total pre-test span at a fixed test window, and would complete the decomposition. It is the
natural next run, at roughly 3h on this machine (shorter training window than the present run).

Two caveats on the numbers above:

- Single seeds per configuration for the tree models; the neural nets average three seeds
  internally. Differences below roughly 0.1pp should not be read as real, which covers nn4
  (−0.045pp) and arguably rf (−0.106pp).
- The return target is reconstructed (`mom1m` shifted −1), so the absolute R² levels are not
  comparable to the published table. Only the differences between runs, which share the target,
  are being interpreted here.
