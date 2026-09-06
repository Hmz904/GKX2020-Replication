"""
Build a GKX-style stock-month panel from PUBLIC data only.

Inputs
  datashare.csv      Xiu's characteristic datashare (permno, DATE, 94 chars, sic2)
  Data2025.xlsx      Goyal-Welch-Zafirov predictor data, sheet "Monthly"

Output
  subsample_panel.parquet

IMPORTANT DEVIATION FROM GU-KELLY-XIU (2020)
  datashare.csv contains no realized returns. The target here is reconstructed by
  shifting the `mom1m` characteristic forward one month per permno. Per the datashare
  README and FINDINGS.md section 3, a row dated T carries information known at the end
  of month T-1 and its response is the return realized DURING month T; mom1m(T) is the
  month T-1 return, so target(T) = mom1m(T+1). This is NOT the CRSP return series used
  in the paper:
    - no delisting-return adjustment
    - the final month of every permno is dropped (no t+1 observation), which
      removes delisting months and biases realized returns UPWARD
    - risk-free rate is Goyal-Welch `Rfree`, not the Ibbotson 1-month T-bill
  Any R^2 produced from this panel is NOT comparable to the paper's 0.40%.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATASHARE = Path(r"D:\Google Download\GKX2020\datashare.csv")
MACRO_XLSX = Path(r"D:\hms\Data2025.xlsx")
OUT_PATH = Path(r"D:\projects\gkx\data\processed\subsample_panel.parquet")
CONFIG_PATH = Path(r"D:\projects\gkx\configs\subsample.yaml")
PANEL_REL = "data/processed/subsample_panel.parquet"

START_YYYYMM = 199501
END_YYYYMM = 201512
N_PERMNO = 300
SEED = 20260905
CHUNKSIZE = 500_000

ID_COLS = ["permno", "DATE", "sic2"]


def shift_month_forward(yyyymm):
    """Return the yyyymm of the following calendar month."""
    y, m = yyyymm // 100, yyyymm % 100
    return np.where(m == 12, (y + 1) * 100 + 1, yyyymm + 1)


MACRO_SRC = ["d/p", "e/p", "b/m", "ntis", "tbl", "tms", "dfy", "svar"]
MACRO_OUT = ["macro_dp", "macro_ep", "macro_bm", "macro_ntis",
             "macro_tbl", "macro_tms", "macro_dfy", "macro_svar"]

# ----------------------------------------------------------------------------
# Pass 1: read only the id columns to build the permno universe
# ----------------------------------------------------------------------------
print("pass 1: scanning permno universe ...", flush=True)
ids = pd.read_csv(DATASHARE, usecols=["permno", "DATE"],
                  dtype={"permno": "int32", "DATE": "int32"})
ids["yyyymm"] = ids["DATE"] // 100
ids = ids[(ids["yyyymm"] >= START_YYYYMM) & (ids["yyyymm"] <= END_YYYYMM)]

universe = np.sort(ids["permno"].unique())
print(f"  rows in window: {len(ids):,}")
print(f"  permnos in window: {len(universe):,}")

# Random sample over the WHOLE universe. Deliberately NOT filtered on history
# length: selecting long-lived permnos would inject survivorship bias.
rng = np.random.default_rng(SEED)
if len(universe) <= N_PERMNO:
    keep = set(universe.tolist())
else:
    keep = set(rng.choice(universe, size=N_PERMNO, replace=False).tolist())
print(f"  sampled permnos: {len(keep)}")

months_per_permno = ids[ids["permno"].isin(keep)].groupby("permno").size()
print(f"  months per sampled permno: min={months_per_permno.min()} "
      f"median={int(months_per_permno.median())} max={months_per_permno.max()}")
del ids

# ----------------------------------------------------------------------------
# Pass 2: stream the full file, keeping only the sampled permnos
# ----------------------------------------------------------------------------
print("pass 2: reading characteristics ...", flush=True)
parts = []
for i, chunk in enumerate(pd.read_csv(DATASHARE, chunksize=CHUNKSIZE, low_memory=False)):
    chunk = chunk[chunk["permno"].isin(keep)]
    if len(chunk):
        yyyymm = chunk["DATE"] // 100
        chunk = chunk[(yyyymm >= START_YYYYMM) & (yyyymm <= END_YYYYMM)]
        if len(chunk):
            parts.append(chunk)
    if (i + 1) % 5 == 0:
        print(f"  chunk {i + 1} ...", flush=True)

panel = pd.concat(parts, ignore_index=True)
del parts
print(f"  panel rows: {len(panel):,}")

char_cols = [c for c in panel.columns if c not in ID_COLS]
print(f"  characteristic columns: {len(char_cols)}")
assert len(char_cols) == 94, f"expected 94 characteristics, found {len(char_cols)}"
assert "mom1m" in char_cols, "mom1m missing; cannot reconstruct returns"
assert "mvel1" in char_cols, "mvel1 missing; cannot build mve"

panel["yyyymm"] = panel["DATE"] // 100
panel = panel.sort_values(["permno", "yyyymm"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# Reconstruct the return target
# ----------------------------------------------------------------------------
# mom1m at month t is the realized return during month t. The row dated t must
# predict the return of month t+1, so the target is mom1m shifted BACK by one
# position within each permno (i.e. next month's value lands on this row).
print("reconstructing returns ...", flush=True)
panel["ret"] = panel.groupby("permno", sort=False)["mom1m"].shift(-1)

# Only accept the shifted value when the next row really is the next calendar
# month. Gaps in a permno's history would otherwise silently splice together
# returns from months that are years apart.
next_month = panel.groupby("permno", sort=False)["yyyymm"].shift(-1)
expected = shift_month_forward(panel["yyyymm"].to_numpy())
panel.loc[next_month != expected, "ret"] = np.nan

n_before = len(panel)
dropped = panel["ret"].isna()
print(f"  rows dropped for missing t+1: {dropped.sum():,} "
      f"({100 * dropped.mean():.2f}%)")
by_year = panel.assign(_y=panel["yyyymm"] // 100, _d=dropped).groupby("_y")["_d"].mean()
print("  drop rate by year:")
for y, r in by_year.items():
    print(f"    {y}: {100 * r:5.2f}%")
panel = panel[~dropped].reset_index(drop=True)
print(f"  rows kept: {len(panel):,} of {n_before:,}")

panel["mve"] = panel["mvel1"]

# ----------------------------------------------------------------------------
# Macro predictors
# ----------------------------------------------------------------------------
print("merging macro predictors ...", flush=True)
macro = pd.read_excel(MACRO_XLSX, sheet_name="Monthly")
# Read one month before the window: after the one-month lag below, the first
# panel month needs the macro observation that precedes it.
macro_start = int(START_YYYYMM - 89) if START_YYYYMM % 100 == 1 else int(START_YYYYMM - 1)
macro = macro[(macro["yyyymm"] >= macro_start) & (macro["yyyymm"] <= END_YYYYMM)].copy()

# GKX define dp and ep in logs; the spreadsheet ships them as plain ratios.
macro["macro_dp"] = np.log(macro["d12"]) - np.log(macro["price"])
macro["macro_ep"] = np.log(macro["e12"]) - np.log(macro["price"])
macro["macro_bm"] = macro["b/m"]
macro["macro_ntis"] = macro["ntis"]
macro["macro_tbl"] = macro["tbl"]
macro["macro_tms"] = macro["tms"]
macro["macro_dfy"] = macro["dfy"]
macro["macro_svar"] = macro["svar"]

market = macro[["yyyymm", "ret", "Rfree"]].rename(columns={"ret": "mkt_ret"})
macro = macro[["yyyymm"] + MACRO_OUT]
assert macro.notna().all().all(), "macro block contains NaN over the sample window"
assert len(macro) == len(macro["yyyymm"].unique()), "duplicate macro months"

# A row dated T carries information known at the end of month T-1, so the macro
# block must be lagged one month. Merging on the same yyyymm would let eight macro
# series -- and, under interact mode, all 846 interaction columns -- see the month
# in which the target return is realized.
macro = macro.sort_values("yyyymm").reset_index(drop=True)
macro["yyyymm"] = shift_month_forward(macro["yyyymm"].to_numpy())

panel = panel.merge(macro, on="yyyymm", how="left", validate="many_to_one")
panel = panel.merge(market, on="yyyymm", how="left", validate="many_to_one")
assert panel[MACRO_OUT].notna().all().all(), "macro merge left gaps"

# Rfree is already a monthly decimal rate in this spreadsheet, not a percentage.
assert market["Rfree"].abs().max() < 0.05, "Rfree looks like percent, not decimal"
panel["ret_excess"] = panel["ret"] - panel["Rfree"]

# ----------------------------------------------------------------------------
# Leakage check: does the reconstructed cross-section track the market?
# ----------------------------------------------------------------------------
# `ret` on a row dated T is the return realized DURING month T, so the equal-weighted
# cross-section must line up with the month-T market return at LAG 0.
#   peak at +1  ->  shift(0) was used: the target is the month T-1 return, contemporaneous
#                   with same-month volatility characteristics. This is the look-ahead
#                   failure described in FINDINGS.md section 3 and it inflates R^2.
#   peak at -1  ->  shift(-2) was used: a month of information is thrown away.
print("checking shift direction against the market return ...", flush=True)
mkt = market.set_index("yyyymm")["mkt_ret"].sort_index().rename("mkt")
ew = panel.groupby("yyyymm")["ret"].mean().rename("ew")
chk = pd.concat([ew, mkt], axis=1).dropna()
corrs = {lag: chk["ew"].corr(chk["mkt"].shift(lag)) for lag in (-2, -1, 0, 1, 2)}
for lag, c in corrs.items():
    print(f"    corr(ew_ret, mkt_ret shifted {lag:+d}) = {c:+.4f}")
best = max(corrs, key=lambda k: corrs[k])
assert best == 0, (
    f"shift direction looks wrong: correlation peaks at lag {best:+d}, not 0. "
    "A peak at +1 means shift(0) was used and same-month volatility characteristics "
    "predict their own month's return. Do not use this panel."
)
assert corrs[0] > 0.5, f"lag-0 correlation is only {corrs[0]:.3f}; check the target"

# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
# The pipeline config addresses months through `date_col: date`.
panel["date"] = pd.to_datetime(panel["yyyymm"].astype(str) + "01", format="%Y%m%d")

out = panel[["permno", "date", "yyyymm", "sic2", "ret", "ret_excess", "mve"]
            + MACRO_OUT + char_cols].copy()
for c in out.columns:
    if c not in ("permno", "date", "yyyymm", "sic2"):
        out[c] = out[c].astype("float32")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)

print()
print(f"wrote {OUT_PATH}")
print(f"  rows      : {len(out):,}")
print(f"  permnos   : {out['permno'].nunique()}")
print(f"  months    : {out['yyyymm'].nunique()}")
print(f"  sic2 codes: {out['sic2'].nunique()}")
print(f"  columns   : {len(out.columns)}")
print()
print("SET THESE IN YOUR CONFIG:")
print(f"  expected_num_industries: {out['sic2'].nunique()}")
print(f"  expected_num_features: {94 * 9 + out['sic2'].nunique()}")


# ----------------------------------------------------------------------------
# Emit a matching run config
# ----------------------------------------------------------------------------
# The 94 characteristic names and the realized industry count are only known once
# the panel exists, and the pipeline refuses to infer either. Writing the config
# here keeps the two in lockstep.
n_ind = int(out["sic2"].nunique())
char_yaml = "\n".join(f"    - {c}" for c in char_cols)

config = f"""# Generated by scripts/build_public_panel.py -- do not hand-edit the feature block.
# Public-data subsample: {N_PERMNO} permnos, {START_YYYYMM} to {END_YYYYMM}.
# The target is reconstructed from mom1m and is NOT the CRSP return used by GKX.
run_name: gkx_subsample
seed: {SEED}
tuning_n_jobs: 4

checkpoint:
  resume: true

data:
  path: {PANEL_REL}
  date_col: date
  id_col: permno
  target_col: ret_excess
  market_cap_col: mve
  return_col: ret
  min_valid_market_cap_fraction: 0.01
  file_glob: "**/*"
  downcast_float32: true

features:
  mode: interact
  characteristic_columns:
{char_yaml}
  macro_columns: [{", ".join(MACRO_OUT)}]
  industry_code_col: sic2
  industry_values: null
  expected_num_characteristics: 94
  expected_num_macros: 8
  expected_num_industries: {n_ind}
  expected_num_features: {94 * 9 + n_ind}
  fill_missing_monthly_median: true
  drop_missing_industry: true

split:
  train_start: 1995-01-01
  initial_train_end: 2005-12-31
  initial_validation_start: 2006-01-01
  initial_validation_end: 2010-12-31
  test_start: 2011-01-01
  test_end: 2015-12-31
  refit_frequency: yearly

models:
  names: [ols, ols3, enet_huber, pcr, pls, glm, rf, gbrt_huber, nn1, nn2, nn3, nn4, nn5]
  # eta0 lowered from the 1e-3 default: on real characteristic distributions that
  # step size diverges at small sklearn_alpha (FINDINGS.md section 1).
  ols_huber_quantile: 0.999
  ols_eta0: 0.0001
  ols_max_iter: 5000
  ols3_features: [mvel1, bm, mom12m]
  ols3_huber_quantile: 0.999
  ols3_eta0: 0.0001
  ols3_max_iter: 5000
  enet_huber:
    sklearn_alpha: [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    paper_rho: 0.5
    huber_quantile: 0.999
    eta0: 0.0001
    max_iter: 5000
  pcr:
    n_components: [5, 10, 20, 30, 40, 50]
  pls:
    n_components: [1, 2, 3, 5, 10]
  glm:
    n_knots: 3
    transform_batch_size: 4096
    streaming_epochs: 5
    sklearn_alpha: [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    paper_rho: 0.5
    huber_quantile: 0.999
    eta0: 0.0001
  rf:
    n_estimators: 300
    max_depth: [1, 2, 3, 4, 5, 6]
    max_features: [3, 5, 10, 20, 30, 50]
    min_samples_leaf: 1
    forest_n_jobs: -1
  gbrt_huber:
    n_estimators: [1, 10, 50, 100, 300, 500, 1000]
    learning_rate: [0.01, 0.1]
    max_depth: [1, 2]
    min_samples_leaf: 1
    huber_quantile: 0.999
  neural_net:
    epochs: 100
    batch_size: 10000
    inference_batch_size: 65536
    learning_rate: [0.001, 0.01]
    l1_lambda: [0.00001, 0.0001, 0.001]
    patience: 5
    output_init_scale: 0.01
    # Cut from the paper's 10 seeds to bound runtime on one machine.
    seeds: [1, 2, 3]
    device: auto

evaluation:
  top_bottom_n: 1000
  annualization: 12
  portfolio_bins: 10
  newey_west_lags: 6
  save_predictions: true
"""

CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.write_text(config, encoding="utf-8")
print(f"wrote {CONFIG_PATH}")
