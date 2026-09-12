"""
Build a GKX-style stock-month panel from PUBLIC data only.

Inputs
  datashare.csv      Xiu's characteristic datashare (permno, DATE, 94 chars, sic2)
  Data2025.xlsx      Goyal-Welch-Zafirov predictor data, sheet "Monthly"

Outputs
  <--out>            the stock-month panel (100 columns, pre-Kronecker)
  <--config>         a matching GKX2020 run config, unless --no-config
  <--artifact-out>   the CA artifact + manifest, when --emit-artifact is passed

THIS PANEL IS PRE-KRONECKER
  The 892/905-column characteristic x macro expansion is NOT written here. It is
  formed at load time by the GKX2020 pipeline when features.mode == interact. The
  file on disk always holds the 94 raw characteristics, so the CA artifact is a
  column subset of this panel, never a reconstruction of one.

SELECTION RULE (see --min-months)
  Two rules are available and they are NOT interchangeable.

  random (default)   Draw --n-permno permnos uniformly from the whole in-window
                     universe, with no condition on how long each one survives.
                     This is the survivorship-neutral rule: a permno delisted in
                     1998 is as likely to be drawn as one alive throughout. The
                     cost is width -- an unrestricted draw of 300 names yields
                     only ~88 per monthly cross-section, because the draws do not
                     overlap in time.

  --min-months N     Keep only permnos observed in at least N months of the
                     window. This buys width (>=189 months gives 3,032 permnos and
                     a median cross-section of ~3,012) by conditioning on survival,
                     which is exactly the bias the random rule avoids. Anything
                     built this way carries a survivorship deviation and must be
                     documented as such; it is defensible for factor-structure work
                     (CA/IPCA loadings, regularization behaviour) and NOT for
                     return predictability claims.

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

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"D:\projects\gkx")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--datashare", default=r"D:\Google Download\GKX2020\datashare.csv")
parser.add_argument("--macro-xlsx", default=r"D:\hms\Data2025.xlsx")
parser.add_argument("--out", default=str(REPO_ROOT / r"data\processed\subsample_panel.parquet"))
parser.add_argument("--panel-rel", default=None,
                    help="path the config uses to address the panel; derived from --out by default")
parser.add_argument("--config", default=str(REPO_ROOT / r"configs\subsample.yaml"))
parser.add_argument("--no-config", action="store_true",
                    help="skip the run config; use for panels too wide for the GKX2020 pipeline")
parser.add_argument("--run-name", default="gkx_subsample")
parser.add_argument("--n-permno", type=int, default=None,
                    help="how many permnos to draw; defaults to 300 when --min-months is absent")
parser.add_argument("--min-months", type=int, default=None,
                    help="keep only permnos observed at least this many months (survivorship-conditioned)")
parser.add_argument("--seed", type=int, default=20260905)
parser.add_argument("--start", type=int, default=199501)
parser.add_argument("--end", type=int, default=201512)
parser.add_argument("--chunksize", type=int, default=500_000)
parser.add_argument("--top-bottom-n", type=int, default=None,
                    help="long/short leg size for evaluation; defaults to 10%% of the median cross-section")
parser.add_argument("--emit-artifact", action="store_true",
                    help="also write the CA artifact via the authoritative branch path")
parser.add_argument("--artifact-out",
                    default=r"D:\projects\Autoencoder\data\processed\gkx_ca_panel_branch.parquet")
parser.add_argument("--artifact-manifest",
                    default=r"D:\projects\Autoencoder\data\processed\gkx_ca_panel_branch.manifest.json")
parser.add_argument("--exporter", default=r"D:\projects\Autoencoder\tools\gkx-side\export_ca_panel.py")
parser.add_argument("--source-version", default=None,
                    help="recorded in the artifact manifest; defaults to this repo's short git SHA")
args = parser.parse_args()

DATASHARE = Path(args.datashare)
MACRO_XLSX = Path(args.macro_xlsx)
OUT_PATH = Path(args.out)
CONFIG_PATH = Path(args.config)
START_YYYYMM = args.start
END_YYYYMM = args.end
CHUNKSIZE = args.chunksize
SEED = args.seed

# Neither flag given => the historical default: 300 names, no history condition.
N_PERMNO = args.n_permno
MIN_MONTHS = args.min_months
if N_PERMNO is None and MIN_MONTHS is None:
    N_PERMNO = 300

if args.panel_rel is not None:
    PANEL_REL = args.panel_rel
else:
    try:
        PANEL_REL = OUT_PATH.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        PANEL_REL = OUT_PATH.as_posix()

ID_COLS = ["permno", "DATE", "sic2"]

MACRO_SRC = ["d/p", "e/p", "b/m", "ntis", "tbl", "tms", "dfy", "svar"]
MACRO_OUT = ["macro_dp", "macro_ep", "macro_bm", "macro_ntis",
             "macro_tbl", "macro_tms", "macro_dfy", "macro_svar"]


def shift_month_forward(yyyymm):
    """Return the yyyymm of the following calendar month."""
    y, m = yyyymm // 100, yyyymm % 100
    return np.where(m == 12, (y + 1) * 100 + 1, yyyymm + 1)


# Load the CA exporter BEFORE the 3.6GB read, so a bad --exporter path fails in
# seconds rather than after the whole build.
exporter = None
if args.emit_artifact:
    spec = importlib.util.spec_from_file_location("export_ca_panel", args.exporter)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load exporter module from {args.exporter}")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    print(f"loaded CA exporter from {args.exporter}", flush=True)

source_version = args.source_version
if source_version is None:
    try:
        source_version = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        source_version = "unknown"

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

months_all = ids.groupby("permno")["yyyymm"].nunique()

if MIN_MONTHS is None:
    eligible = universe
    selection_rule = f"random draw of {N_PERMNO} from the full universe, no history condition"
else:
    eligible = np.sort(months_all.index[months_all >= MIN_MONTHS].to_numpy())
    selection_rule = f"permnos observed in >= {MIN_MONTHS} of {ids['yyyymm'].nunique()} months"
    print(f"  permnos with >= {MIN_MONTHS} months: {len(eligible):,}")

rng = np.random.default_rng(SEED)
if N_PERMNO is None or len(eligible) <= N_PERMNO:
    keep = set(eligible.tolist())
else:
    keep = set(rng.choice(eligible, size=N_PERMNO, replace=False).tolist())
    selection_rule += f", then a random draw of {N_PERMNO}"
print(f"  selection rule: {selection_rule}")
print(f"  selected permnos: {len(keep):,}")

months_per_permno = months_all.loc[sorted(keep)]
print(f"  months per selected permno: min={months_per_permno.min()} "
      f"median={int(months_per_permno.median())} max={months_per_permno.max()}")
del ids

# ----------------------------------------------------------------------------
# Pass 2: stream the full file, keeping only the selected permnos
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

# Order matters: char_cols must stay in datashare column order, because the CA
# artifact manifest records this list verbatim and `export_ca_panel.py verify`
# compares the two lists for equality.
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

# mvel1 in this datashare is market equity as a LEVEL in $thousands (median ~2.1e5,
# min ~8.9, max ~5.1e8), not log size, despite what the datashare README says. It is
# therefore usable directly as the value-weight column. Do NOT exponentiate: the CA
# exporters only check that the weight column is positive and is not one of the 94
# ranked names, so a wrong transform here passes every downstream guard silently.
me = panel["mvel1"].astype("float64")
assert me.notna().any(), "mvel1 is entirely missing"
assert (me.dropna() > 0).all(), "mvel1 must be a strictly positive level to weight by"
assert me.median() > 1e3, (
    f"mvel1 median is {me.median():.4g}; expected a market-equity LEVEL in $thousands. "
    "If this column is log size, weighting by it is meaningless -- fix before exporting."
)
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

monthly_n = out.groupby("yyyymm")["permno"].nunique()
n_ind = int(out["sic2"].nunique())
n_permno_actual = int(out["permno"].nunique())

print()
print(f"wrote {OUT_PATH}")
print(f"  rows        : {len(out):,}")
print(f"  permnos     : {n_permno_actual:,}")
print(f"  months      : {out['yyyymm'].nunique()}")
print(f"  sic2 codes  : {n_ind}")
print(f"  columns     : {len(out.columns)}")
print(f"  monthly N   : min={monthly_n.min()} median={int(monthly_n.median())} max={monthly_n.max()}")
print(f"  N/P (P=95)  : min={monthly_n.min() / 95:.1f} median={monthly_n.median() / 95:.1f}")
print()
print("SET THESE IN YOUR CONFIG:")
print(f"  expected_num_industries: {n_ind}")
print(f"  expected_num_features: {94 * 9 + n_ind}")

# ----------------------------------------------------------------------------
# Emit the CA artifact (branch path)
# ----------------------------------------------------------------------------
# This is the authoritative path described in export_ca_panel.py: hand the frame
# to the exporter while the raw characteristics are still present, rather than
# recovering them afterwards from an expanded panel. The column set and order
# match what scripts/export_gkx_ca_artifact.py produced, so the two artifacts can
# be compared with `export_ca_panel.py verify`.
if args.emit_artifact:
    print()
    print("emitting CA artifact (branch path) ...", flush=True)
    _, manifest = exporter.emit_from_prekronecker(
        out,
        char_cols,
        args.artifact_out,
        args.artifact_manifest,
        target_provenance="reconstructed_from_mom1m_shift_-1",
        source_version=source_version,
        id_col="permno",
        month_col="yyyymm",
        return_col="ret_excess",
        value_weight_col="mve",
        value_weight_provenance="lagged market equity level from GKX build",
        industry_col="sic2",
        extra_columns=("ret",),
        source_panel=str(OUT_PATH),
        build_path="branch",
    )
    print(f"  rows / months / assets : {manifest['rows']:,} / {manifest['months']}"
          f" / {manifest['assets']:,}")
    print(f"  monthly N min/med/max  : {manifest['monthly_n_min']}"
          f" / {manifest['monthly_n_median']} / {manifest['monthly_n_max']}")
    print(f"  N/P min / median       : {manifest['n_over_p_min']:.2f}"
          f" / {manifest['n_over_p_median']:.2f}")
    print(f"  noise ceiling (max)    : {manifest['noise_projection_ceiling_max']:.3f}")
    if manifest["n_over_p_min"] < 20:
        print("  WARNING: N/P below 20 -- total R2 is diagnostic only on this panel")
    print(f"  {args.artifact_out}")
    print(f"  {args.artifact_manifest}")

# ----------------------------------------------------------------------------
# Emit a matching run config
# ----------------------------------------------------------------------------
# The 94 characteristic names and the realized industry count are only known once
# the panel exists, and the pipeline refuses to infer either. Writing the config
# here keeps the two in lockstep.
if args.no_config:
    print()
    print("skipped the run config (--no-config)")
else:
    # The long/short legs have to scale with the cross-section: 30 was chosen when
    # a month held ~88 names. On a panel of a few thousand, fixed legs of 30 would
    # be a different experiment, not the same one at a new width.
    top_bottom = args.top_bottom_n
    if top_bottom is None:
        top_bottom = max(30, int(round(0.10 * monthly_n.median())))
    if 2 * top_bottom > monthly_n.min():
        print(f"  WARNING: top_bottom_n={top_bottom} exceeds half the smallest "
              f"cross-section ({monthly_n.min()}); legs will overlap in thin months")

    char_yaml = "\n".join(f"    - {c}" for c in char_cols)

    config = f"""# Generated by scripts/build_public_panel.py -- do not hand-edit the feature block.
# Public-data subsample: {n_permno_actual} permnos, {START_YYYYMM} to {END_YYYYMM}.
# Selection rule: {selection_rule}.
# Monthly cross-section: min {monthly_n.min()}, median {int(monthly_n.median())}, max {monthly_n.max()}.
# The target is reconstructed from mom1m and is NOT the CRSP return used by GKX.
run_name: {args.run_name}
seed: {SEED}
# Windows: joblib silently deadlocks when tuning_n_jobs > 1 (9h, no traceback). Keep at 1.
tuning_n_jobs: 1

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
  # Legs sized at ~10% of the median cross-section ({int(monthly_n.median())} names).
  top_bottom_n: {top_bottom}
  annualization: 12
  portfolio_bins: 10
  newey_west_lags: 6
  save_predictions: true
"""

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config, encoding="utf-8")
    print()
    print(f"wrote {CONFIG_PATH}")
