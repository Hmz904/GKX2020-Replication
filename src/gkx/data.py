from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    industry_values: tuple[object, ...] = ()


def _required_input_columns(data_cfg: dict, feat_cfg: dict) -> list[str]:
    cols = {
        data_cfg["date_col"],
        data_cfg["id_col"],
        data_cfg["target_col"],
        data_cfg["market_cap_col"],
    }
    if data_cfg.get("return_col"):
        cols.add(data_cfg["return_col"])

    mode = feat_cfg.get("mode", "prebuilt")
    if mode == "prebuilt":
        features = feat_cfg.get("columns")
        if not features:
            raise ValueError(
                "features.columns is required in prebuilt mode; implicit numeric-column "
                "discovery is disabled to prevent target/metadata leakage."
            )
        cols.update(features)
    elif mode == "interact":
        chars = feat_cfg.get("characteristic_columns")
        macros = feat_cfg.get("macro_columns")
        industry = feat_cfg.get("industry_code_col")
        if not chars or not macros or not industry:
            raise ValueError(
                "interact mode requires characteristic_columns, macro_columns, and "
                "industry_code_col explicitly."
            )
        cols.update(chars)
        cols.update(macros)
        cols.add(industry)
    else:
        raise ValueError(f"Unknown features.mode={mode!r}")
    return sorted(cols)


def _read_one(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    columns = list(columns) if columns is not None else None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".feather":
        return pd.read_feather(path, columns=columns)
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
        return df if columns is None else df.loc[:, columns]
    if path.name.lower().endswith(".csv.gz") or suffix == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported file: {path}")


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        text = s.astype("Int64").astype(str)
        text = text.where(text.str.len() != 6, text + "01")
        return pd.to_datetime(text, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def load_panel(data_cfg: dict, feat_cfg: dict) -> pd.DataFrame:
    path = Path(data_cfg["path"])
    columns = _required_input_columns(data_cfg, feat_cfg)
    if path.is_file():
        df = _read_one(path, columns)
    else:
        supported = {".parquet", ".feather", ".pkl", ".pickle", ".csv", ".gz"}
        files = [
            p
            for p in path.glob(data_cfg.get("file_glob", "**/*"))
            if p.is_file() and p.suffix.lower() in supported
        ]
        if not files:
            raise FileNotFoundError(f"No supported data files under {path}")
        df = pd.concat((_read_one(p, columns) for p in sorted(files)), ignore_index=True)

    dc = data_cfg["date_col"]
    df[dc] = _parse_dates(df[dc])
    required_nonmissing = [dc, data_cfg["id_col"], data_cfg["target_col"]]
    if feat_cfg.get("mode") == "interact" and feat_cfg.get("drop_missing_industry", True):
        required_nonmissing.append(feat_cfg["industry_code_col"])
    df = df.dropna(subset=required_nonmissing).sort_values([dc, data_cfg["id_col"]])
    if data_cfg.get("downcast_float32", True):
        df = _downcast_float_columns(df)
    return df.reset_index(drop=True)


def _rank_one_monthly(
    values: pd.Series,
    dates: pd.Series,
    fill_monthly_median: bool,
) -> np.ndarray:
    ranks = values.groupby(dates, sort=False).rank(method="average", na_option="keep")
    counts = values.notna().groupby(dates, sort=False).transform("sum").astype(float)
    scaled = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    scaled = scaled.where(counts > 1, 0.0)
    if fill_monthly_median:
        med = scaled.groupby(dates, sort=False).transform("median")
        scaled = scaled.fillna(med).fillna(0.0)
    return scaled.to_numpy(dtype=np.float32, copy=False)


def rank_characteristics(
    df: pd.DataFrame,
    date_col: str,
    characteristic_columns: list[str],
    fill_monthly_median: bool = True,
) -> np.ndarray:
    """GKX monthly cross-sectional rank transform mapped to [-1, 1]."""
    out = np.empty((len(df), len(characteristic_columns)), dtype=np.float32)
    dates = df[date_col]
    for j, col in enumerate(characteristic_columns):
        out[:, j] = _rank_one_monthly(df[col], dates, fill_monthly_median)
    return out


def rank_characteristics_inplace(
    df: pd.DataFrame,
    date_col: str,
    characteristic_columns: list[str],
    fill_monthly_median: bool = True,
) -> None:
    """Rank each characteristic once for the full panel, month by month, in place."""
    dates = df[date_col]
    for col in characteristic_columns:
        df[col] = _rank_one_monthly(df[col], dates, fill_monthly_median)


def resolve_feature_schema(
    df: pd.DataFrame,
    feat_cfg: dict,
    *,
    date_col: str | None = None,
    industry_cutoff: str | pd.Timestamp | None = None,
) -> FeatureSchema:
    mode = feat_cfg.get("mode", "prebuilt")
    if mode == "prebuilt":
        names = tuple(feat_cfg["columns"])
        expected = feat_cfg.get("expected_num_features")
        if expected is not None and len(names) != int(expected):
            raise ValueError(f"Expected {expected} features, configured {len(names)}")
        return FeatureSchema(names)

    chars = list(feat_cfg["characteristic_columns"])
    macros = list(feat_cfg["macro_columns"])
    industry_col = feat_cfg["industry_code_col"]
    expected_chars = feat_cfg.get("expected_num_characteristics")
    expected_macros = feat_cfg.get("expected_num_macros")
    if expected_chars is not None and len(chars) != int(expected_chars):
        raise ValueError(f"Expected {expected_chars} characteristics, configured {len(chars)}")
    if expected_macros is not None and len(macros) != int(expected_macros):
        raise ValueError(f"Expected {expected_macros} macro predictors, configured {len(macros)}")

    configured_values = feat_cfg.get("industry_values")
    if configured_values:
        industry_values = tuple(configured_values)
    else:
        source = df
        if industry_cutoff is not None:
            if date_col is None:
                raise ValueError("date_col is required when industry_cutoff is supplied")
            source = df.loc[df[date_col] <= pd.Timestamp(industry_cutoff)]
        industry_values = tuple(sorted(source[industry_col].dropna().unique().tolist()))

    expected_industries = feat_cfg.get("expected_num_industries")
    if expected_industries is not None and len(industry_values) != int(expected_industries):
        scope = "pre-test sample" if industry_cutoff is not None else "configured sample"
        raise ValueError(
            f"Expected {expected_industries} SIC2 categories, found {len(industry_values)} in "
            f"the {scope}. Supply features.industry_values explicitly if the intended 74-category "
            "schema is known."
        )

    names: list[str] = []
    for char in chars:
        names.append(char)
        names.extend(f"{char}__x__{macro}" for macro in macros)
    names.extend(f"{industry_col}__{value}" for value in industry_values)
    expected = feat_cfg.get("expected_num_features")
    if expected is not None and len(names) != int(expected):
        raise ValueError(
            f"GKX design has {len(names)} features, expected {expected}. "
            f"Check the 94 characteristics, 8 macro variables, and 74 SIC2 categories."
        )
    return FeatureSchema(tuple(names), industry_values)


def build_design_matrix(
    df: pd.DataFrame,
    feat_cfg: dict,
    date_col: str,
    schema: FeatureSchema,
    *,
    characteristics_ranked: bool = False,
) -> pd.DataFrame:
    mode = feat_cfg.get("mode", "prebuilt")
    if mode == "prebuilt":
        x = df.loc[:, list(schema.feature_names)]
        if x.isna().any().any():
            raise ValueError(
                "Prebuilt design contains missing values. Supply a fully preprocessed design or "
                "use interact mode for GKX monthly rank/median preprocessing."
            )
        # Avoid pandas' deprecated astype(copy=...) keyword; the returned float32 frame is explicit.
        return x.astype("float32")

    chars = list(feat_cfg["characteristic_columns"])
    macros = list(feat_cfg["macro_columns"])
    industry_col = feat_cfg["industry_code_col"]
    if characteristics_ranked:
        ranked = df.loc[:, chars].to_numpy(dtype=np.float32, copy=False)
    else:
        ranked = rank_characteristics(
            df,
            date_col,
            chars,
            fill_monthly_median=feat_cfg.get("fill_missing_monthly_median", True),
        )
    macro = df.loc[:, macros].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(macro).all():
        raise ValueError("Macro predictors contain missing or non-finite values")

    n = len(df)
    p = len(schema.feature_names)
    arr = np.empty((n, p), dtype=np.float32)
    pos = 0
    for j, _char in enumerate(chars):
        base = ranked[:, j]
        arr[:, pos] = base
        pos += 1
        for k, _macro in enumerate(macros):
            arr[:, pos] = base * macro[:, k]
            pos += 1

    industry = df[industry_col].to_numpy()
    for value in schema.industry_values:
        arr[:, pos] = (industry == value).astype(np.float32)
        pos += 1

    if pos != p:
        raise RuntimeError(f"Feature construction wrote {pos} columns, expected {p}")
    return pd.DataFrame(arr, index=df.index, columns=schema.feature_names)


def validate_panel(
    df: pd.DataFrame,
    data_cfg: dict,
    feat_cfg: dict,
    schema: FeatureSchema,
) -> None:
    required = [
        data_cfg["date_col"],
        data_cfg["id_col"],
        data_cfg["target_col"],
        data_cfg["market_cap_col"],
    ]
    if data_cfg.get("return_col"):
        required.append(data_cfg["return_col"])
    if feat_cfg.get("mode") == "prebuilt":
        required.extend(schema.feature_names)
    else:
        required.extend(feat_cfg["characteristic_columns"])
        required.extend(feat_cfg["macro_columns"])
        required.append(feat_cfg["industry_code_col"])
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:20]}")

    reserved = {data_cfg["date_col"], data_cfg["id_col"], data_cfg["target_col"]}
    if data_cfg.get("return_col"):
        reserved.add(data_cfg["return_col"])
    if feat_cfg.get("mode") == "prebuilt":
        configured_features = set(schema.feature_names)
    else:
        configured_features = set(feat_cfg["characteristic_columns"]) | set(
            feat_cfg["macro_columns"]
        )
        reserved.add(feat_cfg["industry_code_col"])
    leakage = sorted(configured_features & reserved)
    if leakage:
        raise ValueError(f"Configured predictive features include reserved/leaky columns: {leakage}")

    if df.duplicated([data_cfg["date_col"], data_cfg["id_col"]]).any():
        raise ValueError("Duplicate security-month rows found")

    target = pd.to_numeric(df[data_cfg["target_col"]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("target_col must be finite after loading")

    return_col = data_cfg.get("return_col")
    if return_col:
        total_return = pd.to_numeric(df[return_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(total_return).all():
            raise ValueError("return_col must be finite after loading when configured")

    market_cap = pd.to_numeric(
        df[data_cfg["market_cap_col"]], errors="coerce"
    ).to_numpy(dtype=float)
    valid_cap = np.isfinite(market_cap) & (market_cap > 0)
    valid_fraction = float(valid_cap.mean()) if len(valid_cap) else 0.0
    min_fraction = float(data_cfg.get("min_valid_market_cap_fraction", 0.01))
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_valid_market_cap_fraction must be in [0, 1]")
    if valid_fraction < min_fraction:
        raise ValueError(
            f"market_cap_col has only {valid_fraction:.2%} finite positive observations, below "
            f"the configured minimum {min_fraction:.2%}. Check market_cap_col mapping or data."
        )
    invalid_count = int((~valid_cap).sum())
    if invalid_count:
        warnings.warn(
            f"market_cap_col has {invalid_count} missing/nonpositive rows "
            f"({1.0 - valid_fraction:.2%}); they remain eligible for prediction but are excluded "
            "where size ranking or value weighting requires a valid market cap.",
            UserWarning,
            stacklevel=2,
        )
