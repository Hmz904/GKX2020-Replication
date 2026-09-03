from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def read_sample(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, nrows=5, low_memory=False)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/inspect_data.py <file-or-directory>")
    root = Path(sys.argv[1])
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        try:
            sample = read_sample(path)
            print(
                f"\n{path} shape={sample.shape}\n"
                f"columns={list(sample.columns)}\n{sample.head(2)}"
            )
        except Exception as exc:  # inspection utility should continue across heterogeneous files
            print(f"skip {path}: {exc}")


if __name__ == "__main__":
    main()
