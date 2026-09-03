from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(path, n_ids=250, start="2000-01-31", periods=120, p=20, seed=42):
    if p < 5:
        raise ValueError("Synthetic generator requires p >= 5.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="ME")
    rows = []
    macro = rng.normal(size=periods)
    for t, date in enumerate(dates):
        x = rng.normal(size=(n_ids, p))
        signal = (
            0.01 * x[:, 0]
            - 0.008 * x[:, 1]
            + 0.012 * np.maximum(x[:, 2], 0) * x[:, 3]
            + 0.005 * macro[t] * x[:, 4]
        )
        y = signal + rng.standard_t(4, size=n_ids) * 0.06
        size_driver = x[:, 5] if p > 5 else x[:, 0]
        mve = np.exp(8 + size_driver + rng.normal(scale=0.3, size=n_ids))
        z = pd.DataFrame(x, columns=[f"x{i}" for i in range(p)])
        z.insert(0, "mve", mve)
        z.insert(0, "ret_total", y + 0.002)
        z.insert(0, "ret_excess", y)
        z.insert(0, "id", np.arange(n_ids))
        z.insert(0, "date", date)
        rows.append(z)
    df = pd.concat(rows, ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
