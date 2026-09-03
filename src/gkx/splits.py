from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    refit_year: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def recursive_windows(cfg: dict):
    if cfg.get("refit_frequency", "yearly") != "yearly":
        raise NotImplementedError("Only yearly refits are implemented")

    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])
    train_end0 = pd.Timestamp(cfg["initial_train_end"])
    val_start0 = pd.Timestamp(cfg["initial_validation_start"])
    val_end0 = pd.Timestamp(cfg["initial_validation_end"])
    if not (train_end0 < val_start0 <= val_end0 < test_start):
        raise ValueError("Initial train/validation/test dates are not temporally ordered")

    for year in range(test_start.year, test_end.year + 1):
        offset = year - test_start.year
        shift = pd.DateOffset(years=offset)
        test_s = max(pd.Timestamp(year, 1, 1), test_start)
        test_e = min(pd.Timestamp(year, 12, 31), test_end)
        yield Window(
            refit_year=year,
            train_end=train_end0 + shift,
            val_start=val_start0 + shift,
            val_end=val_end0 + shift,
            test_start=test_s,
            test_end=test_e,
        )
