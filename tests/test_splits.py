import pytest

from gkx.splits import recursive_windows


def split_config():
    return {
        "test_start": "1987-01-01",
        "test_end": "1988-12-31",
        "initial_train_end": "1974-12-31",
        "initial_validation_start": "1975-01-01",
        "initial_validation_end": "1986-12-31",
        "refit_frequency": "yearly",
    }


def test_recursive_windows_use_all_initial_dates():
    windows = list(recursive_windows(split_config()))
    assert windows[0].train_end.year == 1974
    assert windows[0].val_start.year == 1975
    assert windows[0].val_end.year == 1986
    assert windows[1].train_end.year == 1975
    assert windows[1].val_start.year == 1976
    assert windows[1].val_end.year == 1987


def test_non_yearly_refit_is_not_silent():
    cfg = split_config()
    cfg["refit_frequency"] = "monthly"
    with pytest.raises(NotImplementedError):
        list(recursive_windows(cfg))
