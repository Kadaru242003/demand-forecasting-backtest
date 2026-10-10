import numpy as np
import pandas as pd

from demand_forecast.backtest import Fold
from demand_forecast.intervals import conformal_intervals


def _setup(n_series=400, n_folds=4, h=2, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-04", periods=n_folds * h + 1, freq="7D")
    folds = [Fold(k, weeks[k * h], tuple(weeks[k * h + 1 : k * h + 1 + h])) for k in range(n_folds)]
    rows = []
    for f in folds:
        for s in range(n_series):
            for hh in range(1, h + 1):
                scale = 1 + s % 5
                rows.append((f.fold, f"s{s}", hh, 50.0, 50 + scale * rng.normal(), scale))
    df = pd.DataFrame(rows, columns=["fold", "series_id", "horizon", "yhat", "y", "scale"])
    df["model"] = "m"
    scales = df[["fold", "series_id", "scale"]].drop_duplicates()
    return df.drop(columns="scale"), scales, folds


def test_first_fold_has_no_interval_and_later_folds_cover_nominal():
    fc, scales, folds = _setup()
    out = conformal_intervals(fc, scales, folds, levels=(0.8,))
    assert out.loc[out["fold"] == 0, "lo_80"].isna().all()
    later = out[out["fold"] > 0]
    cov = ((later["y"] >= later["lo_80"]) & (later["y"] <= later["hi_80"])).mean()
    assert 0.76 < cov < 0.84


def test_calibration_never_uses_current_or_later_folds():
    fc, scales, folds = _setup()
    base = conformal_intervals(fc, scales, folds, levels=(0.8,))
    changed = fc.copy()
    changed.loc[changed["fold"] >= 2, "y"] += 1000.0  # perturb fold 2 and later
    new = conformal_intervals(changed, scales, folds, levels=(0.8,))
    early = base["fold"] <= 2
    np.testing.assert_allclose(base.loc[early, "hi_80"], new.loc[early, "hi_80"])
    assert not np.allclose(base.loc[base["fold"] == 3, "hi_80"], new.loc[new["fold"] == 3, "hi_80"])
