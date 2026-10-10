"""Prediction intervals by scaled split-conformal calibration on earlier backtest folds.

For each model and horizon we collect the scaled residuals
``s = (y - yhat) / scale_series`` from *earlier* folds whose test windows ended at or
before the current forecast origin, and set

    lo = yhat + q_{alpha/2}(s) * scale,   hi = yhat + q_{1 - alpha/2}(s) * scale.

Scaling by each series' naive MAE makes the interval width proportional to how volatile
the series has been; using separate lower/upper quantiles allows asymmetric intervals
for skewed count data. Calibration data always precede the origin, so no future
information is used. The guarantee is only *marginal* (averaged over series and weeks).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .backtest import Fold


def _quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample corrected empirical quantile."""
    n = len(scores)
    k = np.clip(np.ceil((n + 1) * level) / n, 0, 1)
    return float(np.quantile(scores, k, method="higher"))


def conformal_intervals(
    forecasts: pd.DataFrame,
    scales: pd.DataFrame,
    folds: Sequence[Fold],
    levels: Sequence[float] = (0.8, 0.95),
) -> pd.DataFrame:
    """Add ``lo_XX`` / ``hi_XX`` columns for each level; rows without calibration data get NaN.

    ``forecasts`` needs fold, model, series_id, horizon, y, yhat; ``scales`` needs
    fold, series_id, scale.
    """
    df = forecasts.merge(scales, on=["fold", "series_id"], how="left")
    df["scale"] = df["scale"].fillna(1.0)
    df["score"] = (df["y"] - df["yhat"]) / df["scale"]
    by_id = {f.fold: f for f in folds}
    for level in levels:
        tag = f"{round(level * 100)}"
        df[f"lo_{tag}"] = np.nan
        df[f"hi_{tag}"] = np.nan
    for fold in folds:
        calib_folds = [f.fold for f in by_id.values() if f.test_weeks[-1] <= fold.train_end]
        if not calib_folds:
            continue
        cur = df["fold"] == fold.fold
        calib = df[df["fold"].isin(calib_folds)]
        for (model, h), cal in calib.groupby(["model", "horizon"]):
            rows = cur & (df["model"] == model) & (df["horizon"] == h)
            s = cal["score"].to_numpy()
            for level in levels:
                alpha = 1 - level
                q_lo = -_quantile(-s, 1 - alpha / 2)  # lower alpha/2 quantile
                q_hi = _quantile(s, 1 - alpha / 2)
                tag = f"{round(level * 100)}"
                base, sc = df.loc[rows, "yhat"], df.loc[rows, "scale"]
                df.loc[rows, f"lo_{tag}"] = np.clip(base + q_lo * sc, 0, None)
                df.loc[rows, f"hi_{tag}"] = base + q_hi * sc
    return df.drop(columns=["score"])
