"""Point-forecast and interval metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def naive_scale(history: pd.DataFrame) -> pd.Series:
    """Per-series MASE denominator: in-sample MAE of the one-step naive forecast.

    This is Hyndman & Koehler's (2006) non-seasonal scale, computed on the training
    window of each fold only. Series with a zero scale (constant history) get NaN and
    are excluded from MASE.
    """
    hist = history.sort_values(["series_id", "week"])
    diffs = hist.groupby("series_id", sort=False)["sales"].diff().abs()
    scale = diffs.groupby(hist["series_id"], sort=False).mean()
    return scale.where(scale > 0)


def mase(y, yhat, series_id, scale: pd.Series) -> float:
    """Mean over series of (MAE of the series / its naive scale)."""
    df = pd.DataFrame({"sid": series_id, "ae": np.abs(np.asarray(y) - np.asarray(yhat))})
    per_series = df.groupby("sid")["ae"].mean() / scale.reindex(df["sid"].unique())
    return float(per_series.dropna().mean())


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def interval_score(y, lo, hi, alpha: float) -> float:
    """Winkler / interval score (lower is better) for a central (1 - alpha) interval."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    width = hi - lo
    below = (2 / alpha) * (lo - y) * (y < lo)
    above = (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(width + below + above))
