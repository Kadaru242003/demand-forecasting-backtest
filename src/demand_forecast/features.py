"""Feature engineering for the direct multi-horizon gradient-boosting model.

Features for a target week ``t`` forecast ``h`` weeks ahead are built from

* *anchor* features: lags and rolling statistics of sales ending at week ``t - h``
  (the forecast origin), so they only use information available at forecast time;
* seasonal lags at ``t - 52`` (always <= origin for h <= 52);
* covariates of week ``t`` that are known in advance: calendar, holidays, SNAP days and
  the planned sell price / promotion proxy.

Sales of rows that are not yet observed must be NaN; every rolling statistic is computed
on past values only, so NaNs never leak into anchor features of valid rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MAJOR_HOLIDAYS

ROLL_WINDOWS = (4, 8, 13, 26, 52)
N_LAGS = 4
SEASON = 52

HOLIDAY_COLS = [f"hol_{h}" for h in MAJOR_HOLIDAYS]
KNOWN_COVARIATES = [
    "sell_price",
    "discount",
    "is_promo",
    "n_events",
    "snap_days",
    "is_holiday_week",
    "holiday_next_week",
    *HOLIDAY_COLS,
]
CATEGORICALS = ["dept_id", "cat_id"]


def anchor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Statistics of each series' sales up to and including each week (no shift)."""
    g = frame.groupby("series_id", sort=False)["sales"]
    feats = {}
    for k in range(N_LAGS):
        feats[f"lag_{k}"] = g.shift(k)
    for w in ROLL_WINDOWS:
        feats[f"rmean_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    feats["rstd_8"] = g.transform(lambda s: s.rolling(8, min_periods=2).std())
    feats["zero_frac_13"] = g.transform(lambda s: s.eq(0).where(s.notna()).rolling(13, 1).mean())
    feats["age"] = g.cumcount()
    return pd.DataFrame(feats, index=frame.index)


def build_features(frame: pd.DataFrame, horizon: int, anchors: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for target weeks forecast ``horizon`` steps after their anchor."""
    by_series = frame["series_id"]
    shifted = anchors.groupby(by_series, sort=False).shift(horizon)
    shifted.columns = [f"a_{c}" for c in shifted.columns]
    g = frame.groupby("series_id", sort=False)
    X = shifted
    X["seas_lag"] = g["sales"].shift(SEASON)
    X["seas_mean3"] = (
        g["sales"].shift(SEASON - 1) + g["sales"].shift(SEASON) + g["sales"].shift(SEASON + 1)
    ) / 3
    X["seas_ratio"] = X["seas_lag"] / (X["a_rmean_52"] + 1)

    week = frame["week"]
    X["weekofyear"] = week.dt.isocalendar().week.astype(int).to_numpy()
    X["month"] = week.dt.month.to_numpy()
    for col in KNOWN_COVARIATES:
        X[col] = frame[col].to_numpy()
    price = g["sell_price"]
    X["price_rel_mean"] = frame["sell_price"] / price.transform(
        lambda s: s.shift(1).rolling(52, min_periods=1).mean()
    )
    X["price_change"] = frame["sell_price"] / price.shift(1) - 1
    X["horizon"] = horizon
    for col in CATEGORICALS:
        X[col] = frame[col].astype("category")
    return X.replace([np.inf, -np.inf], np.nan)
