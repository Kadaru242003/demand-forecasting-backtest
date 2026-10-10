"""Forecasting models, from simple baselines to gradient boosting.

All models follow the same two-method interface used by :mod:`demand_forecast.backtest`:

* ``fit(history)``: ``history`` holds ``series_id``, ``week``, ``sales`` and covariates
  for weeks up to the forecast origin;
* ``predict(future)``: ``future`` holds the rows to forecast (``series_id``, ``week``,
  ``horizon`` and known covariates, **without** ``sales``) and returns one forecast per row.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICALS, anchor_features, build_features


def _last_values(history: pd.DataFrame) -> dict[str, np.ndarray]:
    return {sid: g.to_numpy() for sid, g in history.groupby("series_id", sort=False)["sales"]}


class SeasonalNaive:
    """Forecast = value from the same week one season (52 weeks) earlier.

    Series with less than a season of history fall back to the last observed value.
    """

    def __init__(self, season: int = 52):
        self.season = season

    def fit(self, history: pd.DataFrame) -> SeasonalNaive:
        self.series_ = _last_values(history.sort_values(["series_id", "week"]))
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(future))
        for i, (sid, h) in enumerate(zip(future["series_id"], future["horizon"], strict=True)):
            y = self.series_[sid]
            # y[-1] is the origin; target is origin + h, one season back is index h - season
            idx = len(y) - 1 + h - self.season
            out[i] = y[idx] if 0 <= idx < len(y) else y[-1]
        return out


class MovingAverage:
    """Forecast = mean of the last ``window`` weeks (flat over the horizon)."""

    def __init__(self, window: int = 8):
        self.window = window

    def fit(self, history: pd.DataFrame) -> MovingAverage:
        hist = history.sort_values(["series_id", "week"])
        self.level_ = hist.groupby("series_id")["sales"].apply(
            lambda s: s.iloc[-self.window :].mean()
        )
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return future["series_id"].map(self.level_).to_numpy(dtype=float)


def _fit_ets(args: tuple[np.ndarray, int]) -> np.ndarray:
    """Fit ETS(A,N,N) and ETS(A,Ad,N), keep the lower AICc, forecast ``h`` steps."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y, h = args
    fallback = np.full(h, y[-8:].mean())
    if len(y) < 10 or np.allclose(y, y[0]):
        return fallback
    candidates = [{"error": "add"}]
    if len(y) >= 26:
        candidates.append({"error": "add", "trend": "add", "damped_trend": True})
    best, best_aicc = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in candidates:
            try:
                res = ETSModel(y, **spec).fit(disp=False, maxiter=200)
            except Exception:  # noqa: BLE001 - optimiser failures fall back below
                continue
            if np.isfinite(res.aicc) and res.aicc < best_aicc:
                best, best_aicc = res, res.aicc
    if best is None:
        return fallback
    fc = np.asarray(best.forecast(h))
    return fc if np.all(np.isfinite(fc)) else fallback


class ETS:
    """Per-series exponential smoothing with automatic choice between
    simple exponential smoothing, ETS(A,N,N), and damped-trend ETS(A,Ad,N) by AICc.

    A 52-week seasonal component is not used: with ~4-5 years of noisy, often intermittent
    item-level weekly data, estimating 52 seasonal states per item overfits badly.
    """

    def fit(self, history: pd.DataFrame) -> ETS:
        self.series_ = _last_values(history.sort_values(["series_id", "week"]))
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        max_h = future.groupby("series_id")["horizon"].max()
        jobs = [(self.series_[sid], int(h)) for sid, h in max_h.items()]
        fcs = [_fit_ets(j) for j in jobs]
        lookup = dict(zip(max_h.index, fcs, strict=True))
        return np.array(
            [lookup[s][h - 1] for s, h in zip(future["series_id"], future["horizon"], strict=True)]
        )


DEFAULT_LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}


class LightGBMDirect:
    """Global LightGBM model, one booster per horizon ("direct" multi-step strategy).

    The booster for horizon ``h`` only sees sales features anchored ``h`` weeks before the
    target week, so every feature it uses at prediction time is observable at the origin.
    Hyper-parameters are fixed a priori (no tuning on backtest folds).
    """

    def __init__(
        self,
        max_horizon: int = 8,
        num_boost_round: int = 300,
        params: dict | None = None,
        train_weeks: int | None = 156,
    ):
        self.max_horizon = max_horizon
        self.num_boost_round = num_boost_round
        self.params = {**DEFAULT_LGB_PARAMS, **(params or {})}
        self.train_weeks = train_weeks

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["series_id", "week"]).reset_index(drop=True)
        for col in CATEGORICALS:
            frame[col] = frame[col].astype(str)
        return frame

    def fit(self, history: pd.DataFrame) -> LightGBMDirect:
        hist = self._prepare(history.drop(columns=["horizon"], errors="ignore"))
        self.history_ = hist
        anchors = anchor_features(hist)
        cutoff = None
        if self.train_weeks is not None:
            weeks = np.sort(hist["week"].unique())
            cutoff = weeks[max(0, len(weeks) - self.train_weeks)]
        self.boosters_ = {}
        for h in range(1, self.max_horizon + 1):
            X = build_features(hist, h, anchors)
            mask = X["a_lag_0"].notna().to_numpy().copy()
            if cutoff is not None:
                mask &= (hist["week"] >= cutoff).to_numpy()
            self.feature_names_ = list(X.columns)
            ds = lgb.Dataset(X[mask], label=hist.loc[mask, "sales"], free_raw_data=True)
            self.boosters_[h] = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        fut = future.copy()
        fut["sales"] = np.nan  # unknown by construction
        frame = pd.concat(
            [self.history_.assign(_row=-1), fut.assign(_row=np.arange(len(fut)))],
            ignore_index=True,
        )
        frame = self._prepare(frame)
        anchors = anchor_features(frame)
        out = np.full(len(fut), np.nan)
        for h, booster in self.boosters_.items():
            X = build_features(frame, h, anchors)
            rows = (frame["horizon"] == h).to_numpy()
            if rows.any():
                out[frame.loc[rows, "_row"].to_numpy(dtype=int)] = booster.predict(
                    X.loc[rows, self.feature_names_]
                )
        if np.isnan(out).any():
            raise ValueError("Requested a horizon beyond max_horizon")
        return out
