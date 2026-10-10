"""The backtest must never let a model see data from after its forecast origin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecast.backtest import TARGET, make_folds, run_backtest, split
from demand_forecast.features import anchor_features, build_features
from demand_forecast.models import ETS, LightGBMDirect, MovingAverage, SeasonalNaive

HORIZON = 4


@pytest.fixture(scope="module")
def folds(panel):
    return make_folds(panel["week"], n_folds=3, horizon=HORIZON)


def test_folds_tile_the_end_of_the_data_without_overlap(panel, folds):
    weeks = sorted(panel["week"].unique())
    assert folds[-1].test_weeks[-1] == weeks[-1]
    for f in folds:
        assert len(f.test_weeks) == HORIZON
        assert f.train_end < f.test_weeks[0]
        # test weeks are the consecutive weeks right after the origin
        i = weeks.index(f.train_end)
        assert list(f.test_weeks) == weeks[i + 1 : i + 1 + HORIZON]
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_weeks[-1] <= b.train_end  # earlier test windows are fully observed


def test_split_separates_past_from_future(panel, folds):
    for f in folds:
        history, future, actuals = split(panel, f)
        assert history["week"].max() == f.train_end
        assert TARGET not in future.columns
        assert set(future["week"]) == set(f.test_weeks)
        assert set(future["series_id"]) <= set(history["series_id"])
        assert future["horizon"].between(1, HORIZON).all()
        assert len(actuals) == len(future)


class SpyModel:
    """Records exactly what the backtest hands to a model."""

    calls: list = []

    def fit(self, history):
        self.history = history
        return self

    def predict(self, future):
        SpyModel.calls.append(
            (self.history["week"].max(), future["week"].min(), set(future.columns))
        )
        return np.zeros(len(future))


def test_models_only_receive_past_targets(panel, folds):
    SpyModel.calls = []
    run_backtest(panel, {"spy": SpyModel}, folds)
    assert len(SpyModel.calls) == len(folds)
    for (seen_max, future_min, cols), f in zip(SpyModel.calls, folds, strict=True):
        assert seen_max == f.train_end < future_min
        assert TARGET not in cols


MODELS = {
    "snaive": lambda: SeasonalNaive(52),
    "ma": lambda: MovingAverage(8),
    "ets": lambda: ETS(),
    "lgb": lambda: LightGBMDirect(
        max_horizon=HORIZON, num_boost_round=20, params={"min_data_in_leaf": 5, "num_threads": 1}
    ),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_forecasts_do_not_change_when_the_future_is_scrambled(panel, folds, name):
    """Strongest leakage check: replace every value after the origin with garbage.

    If any model, feature or the backtest itself used future targets, forecasts would move.
    """
    fold = folds[0]
    rng = np.random.default_rng(1)
    poisoned = panel.copy()
    after = poisoned["week"] > fold.train_end
    poisoned.loc[after, TARGET] = rng.integers(0, 10_000, after.sum()).astype(float)

    clean = run_backtest(panel, {name: MODELS[name]}, [fold])
    dirty = run_backtest(poisoned, {name: MODELS[name]}, [fold])
    np.testing.assert_allclose(clean["yhat"].to_numpy(), dirty["yhat"].to_numpy())
    # sanity: the actuals did change, so the comparison is meaningful
    assert not np.allclose(clean["y"].to_numpy(), dirty["y"].to_numpy())


@pytest.mark.parametrize("h", [1, 3, 8])
def test_features_for_horizon_h_only_use_sales_up_to_t_minus_h(panel, h):
    frame = panel.sort_values(["series_id", "week"]).reset_index(drop=True)
    X_full = build_features(frame, h, anchor_features(frame))
    weeks = sorted(frame["week"].unique())
    target_week = weeks[-1]
    anchor_week = weeks[-1 - h]
    masked = frame.copy()
    masked.loc[masked["week"] > anchor_week, TARGET] = np.nan
    X_masked = build_features(masked, h, anchor_features(masked))
    rows = (frame["week"] == target_week).to_numpy()
    pd.testing.assert_frame_equal(
        X_full[rows].reset_index(drop=True), X_masked[rows].reset_index(drop=True)
    )
