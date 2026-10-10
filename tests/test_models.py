import numpy as np
import pandas as pd

from demand_forecast.backtest import make_folds, split
from demand_forecast.models import ETS, MovingAverage, SeasonalNaive


def test_seasonal_naive_uses_value_one_season_back(panel):
    fold = make_folds(panel["week"], 1, 4)[0]
    history, future, _ = split(panel, fold)
    yhat = SeasonalNaive(52).fit(history).predict(future)
    for (sid, week), pred in zip(
        future[["series_id", "week"]].itertuples(index=False), yhat, strict=True
    ):
        past = history[
            (history["series_id"] == sid) & (history["week"] == week - pd.Timedelta(weeks=52))
        ]
        expected = (
            past["sales"].iloc[0]
            if len(past)
            else history[history["series_id"] == sid]["sales"].iloc[-1]
        )
        assert pred == expected


def test_moving_average_is_flat_mean_of_last_window(panel):
    fold = make_folds(panel["week"], 1, 4)[0]
    history, future, _ = split(panel, fold)
    yhat = MovingAverage(8).fit(history).predict(future)
    sid = future["series_id"].iloc[0]
    expected = history[history["series_id"] == sid].sort_values("week")["sales"].iloc[-8:].mean()
    np.testing.assert_allclose(yhat[future["series_id"] == sid], expected)


def test_ets_tracks_a_level_shift():
    weeks = pd.date_range("2020-01-04", periods=60, freq="7D")
    y = np.r_[np.full(40, 10.0), np.full(20, 30.0)] + np.random.default_rng(0).normal(0, 0.5, 60)
    hist = pd.DataFrame({"series_id": "a", "week": weeks, "sales": y})
    fut = pd.DataFrame(
        {"series_id": "a", "week": weeks[-1] + pd.Timedelta(weeks=1), "horizon": [1]}
    )
    pred = ETS().fit(hist).predict(fut)
    assert abs(pred[0] - 30) < 3
