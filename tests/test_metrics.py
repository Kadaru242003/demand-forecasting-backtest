import numpy as np
import pandas as pd
import pytest

from demand_forecast.metrics import coverage, interval_score, mae, mase, naive_scale, rmse


def test_mae_rmse():
    y, yhat = np.array([1.0, 2, 3]), np.array([2.0, 2, 1])
    assert mae(y, yhat) == pytest.approx(1.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))


def test_naive_scale_and_mase():
    hist = pd.DataFrame(
        {
            "series_id": ["a"] * 4 + ["b"] * 3 + ["c"] * 3,
            "week": list(range(4)) + list(range(3)) * 2,
            "sales": [0, 2, 0, 2, 5, 5, 5, 1, 2, 4],
        }
    )
    scale = naive_scale(hist)
    assert scale["a"] == pytest.approx(2.0)
    assert np.isnan(scale["b"])  # constant history is excluded
    assert scale["c"] == pytest.approx(1.5)
    y = np.array([1.0, 1, 5, 3])
    yhat = np.array([3.0, 1, 0, 0])
    sid = np.array(["a", "a", "b", "c"])
    # a: MAE 1 / 2 = 0.5 ; c: MAE 3 / 1.5 = 2.0 ; b excluded
    assert mase(y, yhat, sid, scale) == pytest.approx(1.25)


def test_coverage_and_interval_score():
    y = np.array([0.0, 5, 10])
    lo, hi = np.array([0.0, 0, 0]), np.array([4.0, 6, 8])
    assert coverage(y, lo, hi) == pytest.approx(2 / 3)
    # widths 4, 6, 8 ; one miss of 2 above at alpha=0.2 -> penalty 2/0.2*2 = 20
    assert interval_score(y, lo, hi, 0.2) == pytest.approx((4 + 6 + 8 + 20) / 3)
