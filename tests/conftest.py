"""Synthetic M5-shaped raw tables so the tests run without downloading the dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data import build_weekly_panel


def make_raw(n_items: int = 6, n_weeks: int = 130, seed: int = 0, extra_days: int = 3):
    rng = np.random.default_rng(seed)
    n_days = n_weeks * 7 + extra_days  # trailing partial week must be dropped
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")  # a Saturday, like M5
    week_ids = 11101 + np.arange(n_days) // 7
    events = np.full(n_days, None, dtype=object)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day == 25:
            events[i] = "Christmas"
        elif d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:
            events[i] = "Thanksgiving"
    calendar = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": week_ids,
            "event_name_1": events,
            "event_name_2": None,
            "snap_CA": (dates.day <= 10).astype(int),
            "d": [f"d_{i + 1}" for i in range(n_days)],
        }
    )
    items = [f"FOODS_1_{i:03d}" if i % 2 else f"HOBBIES_1_{i:03d}" for i in range(n_items)]
    t = np.arange(n_days)
    rows, prices = [], []
    for i, item in enumerate(items):
        base = 0.5 + 3 * i
        season = 1 + 0.3 * np.sin(2 * np.pi * t / 364)
        lam = base * season
        y = rng.poisson(lam).astype(float)
        launch_week = 10 * (i % 3)  # some items launch later
        y[: launch_week * 7] = 0
        rows.append([item, item.rsplit("_", 1)[0], item.split("_")[0], "CA_1", "CA", *y])
        for w in range(launch_week, n_weeks + 1):
            price = 2.0 + i
            if w % 17 == 0:
                price *= 0.8  # a promotion
            prices.append(("CA_1", item, 11101 + w, price))
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"] + list(calendar["d"])
    sales = pd.DataFrame(rows, columns=cols)
    prices = pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    return sales, calendar, prices


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture(scope="session")
def panel(raw):
    return build_weekly_panel(*raw)
