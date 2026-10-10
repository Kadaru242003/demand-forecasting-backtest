import numpy as np

from demand_forecast.data import build_weekly_panel


def test_weekly_aggregation_and_cleaning(raw):
    sales, calendar, prices = raw
    panel = build_weekly_panel(sales, calendar, prices)
    # partial trailing week dropped: 130 complete weeks
    assert panel["week"].nunique() == 130
    assert (panel["week"].dt.dayofweek == 5).all()  # Walmart weeks start on Saturday
    # weekly totals equal the sum of the daily values
    item = sales.iloc[0]
    days = [f"d_{i}" for i in range(1, 8)]
    first = panel[panel["series_id"] == item["item_id"]].iloc[0]
    assert first["sales"] == item[days].sum()
    # later-launching items start at their first priced week
    starts = panel.groupby("series_id")["wm_yr_wk"].min()
    first_price = prices.groupby("item_id")["wm_yr_wk"].min()
    assert (starts == first_price.reindex(starts.index)).all()
    # no gaps inside a series
    sizes = panel.groupby("series_id")["wm_yr_wk"].agg(lambda s: s.max() - s.min() + 1)
    assert (sizes == panel.groupby("series_id").size()).all()


def test_holiday_and_promo_flags(panel):
    christmas = panel[panel["hol_Christmas"] == 1]["week"].unique()
    assert len(christmas) == 2  # Dec 2011 and Dec 2012 within 130 weeks
    for w in christmas:
        assert w.month == 12 and 19 <= w.day <= 25
    assert (panel.loc[panel["holiday_next_week"] == 1, "is_holiday_week"].notna()).all()
    promo = panel[panel["is_promo"] == 1]
    assert len(promo) > 0
    assert np.allclose(promo["discount"], 0.2)
