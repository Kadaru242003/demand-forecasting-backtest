"""Download, clean and aggregate the M5 (Walmart) dataset to a weekly item-store panel.

The M5 competition data (Makridakis et al., 2022) contains daily unit sales for 3,049
products across 10 Walmart stores in California, Texas and Wisconsin from 2011-01-29 to
2016-05-22, together with a calendar (events, SNAP days) and weekly sell prices.

We download a public mirror of the original files published by Nixtla and verify its
checksum. The raw files are not committed to this repository.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

M5_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
M5_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Holidays that move demand at Walmart. The M5 calendar marks the *day* of each event;
# we flag the Walmart week (Saturday-Friday) that contains it.
MAJOR_HOLIDAYS = (
    "SuperBowl",
    "ValentinesDay",
    "Easter",
    "MemorialDay",
    "IndependenceDay",
    "LaborDay",
    "Halloween",
    "Thanksgiving",
    "Christmas",
    "NewYear",
)

# A week is flagged as a promotion week when the price is at least this much below the
# item's highest price over the previous 52 weeks. M5 has no promotion field, so this is
# a proxy (see README limitations).
PROMO_DISCOUNT_THRESHOLD = 0.05


def download_m5(raw_dir: Path = RAW_DIR, url: str = M5_URL) -> Path:
    """Download and verify the M5 zip archive, returning its local path (cached)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "m5.zip"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 (fixed https URL)
            payload = resp.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != M5_SHA256:
            raise ValueError(f"Checksum mismatch for {url}: got {digest}")
        path.write_bytes(payload)
    return path


def _read_from_zip(zip_path: Path, name: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf, zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), **kwargs)


def load_raw(zip_path: Path, store_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load daily sales (one store), calendar and sell prices (one store) from the archive."""
    calendar = _read_from_zip(zip_path, "calendar.csv", parse_dates=["date"])
    calendar = calendar.sort_values("date").reset_index(drop=True)
    if "d" not in calendar.columns:  # the mirror drops the day key; d_1 is the first date
        calendar["d"] = "d_" + (calendar.index + 1).astype(str)
    sales = _read_from_zip(zip_path, "sales_train_evaluation.csv")
    sales = sales[sales["store_id"] == store_id].reset_index(drop=True)
    if sales.empty:
        raise ValueError(f"Unknown store_id {store_id!r}")
    prices = _read_from_zip(zip_path, "sell_prices.csv")
    prices = prices[prices["store_id"] == store_id].reset_index(drop=True)
    return sales, calendar, prices


def weekly_calendar(calendar: pd.DataFrame, state_id: str) -> pd.DataFrame:
    """Aggregate the daily calendar to Walmart weeks (Saturday to Friday)."""
    cal = calendar.copy()
    events = cal[["event_name_1", "event_name_2"]]
    cal["n_events"] = events.notna().sum(axis=1)
    for holiday in MAJOR_HOLIDAYS:
        cal[f"hol_{holiday}"] = (events == holiday).any(axis=1).astype(int)
    cal["snap"] = cal[f"snap_{state_id}"]
    agg = {
        "date": ["min", "count"],
        "n_events": "sum",
        "snap": "sum",
        **{f"hol_{h}": "max" for h in MAJOR_HOLIDAYS},
    }
    wk = cal.groupby("wm_yr_wk").agg(agg)
    wk.columns = ["week_start", "n_days", "n_events", "snap_days"] + [
        f"hol_{h}" for h in MAJOR_HOLIDAYS
    ]
    wk = wk.reset_index()
    hol_cols = [f"hol_{h}" for h in MAJOR_HOLIDAYS]
    wk["is_holiday_week"] = wk[hol_cols].max(axis=1)
    # demand often moves the week *before* a holiday (e.g. Thanksgiving shopping)
    wk["holiday_next_week"] = wk["is_holiday_week"].shift(-1).fillna(0).astype(int)
    wk["holiday"] = np.where(
        wk["is_holiday_week"] == 1,
        wk[hol_cols].idxmax(axis=1).str.removeprefix("hol_"),
        "",
    )
    return wk


def build_weekly_panel(
    sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Turn the raw M5 tables into a clean weekly item-store panel.

    Cleaning steps (each documented in the README):
      1. daily sales -> Walmart weeks; partial weeks at the end of the data are dropped;
      2. each item's history starts at its first week with a sell price (product launch),
         removing the structural leading zeros M5 is known for;
      3. weeks after launch without a price (item not on shelf) keep sales (always 0 in M5)
         and carry the last known price forward, with a flag;
      4. a promotion proxy is derived from price drops versus the trailing 52-week max.
    """
    state_id = sales["state_id"].iloc[0]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_to_week = calendar.set_index("d")["wm_yr_wk"]

    values = sales[day_cols].to_numpy(dtype=np.float64)
    week_of_day = day_to_week.loc[day_cols].to_numpy()
    weeks, inverse = np.unique(week_of_day, return_inverse=True)
    weekly = np.zeros((len(sales), len(weeks)))
    np.add.at(weekly.T, inverse, values.T)
    days_per_week = np.bincount(inverse)
    complete = days_per_week == 7
    weekly, weeks = weekly[:, complete], weeks[complete]

    panel = pd.DataFrame(weekly, columns=weeks)
    panel.insert(0, "item_id", sales["item_id"].to_numpy())
    panel = panel.melt(id_vars="item_id", var_name="wm_yr_wk", value_name="sales")
    panel["wm_yr_wk"] = panel["wm_yr_wk"].astype(int)
    meta = sales[["item_id", "dept_id", "cat_id", "store_id"]]
    panel = panel.merge(meta, on="item_id", how="left")

    price = prices[["item_id", "wm_yr_wk", "sell_price"]]
    panel = panel.merge(price, on=["item_id", "wm_yr_wk"], how="left")
    panel = panel.sort_values(["item_id", "wm_yr_wk"]).reset_index(drop=True)

    # 2. drop weeks before the first priced week of each item
    has_price = panel["sell_price"].notna()
    launched = has_price.groupby(panel["item_id"]).cummax()
    panel = panel[launched].reset_index(drop=True)

    # 3. off-shelf weeks
    panel["price_missing"] = panel["sell_price"].isna().astype(int)
    panel["sell_price"] = panel.groupby("item_id")["sell_price"].ffill()

    # 4. promotion proxy (uses prices only, never sales)
    prev_max = panel.groupby("item_id")["sell_price"].transform(
        lambda s: s.shift(1).rolling(52, min_periods=1).max()
    )
    panel["discount"] = (1 - panel["sell_price"] / prev_max).clip(lower=0).fillna(0.0)
    panel["is_promo"] = (panel["discount"] >= PROMO_DISCOUNT_THRESHOLD).astype(int)

    wk_cal = weekly_calendar(calendar, state_id)
    panel = panel.merge(wk_cal.drop(columns="n_days"), on="wm_yr_wk", how="left")
    panel = panel.rename(columns={"item_id": "series_id"})
    panel["week"] = panel["week_start"]
    cols = ["series_id", "week", "sales"] + [
        c for c in panel.columns if c not in ("series_id", "week", "sales", "week_start")
    ]
    return panel[cols].sort_values(["series_id", "week"]).reset_index(drop=True)


def load_weekly_panel(store_id: str = "CA_1", refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned weekly panel for one store, building and caching it if needed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED_DIR / f"weekly_{store_id}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    zip_path = download_m5()
    sales, calendar, prices = load_raw(zip_path, store_id)
    panel = build_weekly_panel(sales, calendar, prices)
    panel.to_parquet(cache, index=False)
    return panel
