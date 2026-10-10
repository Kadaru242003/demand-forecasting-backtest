"""Turn raw backtest forecasts into metric tables and error-analysis segments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .backtest import Fold, split
from .metrics import coverage, interval_score, mae, mase, naive_scale, rmse

BASELINE = "Seasonal naive"

# Mean weekly units over the training window of each fold.
VOLUME_BINS = [-np.inf, 2, 10, np.inf]
VOLUME_LABELS = ["low (<2/wk)", "medium (2-10/wk)", "high (>10/wk)"]


def fold_context(panel: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """Per fold and series: MASE scale and training-window volume segment."""
    out = []
    for fold in folds:
        history, _, _ = split(panel, fold)
        scale = naive_scale(history)
        mean = history.groupby("series_id")["sales"].mean()
        df = pd.DataFrame({"scale": scale, "train_mean": mean.reindex(scale.index)})
        df["volume"] = pd.cut(df["train_mean"], VOLUME_BINS, labels=VOLUME_LABELS)
        df["fold"] = fold.fold
        out.append(df.rename_axis("series_id").reset_index())
    return pd.concat(out, ignore_index=True)


def annotate(forecasts: pd.DataFrame, panel: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Attach week type, promo flag, volume segment and scaled error to each forecast row."""
    cols = ["series_id", "week", "dept_id", "cat_id", "is_promo", "is_holiday_week", "holiday"]
    df = forecasts.merge(panel[cols], on=["series_id", "week"], how="left")
    df = df.merge(context, on=["fold", "series_id"], how="left")
    df["abs_err"] = (df["y"] - df["yhat"]).abs()
    df["scaled_err"] = df["abs_err"] / df["scale"]
    df["week_type"] = np.select(
        [df["is_holiday_week"] == 1, df["is_promo"] == 1], ["holiday week", "promo week"], "regular"
    )
    return df


def metrics_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold), g in df.groupby(["model", "fold"]):
        scale = g.groupby("series_id")["scale"].first()
        rows.append(
            {
                "model": model,
                "fold": fold,
                "MAE": mae(g["y"], g["yhat"]),
                "RMSE": rmse(g["y"], g["yhat"]),
                "MASE": mase(g["y"], g["yhat"], g["series_id"], scale),
            }
        )
    out = pd.DataFrame(rows)
    base = out[out["model"] == BASELINE].set_index("fold")["MAE"]
    out["MAE vs seasonal naive"] = out["MAE"] / out["fold"].map(base)
    return out


def summary_table(by_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds, plus the fold-to-fold range and how often a model beats the baseline."""
    g = by_fold.groupby("model")
    out = g[["MAE", "RMSE", "MASE", "MAE vs seasonal naive"]].mean()
    out["MASE min"] = g["MASE"].min()
    out["MASE max"] = g["MASE"].max()
    base = by_fold[by_fold["model"] == BASELINE].set_index("fold")["MASE"]
    out["folds beating baseline (MASE)"] = g.apply(
        lambda x: f"{int((x.set_index('fold')['MASE'] < base).sum())}/{len(x)}",
        include_groups=False,
    )
    return out.sort_values("MASE")


def segment_errors(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean scaled absolute error per model and segment, plus each segment's share of rows."""
    valid = df[np.isfinite(df["scaled_err"])]
    tab = valid.pivot_table(
        index=by, columns="model", values="scaled_err", aggfunc="mean", observed=True
    )
    n = valid[valid["model"] == valid["model"].iloc[0]].groupby(by, observed=True).size()
    tab["n_rows"] = n
    return tab


def bias_by_segment(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean (forecast - actual) / mean actual per model and segment (positive = over-forecast)."""
    g = df.groupby([by, "model"], observed=True)
    out = (g["yhat"].sum() - g["y"].sum()) / g["y"].sum()
    return out.unstack("model")


def interval_report(df: pd.DataFrame, levels: Sequence[float] = (0.8, 0.95)) -> pd.DataFrame:
    rows = []
    has = df.dropna(subset=[f"lo_{round(levels[0] * 100)}"])
    for level in levels:
        tag = f"{round(level * 100)}"
        for key, by in [
            ("overall", None),
            ("fold", "fold"),
            ("horizon", "horizon"),
            ("volume", "volume"),
            ("week_type", "week_type"),
        ]:
            groups = [("all", has)] if by is None else has.groupby(by, observed=True)
            for name, g in groups:
                rows.append(
                    {
                        "level": level,
                        "by": key,
                        "group": str(name),
                        "coverage": coverage(g["y"], g[f"lo_{tag}"], g[f"hi_{tag}"]),
                        "mean_width": float((g[f"hi_{tag}"] - g[f"lo_{tag}"]).mean()),
                        "interval_score": interval_score(
                            g["y"], g[f"lo_{tag}"], g[f"hi_{tag}"], 1 - level
                        ),
                        "n": len(g),
                    }
                )
    return pd.DataFrame(rows)


def store_total_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Error of the store-level weekly total obtained by summing item forecasts (bottom-up).

    Returns mean absolute percentage error and bias of the weekly totals per model and fold.
    """
    tot = df.groupby(["model", "fold", "week"])[["y", "yhat"]].sum()
    tot["ape"] = (tot["yhat"] - tot["y"]).abs() / tot["y"]
    g = tot.groupby(["model", "fold"])
    out = pd.DataFrame({"MAPE": g["ape"].mean(), "bias": g["yhat"].sum() / g["y"].sum() - 1})
    return out.reset_index()
