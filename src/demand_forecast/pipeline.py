"""End-to-end run: data -> rolling-origin backtest -> metrics, intervals, charts.

    python -m demand_forecast.pipeline --store CA_1

Fold 0 is a warm-up fold used only to calibrate the prediction intervals of fold 1;
point metrics are reported on the evaluation folds 1..N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import plots
from .backtest import make_folds, run_backtest
from .data import PROCESSED_DIR, REPO_ROOT, load_weekly_panel
from .evaluation import (
    annotate,
    bias_by_segment,
    fold_context,
    interval_report,
    metrics_by_fold,
    segment_errors,
    store_total_errors,
    summary_table,
)
from .intervals import conformal_intervals
from .models import ETS, LightGBMDirect, MovingAverage, SeasonalNaive

LEVELS = (0.8, 0.95)


def default_models(horizon: int) -> dict:
    return {
        "Seasonal naive": lambda: SeasonalNaive(52),
        "Moving average (8w)": lambda: MovingAverage(8),
        "ETS": lambda: ETS(),
        "LightGBM": lambda: LightGBMDirect(max_horizon=horizon),
    }


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(floatfmt.format)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="CA_1")
    ap.add_argument("--folds", type=int, default=6, help="evaluation folds")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--reuse", action="store_true", help="reuse cached forecasts if present")
    args = ap.parse_args(argv)

    out, figs = args.out, args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    panel = load_weekly_panel(args.store)
    folds = make_folds(panel["week"], args.folds + 1, args.horizon)

    cache = PROCESSED_DIR / f"forecasts_{args.store}.parquet"
    if args.reuse and cache.exists():
        fc = pd.read_parquet(cache)
    else:
        fc = run_backtest(panel, default_models(args.horizon), folds, verbose=True)
        fc.to_parquet(cache, index=False)

    context = fold_context(panel, folds)
    fc = conformal_intervals(fc, context[["fold", "series_id", "scale"]], folds, LEVELS)
    df = annotate(fc.drop(columns="scale"), panel, context)
    eval_folds = [f for f in folds if f.fold >= 1]
    ev = df[df["fold"] >= 1]

    by_fold = metrics_by_fold(ev)
    summary = summary_table(by_fold)
    best = summary.index[0]
    seg_week = segment_errors(ev, "week_type").reindex(["regular", "promo week", "holiday week"])
    seg_vol = segment_errors(ev, "volume")
    seg_hol = segment_errors(ev[ev["holiday"] != ""], "holiday")
    seg_dept = segment_errors(ev, "dept_id")
    bias_week = bias_by_segment(ev, "week_type")
    bias_vol = bias_by_segment(ev, "volume")
    cov = interval_report(ev[ev["model"] == best], LEVELS)
    store_err = store_total_errors(ev)

    by_fold.to_csv(out / "metrics_by_fold.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv")
    for name, t in [
        ("week_type", seg_week),
        ("volume", seg_vol),
        ("holiday", seg_hol),
        ("dept", seg_dept),
    ]:
        t.to_csv(out / f"segment_error_{name}.csv")
    pd.concat({"week_type": bias_week, "volume": bias_vol}).to_csv(out / "bias_by_segment.csv")
    cov.to_csv(out / "interval_coverage.csv", index=False)
    store_err.to_csv(out / "store_total_errors.csv", index=False)

    plots.plot_fold_mase(by_fold, eval_folds, figs / "fold_mase.png")
    plots.plot_segment_errors(seg_week, seg_vol, figs / "segment_errors.png")
    plots.plot_store_total(ev, panel, eval_folds, figs / "store_total.png")
    plots.plot_coverage(cov[cov["by"] != "fold"], best, figs / "interval_coverage.png")

    n_series = panel["series_id"].nunique()
    meta = {
        "store": args.store,
        "n_series": int(n_series),
        "weeks": [str(panel["week"].min().date()), str(panel["week"].max().date())],
        "horizon": args.horizon,
        "folds": [
            {
                "fold": f.fold,
                "train_end": str(f.train_end.date()),
                "test": [str(f.test_weeks[0].date()), str(f.test_weeks[-1].date())],
            }
            for f in folds
        ],
        "best_model": best,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    summ = summary.reset_index()
    pf = by_fold.pivot(index="fold", columns="model", values="MASE").reset_index()
    cov_overall = cov[cov["by"].isin(["overall", "fold"])][
        ["level", "by", "group", "coverage", "mean_width", "n"]
    ]
    md = [
        f"# Backtest results: store {args.store}, {n_series} items, "
        f"{args.folds} folds x {args.horizon} weeks\n",
        "## Summary (mean over evaluation folds)\n",
        _md(summ),
        "\n## MASE per fold\n",
        _md(pf),
        "\n## Full per-fold metrics\n",
        _md(by_fold),
        "\n## Mean scaled error by week type\n",
        _md(seg_week.reset_index()),
        "\n## Mean scaled error by volume segment\n",
        _md(seg_vol.reset_index()),
        "\n## Mean scaled error by holiday\n",
        _md(seg_hol.reset_index()),
        "\n## Relative bias (sum forecast / sum actual - 1)\n",
        _md(pd.concat({"week_type": bias_week, "volume": bias_vol}).reset_index()),
        "\n## Store-level weekly total (sum of item forecasts): MAPE / bias per fold\n",
        _md(store_err.pivot(index="fold", columns="model", values="MAPE").reset_index()),
        "\n",
        _md(store_err.pivot(index="fold", columns="model", values="bias").reset_index()),
        f"\n## Interval coverage ({best})\n",
        _md(cov_overall),
    ]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(summary.to_string())
    print(cov[cov["by"] == "overall"].to_string())


if __name__ == "__main__":
    main()
