"""Charts for the README / error analysis (static PNG, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed model -> colour mapping (validated categorical palette, slots 1-4 in order).
MODEL_ORDER = ["LightGBM", "ETS", "Moving average (8w)", "Seasonal naive"]
COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "font.size": 9.5,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _models(present) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(present)]


def _grouped_bars(ax, table: pd.DataFrame, models: list[str], fmt="{:.2f}", label_best=True):
    n = len(models)
    width = 0.8 / n
    x = np.arange(len(table.index))
    for i, m in enumerate(models):
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            table[m],
            width=width,
            color=COLORS[m],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=m,
        )
    if label_best:  # label only the best (lowest) bar per group
        for j, (_, row) in enumerate(table[models].iterrows()):
            i = int(np.argmin(row.to_numpy()))
            ax.text(
                x[j] + (i - (n - 1) / 2) * width,
                row.iloc[i],
                fmt.format(row.iloc[i]),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x, table.index)


def plot_fold_mase(by_fold: pd.DataFrame, folds, path: Path) -> None:
    models = _models(by_fold["model"])
    tab = by_fold.pivot(index="fold", columns="model", values="MASE")
    labels = {
        f.fold: f"Fold {f.fold}\n{f.test_weeks[0]:%d %b %y} –\n{f.test_weeks[-1]:%d %b %y}"
        for f in folds
    }
    tab.index = [labels[i] for i in tab.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    _grouped_bars(ax, tab, models)
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("MASE per backtest fold (8-week horizon, 3,049 items, store CA_1)", loc="left")
    ax.legend(ncols=len(models), loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.savefig(path)
    plt.close(fig)


def plot_segment_errors(by_week_type: pd.DataFrame, by_volume: pd.DataFrame, path: Path) -> None:
    models = _models(by_week_type.columns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, tab, title in [
        (axes[0], by_week_type, "By week type"),
        (axes[1], by_volume, "By item volume (training-window mean)"),
    ]:
        t = tab.copy()
        t.index = [f"{i}\n(n={n:,})" for i, n in zip(t.index, t["n_rows"], strict=True)]
        _grouped_bars(ax, t, models)
        ax.set_title(title, loc="left")
        ax.set_ylabel("mean scaled abs. error (|e| / naive MAE)")
    axes[0].legend(ncols=len(models), loc="upper left", bbox_to_anchor=(0, -0.22))
    fig.suptitle(
        "Where the models fail: scaled error by segment",
        y=1.03,
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.savefig(path)
    plt.close(fig)


def plot_store_total(df: pd.DataFrame, panel: pd.DataFrame, folds, path: Path) -> None:
    """Store-level weekly totals: actual vs the sum of each model's item forecasts."""
    models = _models(df["model"])
    tot = df.groupby(["week", "model"])["yhat"].sum().unstack("model")
    actual = df[df["model"] == models[0]].groupby("week")["y"].sum()
    hol = panel.drop_duplicates("week").set_index("week")["holiday"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for w in actual.index:
        if hol.get(w, ""):
            ax.axvspan(w - pd.Timedelta(days=3.5), w + pd.Timedelta(days=3.5), color=GRID, lw=0)
            ax.text(
                w,
                1.0,
                hol[w],
                rotation=90,
                va="top",
                ha="center",
                fontsize=7,
                color=INK_2,
                transform=ax.get_xaxis_transform(),
            )
    for f in folds:
        ax.axvline(f.train_end + pd.Timedelta(days=3.5), color=INK_2, lw=0.6, ls=":")
    for m in models:
        ax.plot(tot.index, tot[m], color=COLORS[m], lw=1.6, label=m)
    ax.plot(actual.index, actual, color=INK, lw=2, label="Actual")
    ax.set_ylabel("units / week (sum over items)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(
        "Store CA_1 weekly total: actual vs summed item forecasts "
        "(dotted = fold origins, shaded = holiday weeks)",
        loc="left",
    )
    ax.legend(ncols=len(models) + 1, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.savefig(path)
    plt.close(fig)


def plot_coverage(report: pd.DataFrame, model: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    series = [("80%", 0.8, "#2a78d6"), ("95%", 0.95, "#eb6834")]
    for ax, by, title in [
        (axes[0], "horizon", "By horizon (weeks ahead)"),
        (axes[1], "volume", "By item volume"),
        (axes[2], "week_type", "By week type"),
    ]:
        for label, level, color in series:
            r = report[(report["by"] == by) & (report["level"] == level)]
            if by == "week_type":
                r = r.set_index("group").reindex(["regular", "promo week", "holiday week"])
                r = r.reset_index()
            x = np.arange(len(r))
            ax.plot(
                x,
                r["coverage"],
                marker="o",
                ms=7,
                lw=1.8 if by == "horizon" else 0,  # no lines between unordered categories
                color=color,
                label=f"{label} interval",
                mec=SURFACE,
                mew=1.5,
            )
            ax.axhline(level, color=color, lw=1, ls="--", alpha=0.7)
            ax.set_xticks(x, [g.replace(" (", "\n(") for g in r["group"]])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("empirical coverage (dashed = nominal)")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(loc="lower left")
    fig.suptitle(
        f"Conformal prediction intervals for {model}: empirical coverage on folds 1–6",
        y=1.03,
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.savefig(path)
    plt.close(fig)
