"""Rolling-origin backtesting.

The contract that prevents leakage lives here, not in the models:

* a model's ``fit`` only ever receives rows with ``week <= train_end``;
* its ``predict`` receives the test rows with the target column *removed*, i.e. only the
  covariates that are known in advance (calendar, planned prices);
* only series that already exist at the forecast origin are forecast (no cold starts).

``tests/test_backtest.py`` checks these properties directly and also verifies that
scrambling the future target values cannot change any forecast.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

TARGET = "sales"
KEYS = ["series_id", "week"]


class Forecaster(Protocol):
    def fit(self, history: pd.DataFrame) -> Forecaster: ...

    def predict(self, future: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: pd.Timestamp
    test_weeks: tuple[pd.Timestamp, ...]

    @property
    def horizon(self) -> int:
        return len(self.test_weeks)


def make_folds(weeks: Sequence, n_folds: int, horizon: int, step: int | None = None) -> list[Fold]:
    """Create expanding-window folds whose test windows tile the end of the data.

    The last fold's test window ends on the last available week; each earlier fold's
    origin is ``step`` weeks (default: ``horizon``) earlier.
    """
    step = horizon if step is None else step
    weeks = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(weeks))))
    n = len(weeks)
    folds = []
    for k in range(n_folds):
        origin_idx = n - horizon - step * (n_folds - 1 - k) - 1
        if origin_idx < 1:
            raise ValueError("Not enough history for the requested folds")
        test = weeks[origin_idx + 1 : origin_idx + 1 + horizon]
        folds.append(Fold(k, weeks[origin_idx], tuple(test)))
    return folds


def split(panel: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (history, future covariates without target, actuals) for one fold."""
    history = panel[panel["week"] <= fold.train_end]
    known = history["series_id"].unique()
    test = panel[panel["week"].isin(fold.test_weeks) & panel["series_id"].isin(known)]
    test = test.sort_values(KEYS).reset_index(drop=True)
    horizon = {w: h + 1 for h, w in enumerate(fold.test_weeks)}
    future = test.drop(columns=[TARGET])
    future.insert(2, "horizon", future["week"].map(horizon).astype(int))
    actuals = test[KEYS + [TARGET]].copy()
    return history.reset_index(drop=True), future, actuals


def run_backtest(
    panel: pd.DataFrame,
    models: dict[str, Callable[[], Forecaster]],
    folds: Sequence[Fold],
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit every model on every fold and return a long table of forecasts vs actuals."""
    out = []
    for fold in folds:
        history, future, actuals = split(panel, fold)
        for name, factory in models.items():
            if verbose:
                print(f"fold {fold.fold} ({fold.train_end.date()}): {name}", flush=True)
            model = factory().fit(history)
            yhat = np.asarray(model.predict(future.copy()), dtype=float)
            if yhat.shape != (len(future),):
                raise ValueError(f"{name} returned shape {yhat.shape}, expected {len(future)}")
            res = future[KEYS + ["horizon"]].copy()
            res["fold"] = fold.fold
            res["model"] = name
            res["y"] = actuals[TARGET].to_numpy()
            res["yhat"] = np.clip(yhat, 0, None)
            out.append(res)
    return pd.concat(out, ignore_index=True)
