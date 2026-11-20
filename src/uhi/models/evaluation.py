
"""Evaluation utilities for UHI machine learning."""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
    }


def evaluate_mean_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Evaluate the training-mean baseline."""

    model = DummyRegressor(strategy="mean")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    return model, metrics


def spatial_cross_validation(
    model,
    X,
    y,
    groups,
    n_splits=5,
):
    """Evaluate a model using GroupKFold spatial CV."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Only {groups.nunique()} groups available "
            f"for {n_splits} folds."
        )

    cv = GroupKFold(n_splits=n_splits)

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups=groups),
        start=1,
    ):

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        fold_model = clone(model)

        fold_model.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = fold_model.predict(X_val_fold)

        metrics = calculate_metrics(
            y_val_fold,
            predictions,
        )

        fold_result = {
            "fold": fold,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "train_blocks": groups.iloc[train_idx].nunique(),
            "validation_blocks": groups.iloc[val_idx].nunique(),
            **metrics,
        }

        results.append(fold_result)

        print(
            f"Fold {fold}: "
            f"R2={metrics['r2']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, "
            f"MAE={metrics['mae']:.3f}"
        )

    results_df = pd.DataFrame(results)

    summary = results_df[
        ["r2", "rmse", "mae"]
    ].agg(["mean", "std"])

    return results_df, summary