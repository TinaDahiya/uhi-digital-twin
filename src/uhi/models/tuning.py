
"""Spatial hyperparameter tuning for Random Forest."""

import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
)

from uhi.models.random_forest import build_random_forest


def tune_random_forest(
    X_train,
    y_train,
    groups,
    n_iter=25,
    n_splits=5,
    random_state=42,
):
    """Tune Random Forest using geographically grouped CV."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Only {groups.nunique()} groups available "
            f"for {n_splits} folds."
        )

    param_distributions = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [0.5, 0.7, 1.0],
    }

    cv = GroupKFold(n_splits=n_splits)

    model = build_random_forest(
        random_state=random_state,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        refit="rmse",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    best_index = search.best_index_

    best_cv_metrics = {
        "mean_r2": float(
            search.cv_results_["mean_test_r2"][best_index]
        ),
        "mean_rmse": float(
            -search.cv_results_["mean_test_rmse"][best_index]
        ),
        "mean_mae": float(
            -search.cv_results_["mean_test_mae"][best_index]
        ),
        "std_rmse": float(
            search.cv_results_["std_test_rmse"][best_index]
        ),
    }

    results_df = pd.DataFrame(search.cv_results_)

    return (
        search,
        search.best_estimator_,
        best_cv_metrics,
        results_df,
    )