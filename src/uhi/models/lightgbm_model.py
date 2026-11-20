
"""LightGBM model for SUHII prediction."""

from lightgbm import LGBMRegressor


def build_lightgbm(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
):
    """Create a LightGBM regression model."""

    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def train_lightgbm(
    X_train,
    y_train,
    **model_params,
):
    """Train LightGBM using supplied training data."""

    model = build_lightgbm(**model_params)
    model.fit(X_train, y_train)

    return model