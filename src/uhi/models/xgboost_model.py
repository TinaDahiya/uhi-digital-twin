
"""XGBoost model for SUHII prediction."""

from xgboost import XGBRegressor


def build_xgboost(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
):
    """Create an XGBoost regression model."""

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        objective="reg:squarederror",
    )


def train_xgboost(
    X_train,
    y_train,
    **model_params,
):
    """Train XGBoost using supplied training data."""

    model = build_xgboost(**model_params)
    model.fit(X_train, y_train)

    return model
