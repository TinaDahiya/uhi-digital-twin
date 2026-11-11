
"""Gradient Boosting model for SUHII prediction."""

from sklearn.ensemble import GradientBoostingRegressor


def build_gradient_boosting(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    subsample=0.8,
    max_features=None,
    random_state=42,
):
    """Create a Gradient Boosting regression model."""

    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state,
    )


def train_gradient_boosting(
    X_train,
    y_train,
    **model_params,
):
    """Train Gradient Boosting using supplied data."""

    model = build_gradient_boosting(**model_params)
    model.fit(X_train, y_train)

    return model