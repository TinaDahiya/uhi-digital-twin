
"""Random Forest model for SUHII prediction."""

from sklearn.ensemble import RandomForestRegressor


def build_random_forest(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
):
    """Create a Random Forest regression model."""

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_random_forest(
    X_train,
    y_train,
    **model_params,
):
    """Train Random Forest using supplied training data."""

    model = build_random_forest(**model_params)

    model.fit(X_train, y_train)

    return model
