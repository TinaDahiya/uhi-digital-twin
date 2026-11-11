"""Reusable model training utilities."""

from uhi.models.registry import get_model


def train_model(
    model_name,
    X_train,
    y_train,
    model_params=None,
):
    """Build and train a selected model."""

    if model_params is None:
        model_params = {}

    model = get_model(
        model_name,
        **model_params,
    )

    model.fit(X_train, y_train)

    return model