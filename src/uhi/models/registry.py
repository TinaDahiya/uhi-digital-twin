
"""Central registry for supported machine learning models."""

from uhi.models.random_forest import build_random_forest
from uhi.models.xgboost_model import build_xgboost
from uhi.models.gradient_boosting import build_gradient_boosting

MODEL_REGISTRY = {
    "rf": build_random_forest,
    "xgboost": build_xgboost,
}


def get_model(model_name, **model_params):
    """Construct a model using its registered builder."""

    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    return MODEL_REGISTRY[model_name](**model_params)


"""Central registry for supported machine learning models."""

from uhi.models.random_forest import build_random_forest
from uhi.models.xgboost_model import build_xgboost
from uhi.models.lightgbm_model import build_lightgbm


MODEL_REGISTRY = {
    "rf": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "gradient_boosting": build_gradient_boosting,
}


def get_model(model_name, **model_params):
    """Construct a model using its registered builder."""

    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    return MODEL_REGISTRY[model_name](**model_params)
