"""Registry / factory for groundwater level models.

LSTM is imported lazily so RF/XGBoost work on machines without PyTorch.
"""
from __future__ import annotations

from typing import Any, Callable

from .data_loader import SplitData
from .models.base_model import BaseGroundwaterLevelModel
import numpy as np


def _rf(params): 
    from .models.rf_model import RandomForestGroundwaterModel
    return RandomForestGroundwaterModel(params)


def _xgb(params):
    from .models.xgb_model import XGBoostGroundwaterModel
    return XGBoostGroundwaterModel(params)


def _lstm(params):
    from .models.lstm_model import LSTMGroundwaterModel
    return LSTMGroundwaterModel(params)


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any]], BaseGroundwaterLevelModel]] = {
    "lstm": _lstm,
    "rf": _rf,
    "xgb": _xgb,
}


def register_model(name: str, builder: Callable[[dict[str, Any]], BaseGroundwaterLevelModel]) -> None:
    MODEL_REGISTRY[name] = builder


def create_model(name: str, config: dict[str, Any]) -> BaseGroundwaterLevelModel:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](dict(config.get("models", {}).get(key, {})))


def get_inputs(split: SplitData, model: BaseGroundwaterLevelModel) -> np.ndarray:
    """Pick the right array view for a model's input_kind."""
    return split.X_seq if model.input_kind == "sequence" else split.X_tab
