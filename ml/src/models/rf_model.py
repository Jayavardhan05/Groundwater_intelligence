"""Random Forest regressor for groundwater level prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseGroundwaterLevelModel


class RandomForestGroundwaterModel(BaseGroundwaterLevelModel):
    name = "rf"
    input_kind = "tabular"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # RF has no early stopping; validation data is accepted for interface parity.
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).astype(np.float32)

    def save(self, artifact_dir) -> None:
        self._check_fitted()
        d = self._prep_dir(artifact_dir)
        joblib.dump({"model": self.model, "params": self.params}, d / "rf_model.joblib")

    def load(self, artifact_dir):
        blob = joblib.load(Path(artifact_dir) / "rf_model.joblib")
        self.model, self.params = blob["model"], blob["params"]
        self.is_fitted = True
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_fitted()
        return self.model.feature_importances_
