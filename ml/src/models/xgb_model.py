"""XGBoost regressor for groundwater level prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor

from .base_model import BaseGroundwaterLevelModel


class XGBoostGroundwaterModel(BaseGroundwaterLevelModel):
    name = "xgb"
    input_kind = "tabular"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        # early_stopping_rounds is a constructor arg in xgboost >= 2.0
        self.model = XGBRegressor(objective="reg:squarederror", **self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        kwargs = {}
        if X_val is not None and y_val is not None:
            kwargs["eval_set"] = [(X_val, y_val)]
            kwargs["verbose"] = False
        else:
            # early stopping needs an eval set; disable it when none is given
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).astype(np.float32)

    def save(self, artifact_dir) -> None:
        self._check_fitted()
        d = self._prep_dir(artifact_dir)
        self.model.save_model(str(d / "xgb_model.json"))
        joblib.dump({"params": self.params}, d / "xgb_meta.joblib")

    def load(self, artifact_dir):
        d = Path(artifact_dir)
        self.params = joblib.load(d / "xgb_meta.joblib")["params"]
        self.model = XGBRegressor(objective="reg:squarederror", **self.params)
        self.model.load_model(str(d / "xgb_model.json"))
        self.is_fitted = True
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_fitted()
        return self.model.feature_importances_
