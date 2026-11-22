"""Common contract for all groundwater level models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

InputKind = Literal["sequence", "tabular"]


class BaseGroundwaterLevelModel(ABC):
    """Unified interface: every model predicts groundwater level (scaled target).

    ``input_kind`` tells the training pipeline which array to feed:
    ``"sequence"`` -> ``SplitData.X_seq`` (n, lookback, features),
    ``"tabular"``  -> ``SplitData.X_tab`` (n, features).
    Predictions are returned in the same (scaled) space as ``y``; use
    ``PreparedData.inverse_transform_target`` to get original units.
    """

    name: ClassVar[str]
    input_kind: ClassVar[InputKind]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseGroundwaterLevelModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of shape (n,)."""

    @abstractmethod
    def save(self, artifact_dir: str | Path) -> None: ...

    @abstractmethod
    def load(self, artifact_dir: str | Path) -> "BaseGroundwaterLevelModel": ...

    # ---- helpers -----------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} model is not fitted or loaded")

    @staticmethod
    def _prep_dir(artifact_dir: str | Path) -> Path:
        p = Path(artifact_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
