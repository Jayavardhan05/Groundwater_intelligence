"""Inference service: loads a trained groundwater level model for FastAPI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data_loader import GroundwaterDataLoader, load_config
from .factory import create_model, get_inputs


class GroundwaterLevelPredictor:
    """Loads artifacts once; call ``predict_next`` per request.

    ``model_name=None`` uses the model recorded in ``best_model.json``.
    """

    def __init__(self, artifact_dir: str | Path = "ml/artifacts", model_name: str | None = None) -> None:
        self.dir = Path(artifact_dir)
        if model_name is None:
            model_name = json.loads((self.dir / "best_model.json").read_text())["best_model"]
        self.model_name = model_name

        self.cfg = load_config(self.dir / "model_config.yaml")
        pre = joblib.load(self.dir / "data_preprocessing.joblib")
        self.scaler = pre["input_scaler"]
        self.scale_target: bool = pre["scale_target"]
        self.target_idx: int = pre["target_idx"]
        self.loader = GroundwaterDataLoader(self.cfg)
        self.model = create_model(model_name, self.cfg).load(self.dir / model_name)

    def _to_original(self, y):
        if not self.scale_target:
            return y
        return y * self.scaler.scale_[self.target_idx] + self.scaler.mean_[self.target_idx]

    def predict_next(self, history: pd.DataFrame) -> dict[str, Any]:
        """Forecast one step (``horizon``) ahead for every well in ``history``.

        ``history`` needs the configured well/time/target/covariate columns with
        enough recent rows per well (see ``required_history_steps``).
        """
        samples, skipped = self.loader.build_inference_samples(history, self.scaler)
        meta = samples["meta"]
        preds = []
        if len(meta):
            kind_x = samples["X_seq"] if self.model.input_kind == "sequence" else samples["X_tab"]
            values = self._to_original(self.model.predict(kind_x))
            wells = meta[self.loader.well_col].astype(str).tolist()
            stamps = pd.to_datetime(meta["target_time"]).tolist()
            for wid, ts, v in zip(wells, stamps, values):
                preds.append({
                    "well_id": wid,
                    "forecast_timestamp": pd.Timestamp(ts).isoformat(),
                    "predicted_groundwater_level": float(v),
                    "model_name": self.model_name,
                })
        return {"predictions": preds, "skipped_wells": skipped}

    @property
    def required_history_steps(self) -> int:
        return self.loader._history + 1
