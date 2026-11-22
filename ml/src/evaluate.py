"""Evaluation metrics and multi-model benchmarking for groundwater level models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import PreparedData, SplitData
from .factory import get_inputs
from .models.base_model import BaseGroundwaterLevelModel


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.abs(obs - sim)))


def r2(obs: np.ndarray, sim: np.ndarray) -> float:
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency: 1 is perfect, 0 equals predicting the observed mean,
    negative is worse than that mean. (Algebraically the same as R2 on the same data,
    but kept separate because NSE is the standard hydrology term.)"""
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else float("nan")


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    return {"rmse": rmse(obs, sim), "mae": mae(obs, sim), "r2": r2(obs, sim), "nse": nse(obs, sim)}


def predict_original_units(model: BaseGroundwaterLevelModel, data: PreparedData, split: SplitData) -> np.ndarray:
    return data.inverse_transform_target(model.predict(get_inputs(split, model)))


def evaluate_model(
    model: BaseGroundwaterLevelModel, data: PreparedData, split_name: str = "test", per_well: bool = False
) -> dict:
    """Metrics in ORIGINAL units (e.g. metres), not scaled units."""
    split = data.splits[split_name]
    pred = predict_original_units(model, data, split)
    out: dict = {"model": model.name, "split": split_name, "n_samples": len(split),
                 **compute_metrics(split.y_raw, pred)}
    if per_well:
        well_col = split.meta.columns[0]
        rows = []
        for wid, idx in split.meta.groupby(well_col).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) >= 2:
                rows.append({"well_id": wid, "n": len(idx), **compute_metrics(split.y_raw[idx], pred[idx])})
        out["per_well"] = rows
    return out


def benchmark(results: list[dict]) -> pd.DataFrame:
    """One row per (model, split); sorted by test RMSE where available."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_well"} for r in results])
    wide = df.pivot(index="model", columns="split", values=["rmse", "mae", "r2", "nse"])
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    sort_col = "test_rmse" if "test_rmse" in wide else wide.columns[0]
    return wide.sort_values(sort_col)


def save_report(results: list[dict], table: pd.DataFrame, artifacts_dir: str | Path) -> None:
    d = Path(artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    table.to_csv(d / "benchmark.csv")
    (d / "evaluation_report.json").write_text(json.dumps(results, indent=2, default=float))
