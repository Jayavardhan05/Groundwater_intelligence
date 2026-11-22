"""End-to-end: train -> benchmark -> artifacts -> inference parity."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.src.data_loader import GroundwaterDataLoader, load_config
from ml.src.evaluate import compute_metrics, nse
from ml.src.inference import GroundwaterLevelPredictor
from ml.src.train import run_training
from ml.tests.test_data_loader import _synthetic

CFG_PATH = "ml/tests/test_config.yaml"


def test_metric_sanity():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_metrics(obs, obs)["rmse"] == 0 and nse(obs, obs) == 1.0
    assert nse(obs, np.full(4, obs.mean())) == pytest.approx(0.0)       # mean predictor
    assert nse(obs, obs[::-1]) < 0                                       # worse than mean


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    cfg = load_config(CFG_PATH)
    cfg["models"]["rf"].update(n_estimators=30)
    cfg["models"]["xgb"].update(n_estimators=60, early_stopping_rounds=10)
    csv = _synthetic(tmp_path_factory.mktemp("data"), n_wells=4, n=150)
    art = tmp_path_factory.mktemp("artifacts")
    table = run_training(cfg, ["rf", "xgb"], str(csv), art, CFG_PATH)
    return cfg, csv, art, table


def test_artifacts_and_benchmark(trained):
    _, _, art, table = trained
    for f in ("benchmark.csv", "evaluation_report.json", "best_model.json",
              "data_preprocessing.joblib", "model_config.yaml", "rf/metrics.json", "xgb/xgb_model.json"):
        assert (art / f).exists(), f
    assert {"test_rmse", "test_nse", "val_rmse"} <= set(table.columns)
    assert json.loads((art / "best_model.json").read_text())["best_model"] in ("rf", "xgb")


def test_all_models_have_good_nse_on_autoregressive_data(trained):
    _, _, _, table = trained
    assert (table["test_nse"] > 0.5).all()


@pytest.mark.parametrize("name", ["rf", "xgb"])
def test_inference_matches_batch_pipeline(trained, name):
    """Predicting from truncated history must equal the batch prediction for that origin."""
    cfg, csv, art, _ = trained
    data = GroundwaterDataLoader(cfg).prepare(csv)
    predictor = GroundwaterLevelPredictor(art, model_name=name)

    test = data.splits["test"]
    row = test.meta.iloc[5]
    raw = pd.read_csv(csv, parse_dates=["timestamp"])
    hist = raw[(raw.well_id == row.well_id) & (raw.timestamp <= row.origin_time)]

    out = predictor.predict_next(hist)
    assert out["skipped_wells"] == [] and len(out["predictions"]) == 1
    p = out["predictions"][0]
    assert p["well_id"] == row.well_id and p["model_name"] == name
    assert pd.Timestamp(p["forecast_timestamp"]) == row.target_time

    batch = data.inverse_transform_target(predictor.model.predict(test.X_tab[[5]]))[0]
    assert p["predicted_groundwater_level"] == pytest.approx(float(batch), rel=1e-4, abs=1e-4)


def test_default_model_is_best_and_short_history_skipped(trained):
    _, csv, art, _ = trained
    predictor = GroundwaterLevelPredictor(art)
    best = json.loads((art / "best_model.json").read_text())["best_model"]
    assert predictor.model_name == best
    raw = pd.read_csv(csv, parse_dates=["timestamp"])
    short = raw[raw.well_id == "W0"].head(3)
    out = predictor.predict_next(short)
    assert out["predictions"] == [] and out["skipped_wells"] == ["W0"]
