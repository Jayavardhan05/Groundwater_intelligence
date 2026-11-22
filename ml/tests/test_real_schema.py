"""Pipeline on a CSV that uses the real column names, quality flags and static well attributes."""
import numpy as np
import pandas as pd
import pytest

from ml.src.data_loader import GroundwaterDataLoader, load_config
from ml.src.inference import GroundwaterLevelPredictor
from ml.src.train import run_training

CFG = "ml/configs/model_config.yaml"


def _csv(tmp_path, n_wells=4, n=120):
    rng = np.random.default_rng(1)
    parts = []
    for w in range(n_wells):
        t = pd.date_range("2005-01-01", periods=n, freq="MS")
        lvl = 8 + w + 2 * np.sin(2 * np.pi * t.month / 12) + np.cumsum(rng.normal(0, .05, n))
        df = pd.DataFrame({
            "_id": range(n), "Well ID": 10001 + w, "Site Name": f"Site{w}",
            "Latitude": 18.4 + w * .1, "Longitude": 83.9 + w * .1, "Mode": "manual",
            "Date_original": t.strftime("%d-%m-%Y"), "Water Level": lvl.round(2),
            "Water_Level_numeric": lvl.round(2), "Well_Depth_numeric": 31.0 + w,
            "Measurement_Status": "numeric_measurement",
            "Flag_Negative_Level": False, "Flag_Level_Over_100m": False,
            "Flag_Level_Outside_0_100m": False, "Flag_Level_Exceeds_Well_Depth": False,
            "Date": t.strftime("%Y-%m-%d"),
        })
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # inject bad rows that must be filtered out
    df.loc[10, ["Water_Level_numeric", "Flag_Level_Over_100m", "Flag_Level_Outside_0_100m"]] = [500.0, True, True]
    df.loc[20, ["Water_Level_numeric", "Flag_Negative_Level", "Flag_Level_Outside_0_100m"]] = [-3.0, True, True]
    df.loc[30, ["Water_Level_numeric", "Measurement_Status"]] = [np.nan, "no_measurement"]
    df.loc[40, ["Water_Level_numeric", "Flag_Level_Exceeds_Well_Depth"]] = [90.0, True]
    p = tmp_path / "groundwater_levels.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def cfg():
    c = load_config(CFG)
    c["models"]["rf"].update(n_estimators=30)
    c["models"]["xgb"].update(n_estimators=60, early_stopping_rounds=10)
    return c


def test_flagged_rows_removed(tmp_path, cfg):
    loader = GroundwaterDataLoader(cfg)
    df = loader.load_raw(_csv(tmp_path))
    assert df[loader.target_col].between(0, 100).all()
    assert len(df) == 4 * 120 - 4


def test_static_cols_are_inputs_without_lags(tmp_path, cfg):
    data = GroundwaterDataLoader(cfg).prepare(_csv(tmp_path))
    names = data.tab_feature_names
    for c in ("Latitude", "Longitude", "Well_Depth_numeric"):
        assert c in names and f"{c}_lag1" not in names
    assert "Latitude" in data.seq_feature_names
    assert data.splits["train"].y_raw.max() < 100


def test_train_and_infer_on_real_schema(tmp_path, cfg):
    csv, art = _csv(tmp_path), tmp_path / "art"
    table = run_training(cfg, ["rf", "xgb"], str(csv), art, CFG)
    assert (table["test_nse"] > 0.5).all()
    raw = pd.read_csv(csv)
    hist = raw[raw["Well ID"] == 10002]
    out = GroundwaterLevelPredictor(art).predict_next(hist)
    assert len(out["predictions"]) == 1 and out["predictions"][0]["well_id"] == "10002"
    assert 0 < out["predictions"][0]["predicted_groundwater_level"] < 100
