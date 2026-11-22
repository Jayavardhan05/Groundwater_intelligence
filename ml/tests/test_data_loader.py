import numpy as np
import pandas as pd
import pytest

from ml.src.data_loader import GroundwaterDataLoader, load_config

CFG_PATH = "ml/tests/test_config.yaml"


def _synthetic(tmp_path, n_wells=4, n=120):
    rng = np.random.default_rng(0)
    rows = []
    for w in range(n_wells):
        t = pd.date_range("2010-01-01", periods=n, freq="MS")
        season = np.sin(2 * np.pi * t.month / 12)
        rain = rng.gamma(2, 20, n)
        level = 10 + w + 2 * season - 0.01 * rain + np.cumsum(rng.normal(0, 0.05, n))
        rows.append(pd.DataFrame({
            "well_id": f"W{w}", "timestamp": t, "gw_level": level, "rainfall": rain,
            "temperature": 25 + 5 * season, "evapotranspiration": rng.normal(3, .5, n),
            "recharge": rain * 0.1, "pumping": rng.normal(1, .1, n)}))
    df = pd.concat(rows)
    p = tmp_path / "gw.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def prepared(tmp_path):
    cfg = load_config(CFG_PATH)
    return GroundwaterDataLoader(cfg).prepare(_synthetic(tmp_path)), cfg


def test_shapes_aligned(prepared):
    data, cfg = prepared
    L = cfg["sequence"]["lookback"]
    for s in data.splits.values():
        assert s.X_seq.shape[1] == L
        assert len(s.X_seq) == len(s.X_tab) == len(s.y) == len(s.meta)
        assert not np.isnan(s.X_seq).any() and not np.isnan(s.X_tab).any()
    assert data.splits["train"].X_tab.shape[1] == len(data.tab_feature_names)


def test_temporal_no_leakage(prepared):
    data, _ = prepared
    tr, va, te = (data.splits[k].meta for k in ("train", "val", "test"))
    for wid in tr["well_id"].unique():
        a = tr[tr.well_id == wid].target_time.max()
        b = va[va.well_id == wid].target_time
        c = te[te.well_id == wid].target_time
        assert a < b.min() and b.max() < c.min()


def test_inputs_strictly_past_of_target(prepared):
    data, cfg = prepared
    h = cfg["sequence"]["horizon"]
    for s in data.splits.values():
        assert (s.meta.target_time > s.meta.origin_time).all()


def test_target_inverse_transform(prepared):
    data, _ = prepared
    s = data.splits["val"]
    np.testing.assert_allclose(data.inverse_transform_target(s.y), s.y_raw, rtol=1e-4, atol=1e-4)


def test_well_split_holds_out_whole_wells(tmp_path):
    cfg = load_config(CFG_PATH)
    cfg["split"]["strategy"] = "well"
    data = GroundwaterDataLoader(cfg).prepare(_synthetic(tmp_path, n_wells=10))
    sets = {k: set(v.meta.well_id) for k, v in data.splits.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
