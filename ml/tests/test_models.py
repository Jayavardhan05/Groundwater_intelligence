"""Shape, learning and save/load parity checks for all three models."""
import numpy as np
import pytest

from ml.src.data_loader import GroundwaterDataLoader, load_config
from ml.src.factory import create_model, get_inputs, MODEL_REGISTRY
from ml.tests.test_data_loader import _synthetic

CFG_PATH = "ml/tests/test_config.yaml"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


MODELS = ["rf", "xgb", pytest.param("lstm", marks=pytest.mark.skipif(not _torch_available(), reason="torch not installed"))]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_config(CFG_PATH)
    # keep tests fast
    cfg["models"]["rf"].update(n_estimators=30)
    cfg["models"]["xgb"].update(n_estimators=60, early_stopping_rounds=10)
    cfg["models"]["lstm"].update(epochs=5, hidden_size=16, patience=3)
    path = _synthetic(tmp_path_factory.mktemp("d"), n_wells=4, n=150)
    return cfg, GroundwaterDataLoader(cfg).prepare(path)


def test_registry_has_all_three():
    assert {"lstm", "rf", "xgb"} <= set(MODEL_REGISTRY)


def test_unknown_model_raises(setup):
    with pytest.raises(KeyError):
        create_model("nope", setup[0])


@pytest.mark.parametrize("name", MODELS)
def test_fit_predict_save_load_parity(name, setup, tmp_path):
    cfg, data = setup
    model = create_model(name, cfg)
    tr, va, te = (data.splits[k] for k in ("train", "val", "test"))
    model.fit(get_inputs(tr, model), tr.y, get_inputs(va, model), va.y)

    pred = model.predict(get_inputs(te, model))
    assert pred.shape == (len(te),) and np.isfinite(pred).all()

    model.save(tmp_path)
    reloaded = create_model(name, cfg).load(tmp_path)
    np.testing.assert_allclose(reloaded.predict(get_inputs(te, model)), pred, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["rf", "xgb"])
def test_beats_mean_baseline(name, setup):
    """Synthetic series is strongly autoregressive; trees must beat predicting the train mean."""
    cfg, data = setup
    model = create_model(name, cfg)
    tr, va, te = (data.splits[k] for k in ("train", "val", "test"))
    model.fit(tr.X_tab, tr.y, va.X_tab, va.y)
    mse_model = np.mean((model.predict(te.X_tab) - te.y) ** 2)
    mse_base = np.mean((tr.y.mean() - te.y) ** 2)
    assert mse_model < mse_base


def test_fit_without_validation_set(setup):
    cfg, data = setup
    tr = data.splits["train"]
    for name in ("rf", "xgb"):
        m = create_model(name, cfg).fit(tr.X_tab, tr.y)
        assert m.predict(tr.X_tab[:5]).shape == (5,)
