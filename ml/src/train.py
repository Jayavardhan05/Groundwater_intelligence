"""Unified training CLI.

    python -m ml.src.train --model rf
    python -m ml.src.train --model lstm
    python -m ml.src.train --model all --data data/processed/groundwater_levels.parquet

Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from .data_loader import GroundwaterDataLoader, load_config
from .evaluate import benchmark, evaluate_model, save_report
from .factory import MODEL_REGISTRY, create_model, get_inputs

log = logging.getLogger("ml.train")


def run_training(
    cfg: dict, model_names: list[str], data_path: str | None = None,
    artifacts_dir: str | Path | None = None, config_path: str | Path | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    artifacts_dir = Path(artifacts_dir or cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = GroundwaterDataLoader(cfg).prepare(data_path)
    data.save_scalers(artifacts_dir)
    (artifacts_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2, default=str))
    if config_path:
        shutil.copy(config_path, artifacts_dir / "model_config.yaml")
    log.info("samples: %s", {k: len(v) for k, v in data.splits.items()})

    results: list[dict] = []
    for name in model_names:
        try:
            model = create_model(name, cfg)
        except (ImportError, OSError) as exc:   # e.g. PyTorch not installed
            if strict:
                raise
            log.warning("Skipping %s: %s", name, exc)
            continue

        tr, va = data.splits["train"], data.splits["val"]
        log.info("training %s", name)
        model.fit(get_inputs(tr, model), tr.y, get_inputs(va, model), va.y)
        model.save(artifacts_dir / name)

        per_model = []
        for split in ("train", "val", "test"):
            r = evaluate_model(model, data, split, per_well=(split == "test"))
            per_model.append(r)
            results.append(r)
        (artifacts_dir / name / "metrics.json").write_text(json.dumps(per_model, indent=2, default=float))
        log.info("%s test: %s", name, {k: round(v, 4) for k, v in per_model[-1].items() if k in ("rmse", "mae", "r2", "nse")})

    if not results:
        raise RuntimeError("No models were trained")

    table = benchmark(results)
    save_report(results, table, artifacts_dir)

    # Best model chosen on VALIDATION rmse so the test set is not used for selection.
    best = table["val_rmse"].idxmin()
    (artifacts_dir / "best_model.json").write_text(
        json.dumps({"best_model": best, "selected_by": "val_rmse", "val_rmse": float(table.loc[best, "val_rmse"])}, indent=2)
    )
    log.info("benchmark:\n%s", table.round(4).to_string())
    log.info("best model (val RMSE): %s", best)
    return table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train groundwater level models")
    ap.add_argument("--model", default="all", help=f"one of {sorted(MODEL_REGISTRY)} or 'all'")
    ap.add_argument("--config", default="ml/configs/model_config.yaml")
    ap.add_argument("--data", default=None, help="override data.processed_path")
    ap.add_argument("--artifacts-dir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    names = sorted(MODEL_REGISTRY) if args.model == "all" else [args.model]
    run_training(cfg, names, args.data, args.artifacts_dir, args.config, strict=(args.model != "all"))


if __name__ == "__main__":
    main()
