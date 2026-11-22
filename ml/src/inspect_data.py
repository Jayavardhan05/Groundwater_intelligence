"""Check how regular your well records are BEFORE choosing `data.freq`.

    python -m ml.src.inspect_data [--config ml/configs/model_config.yaml] [--data path]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data_loader import GroundwaterDataLoader, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/model_config.yaml")
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)

    loader = GroundwaterDataLoader(load_config(args.config))
    path = args.data or loader.cfg["data"]["processed_path"]
    raw = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)
    df = loader.load_raw(args.data)
    t, w = loader.time_col, loader.well_col

    print(f"rows in file: {len(raw):,} | rows kept after filters: {len(df):,} "
          f"({100 * len(df) / max(len(raw), 1):.1f}%)")
    print(f"wells: {df[w].nunique():,} | dates: {df[t].min().date()} -> {df[t].max().date()}")

    df = df.sort_values([w, t]).drop_duplicates([w, t])
    per_well = df.groupby(w)[t].count()
    print("\nobservations per well:", per_well.describe().round(1)[["min", "25%", "50%", "75%", "max"]].to_dict())

    gaps = df.groupby(w)[t].diff().dt.days.dropna()
    q = gaps.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict()
    print("days between consecutive readings (quantiles):", q)

    print("\nreadings per calendar month (all wells):")
    print(df[t].dt.month.value_counts().sort_index().to_string())

    med = float(gaps.median()) if len(gaps) else float("nan")
    if med <= 2:
        hint = "daily data -> freq: D"
    elif 25 <= med <= 35:
        hint = "monthly data -> freq: MS"
    elif 80 <= med <= 100:
        hint = "quarterly-ish (e.g. 4 readings/year). A fixed monthly grid will leave most steps empty."
    else:
        hint = "irregular or seasonal spacing. A fixed pandas freq may not fit."
    print(f"\nmedian gap: {med} days -> {hint}")
    ok = (per_well >= loader.min_rows).sum()
    print(f"wells with >= min_rows_per_well ({loader.min_rows}) readings: {ok:,} / {len(per_well):,}")


if __name__ == "__main__":
    main()
