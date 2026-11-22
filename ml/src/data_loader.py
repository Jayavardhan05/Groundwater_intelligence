"""Data loading for groundwater level forecasting.

Builds two views of the same samples so LSTM, Random Forest and XGBoost are
trained and compared on *identical* rows:

* ``X_seq``: 3D sliding windows ``(samples, lookback, n_inputs)`` for the LSTM
* ``X_tab``: 2D lagged matrix ``(samples, n_features)`` for RF / XGBoost

Leakage rules enforced here:
1. Every well is processed independently, on its own regular time grid, so
   windows never cross well boundaries or gaps.
2. Splits are chronological per well (or by whole wells). A sample belongs to
   a split by the time of its *target*; inputs are always strictly in the past,
   so validation/test windows may look back into earlier data but never ahead.
3. Scalers are fit on training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "val", "test")


def load_config(path: str | Path = "ml/configs/model_config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SplitData:
    """Samples for one split. ``meta`` aligns row-by-row with the arrays."""

    X_seq: np.ndarray            # (n, lookback, n_inputs)
    X_tab: np.ndarray            # (n, n_tab_features)
    y: np.ndarray                # (n,) scaled if scaling.scale_target else raw
    y_raw: np.ndarray            # (n,) always in original units
    meta: pd.DataFrame           # well_id, origin_time, target_time

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class PreparedData:
    splits: dict[str, SplitData]
    seq_feature_names: list[str]
    tab_feature_names: list[str]
    input_scaler: StandardScaler           # fit on train rows of [target, *features]
    target_idx: int = 0                    # column of the target inside input_scaler
    scale_target: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    # ---- target scaling helpers -------------------------------------------------
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self.scale_target:
            return np.asarray(y_scaled)
        s = self.input_scaler
        return np.asarray(y_scaled) * s.scale_[self.target_idx] + s.mean_[self.target_idx]

    # ---- persistence ------------------------------------------------------------
    def save_scalers(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "input_scaler": self.input_scaler,
                "seq_feature_names": self.seq_feature_names,
                "tab_feature_names": self.tab_feature_names,
                "scale_target": self.scale_target,
                "target_idx": self.target_idx,
            },
            artifact_dir / "data_preprocessing.joblib",
        )


class GroundwaterDataLoader:
    """Loads processed well time series and produces leakage-safe splits."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        d, s, t = config["data"], config["sequence"], config["tabular"]
        self.well_col: str = d["well_col"]
        self.time_col: str = d["time_col"]
        self.target_col: str = d["target_col"]
        # dynamic covariates change over time; static ones (lat/lon, well depth) are constant per well
        self.static_cols: list[str] = list(d.get("static_cols") or [])
        dynamic = [c for c in (d.get("feature_cols") or []) if c not in self.static_cols]
        self.feature_cols: list[str] = dynamic + self.static_cols
        # row-quality filters applied when loading raw data
        self.keep_where: dict[str, list] = dict(d.get("keep_where") or {})
        self.drop_where_true: list[str] = list(d.get("drop_where_true") or [])
        self.freq: str = d["freq"]
        self.max_gap: int = int(d.get("max_interp_gap", 0))
        self.min_rows: int = int(d.get("min_rows_per_well", 0))
        self.lookback: int = int(s["lookback"])
        self.horizon: int = int(s["horizon"])
        self.use_target_input: bool = bool(s.get("include_target_as_input", True))
        self.target_lags: list[int] = sorted(t["target_lags"])
        self.feature_lags: list[int] = sorted(t.get("feature_lags", []))
        self.rolling: list[int] = sorted(t.get("rolling_windows", []))
        self.seasonal: bool = bool(t.get("seasonal_features", False))
        if self.horizon < 1 or self.lookback < 1:
            raise ValueError("horizon and lookback must be >= 1")

    # ------------------------------------------------------------------ loading
    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        path = Path(path or self.cfg["data"]["processed_path"])
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        needed = {self.well_col, self.time_col, self.target_col, *self.feature_cols,
                  *self.keep_where, *self.drop_where_true}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
        df = self._apply_filters(df)
        df = df[[self.well_col, self.time_col, self.target_col, *self.feature_cols]].copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        for c in [self.target_col, *self.feature_cols]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[self.time_col, self.target_col])
        if df.empty:
            raise ValueError("No rows left after filtering - check keep_where / drop_where_true")
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """keep_where: column value must be in the allowed list (case-insensitive).
        drop_where_true: drop rows where a flag column is True."""
        mask = pd.Series(True, index=df.index)
        for col, allowed in self.keep_where.items():
            allowed_l = {str(a).strip().lower() for a in allowed}
            mask &= df[col].astype(str).str.strip().str.lower().isin(allowed_l)
        for col in self.drop_where_true:
            mask &= ~(df[col].astype(str).str.strip().str.lower() == "true")
        return df[mask]

    def _regularise(self, df: pd.DataFrame, min_rows: int | None = None) -> dict[str, pd.DataFrame]:
        """Per-well regular grid; short gaps interpolated, long gaps left as NaN."""
        min_rows = self.min_rows if min_rows is None else min_rows
        wells: dict[str, pd.DataFrame] = {}
        cols = [self.target_col, *self.feature_cols]
        for wid, g in df.groupby(self.well_col, sort=True):
            g = g.sort_values(self.time_col).drop_duplicates(self.time_col, keep="last")
            g = g.set_index(self.time_col)[cols]
            g = g.resample(self.freq).mean()  # regular grid, NaN where missing
            for c in self.static_cols:        # constant per well: fill the new grid rows
                g[c] = g[c].ffill().bfill()
            if self.max_gap > 0:
                g = g.interpolate(method="time", limit=self.max_gap, limit_area="inside")
            if g[self.target_col].notna().sum() >= min_rows:
                wells[str(wid)] = g
        if not wells:
            raise ValueError("No wells left after min_rows filtering")
        return wells

    # ---------------------------------------------------------------- splitting
    def _well_split_bounds(self, n: int) -> dict[str, tuple[int, int]]:
        """Target-index ranges [lo, hi) per split for one well (temporal strategy)."""
        sp = self.cfg["split"]
        n_tr = int(n * sp["train_frac"])
        n_va = int(n * (sp["train_frac"] + sp["val_frac"]))
        return {"train": (0, n_tr), "val": (n_tr, n_va), "test": (n_va, n)}

    def _assign_wells(self, well_ids: list[str]) -> dict[str, str]:
        sp = self.cfg["split"]
        rng = np.random.default_rng(sp.get("seed", 42))
        ids = list(well_ids)
        rng.shuffle(ids)
        n_tr = max(1, int(len(ids) * sp["train_frac"]))
        n_va = max(1, int(len(ids) * sp["val_frac"]))
        out = {w: "train" for w in ids[:n_tr]}
        out.update({w: "val" for w in ids[n_tr : n_tr + n_va]})
        out.update({w: "test" for w in ids[n_tr + n_va :]})
        return out

    # ------------------------------------------------------------ feature build
    @property
    def _history(self) -> int:
        """Rows of past context needed before a forecast origin."""
        need = [self.lookback - 1]
        need += self.target_lags + self.feature_lags
        need += [w - 1 for w in self.rolling]
        return max(need)

    def _tabular_frame(self, scaled: pd.DataFrame) -> pd.DataFrame:
        """Lagged features at each forecast origin t (all use data <= t)."""
        tgt = scaled[self.target_col]
        cols: dict[str, pd.Series] = {}
        for lag in self.target_lags:
            cols[f"{self.target_col}_lag{lag}"] = tgt.shift(lag)
        for w in self.rolling:
            cols[f"{self.target_col}_rollmean{w}"] = tgt.rolling(w).mean()
        for c in self.feature_cols:
            cols[c] = scaled[c]
            if c in self.static_cols:
                continue
            for lag in self.feature_lags:
                cols[f"{c}_lag{lag}"] = scaled[c].shift(lag)
        out = pd.DataFrame(cols, index=scaled.index)
        if self.seasonal:
            tgt_month = scaled.index.shift(self.horizon, freq=self.freq).month
            out["month_sin"] = np.sin(2 * np.pi * tgt_month / 12)
            out["month_cos"] = np.cos(2 * np.pi * tgt_month / 12)
        return out

    def _seq_array(self, mat: np.ndarray) -> np.ndarray:
        """(n, lookback, d) windows ending at each origin; NaN where incomplete."""
        n, d = mat.shape
        out = np.full((n, self.lookback, d), np.nan, dtype=np.float32)
        if n >= self.lookback:
            win = sliding_window_view(mat, self.lookback, axis=0)  # (n-L+1, d, L)
            out[self.lookback - 1 :] = win.transpose(0, 2, 1)
        return out

    # --------------------------------------------------------------------- main
    def prepare(self, path: str | Path | None = None) -> PreparedData:
        wells = self._regularise(self.load_raw(path))
        strategy = self.cfg["split"]["strategy"]
        if strategy not in ("temporal", "well"):
            raise ValueError(f"Unknown split strategy: {strategy}")
        well_split = self._assign_wells(list(wells)) if strategy == "well" else {}

        in_cols = [self.target_col, *self.feature_cols]
        if not self.use_target_input:
            seq_cols = list(self.feature_cols)
        else:
            seq_cols = in_cols

        # ---- 1. fit scaler on TRAIN rows only
        train_rows = []
        for wid, g in wells.items():
            if strategy == "temporal":
                lo, hi = self._well_split_bounds(len(g))["train"]
                train_rows.append(g.iloc[lo:hi][in_cols])
            elif well_split[wid] == "train":
                train_rows.append(g[in_cols])
        train_df = pd.concat(train_rows).dropna()
        if train_df.empty:
            raise ValueError("No training rows available to fit scaler")
        scaler = StandardScaler().fit(train_df.values)
        target_idx = 0

        # ---- 2. build samples per well
        buckets: dict[str, list[tuple]] = {k: [] for k in SPLITS}
        tab_names: list[str] = []
        for wid, g in wells.items():
            scaled = pd.DataFrame(scaler.transform(g[in_cols].values), index=g.index, columns=in_cols)
            tab = self._tabular_frame(scaled)
            tab_names = list(tab.columns)
            seq = self._seq_array(scaled[seq_cols].values)

            n, h = len(g), self.horizon
            y_s = scaled[self.target_col].shift(-h).values        # target at origin + h
            y_r = g[self.target_col].shift(-h).values
            origin_idx = np.arange(n)
            target_idx_arr = origin_idx + h
            times = g.index
            target_time = times.shift(h, freq=self.freq)

            valid = (
                (origin_idx >= self._history)
                & (target_idx_arr < n)
                & ~np.isnan(y_s)
                & ~np.isnan(tab.values).any(axis=1)
                & ~np.isnan(seq).any(axis=(1, 2))
            )

            if strategy == "temporal":
                bounds = self._well_split_bounds(n)
                masks = {k: valid & (target_idx_arr >= lo) & (target_idx_arr < hi) for k, (lo, hi) in bounds.items()}
            else:
                masks = {k: valid if well_split[wid] == k else np.zeros(n, bool) for k in SPLITS}

            for k, m in masks.items():
                if not m.any():
                    continue
                meta = pd.DataFrame(
                    {self.well_col: wid, "origin_time": times[m], "target_time": target_time[m]}
                )
                buckets[k].append((seq[m], tab.values[m].astype(np.float32), y_s[m], y_r[m], meta))

        # ---- 3. concatenate
        splits: dict[str, SplitData] = {}
        scale_target = bool(self.cfg["scaling"].get("scale_target", True))
        for k in SPLITS:
            parts = buckets[k]
            if not parts:
                raise ValueError(f"Split '{k}' is empty - check data length / split fractions")
            X_seq = np.concatenate([p[0] for p in parts])
            X_tab = np.concatenate([p[1] for p in parts])
            y_s = np.concatenate([p[2] for p in parts]).astype(np.float32)
            y_r = np.concatenate([p[3] for p in parts]).astype(np.float32)
            meta = pd.concat([p[4] for p in parts], ignore_index=True)
            splits[k] = SplitData(X_seq, X_tab, y_s if scale_target else y_r, y_r, meta)

        return PreparedData(
            splits=splits,
            seq_feature_names=seq_cols,
            tab_feature_names=tab_names,
            input_scaler=scaler,
            target_idx=target_idx,
            scale_target=scale_target,
            config=self.cfg,
        )

    # --------------------------------------------------------------- inference
    def build_inference_samples(self, df: pd.DataFrame, prepared_or_scaler) -> tuple[dict[str, np.ndarray | pd.DataFrame], list[str]]:
        """Build one sample per well at its *latest* forecast origin.

        ``df`` holds recent history (columns: well, time, target, covariates).
        ``prepared_or_scaler`` is the fitted ``StandardScaler`` from training.
        Returns ``({"X_seq", "X_tab", "meta"}, skipped_well_ids)``; wells whose
        history is too short or has unfilled gaps near the end are skipped.
        """
        scaler = prepared_or_scaler
        in_cols = [self.target_col, *self.feature_cols]
        seq_cols = in_cols if self.use_target_input else list(self.feature_cols)
        d = df[[self.well_col, self.time_col, *in_cols]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col])
        wells = self._regularise(d, min_rows=1)

        seqs, tabs, metas, skipped = [], [], [], []
        for wid, g in wells.items():
            scaled = pd.DataFrame(scaler.transform(g[in_cols].values), index=g.index, columns=in_cols)
            tab = self._tabular_frame(scaled)
            seq = self._seq_array(scaled[seq_cols].values)
            last = len(g) - 1
            if (
                last < self._history
                or np.isnan(tab.values[last]).any()
                or np.isnan(seq[last]).any()
            ):
                skipped.append(wid)
                continue
            origin = g.index[last]
            seqs.append(seq[last][None])
            tabs.append(tab.values[last][None].astype(np.float32))
            metas.append({
                self.well_col: wid,
                "origin_time": origin,
                "target_time": origin + pd.tseries.frequencies.to_offset(self.freq) * self.horizon,
            })
        if not metas:
            return {"X_seq": np.empty((0, self.lookback, len(seq_cols))), "X_tab": np.empty((0, len(tab.columns))),
                    "meta": pd.DataFrame(columns=[self.well_col, "origin_time", "target_time"])}, skipped
        return {"X_seq": np.concatenate(seqs), "X_tab": np.concatenate(tabs), "meta": pd.DataFrame(metas)}, skipped
