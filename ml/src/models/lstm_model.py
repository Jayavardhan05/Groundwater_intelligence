"""Multi-layer LSTM for groundwater level sequence forecasting (PyTorch)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseGroundwaterLevelModel


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMGroundwaterModel(BaseGroundwaterLevelModel):
    name = "lstm"
    input_kind = "sequence"

    DEFAULTS = dict(hidden_size=64, num_layers=2, dropout=0.2, learning_rate=1e-3,
                    batch_size=64, epochs=100, patience=10, seed=42)

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULTS, **(params or {})})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _LSTMNet | None = None
        self.n_features: int | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _build(self, n_features: int) -> _LSTMNet:
        p = self.params
        self.n_features = n_features
        return _LSTMNet(n_features, p["hidden_size"], p["num_layers"], p["dropout"]).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        p = self.params
        torch.manual_seed(p["seed"])
        self.net = self._build(X_train.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=p["learning_rate"])
        loss_fn = nn.MSELoss()

        tr = DataLoader(
            TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                          torch.as_tensor(y_train, dtype=torch.float32)),
            batch_size=p["batch_size"], shuffle=True,
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)

        best, best_state, bad = float("inf"), None, 0
        self.history = {"train_loss": [], "val_loss": []}
        for _ in range(p["epochs"]):
            self.net.train()
            total, count = 0.0, 0
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(xb)
                count += len(xb)
            self.history["train_loss"].append(total / count)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    v = loss_fn(self.net(Xv), yv).item()
                self.history["val_loss"].append(v)
                if v < best - 1e-8:
                    best, bad = v, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.as_tensor(X[i : i + 1024], dtype=torch.float32, device=self.device)
                out.append(self.net(xb).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def save(self, artifact_dir) -> None:
        self._check_fitted()
        d = self._prep_dir(artifact_dir)
        torch.save(self.net.state_dict(), d / "lstm_weights.pt")
        meta = {"params": self.params, "n_features": self.n_features, "history": self.history}
        (d / "lstm_meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, artifact_dir):
        d = Path(artifact_dir)
        meta = json.loads((d / "lstm_meta.json").read_text())
        self.params, self.history = meta["params"], meta["history"]
        self.net = self._build(meta["n_features"])
        self.net.load_state_dict(torch.load(d / "lstm_weights.pt", map_location=self.device))
        self.net.eval()
        self.is_fitted = True
        return self
