"""Traditional unsupervised anomaly baselines (PCA and IsolationForest)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class TraditionalConfig:
    pca_components: int = 12
    iforest_contamination: float = 0.05
    random_state: int = 42


class TraditionalAnomalyBaselines:
    """Fit and score PCA + IsolationForest on tabular features."""

    def __init__(self, cfg: TraditionalConfig):
        self.cfg = cfg
        self.pca: PCA | None = None
        self.iforest: IsolationForest | None = None

    def fit(self, x_train: np.ndarray) -> dict[str, Any]:
        if x_train.ndim != 2:
            raise ValueError("x_train must be 2-D")
        n_samples, n_features = x_train.shape
        if n_samples < 2:
            raise ValueError("x_train needs at least 2 samples")

        max_components = max(1, min(n_features - 1, n_samples - 1))
        n_components = int(np.clip(self.cfg.pca_components, 1, max_components))

        self.pca = PCA(n_components=n_components, svd_solver="auto", random_state=self.cfg.random_state)
        self.pca.fit(x_train)

        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=float(np.clip(self.cfg.iforest_contamination, 1e-4, 0.49)),
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )
        self.iforest.fit(x_train)

        return {
            "num_train": int(n_samples),
            "num_features": int(n_features),
            "pca_components_used": int(n_components),
            "iforest_contamination": float(self.cfg.iforest_contamination),
        }

    def pca_scores(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted")
        z = self.pca.transform(x)
        x_rec = self.pca.inverse_transform(z)
        err = np.mean((x - x_rec) ** 2, axis=1)
        return err.astype(np.float64)

    def iforest_scores(self, x: np.ndarray) -> np.ndarray:
        if self.iforest is None:
            raise RuntimeError("IsolationForest model is not fitted")
        # score_samples: higher means more normal; convert to anomaly score.
        s = -self.iforest.score_samples(x)
        return s.astype(np.float64)
