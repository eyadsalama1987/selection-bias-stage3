from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class WeightStats:
    mean_weight: float
    max_weight: float
    ess: float


def _effective_sample_size(weights: np.ndarray) -> float:
    total = weights.sum()
    denom = np.square(weights).sum()
    if denom == 0:
        return 0.0
    return float(total**2 / denom)


def compute_importance_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, WeightStats, Pipeline]:
    X_domain = pd.concat([X_train, X_test], axis=0)
    y_domain = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]
    )

    domain_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    domain_model.fit(X_domain, y_domain)

    prob_test = domain_model.predict_proba(X_train)[:, 1]
    prob_train = 1.0 - prob_test

    epsilon = 1e-6
    prob_train = np.clip(prob_train, epsilon, 1.0)
    prob_test = np.clip(prob_test, epsilon, 1.0)

    prior_ratio = len(X_train) / max(len(X_test), 1)
    weights = (prob_test / prob_train) * prior_ratio

    stats = WeightStats(
        mean_weight=float(np.mean(weights)),
        max_weight=float(np.max(weights)),
        ess=_effective_sample_size(weights),
    )
    return weights, stats, domain_model


def clip_weights(weights: np.ndarray, max_clip: float = 10.0) -> np.ndarray:
    percentile_clip = np.percentile(weights, 95)
    cap = min(max_clip, percentile_clip)
    return np.clip(weights, 0.0, cap)
