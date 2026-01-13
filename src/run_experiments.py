from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import apply_age_bias, download_heart_data, train_test_split_unbiased
from src.plot import plot_bias_scatter, plot_performance, plot_weights
from src.weighting import clip_weights, compute_importance_weights


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    return {
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"

    df = download_heart_data(data_dir)
    split = train_test_split_unbiased(df)

    bias_levels = [0.0, 0.3, 0.6, 0.8]
    results: List[Dict[str, float]] = []
    weights_by_bias: Dict[float, np.ndarray] = {}

    for bias_level in bias_levels:
        X_train_biased, y_train_biased = apply_age_bias(
            split.X_train, split.y_train, bias_level=bias_level
        )

        baseline_model = train_logistic_regression(X_train_biased, y_train_biased)
        baseline_metrics = evaluate_model(baseline_model, split.X_test, split.y_test)

        weights, stats, _ = compute_importance_weights(X_train_biased, split.X_test)
        iw_model = train_logistic_regression(
            X_train_biased, y_train_biased, sample_weight=weights
        )
        iw_metrics = evaluate_model(iw_model, split.X_test, split.y_test)

        clipped_weights = clip_weights(weights)
        iw_clip_model = train_logistic_regression(
            X_train_biased, y_train_biased, sample_weight=clipped_weights
        )
        iw_clip_metrics = evaluate_model(iw_clip_model, split.X_test, split.y_test)

        weights_by_bias[bias_level] = weights

        results.append(
            {
                "bias_level": bias_level,
                "baseline_acc": baseline_metrics["acc"],
                "baseline_f1": baseline_metrics["f1"],
                "iw_acc": iw_metrics["acc"],
                "iw_f1": iw_metrics["f1"],
                "iw_clip_acc": iw_clip_metrics["acc"],
                "iw_clip_f1": iw_clip_metrics["f1"],
                "mean_w": stats.mean_weight,
                "max_w": stats.max_weight,
                "ess": stats.ess,
            }
        )

    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "results.csv", index=False)

    plot_performance(results_df, output_dir / "performance.png")
    plot_weights(weights_by_bias, output_dir / "weights.png")

    bias_scatter_path = output_dir / "data_bias_scatter.png"
    plot_bias_scatter(
        X_train_biased,
        split.X_test,
        bias_scatter_path,
    )


if __name__ == "__main__":
    run()
