from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_performance(results: pd.DataFrame, output_path: Path) -> None:
    bias_levels = results["bias_level"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(bias_levels, results["baseline_acc"], marker="o", label="Baseline")
    axes[0].plot(bias_levels, results["iw_acc"], marker="o", label="IW")
    axes[0].plot(bias_levels, results["iw_clip_acc"], marker="o", label="IW + Clipping")
    axes[0].set_xlabel("Bias level")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Bias")
    axes[0].legend()

    axes[1].plot(bias_levels, results["baseline_f1"], marker="o", label="Baseline")
    axes[1].plot(bias_levels, results["iw_f1"], marker="o", label="IW")
    axes[1].plot(bias_levels, results["iw_clip_f1"], marker="o", label="IW + Clipping")
    axes[1].set_xlabel("Bias level")
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("F1-score vs Bias")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_weights(weights_by_bias: Dict[float, np.ndarray], output_path: Path) -> None:
    labels: List[str] = []
    weight_arrays: List[np.ndarray] = []
    for bias_level, weights in weights_by_bias.items():
        labels.append(str(bias_level))
        weight_arrays.append(weights)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(weight_arrays, labels=labels, showfliers=False)
    ax.set_xlabel("Bias level")
    ax.set_ylabel("Importance weights")
    ax.set_title("Weight distribution by bias level")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bias_scatter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_path: Path,
) -> None:
    combined = pd.concat([X_train, X_test], axis=0)
    labels = np.array(["train"] * len(X_train) + ["test"] * len(X_test))

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(combined_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            components[mask, 0],
            components[mask, 1],
            alpha=0.6,
            label=label,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Biased train vs unbiased test (PCA)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
