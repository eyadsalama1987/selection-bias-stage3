from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
GITHUB_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def download_heart_data(data_dir: Path) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / "heart.csv"

    if target_path.exists():
        return pd.read_csv(target_path)

    try:
        request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            df = pd.read_csv(response, header=None, names=COLUMNS)
    except (URLError, OSError):
        try:
            archive_path = data_dir / "heart_disease.zip"
            request = Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response, open(archive_path, "wb") as file_handle:
                file_handle.write(response.read())
            with zipfile.ZipFile(archive_path) as zip_file:
                with zip_file.open("processed.cleveland.data") as file_handle:
                    df = pd.read_csv(file_handle, header=None, names=COLUMNS)
        except (URLError, OSError):
            request = Request(GITHUB_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response:
                df = pd.read_csv(response, encoding="utf-8-sig")
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(target_path, index=False)
    return df


def train_test_split_unbiased(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> DatasetSplit:
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def apply_age_bias(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    bias_level: float,
    age_threshold: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    if bias_level <= 0:
        return X_train.copy(), y_train.copy()

    rng = np.random.default_rng(random_state)
    young_mask = X_train["age"] < age_threshold
    young_indices = X_train[young_mask].index.to_numpy()
    retain_frac = max(0.0, 1.0 - bias_level)
    retain_count = int(len(young_indices) * retain_frac)
    if retain_count < 1:
        retained_young = np.array([], dtype=int)
    else:
        retained_young = rng.choice(young_indices, size=retain_count, replace=False)

    keep_indices = np.concatenate([X_train[~young_mask].index.to_numpy(), retained_young])
    biased_X = X_train.loc[keep_indices].sort_index()
    biased_y = y_train.loc[keep_indices].sort_index()
    return biased_X, biased_y
