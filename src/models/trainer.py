"""Model training routines that build and fit sklearn pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    scale_numeric: bool,
    pca_enabled: bool = False,
    pca_n_components: int | None = None,
) -> Pipeline:
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
    model = LogisticRegression(**model_params)
    steps = [("preprocessor", preprocessor)]
    if pca_enabled:
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))
    steps.append(("model", model))
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)
    return pipeline
