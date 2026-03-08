from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    scale_numeric: bool,
) -> Pipeline:
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
    model = LogisticRegression(**model_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline
