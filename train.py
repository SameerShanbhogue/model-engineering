from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.eval.metrics import classification_metrics
from src.models.trainer import train_logistic_regression
from src.utils.common import ensure_dir, load_yaml, set_seed


def _ensure_training_data(data_cfg: dict) -> pd.DataFrame:
    dataset_path = Path(data_cfg["dataset_path"])
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": data_cfg["target_column"]}, inplace=True)
    df.to_csv(dataset_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", default="configs/train/default.yaml")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(train_cfg["paths"]["data_config"])
    feature_cfg = load_yaml(train_cfg["paths"]["feature_config"])
    model_cfg = load_yaml(train_cfg["paths"]["model_config"])

    set_seed(train_cfg["seed"])

    df = _ensure_training_data(data_cfg)
    target_col = data_cfg["target_column"]
    drop_cols = feature_cfg.get("drop_columns", [])

    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_cfg.get("test_size", 0.2)),
        random_state=int(data_cfg.get("random_state", 42)),
        stratify=y,
    )

    model = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        model_params=model_cfg.get("params", {}),
        scale_numeric=bool(feature_cfg.get("scale_numeric", True)),
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_prob)

    run_dir = ensure_dir(Path(train_cfg["paths"]["runs_dir"]) / train_cfg["run_name"])

    model_file = train_cfg["artifacts"]["model_file"]
    metrics_file = train_cfg["artifacts"]["metrics_file"]
    params_file = train_cfg["artifacts"]["params_file"]
    predictions_file = train_cfg["artifacts"]["predictions_file"]

    joblib.dump(model, run_dir / model_file)
    (run_dir / metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    params = {
        "train_config": args.config,
        "data_config": train_cfg["paths"]["data_config"],
        "feature_config": train_cfg["paths"]["feature_config"],
        "model_config": train_cfg["paths"]["model_config"],
    }
    (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    if y_prob is not None:
        pred_df["y_score"] = y_prob
    pred_df.to_csv(run_dir / predictions_file, index=False)

    holdout = X_test.copy()
    holdout[target_col] = y_test.values
    holdout.to_csv(run_dir / "holdout.csv", index=False)

    summary_path = Path(train_cfg["paths"]["runs_dir"]) / "summary.csv"
    summary_row = {
        "run_name": train_cfg["run_name"],
        "model": model_cfg.get("model_type", "unknown"),
        "accuracy": metrics.get("accuracy", ""),
        "f1": metrics.get("f1", ""),
        "roc_auc": metrics.get("roc_auc", ""),
        "notes": "baseline run",
    }
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved run artifacts to: {run_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
