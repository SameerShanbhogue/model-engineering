from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.metrics import classification_metrics
from src.inference.predictor import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved run")
    parser.add_argument("--run-dir", default="runs/run_001")
    parser.add_argument("--target-col", default="target")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model = load_model(run_dir)

    holdout_path = run_dir / "holdout.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing holdout file: {holdout_path}")

    df = pd.read_csv(holdout_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in holdout.csv")

    X = df.drop(columns=[args.target_col])
    y_true = df[args.target_col]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(y_true, y_pred, y_prob)

    out_path = run_dir / "evaluation_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Evaluation metrics saved to: {out_path}")
    print(metrics)


if __name__ == "__main__":
    main()
