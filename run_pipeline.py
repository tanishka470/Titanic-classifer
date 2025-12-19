import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ds_pipeline.data import load_data, make_splits
from ds_pipeline.features import build_preprocessor
from ds_pipeline.metrics import classification_report_dict
from ds_pipeline.model import build_model


def run_training(args):
    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = make_splits(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    preprocessor = build_preprocessor()
    pipeline = build_model(preprocessor)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_report_dict(y_test, y_pred, y_proba)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = Path(args.output_dir) / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = Path(args.output_dir) / f"model_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)

    preds_path = Path(args.output_dir) / f"predictions_{timestamp}.csv"
    pd.DataFrame({"pred": y_pred, "actual": y_test.values}).to_csv(
        preds_path, index=False
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
    cm_path = Path(args.output_dir) / f"confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved model to {model_path}")
    print(f"Saved predictions to {preds_path}")
    print(f"Saved confusion matrix to {cm_path}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "titanic_sample.csv"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
