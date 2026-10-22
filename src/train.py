"""Train and evaluate baseline classifiers for the Kaggle Iris dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "Iris.csv"
DEFAULT_OUTPUT = ROOT / "reports"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the Iris CSV, excluding the row ID from predictors."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    raw_y = frame[TARGET]
    if X.isna().any().any() or raw_y.isna().any():
        raise ValueError("Dataset contains missing values; handle them before training.")
    y = raw_y.astype(str)
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    return X, y


def build_models(random_state: int) -> dict[str, object]:
    """Return three baseline classifiers for a compact model comparison."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "support_vector_machine": Pipeline(
            [("scale", StandardScaler()), ("model", SVC())]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        ),
    }


def run(data_path: Path, output_dir: Path, random_state: int = 42) -> dict[str, object]:
    """Compare models on training folds and evaluate the selected model once on holdout data."""
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = build_models(random_state)
    candidate_scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        candidate_scores[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }

    selected_name = max(
        candidate_scores,
        key=lambda name: candidate_scores[name]["mean_accuracy"],
    )
    selected_model = models[selected_name]
    selected_model.fit(X_train, y_train)
    predictions = selected_model.predict(X_test)
    labels = sorted(y.unique())

    result: dict[str, object] = {
        "dataset": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "rows": int(len(X)),
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "random_state": random_state,
        "test_size": 0.2,
        "selected_model": selected_name,
        "cross_validation": candidate_scores,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "classification_report": classification_report(
                y_test, predictions, labels=labels, output_dict=True, zero_division=0
            ),
            "confusion_matrix_labels": labels,
            "confusion_matrix": [
                [int(value) for value in row]
                for row in confusion_matrix(y_test, predictions, labels=labels)
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        display_labels=labels,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    plt.title(f"Iris test-set confusion matrix\n{selected_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()
    dump(selected_model, output_dir / "iris_model.joblib")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Iris.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = run(args.data, args.output_dir, args.random_state)
    metrics = result["test_metrics"]
    print(f"Selected model: {result['selected_model']}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Reports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
