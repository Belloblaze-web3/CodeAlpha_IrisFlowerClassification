"""Demonstrate the trained Iris classifier with real dataset examples or custom measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "reports" / "iris_model.joblib"
DATA_PATH = ROOT / "data" / "Iris.csv"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sepal-length", type=float)
    parser.add_argument("--sepal-width", type=float)
    parser.add_argument("--petal-length", type=float)
    parser.add_argument("--petal-width", type=float)
    args = parser.parse_args()

    values = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
    supplied = [value is not None for value in values]
    if any(supplied) and not all(supplied):
        parser.error("Provide all four measurements, or omit all four to run the built-in examples.")
    if not MODEL_PATH.exists():
        parser.error("Trained model not found. Run `python src/train.py` first.")

    model = load(MODEL_PATH)
    if all(supplied):
        examples = pd.DataFrame([values], columns=FEATURES)
        labels = ["custom input"]
    else:
        data = pd.read_csv(DATA_PATH)
        examples = data.groupby(TARGET, sort=True).head(1).reset_index(drop=True)[FEATURES]
        labels = data.groupby(TARGET, sort=True).head(1)[TARGET].astype(str).tolist()

    predictions = model.predict(examples)
    probabilities = model.predict_proba(examples) if hasattr(model, "predict_proba") else None

    print("Iris Flower Classification — demo")
    print("Measurements: sepal length, sepal width, petal length, petal width (cm)\n")
    for index, (reference, prediction) in enumerate(zip(labels, predictions, strict=True)):
        row = examples.iloc[index]
        measurements = ", ".join(f"{name}={row[name]:.1f}" for name in FEATURES)
        print(f"Example: {reference}")
        print(f"  {measurements}")
        print(f"  Predicted species: {prediction}")
        if probabilities is not None:
            best_probability = float(probabilities[index].max())
            print(f"  Highest class probability: {best_probability:.1%}")
        print()


if __name__ == "__main__":
    main()
