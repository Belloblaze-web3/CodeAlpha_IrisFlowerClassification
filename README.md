# CodeAlpha — Iris Flower Classification

A reproducible multiclass classification project for CodeAlpha Data Science Task 1. It predicts Iris species from sepal and petal measurements using version 1 of the Kaggle `Iris.csv` dataset.

## Result

The support vector machine (SVM) had the highest mean accuracy in five-fold cross-validation on the training split (96.67%). On the held-out 20% test split, it achieved 96.67% accuracy, 96.67% balanced accuracy, and 96.66% macro F1. It correctly classified 29 of 30 test observations. One Iris-versicolor was predicted as Iris-virginica. Because the dataset is small and this is one fixed split, these scores are a demonstration result, not a guarantee of performance on new populations.

The full candidate scores, per-class report, and confusion matrix are saved in `reports/metrics.json` and `reports/confusion_matrix.png`.

## Dataset

The included CSV is version 1 of [Iris.csv by saurabh00007 on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv/versions/1). It contains 150 observations, four measurement features, an `Id` column, and three balanced species classes. The Kaggle metadata identifies the dataset license as [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The `Id` column is intentionally excluded from model features because it is a row identifier, not a flower measurement.

See [`data/README.md`](data/README.md) for dataset provenance and license details.

## Workflow

1. Validate the CSV schema and convert the four measurements to numeric features.
2. Split the data into stratified 80% training and 20% test sets with a fixed random seed.
3. Compare logistic regression, support vector machine, and random forest with five-fold stratified cross-validation on the training split.
4. Select the model with the highest mean cross-validation accuracy, fit it on the training split, and evaluate it once on the held-out test split.
5. Save the selected model, metrics JSON, and confusion-matrix image under `reports/`.

The test split is not used for model selection. The code reports accuracy, balanced accuracy, macro F1, per-class metrics, and the confusion matrix.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/train.py
python src/demo.py
```

The demo script predicts from one custom measurement vector or prints predictions for one real row from each species in the included dataset. Example custom prediction:

```bash
python src/demo.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

To select another CSV or output directory:

```bash
python src/train.py --data path/to/Iris.csv --output-dir reports
```

## Outputs

- `reports/metrics.json` — dataset summary, cross-validation scores, and held-out test metrics.
- `reports/confusion_matrix.png` — held-out predictions by species.
- `reports/iris_model.joblib` — fitted scikit-learn pipeline or estimator.

The trained model file is ignored by Git by default so generated binaries do not clutter the source repository. Recreate it by running the training script.

## Source layout

```text
├── assets/iris_cover.jpg
├── data/
│   ├── Iris.csv
│   └── README.md
├── notebooks/iris_classification_demo.ipynb
├── reports/
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/
│   ├── demo.py
│   └── train.py
├── requirements.txt
└── README.md
```
