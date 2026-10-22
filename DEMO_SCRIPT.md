# Project demo script

## Live command-line demo

From the repository root, run:

```bash
python src/train.py
python src/demo.py
```

The training command compares logistic regression, SVM, and random forest, selects the strongest model by five-fold cross-validation on the training split, evaluates it on held-out data, and writes the metrics and confusion matrix to `reports/`.

The demo command loads that saved model and predicts one real example from each species in the supplied CSV. To demonstrate a custom flower measurement, run:

```bash
python src/demo.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

## 60-second explanation

“This project predicts one of three Iris species from four flower measurements. I used the Kaggle Iris CSV, checked that it has 150 rows with 50 examples in each class, and excluded the row ID from the inputs. I compared logistic regression, support vector machine, and random forest using stratified five-fold cross-validation on the training data. The support vector machine had the strongest mean cross-validation score. On a separate 30-row test split, it classified 29 flowers correctly, for 96.67% accuracy and 96.66% macro F1. The confusion matrix shows the only error: one versicolor was classified as virginica. This dataset is small, so the score is a useful learning benchmark, not a claim of guaranteed real-world performance. The included demo can classify real examples from the dataset or a custom measurement vector.”
