# LinkedIn post draft — review before publishing

## Post copy

🌸 **CodeAlpha Data Science Task 1: Iris Flower Classification**

I built a reproducible machine-learning workflow to classify Iris flowers from four measurements: sepal length, sepal width, petal length, and petal width.

**What I did**
- Audited the 150-row Kaggle dataset and confirmed the three species are evenly represented.
- Excluded the `Id` column so the model learns from flower measurements, not row order.
- Used a stratified 80/20 train-test split and five-fold stratified cross-validation on the training data.
- Compared logistic regression, a support vector machine, and a random forest.

**Result**
The support vector machine led model selection with 96.67% mean cross-validation accuracy. On the held-out test set, it reached 96.67% accuracy and 96.66% macro F1, correctly classifying 29 of 30 flowers. The single error was a versicolor predicted as virginica.

This is a small benchmark dataset, so I’m treating the score as a learning result—not a guarantee of real-world performance. The project includes the source CSV, reproducible code, metrics, confusion matrix, and a runnable demo.

GitHub: [add repository link after publishing]
Kaggle: [add notebook link after publishing]

#DataScience #MachineLearning #Python #ScikitLearn #CodeAlpha

## Video explanation outline

1. Introduce the classification problem and show the four input measurements.
2. Show the dataset check: 150 rows and 50 examples per species.
3. Explain why `Id` is excluded and why the split is stratified.
4. Compare the three candidate models using cross-validation.
5. Show the held-out confusion matrix and explain the one versicolor/virginica error.
6. Run `python src/demo.py` and then demonstrate a custom measurement input.
7. Close by showing the GitHub and Kaggle links after they are published.

## Before posting

Replace both link placeholders with the actual URLs. Review the first-person wording and confirm it matches your internship status. Do not post until the linked project artifacts are ready.
