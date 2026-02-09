from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

# Create a sample classification dataset
X, y = make_classification(
    n_samples=10,
    n_features=4,
    n_classes=2
)

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []

    for i in range(k):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.concatenate([
            indices[:i * fold_size],
            indices[(i + 1) * fold_size:]
        ])
        folds.append((train_indices, test_indices))

    return folds


# Number of folds
k = 5

# Get fold indices
fold_indices = kfold_indices(X, k)

# Initialize model
model = DecisionTreeClassifier()

# Store scores
scores = []

# K-Fold Cross Validation
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("X_train samples")
    print(X_train)

    print("X_test samples")
    print(X_test)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    fold_score = accuracy_score(y_test, y_pred)
    scores.append(fold_score)

# Final results
mean_accuracy = np.mean(scores)

print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", mean_accuracy)