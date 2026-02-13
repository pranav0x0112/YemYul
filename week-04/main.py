import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Actual values
# A = 1 → Positive class
# B = 0 → Negative class
actual = [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]

# Predicted values
predicted = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

# Confusion matrix
cm = confusion_matrix(actual, predicted, labels=[1, 0])
print("Confusion Matrix:\n", cm)

# Accuracy
acc = accuracy_score(actual, predicted)
print("Accuracy =", acc)

# Classification report
report = classification_report(actual, predicted, labels=[1, 0])
print("\nClassification Report:\n")
print(report)

# ROC curve
fpr, tpr, _ = metrics.roc_curve(actual, predicted)
print("FPR =", fpr)
print("TPR =", tpr)

# Plot ROC
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
