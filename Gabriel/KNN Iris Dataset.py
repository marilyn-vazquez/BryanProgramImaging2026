# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

iris = load_iris()

# Use Petal Length and Petal Width
X = iris.data[:, [2, 3]]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn.fit(
    X_train,
    y_train
)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\nAccuracy:", accuracy)

print("F1 Score:", f1)

colors = ListedColormap([
    "#FF0000",
    "#00AA00",
    "#0000FF"
])


plt.figure(figsize=(8, 6))


DecisionBoundaryDisplay.from_estimator(
    knn,
    X,
    cmap=colors,
    alpha=0.3
)


# Plot training data
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=colors,
    edgecolors="k",
    label="Training Data"
)


# Plot testing data
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap=colors,
    edgecolors="k",
    alpha=0.5,
    label="Testing Data"
)


plt.xlabel("Petal Length (cm)")

plt.ylabel("Petal Width (cm)")

plt.title("KNN Classification - Iris Dataset")

plt.legend()

plt.tight_layout()

plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names
)


plt.title("KNN Confusion Matrix")

plt.tight_layout()

plt.show()

