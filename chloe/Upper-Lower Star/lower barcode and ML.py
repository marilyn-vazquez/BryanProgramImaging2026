# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:48:16 2026

@author: chloe
"""

import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Silence UI and warning noise
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

# =====================================================================
# 1. BARCODE VECTORIZATION (FROM RAW LOWER-STAR DIAGRAMS)
# =====================================================================

def vectorize_persistence_diagrams(diagram_paths):
    """
    Loads raw persistence diagrams (.npy files) and summarizes their 
    topological features into a 10-dimensional barcode vector.
    """
    vectorized_features = []
    y_labels = []
    
    for path in diagram_paths:
        path = Path(path)
        
        # Load the raw [dim, birth, death] array
        ph = np.load(path)
        
        # Filter out infinite topological features
        finite_mask = np.isfinite(ph[:, 2])
        ph_finite = ph[finite_mask]
        
        births = ph_finite[:, 1]
        deaths = ph_finite[:, 2]
        persistence = deaths - births
        
        if len(persistence) == 0:
            summary_vector = np.zeros(10)
        else:
            summary_vector = np.array([
                np.mean(births),       # 1. Mean birth time
                np.std(births),        # 2. Birth standard deviation
                np.median(births),     # 3. Median birth time
                np.max(births),        # 4. Maximum birth time
                np.mean(deaths),       # 5. Mean death time
                np.std(deaths),        # 6. Death standard deviation
                np.max(deaths),        # 7. Maximum death time
                np.mean(persistence),  # 8. Mean feature lifetime
                np.std(persistence),   # 9. Lifetime standard deviation
                np.sum(persistence)    # 10. Total persistent mass
            ])
            
        vectorized_features.append(summary_vector)
        
        # Label mapping: 1 for microgravity, 0 otherwise (control)
        label = 1 if "microgravity" in path.name.lower() else 0
        y_labels.append(label)
        
    return np.array(vectorized_features), np.array(y_labels)

# =====================================================================
# 2. MACHINE LEARNING BENCHMARKS & EVALUATION
# =====================================================================

def run_ml_benchmark(X_tda, y, output_dir, dataset_title="Microscopy Dataset"):
    """Trains and compares Linear SVM, RBF SVM, and an MLP Neural Network using 10D features."""
    names = [
        "Linear SVM",
        "RBF SVM",
        "Neural Network (MLP)"
    ]
    classifiers = [
        SVC(kernel="linear", C=1.0, random_state=42),
        SVC(kernel="rbf", gamma=2, C=1, random_state=42),
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    ]

    # Split dataset for training and validation
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_tda, y, test_size=0.4, random_state=42
    )

    pca = PCA(n_components=2, random_state=42)
    X_train_vis = pca.fit_transform(X_train_full)
    X_test_vis = pca.transform(X_test_full)

    x_min, x_max = X_train_vis[:, 0].min() - 1.0, X_train_vis[:, 0].max() + 1.0
    y_min, y_max = X_train_vis[:, 1].min() - 1.0, X_train_vis[:, 1].max() + 1.0

    cm_standard = mcm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    
    num_classifiers = len(classifiers)
    fig = plt.figure(figsize=(3 * num_classifiers + 3, 4))
    
    ax = plt.subplot(1, num_classifiers + 1, 1)
    ax.set_title(f"{dataset_title}\n(Data PCA)", fontsize=9, weight="bold")
    ax.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", s=35)
    ax.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, cmap=cm_bright, alpha=0.5, edgecolors="k", s=35)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    metrics_records = []

    for idx, (name, clf) in enumerate(zip(names, classifiers), start=2):
        ax = plt.subplot(1, num_classifiers + 1, idx)
        model_pipeline = make_pipeline(StandardScaler(), clf)
        
        model_pipeline.fit(X_train_full, y_train)
        y_pred = model_pipeline.predict(X_test_full)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        cm_data = confusion_matrix(y_test, y_pred)
        
        metrics_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "TN": cm_data[0,0],
            "FP": cm_data[0,1],
            "FN": cm_data[1,0],
            "TP": cm_data[1,1]
        })
        
        vis_clf = copy.deepcopy(clf)
        vis_pipeline = make_pipeline(StandardScaler(), vis_clf)
        
        try:
            vis_pipeline.fit(X_train_vis, y_train)
            DecisionBoundaryDisplay.from_estimator(
                vis_pipeline, X_train_vis, cmap=cm_standard, alpha=0.8, ax=ax, eps=0.5
            )
        except Exception:
            pass

        ax.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", s=25)
        ax.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.5, s=25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name, fontsize=9, weight="bold")
            
        metrics_str = f"Acc: {acc:.2f}\nF1: {f1:.2f}".replace("0.", ".")
        ax.text(x_max - 0.2, y_min + (0.35 * (y_max - y_min)), metrics_str, size=9,
                horizontalalignment="right", weight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    plt.show()

    # Save metrics table
    df_metrics = pd.DataFrame(metrics_records)
    csv_path = Path(output_dir) / "microgravity_lower_star_ml_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    print("\n📊 --- MACHINE LEARNING BENCHMARK PERFORMANCE ---")
    print(df_metrics.to_string(index=False))
    print(f"\n✅ Evaluation metrics table saved to: {csv_path}\n")

# =====================================================================
# 3. RUNNER CONTROLLER
# =====================================================================

if __name__ == '__main__':
    random.seed(101)

    # Folder containing your lower-star .npy diagram check-points
    PROCESSED_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
   
    # Gather saved raw lower-star diagram arrays
    diagram_paths = sorted(list(PROCESSED_DIR.glob('*_lower_star_diagram.npy')))
   
    if not diagram_paths:
        raise FileNotFoundError(
            f"Could not find any lower star diagrams (*_lower_star_diagram.npy) in {PROCESSED_DIR}.\n"
            f"Please run your lower-star extraction script first."
        )
       
    print(f"Found {len(diagram_paths)} raw diagrams. Starting Vectorization...")

    # Vectorize the barcode diagrams into features and targets
    X_topological_features, y_experimental_classes = vectorize_persistence_diagrams(diagram_paths)
    
    # Save the extracted 10D feature matrices for record-keeping
    np.save(PROCESSED_DIR / "step2_lower_star_10d_features.npy", X_topological_features)
    np.save(PROCESSED_DIR / "step2_lower_star_labels.npy", y_experimental_classes)
    print("✅ Vectorized 10D arrays cached to disk.")

    # Execute Machine Learning Comparison
    print("\n=== Running Support Vector Machines & Neural Network ===")
    print(f"Dataset Dimensions: {X_topological_features.shape}")
    
    if len(X_topological_features) < 5:
        print("⚠️ Warning: Not enough samples to execute cross-validation splits.")
    else:
        run_ml_benchmark(
            X_tda=X_topological_features, 
            y=y_experimental_classes, 
            output_dir=PROCESSED_DIR, 
            dataset_title="Lower-Star Barcode Machine Learning Experiment"
        )