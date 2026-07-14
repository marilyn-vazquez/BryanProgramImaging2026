# -*- coding: utf-8 -*-
"""
Integrated Lower-Star Topological Data Analysis & Machine Learning Pipeline

This script performs the complete pipeline directly on your preprocessed folder:
1. Computes raw lower-star persistent homology via Cripser for all processed images.
2. Saves raw persistence diagrams (.npy matrices).
3. Summarizes raw diagrams into a 10-dimensional barcode feature vector.
4. Evaluates classification performance using Linear SVM, RBF SVM, and an MLP Network.
"""

import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cripser as cr
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.colors import ListedColormap

from skimage import io
from skimage.util import img_as_float

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
# 1. CRIPSER LOWER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_lower_star_ph(images_paths, output_dir):
    """
    Computes raw lower-star persistent homology via Cripser for a list of images
    and saves each raw persistence diagram matrix as a separate .npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagram_paths = []
    
    for path in images_paths:
        path = Path(path)
        print(f"Computing Lower-Star PH for: {path.name}")
        
        # Load the preprocessed image
        img = img_as_float(io.imread(path, as_gray=True))
        
        # Scale to standard 0-255 domain for numerical stability in Cripser
        if img.max() <= 1.0:
            img = img * 255.0
            
        img_input = np.asarray(img, dtype=np.float64)
        
        # Compute Persistent Homology (Lower-star cubical filtration)
        ph_diagram = cr.computePH(img_input) if hasattr(cr, "computePH") else cr.compute_ph(img_input)
        
        # Save the raw persistence diagram matrix [dim, birth, death] for this specific image
        save_path = output_dir / f"{path.stem}_lower_star_diagram.npy"
        np.save(save_path, ph_diagram)
        diagram_paths.append(save_path)
        
    print(f"\n✅ All lower-star persistence diagrams saved to: {output_dir}")
    return sorted(diagram_paths)

# =====================================================================
# 2. BARCODE VECTORIZATION (FROM RAW LOWER-STAR DIAGRAMS)
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
# 3. MACHINE LEARNING BENCHMARKS & EVALUATION
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
# 4. RUNNER CONTROLLER
# =====================================================================

if __name__ == '__main__':
    random.seed(101)

    # Directly point to the folder containing your preprocessed images
    PROCESSED_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
   
    # Gather processed target images directly from the directory
    processed_paths = sorted(list(PROCESSED_DIR.glob('*_processed.tif')))
   
    if not processed_paths:
        raise FileNotFoundError(f"Could not find any processed images in: {PROCESSED_DIR}.")
       
    print(f"Found {len(processed_paths)} preprocessed images to process.")

    # Phase 1: Execute lower-star TDA computation over target images
    print("\n=== Phase 1: Running Lower-Star Homology Extraction ===")
    saved_diagrams = compute_lower_star_ph(images_paths=processed_paths, output_dir=PROCESSED_DIR)

    # Phase 2: Vectorize the barcode diagrams into features and targets
    print("\n=== Phase 2: Starting Vectorization ===")
    X_topological_features, y_experimental_classes = vectorize_persistence_diagrams(saved_diagrams)
    
    # Save the extracted 10D feature matrices for record-keeping
    np.save(PROCESSED_DIR / "step2_lower_star_10d_features.npy", X_topological_features)
    np.save(PROCESSED_DIR / "step2_lower_star_labels.npy", y_experimental_classes)
    print("✅ Vectorized 10D arrays cached to disk.")

    # Phase 3: Execute Machine Learning Comparison
    print("\n=== Phase 3: Running Support Vector Machines & Neural Network ===")
    print(f"Dataset Dimensions: {X_topological_features.shape}")
    
    if len(X_topological_features) < 5:
        print("⚠️ Warning: Not enough samples to execute cross-validation splits.")
    else:
        run_ml_benchmark(
            X_tda=X_topological_features, 
            y=y_experimental_classes, 
            output_dir=PROCESSED_DIR, 
            dataset_title="Lower-Star Barcode Machine Learning"
        )