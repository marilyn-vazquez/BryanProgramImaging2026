# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:24:35 2026

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

# --- CALLS ON YOUR ORIGINAL CODE ---
import morphology_tda as tda

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

def extract_10d_barcode(pds):
    h0, h1 = pds
    combined = np.vstack([h0, h1]) if (len(h0) > 0 and len(h1) > 0) else (h0 if len(h0) > 0 else h1)
    finite_mask = np.isfinite(combined[:, 1])
    combined_finite = combined[finite_mask]
    births, deaths = combined_finite[:, 0], combined_finite[:, 1]
    persistence = deaths - births
    if len(persistence) == 0:
        return np.zeros(10)
    return np.array([
        np.mean(births), np.std(births), np.median(births), np.max(births),
        np.mean(deaths), np.std(deaths), np.max(deaths),
        np.mean(persistence), np.std(persistence), np.sum(persistence)
    ])

def run_ml_benchmark(X_tda, y, output_dir):
    names = ["Linear SVM", "Neural Network (MLP)"]
    classifiers = [
        SVC(kernel="linear", C=1.0, random_state=42),
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    ]
    X_train, X_test, y_train, y_test = train_test_split(X_tda, y, test_size=0.4, random_state=42)
    pca = PCA(n_components=2, random_state=42)
    X_train_vis = pca.fit_transform(X_train)
    X_test_vis = pca.transform(X_test)
    x_min, x_max = X_train_vis[:, 0].min() - 1.0, X_train_vis[:, 0].max() + 1.0
    y_min, y_max = X_train_vis[:, 1].min() - 1.0, X_train_vis[:, 1].max() + 1.0

    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 3, 1)
    ax.set_title("Closing PCA Data", fontsize=9, weight="bold")
    ax.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolors="k", s=35)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_xticks(()); ax.set_yticks(())

    metrics_records = []
    for idx, (name, clf) in enumerate(zip(names, classifiers), start=2):
        ax = plt.subplot(1, 3, idx)
        model_pipeline = make_pipeline(StandardScaler(), clf)
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        cm_data = confusion_matrix(y_test, y_pred)
        metrics_records.append({"Model": name, "Accuracy": round(acc, 4), "F1-Score": round(f1, 4)})
        
        vis_clf = copy.deepcopy(clf)
        vis_pipeline = make_pipeline(StandardScaler(), vis_clf)
        try:
            vis_pipeline.fit(X_train_vis, y_train)
            DecisionBoundaryDisplay.from_estimator(vis_pipeline, X_train_vis, cmap=mcm.RdBu, alpha=0.8, ax=ax, eps=0.5)
        except Exception: pass

        ax.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolors="k", s=25)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_xticks(()); ax.set_yticks(())
        ax.set_title(name, fontsize=9, weight="bold")
            
    plt.tight_layout()
    plt.show()
    print("\n📊 --- CLOSING EXPERIMENT ---")
    print(pd.DataFrame(metrics_records).to_string(index=False))

if __name__ == '__main__':
    random.seed(101)
    PROCESSED_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
    processed_paths = sorted(list(PROCESSED_DIR.glob('*_processed.tif')))
    
    # Generate Kernel List via your original code
    kernels = tda.get_square_SE_list(maximal_SE_lengths=4)
    
    X_features, y_labels = [], []
    for path in processed_paths:
        img = img_as_float(io.imread(path, as_gray=True))
        
        # Call binary thresholding from original code
        binary_img = tda.biImg_by_threshold_leq(img, threshold=0.5)
        
        # Call original morphological filtration sequence
        pds = tda.persistence_of_morph_filtration(binary_img, kernels, morph_type='closing')
        
        X_features.append(extract_10d_barcode(pds))
        y_labels.append(1 if "microgravity" in path.name.lower() else 0)

    run_ml_benchmark(np.array(X_features), np.array(y_labels), PROCESSED_DIR)