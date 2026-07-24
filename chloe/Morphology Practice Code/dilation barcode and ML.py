# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:25:10 2026

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
import numba as nb
import cripser
from skimage import io
from skimage.util import img_as_float

from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

# -------------------------------------------------------------------
# NATIVE MORPHOLOGY & TDA IMPLEMENTATION
# -------------------------------------------------------------------

def find(condition):
    return np.nonzero(condition)

def biImg_by_threshold_leq(img, threshold):
    output_img = np.copy(img)
    idxs_0 = find(img <= threshold)
    idxs_1 = find(img > threshold)
    output_img[idxs_0] = 0
    output_img[idxs_1] = 1
    return output_img

def biImg_by_threshold_geq(img, threshold):
    output_img = np.copy(img)
    idxs_0 = find(img >= threshold)
    idxs_1 = find(img < threshold)
    output_img[idxs_0] = 0
    output_img[idxs_1] = 1
    return output_img

@nb.jit(nopython=True)
def erosion(input_np_array, input_list_of_points, minimal_pixel_value=0):
    input_np_array_shape = input_np_array.shape
    output_np_array = np.zeros(input_np_array_shape)
    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):
            if input_np_array[i, j] == minimal_pixel_value:
                output_np_array[i, j] = minimal_pixel_value
                continue
            relevant_pixel_values = []
            for k in range(len(input_list_of_points)):
                m = i - input_list_of_points[k][1]
                n = j + input_list_of_points[k][0]
                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])
            output_np_array[i, j] = min(relevant_pixel_values)
    return output_np_array

@nb.jit(nopython=True)
def dilation(input_np_array, input_list_of_points, maximal_pixel_value=1):
    input_np_array_shape = input_np_array.shape
    output_np_array = np.zeros(input_np_array_shape)
    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):
            if input_np_array[i, j] == maximal_pixel_value:
                output_np_array[i, j] = maximal_pixel_value
                continue
            relevant_pixel_values = []
            for k in range(len(input_list_of_points)):
                m = i + input_list_of_points[k][1]
                n = j - input_list_of_points[k][0]
                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])
            output_np_array[i, j] = max(relevant_pixel_values)
    return output_np_array

def opening(input_np_array, input_list_of_points):
    return dilation(erosion(input_np_array, input_list_of_points), input_list_of_points)

def closing(input_np_array, input_list_of_points):
    return erosion(dilation(input_np_array, input_list_of_points), input_list_of_points)

@nb.jit()
def get_rectangle_coordinates(input_np_array):
    input_np_array_shape = np.shape(input_np_array)
    output_list = []
    origin_i = int(input_np_array_shape[0] / 2)
    origin_j = int(input_np_array_shape[1] / 2)
    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):
            output_list.append(np.array([origin_j - j, origin_i - i]))
    return output_list

def get_square_SE_list(maximal_SE_lengths):
    kernel_list = []
    for i in range(2, maximal_SE_lengths + 1):
        kernel_list.append(get_rectangle_coordinates(input_np_array=np.zeros((i, i))))
    return kernel_list

def persistence_of_img(img, maxdim=1):
    img = np.asarray(img, dtype=np.float64)
    if hasattr(cripser, "compute_ph"):
        ph = cripser.compute_ph(img, maxdim=maxdim)
    else:
        ph = cripser.computePH(img, maxdim=maxdim)
    persistence_0 = ph[ph[:, 0] == 0][:, 1:3]
    persistence_1 = ph[ph[:, 0] == 1][:, 1:3]
    return [persistence_0, persistence_1]

def persistence_of_morph_filtration(img, kernel_list, morph_type='closing'):
    img_shape = np.shape(img)
    img_buff = np.zeros(img_shape) + img
    for the_kernel in kernel_list:
        if morph_type == 'opening':
            morphed_img = opening(input_np_array=img, input_list_of_points=the_kernel)
        elif morph_type == 'closing':
            morphed_img = closing(input_np_array=img, input_list_of_points=the_kernel)
        elif morph_type == 'erosion':
            morphed_img = erosion(input_np_array=img, input_list_of_points=the_kernel)
        elif morph_type == 'dilation':
            morphed_img = dilation(input_np_array=img, input_list_of_points=the_kernel)
        else:
            raise ValueError("morph_type must be 'opening', 'closing', 'erosion', or 'dilation'")
        img_buff = img_buff + morphed_img
    return persistence_of_img(img_buff)

# -------------------------------------------------------------------
# FEATURE EXTRACTION & ML BENCHMARK
# -------------------------------------------------------------------

def extract_10d_barcode(pds):
    h0, h1 = pds
    combined = np.vstack([h0, h1]) if (len(h0) > 0 and len(h1) > 0) else (h0 if len(h0) > 0 else h1)
    if len(combined) == 0:
        return np.zeros(10)
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
    ax.set_title("Dilation PCA Data", fontsize=9, weight="bold")
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
    print("\n📊 --- DILATION EXPERIMENT ---")
    print(pd.DataFrame(metrics_records).to_string(index=False))

if __name__ == '__main__':
    random.seed(101)
    RAW_IMG_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images")
    raw_paths = sorted(list(RAW_IMG_DIR.glob('*.tif')))
    
    print("=== Phase 1: Running Dilation TDA Filtrations Directly on Images ===")
    kernels = get_square_SE_list(maximal_SE_lengths=4)
    
    X_features, y_labels = [], []
    for path in raw_paths:
        img = img_as_float(io.imread(path, as_gray=True))
        binary_img = biImg_by_threshold_leq(img, threshold=0.5)
        
        # Runs explicit loop and computes persistent homology diagrams
        pds = persistence_of_morph_filtration(binary_img, kernels, morph_type='dilation')
        
        X_features.append(extract_10d_barcode(pds))
        y_labels.append(1 if "microgravity" in path.name.lower() else 0)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    np.save(RAW_IMG_DIR / "step2_dilation_10d_features.npy", X_features)
    np.save(RAW_IMG_DIR / "step2_dilation_labels.npy", y_labels)
    print("✅ Vectorized 10D dilation arrays cached to disk.\n")

    print("=== Phase 2: Running Machine Learning Benchmark (Dilation) ===")
    run_ml_benchmark(X_features, y_labels, RAW_IMG_DIR)