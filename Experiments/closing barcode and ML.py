# -*- coding: utf-8 -*-
"""
Integrated Closing Morphological TDA & Machine Learning Pipeline
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
import pdb  # Interactive debugging module
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

def closing(input_np_array, input_list_of_points):
    return erosion(dilation(input_np_array, input_list_of_points), input_list_of_points)

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
        morphed_img = closing(input_np_array=img, input_list_of_points=the_kernel)
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
    
    # 🔬 PDB CHECKPOINT: INSPECT ML DATA
    # pdb.set_trace()

    # ... (rest of your plotting and training code remains the same)
    
    print("\n📊 --- CLOSING EXPERIMENT ---")
    print(pd.DataFrame(metrics_records).to_string(index=False))

if __name__ == '__main__':
    random.seed(101)
    RAW_IMG_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images")
    raw_paths = sorted(list(RAW_IMG_DIR.glob('*.tif')))
    
    print("=== Phase 1: Running Closing TDA Filtrations ===")
    kernels = get_square_SE_list(maximal_SE_lengths=4)
    
    X_features, y_labels = [], []
    for path in raw_paths:
        img = img_as_float(io.imread(path, as_gray=True))
        binary_img = biImg_by_threshold_leq(img, threshold=0.5)
        
        pds = persistence_of_morph_filtration(binary_img, kernels, morph_type='closing')
        
        # 🔬 PDB CHECKPOINT: INSPECT PDs PER IMAGE
        # pdb.set_trace()
        
        X_features.append(extract_10d_barcode(pds))
        y_labels.append(1 if "microgravity" in path.name.lower() else 0)

    # ... (rest of your caching and runner code)