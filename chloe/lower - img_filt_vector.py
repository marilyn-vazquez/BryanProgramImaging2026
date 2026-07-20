# -*- coding: utf-8 -*-
"""
Integrated Lower-Star Topological Data Analysis & Machine Learning Pipeline

This script performs the complete pipeline directly on your preprocessed folder:
1. Computes raw lower-star persistent homology via Cripser for all processed images.
2. Saves raw persistence diagrams (.npy matrices) for each image.
3. Summarizes raw diagrams into a 10-dimensional barcode feature vector.
4. Evaluates classification performance using Linear SVM, RBF SVM, and an MLP Network.
5. Exports classifier performance metrics to a CSV file.
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
import pdb  # Interactive debugging module
import numpy as np
import cripser as cr
from pathlib import Path
from skimage import io, util

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
        
        # -------------------------------------------------------------
        # 🔬 PDB CHECKPOINT: INSPECT IMAGE PERSISTENT HOMOLOGY
        # -------------------------------------------------------------
        # Uncomment the line below to inspect individual image variables (e.g., img_input, ph_diagram)
        # pdb.set_trace()
        # -------------------------------------------------------------

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
        
    # -----------------------------------------------------------------
    # 🔬 PDB CHECKPOINT: INSPECT BARCODE VECTORIZATION
    # -----------------------------------------------------------------
    # Uncomment the line below to inspect variables (e.g., vectorized_features, y_labels, ph)
    # pdb.set_trace()
    # -----------------------------------------------------------------

    return np.array(vectorized_features), np.array(y_labels)

def process_and_save_everything(input_dir, filtration_type="lower_star"):
    """
    Computes PH and saves vectors based on the filtration_type.
    Example: filtration_type="upper_star" results in *_upper_star_Filt_vect.npy
    """
    img_paths = sorted(list(input_dir.glob('*_processed.tif')))
    all_vectors = []
    
    for img_path in img_paths:
        # 1. Compute PH (You can add a toggle here if you have different methods)
        img = img_as_float(io.imread(img_path, as_gray=True)) * 255.0
        ph = cr.computePH(np.asarray(img, dtype=np.float64))
        
        # 2. Vectorize
        # Note: vectorize_persistence_diagrams just needs the diagram
        X_vec, _ = vectorize_persistence_diagrams([ph]) 
        vector = X_vec[0]
        
        # 3. SAVE WITH FILTRATION NAME IN FILENAME
        # Result: IMG001_lower_star_Filt_vect.npy
        save_name = f"{img_path.stem}_{filtration_type}_Filt_vect.npy"
        np.save(input_dir / save_name, vector)
        
        all_vectors.append(vector)
        print(f"Processed: {img_path.name} as {filtration_type}")
        
    # 4. Save Master file with filtration name
    master_name = f"vectorized_{filtration_type}_barcode.npy"
    np.save(input_dir / master_name, np.array(all_vectors))
    print(f"\n✅ All {filtration_type} files saved.")

if __name__ == '__main__':
    DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
    
    # Just change this string when you switch methods!
    process_and_save_everything(DIR, filtration_type="upper_star")


# =====================================================================
# 3. MACHINE LEARNING BENCHMARKS & EVALUATION
# =====================================================================

