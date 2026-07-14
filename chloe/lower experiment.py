import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.colors import ListedColormap

import cripser as cr
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

# -------------------------------------------------------------------
# 1. CRIPSER PERSISTENT HOMOLOGY & VECTORIZATION
# -------------------------------------------------------------------

def extract_true_cripser_statistics(images_list):
    """Computes lower-star persistent homology via Cripser and outputs a 10D barcode feature vector."""
    vectorized_features = []
    
    for img in images_list:
        # Scale back to normal unit domain for Cripser if required
        if img.max() <= 1.0:
            img = img * 255.0
            
        img_input = np.asarray(img, dtype=np.float64)
        
        # Compute Persistent Homology (Lower-star filtration)
        ph = cr.computePH(img_input) if hasattr(cr, "computePH") else cr.compute_ph(img_input)
        
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
        
    return np.array(vectorized_features)

# -------------------------------------------------------------------
# 2. MACHINE LEARNING BENCHMARKS & EVALUATION
# -------------------------------------------------------------------

def run_ml_benchmark(X_tda, y, output_dir, dataset_title="Microscopy Dataset"):
    """Trains and compares Linear SVM, RBF SVM, and an MLP Neural Network using 10D Cripser features."""
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
    csv_path = Path(output_dir) / "microgravity_pipeline_metrics_table.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    print("\n📊 --- BENCHMARK PERFORMANCE TABLE ---")
    print(df_metrics.to_string(index=False))
    print(f"\n✅ Metrics table saved to: {csv_path}\n")

# -------------------------------------------------------------------
# 3. PIPELINE MAIN CONTROLLER
# -------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(101)

    # Point to the directory containing preprocessed images (created by Script 1)
    PROCESSED_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
   
    # Gather processed target images
    processed_paths = sorted(list(PROCESSED_DIR.glob('*_processed.tif')))
   
    if not processed_paths:
        raise FileNotFoundError(f"Could not find any processed images in: {PROCESSED_DIR}. Did you run Script 1?")
       
    print(f"Found {len(processed_paths)} preprocessed images to analyze.")

    # -------------------------------------------------------------
    # LOAD AND MAP IMAGES
    # -------------------------------------------------------------
    preprocessed_images_list = []
    y_experimental_classes_list = []

    for path in processed_paths:
        # Load the preprocessed array
        img = img_as_float(io.imread(path, as_gray=True))
        preprocessed_images_list.append(img)
        
        # Labeling mapping: 1 for microgravity, 0 otherwise (control)
        label = 1 if "microgravity" in path.name.lower() else 0
        y_experimental_classes_list.append(label)
        
    y_experimental_classes = np.array(y_experimental_classes_list)

    # -------------------------------------------------------------
    # EXECUTE STEP 2: Cripser Filtration & Barcode Vectorization
    # -------------------------------------------------------------
    print("\n=== Starting Lower-Star PH Vectorization ===")
    X_topological_features = extract_true_cripser_statistics(preprocessed_images_list)
    
    # Save Feature Matrices Checkpoints
    np.save(PROCESSED_DIR / "step2_final_10d_features.npy", X_topological_features)
    np.save(PROCESSED_DIR / "step2_final_labels.npy", y_experimental_classes)
    print("✅ 10D topological barcode arrays written to disk.")

    # -------------------------------------------------------------
    # EXECUTE STEP 3: Linear SVM vs RBF SVM vs Neural Network (MLP)
    # -------------------------------------------------------------
    print("\n=== Starting Machine Learning Comparison ===")
    print(f"Dataset Dimensions: {X_topological_features.shape}")
    
    if len(processed_paths) < 5:
        print("⚠️ Warning: Add more images to perform a robust cross-validation train/test split.")
    else:
        run_ml_benchmark(
            X_tda=X_topological_features, 
            y=y_experimental_classes, 
            output_dir=PROCESSED_DIR, 
            dataset_title="Microgravity Space Biology Analysis"
        )