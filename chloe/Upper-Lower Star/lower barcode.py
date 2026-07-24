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
# 1. CRIPSER LOWER-STAR PERSISTENT HOMOLOGY (FIXED LOOP)
# =====================================================================

def compute_lower_star_ph(images_paths, output_dir):
    """
    Computes raw lower-star persistent homology via Cripser for a list of images
    and saves each raw persistence diagram matrix as a separate .npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagram_paths = []

    # FIX: Added missing for loop initialization
    for path in images_paths:
        path = Path(path)

        # Load the preprocessed image
        img = img_as_float(io.imread(path, as_gray=True))

        # Scale to standard 0-255 domain for numerical stability in Cripser
        if img.max() <= 1.0:
            img = img * 255.0

        img_input = np.asarray(img, dtype=np.float64)

        # Compute Persistent Homology (Lower-star cubical filtration)
        # REMOVED: Unused PCA initialization line that was here

        if hasattr(cr, "computePH"):
            ph_diagram = cr.computePH(img_input)
        elif hasattr(cr, "compute_ph"):
            ph_diagram = cr.compute_ph(img_input)
        else:
            raise AttributeError("No compatible Cripser PH function found.")

        # Save the raw persistence diagram matrix [dim, birth, death]
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

        # --- FIX 1: Strict Global Finite Masking ---
        # Ensure absolutely NO column (dim, birth, or death) contains NaN or Inf
        global_finite_mask = np.all(np.isfinite(ph), axis=1)
        ph_finite = ph[global_finite_mask]

        # --- FIX 2: Handle Edge Case of Unbound/Infinite Features ---
        # If Cripser uses a specific placeholder for infinity (like 1e10 or max_val)
        # remove rows where death is significantly far away or negative.
        if len(ph_finite) > 0:
            # Drop features that have absurdly high death values (unbound components)
            valid_death_mask = ph_finite[:, 2] < 1e9
            ph_finite = ph_finite[valid_death_mask]

        births = ph_finite[:, 1]
        deaths = ph_finite[:, 2]
        persistence = deaths - births

        # --- FIX 3: Catch Negative Persistence/Noise ---
        # Lower-star filtration should have death >= birth, filter any noise
        valid_persistence = persistence > 0
        births = births[valid_persistence]
        deaths = deaths[valid_persistence]
        persistence = persistence[valid_persistence]

        if len(persistence) == 0:
            summary_vector = np.zeros(10)
        else:
            summary_vector = np.array([
                np.mean(births),
                np.std(births),
                np.median(births),
                np.max(births),
                np.mean(deaths),
                np.std(deaths),
                np.max(deaths),
                np.mean(persistence),
                np.std(persistence),
                np.sum(persistence)
            ])

            # --- FIX 4: Safety Check to Prevent Future Pipeline Crashes ---
            # If standard deviation calculation still overflows due to massive scale
            summary_vector = np.nan_to_num(summary_vector, nan=0.0, posinf=0.0, neginf=0.0)

        vectorized_features.append(summary_vector)

        # Label mapping
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
        X_tda, y, test_size=0.2, random_state=42
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

        # FIX: Completed the dictionary compilation from the confusion matrix
        metrics_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "TN": cm_data[0, 0],
            "FP": cm_data[0, 1],
            "FN": cm_data[1, 0],
            "TP": cm_data[1, 1]
        })

    # Optional: Save metrics to CSV as intended by your docstring

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
    figure_path = Path(output_dir) / "lower_star_ml_results.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Save metrics table to disk
    df_metrics = pd.DataFrame(metrics_records)
    csv_path = Path(output_dir) / "microgravity_lower_star_ml_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    print("\n📊 --- MACHINE LEARNING BENCHMARK PERFORMANCE ---")
    print(df_metrics.to_string(index=False))
    print(f"\n✅ Evaluation metrics table saved to: {csv_path}\n")

    # -----------------------------------------------------------------
    # 🔬 PDB CHECKPOINT: INSPECT CLASSIFIERS & METRICS
    # -----------------------------------------------------------------
    # Uncomment the line below to inspect variables (e.g., df_metrics, X_train_full, y_test, y_pred)
    # pdb.set_trace()
    # -----------------------------------------------------------------

# =====================================================================
# 4. RUNNER CONTROLLER
# =====================================================================

# =====================================================================
# 4. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":

    random.seed(101)

    PROCESSED_DIR = Path(
        r"C:\Users\chloe.jamieson\OneDrive - Simpson College\Documents\GitHub\BryanProgramImaging2026\Experiments\IMAGES2.0\All Images\preprocessed_imagesv2"
    )

    # Separate folders for outputs
    DIAGRAM_DIR = PROCESSED_DIR.parent / "lower_star_diagrams"
    BARCODE_DIR = PROCESSED_DIR.parent / "lower_star_barcodes"

    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    BARCODE_DIR.mkdir(parents=True, exist_ok=True)

    processed_paths = sorted(PROCESSED_DIR.glob("*_processed.tif"))

    if not processed_paths:
        raise FileNotFoundError(
            f"No processed images found in {PROCESSED_DIR}"
        )

    print(f"Found {len(processed_paths)} processed images.")

    # ================================================================
    # Phase 1: Compute persistence diagrams
    # ================================================================

    print("\n=== Phase 1: Computing Lower-Star Persistence Diagrams ===")

    saved_diagrams = compute_lower_star_ph(
        images_paths=processed_paths,
        output_dir=DIAGRAM_DIR
    )

    # ================================================================
    # Phase 2: Convert diagrams to barcode vectors
    # ================================================================

    print("\n=== Phase 2: Vectorizing Persistence Diagrams ===")

    X_topological_features, y_experimental_classes = (
        vectorize_persistence_diagrams(saved_diagrams)
    )

    # Save each barcode individually
    for diagram_path, barcode in zip(saved_diagrams, X_topological_features):

        barcode_name = (
            diagram_path.name
            .replace("_lower_star_diagram.npy",
                     "_lower_star_barcode.npy")
        )

        np.save(BARCODE_DIR / barcode_name, barcode)

    # Save master arrays
    np.save(
        BARCODE_DIR / "lower_star_barcode_matrix.npy",
        X_topological_features
    )

    np.save(
        BARCODE_DIR / "lower_star_labels.npy",
        y_experimental_classes
    )

    print("✅ Barcode vectors saved.")

    # ================================================================
    # Phase 3: Machine Learning
    # ================================================================

    print("\n=== Phase 3: Machine Learning ===")

    run_ml_benchmark(
        X_tda=X_topological_features,
        y=y_experimental_classes,
        output_dir=BARCODE_DIR,
        dataset_title="Lower-Star Barcode Machine Learning"
    )
