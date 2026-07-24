"""
Timed Upper-Star Barcode Persistence Homology and Machine Learning Experiment
@author: Chloe
"""

import os
import random
from pathlib import Path
from time import perf_counter

import cripser as cr
import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =====================================================================
# 1. EXPERIMENT SETTINGS
# =====================================================================
FILTRATION_NAME = "Upper_Star"
VECTORIZATION_METHOD = "Barcode_10D"
PREPROCESSING_VERSION = "V2"
EXPERIMENT_FOLDER_NAME = "Upper_Star_Preprocessed_V2"

TEST_SIZE = 0.20
SPLIT_SEED = 42
MLP_RANDOM_STATE = 42
SAVE_RECOMPUTED_PH = True


# =====================================================================
# 2. CORE HOMOLOGY & VECTORIZATION FUNCTIONS
# =====================================================================
def build_upper_star_filtration(loaded_image):
    """
    Converts an input image to float, scales it to standard 0-255 domain,
    and negates values to invert lower-star filtration into upper-star filtration.

    Parameters
    ----------
    loaded_image : numpy.ndarray
        Raw loaded image array.

    Inputs
    ------
    - Loaded image array.

    Outputs
    -------
    - numpy.ndarray: Scaled and negated float64 image array for upper-star filtration.
    """
    image = img_as_float(loaded_image)
    if image.max() <= 1.0:
        image = image * 255.0
    # Negate image values to invert lower-star filtration into upper-star filtration
    inverted_image = -image
    return np.asarray(inverted_image, dtype=np.float64)


def compute_upper_star_ph(filtration_image):
    """
    Computes upper-star persistent homology using available Cripser API on the negated image.

    Parameters
    ----------
    filtration_image : numpy.ndarray
        Inverted/negated filtration image array.

    Inputs
    ------
    - Filtration image array.

    Outputs
    -------
    - numpy.ndarray: Persistent homology diagram.
    """
    if hasattr(cr, "computePH"): return cr.computePH(filtration_image, maxdim=1)
    if hasattr(cr, "compute_ph"): return cr.compute_ph(filtration_image, maxdim=1)
    raise AttributeError("No compatible Cripser persistent-homology function found.")


def vectorize_persistence_diagram(ph_diagram):
    """
    Converts a raw persistence diagram into a clean 10-dimensional summary feature vector.

    Parameters
    ----------
    ph_diagram : numpy.ndarray
        Raw persistent homology diagram.

    Inputs
    ------
    - Persistent homology diagram array.

    Outputs
    -------
    - numpy.ndarray: 10D statistical feature vector.
    """
    ph = np.asarray(ph_diagram, dtype=np.float64)
    if ph.size == 0: return np.zeros(10, dtype=np.float64)

    # Filter out infinite and absurdly large death values (> 1e9)
    ph_finite = ph[np.all(np.isfinite(ph), axis=1)]
    if len(ph_finite) == 0: return np.zeros(10, dtype=np.float64)

    ph_finite = ph_finite[ph_finite[:, 2] < 1e9]
    if len(ph_finite) == 0: return np.zeros(10, dtype=np.float64)

    # Extract births, deaths, and enforce positive persistence
    births, deaths = ph_finite[:, 1], ph_finite[:, 2]
    persistence = deaths - births

    valid_mask = persistence > 0
    births, deaths, persistence = births[valid_mask], deaths[valid_mask], persistence[valid_mask]
    if len(persistence) == 0: return np.zeros(10, dtype=np.float64)

    # Construct 10D vector: [Mean/Std/Med/Max Birth, Mean/Std/Max Death, Mean/Std/Sum Pers]
    summary_vector = np.array([
        np.mean(births), np.std(births), np.median(births), np.max(births),
        np.mean(deaths), np.std(deaths), np.max(deaths),
        np.mean(persistence), np.std(persistence), np.sum(persistence)
    ], dtype=np.float64)

    return np.nan_to_num(summary_vector, nan=0.0, posinf=0.0, neginf=0.0)


# =====================================================================
# 3. UTILITY & IMAGE HELPERS
# =====================================================================
def get_image_id(image_path):
    """
    Removes file extension and trailing '_processed' suffix from the filename.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the image file.

    Inputs
    ------
    - Image file path.

    Outputs
    -------
    - str: Cleaned image identifier string.
    """
    image_id = Path(image_path).stem
    if image_id.lower().endswith("_processed"): image_id = image_id[:-10]
    return image_id


def get_label_from_filename(image_path):
    """
    Determines numeric and string class labels based on the filename.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the image file.

    Inputs
    ------
    - Image file path.

    Outputs
    -------
    - tuple: Numeric label (int) and string label (str).
    """
    filename = Path(image_path).name.lower()
    if "microgravity" in filename: return 1, "Microgravity"
    if "control" in filename: return 0, "Control"
    raise ValueError(f"Could not determine class label from filename: {Path(image_path).name}")


# =====================================================================
# 4. DATASET CONSTRUCTION AND LOOP
# =====================================================================
def build_and_time_upper_star_dataset(
        image_paths, ph_output_dir, image_vector_output_dir,
        filtration_name, vectorization_method, preprocessing_version,
        save_recomputed_ph=True,
):
    """
    Recomputes, times, and processes all image pipeline stages using upper-star filtration.

    Parameters
    ----------
    image_paths : list of pathlib.Path or str
        Collection of image paths.
    ph_output_dir : str or pathlib.Path
        Directory to store persistence diagram arrays.
    image_vector_output_dir : str or pathlib.Path
        Directory to store vector arrays.
    filtration_name : str
        Name of the filtration technique.
    vectorization_method : str
        Name of the vectorization method.
    preprocessing_version : str
        Version identifier for preprocessing.
    save_recomputed_ph : bool, optional
        Whether to save persistence diagram files to disk. Default is True.

    Inputs
    ------
    - Image paths and configuration parameters.

    Outputs
    -------
    - tuple of pandas.DataFrame: Dataset records DataFrame and timing logs DataFrame.
    """
    os.makedirs(ph_output_dir, exist_ok=True)
    os.makedirs(image_vector_output_dir, exist_ok=True)

    dataset_records, timing_records = [], []

    for path in image_paths:
        img_id = get_image_id(path)
        label_num, label_str = get_label_from_filename(path)
        loaded_img = io.imread(path)

        # Stage 1: Filtration (Upper-Star via Negation)
        t0 = perf_counter()
        filt_img = build_upper_star_filtration(loaded_img)
        t1 = perf_counter()

        # Stage 2: Persistent Homology
        ph_diag = compute_upper_star_ph(filt_img)
        t2 = perf_counter()

        # Stage 3: Vectorization
        vec = vectorize_persistence_diagram(ph_diag)
        t3 = perf_counter()

        # Log Durations
        filt_t, ph_t, vec_t = t1 - t0, t2 - t1, t3 - t2
        timing_records.append({
            "Image_ID": img_id, "Filtration_Time": filt_t, "PH_Time": ph_t, "Vectorization_Time": vec_t
        })

        # Save individual assets
        if save_recomputed_ph:
            np.save(os.path.join(ph_output_dir, f"{img_id}_ph.npy"), ph_diag)
        np.save(os.path.join(image_vector_output_dir, f"{img_id}_vec.npy"), vec)

        # Append to Master Dataset Record
        record = {
            "Image_ID": img_id, "Label_Num": label_num, "Label_Str": label_str,
            "Filtration_Name": filtration_name, "Vectorization_Method": vectorization_method,
            "Preprocessing_Version": preprocessing_version
        }
        for i in range(10): record[f"Feature_{i + 1}"] = vec[i]
        dataset_records.append(record)

    return pd.DataFrame(dataset_records), pd.DataFrame(timing_records)


# =====================================================================
# 5. MACHINE LEARNING MODEL PIPELINE
# =====================================================================
def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a standard machine learning model, recording accuracy, F1-score, and performance.

    Parameters
    ----------
    model : estimator object
        Unfitted scikit-learn model instance.
    name : str
        Model descriptor name.
    X_train : numpy.ndarray
        Training feature matrix.
    X_test : numpy.ndarray
        Testing feature matrix.
    y_train : numpy.ndarray
        Training target labels.
    y_test : numpy.ndarray
        Testing target labels.

    Inputs
    ------
    - Model estimator, name, and split feature/target datasets.

    Outputs
    -------
    - dict: Evaluation metrics dictionary including execution time, accuracy, F1-score, and confusion matrix counts.
    """
    t0 = perf_counter()
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    t1 = perf_counter()

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "Model": name, "Training_Testing_Time_Sec": t1 - t0,
        "Accuracy": accuracy_score(y_test, preds), "F1_Score": f1_score(y_test, preds),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }


def run_experiment(image_paths, output_root):
    """
    Executes the full data construction and model assessment pipeline.

    Parameters
    ----------
    image_paths : list of pathlib.Path or str
        Collection of processed image paths.
    output_root : str or pathlib.Path
        Output directory path to store CSV reports and features.

    Inputs
    ------
    - Image paths and output root directory.

    Outputs
    -------
    - Disk files: dataset.csv, image_timing.csv, timing_summary.csv, and model_results.csv.
    """
    ph_dir = os.path.join(output_root, "PH_Arrays")
    vec_dir = os.path.join(output_root, "Vectors")

    # Process images and compute vectors
    df_data, df_times = build_and_time_upper_star_dataset(
        image_paths, ph_dir, vec_dir, FILTRATION_NAME,
        VECTORIZATION_METHOD, PREPROCESSING_VERSION, SAVE_RECOMPUTED_PH
    )

    df_data.to_csv(os.path.join(output_root, "dataset.csv"), index=False)
    df_times.to_csv(os.path.join(output_root, "image_timing.csv"), index=False)

    # Generate aggregate summary stats
    df_times.describe().to_csv(os.path.join(output_root, "timing_summary.csv"))

    # ML Setup
    feature_cols = [f"Feature_{i + 1}" for i in range(10)]
    X = df_data[feature_cols].values
    y = df_data["Label_Num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    # Evaluate Models
    models = [
        (SVC(kernel="linear"), "Linear_SVM"),
        (SVC(kernel="rbf"), "RBF_SVM"),
        (MLPClassifier(random_state=MLP_RANDOM_STATE, max_iter=1000), "Neural_Network")
    ]

    ml_results = [evaluate_model(m, name, X_train, X_test, y_train, y_test) for m, name in models]
    pd.DataFrame(ml_results).to_csv(os.path.join(output_root, "model_results.csv"), index=False)


# =====================================================================
# 6. MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    # Define directories using your exact path pattern
    PROCESSED_DIR = Path(
        r"C:\Users\chloe.jamieson\OneDrive - Simpson College\Documents\GitHub\BryanProgramImaging2026\Experiments\IMAGES2.0\All Images\preprocessed_imagesv2"
    )
    OUTPUT_ROOT = PROCESSED_DIR.parent / EXPERIMENT_FOLDER_NAME

    # Find images matching pattern
    image_paths = sorted(PROCESSED_DIR.glob("*_processed.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No processed TIF images found in {PROCESSED_DIR}")

    print(f"Found {len(image_paths)} images. Commencing Upper-Star pipeline...")

    # Run the experiment
    run_experiment(image_paths, OUTPUT_ROOT)
    print(f"Upper-Star experiment finished successfully. Results saved to: {OUTPUT_ROOT}")
