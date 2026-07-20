# -*- coding: utf-8 -*-
"""
Upper-Star Persistence Binning Timing Experiment

Pipeline:
    1. Load V2 preprocessed microscopy images
    2. Build the upper-star filtration for every image
    3. Recompute persistent homology for every image
    4. Apply persistence binning to every image
    5. Record filtration, PH, and vectorization time per image
    6. Save individual vectors and the combined dataset
    7. Calculate timing summary statistics across all images
    8. Perform one stratified train/test split
    9. Time one Linear SVM run
    10. Time one Neural Network run
    11. Save model timing, Accuracy, and F1 Score

Timing notes:
    - Image loading and file saving are outside the stage timers.
    - Persistent homology is always recomputed.
    - Existing saved PH results are not loaded.
    - Times are measured with time.perf_counter().

@author: Gabriel
"""

from pathlib import Path
from time import perf_counter

import cripser
import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =====================================================================
# 1. EXPERIMENT SETTINGS
# =====================================================================

FILTRATION_NAME = "Upper_Star"
VECTORIZATION_METHOD = "Persistence_Binning"
PREPROCESSING_VERSION = "V2"
EXPERIMENT_FOLDER_NAME = "Upper_Star_Preprocessed_V2"

N_BINS = 3
BIRTH_RANGE = (0.0, 255.0)
PERSISTENCE_RANGE = (0.0, 255.0)

TEST_SIZE = 0.20
SPLIT_SEED = 42
MLP_RANDOM_STATE = 42

SAVE_RECOMPUTED_PH = False

# =====================================================================
# 2. UPPER-STAR FILTRATION
# =====================================================================

def build_upper_star_filtration(loaded_image):
    """
    Prepare the upper-star filtration image.

    This stage converts the loaded image to floating point, scales it
    to 0-255 when necessary, and inverts the intensity values so that
    bright structures receive lower filtration values.
    """
    image = img_as_float(loaded_image)

    if image.max() <= 1.0:
        image = image * 255.0

    image = np.asarray(
        image,
        dtype=np.float64,
    )

    return image.max() - image


# =====================================================================
# 3. UPPER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_upper_star(filtration_image):
    """Compute upper-star persistent homology."""
    if hasattr(cripser, "compute_ph"):
        return cripser.compute_ph(filtration_image, maxdim=1)

    return cripser.computePH(filtration_image, maxdim=1)


# =====================================================================
# 4. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0),
):
    """
    Convert H0 and H1 diagrams into one persistence-binning vector.

    With n_bins=3:
        H0: 3 x 3 = 9 features
        H1: 3 x 3 = 9 features
        Total: 18 features
    """
    birth_bins = np.linspace(
        birth_range[0],
        birth_range[1],
        n_bins + 1,
    )
    persistence_bins = np.linspace(
        persistence_range[0],
        persistence_range[1],
        n_bins + 1,
    )

    feature_blocks = []

    for diagram in persistence_diagrams:
        diagram = np.asarray(diagram, dtype=np.float64)

        if diagram.size == 0:
            feature_blocks.append(
                np.zeros(n_bins * n_bins, dtype=np.float64)
            )
            continue

        finite_mask = (
            np.isfinite(diagram[:, 0])
            & np.isfinite(diagram[:, 1])
        )
        finite_diagram = diagram[finite_mask]

        if len(finite_diagram) == 0:
            feature_blocks.append(
                np.zeros(n_bins * n_bins, dtype=np.float64)
            )
            continue

        births = finite_diagram[:, 0]
        persistences = finite_diagram[:, 1] - finite_diagram[:, 0]

        valid_mask = persistences >= 0
        births = births[valid_mask]
        persistences = persistences[valid_mask]

        bin_matrix, _, _ = np.histogram2d(
            births,
            persistences,
            bins=[birth_bins, persistence_bins],
            weights=persistences,
        )

        feature_blocks.append(bin_matrix.flatten())

    return np.concatenate(feature_blocks)

# =====================================================================
# 5. IMAGE INFORMATION HELPERS
# =====================================================================

def get_image_id(image_path):
    """Remove the extension and trailing '_processed' suffix."""
    image_id = Path(image_path).stem

    if image_id.lower().endswith("_processed"):
        image_id = image_id[:-len("_processed")]

    return image_id


def get_label_from_filename(image_path):
    """Determine the class label from the image filename."""
    filename = Path(image_path).name.lower()

    if "microgravity" in filename:
        return 1, "Microgravity"

    if "control" in filename:
        return 0, "Control"

    raise ValueError(
        f"Could not determine class label from filename: "
        f"{Path(image_path).name}"
    )

# =====================================================================
# 6. BUILD DATASET AND TIME EVERY IMAGE
# =====================================================================

def build_and_time_upper_star_dataset(
    image_paths,
    ph_output_dir,
    image_vector_output_dir,
    filtration_name,
    vectorization_method,
    preprocessing_version,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0),
    save_recomputed_ph=True,
):
    """
    Recompute and time all image-processing stages.

    Timed independently for every image:
        1. Lower-star filtration construction
        2. Persistent homology computation
        3. Persistence-binning vectorization

    Image loading and output saving are outside the stage timers.
    """
    ph_output_dir = Path(ph_output_dir)
    image_vector_output_dir = Path(image_vector_output_dir)

    if save_recomputed_ph:
        ph_output_dir.mkdir(parents=True, exist_ok=True)

    image_vector_output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    labels = []
    image_names = []
    manifest_records = []
    timing_records = []

    for image_number, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)

        print("\n============================================")
        print(f"IMAGE {image_number} OF {len(image_paths)}")
        print(f"Processing: {image_path.name}")

        # File reading is outside the stage timers.
        loaded_image = io.imread(
            image_path,
            as_gray=True,
        )

        # ---------------------------------------------------------
        # TIME UPPER-STAR FILTRATION CONSTRUCTION
        # ---------------------------------------------------------

        filtration_start = perf_counter()

        filtration_image = build_upper_star_filtration(loaded_image)

        filtration_time = perf_counter() - filtration_start

        # ---------------------------------------------------------
        # TIME PERSISTENT HOMOLOGY
        # ---------------------------------------------------------

        ph_start = perf_counter()

        ph_upper = compute_upper_star(filtration_image)

        ph_time = perf_counter() - ph_start

        # Saving occurs after the PH timer stops.
        ph_filename = f"{image_path.stem}_upper_star_ph.npy"
        ph_save_path = ph_output_dir / ph_filename

        if save_recomputed_ph:
            np.save(ph_save_path, ph_upper)

        # Separate H0 and H1 before timing the binning function.
        persistence_0 = ph_upper[ph_upper[:, 0] == 0][:, 1:3]
        persistence_1 = ph_upper[ph_upper[:, 0] == 1][:, 1:3]

        # ---------------------------------------------------------
        # TIME PERSISTENCE BINNING
        # ---------------------------------------------------------

        vectorization_start = perf_counter()

        feature_vector = build_persistence_binning_vector(
            persistence_diagrams=[persistence_0, persistence_1],
            n_bins=n_bins,
            birth_range=birth_range,
            persistence_range=persistence_range,
        )

        vectorization_time = perf_counter() - vectorization_start

        total_computation_time = (
            filtration_time
            + ph_time
            + vectorization_time
        )

        print(f"H0 intervals: {len(persistence_0)}")
        print(f"H1 intervals: {len(persistence_1)}")
        print(f"Feature vector shape: {feature_vector.shape}")
        print(f"Filtration time: {filtration_time:.6f} seconds")
        print(f"PH time: {ph_time:.6f} seconds")
        print(f"Vectorization time: {vectorization_time:.6f} seconds")
        print(
            f"Total timed computation: "
            f"{total_computation_time:.6f} seconds"
        )

        label, class_name = get_label_from_filename(image_path)
        image_id = get_image_id(image_path)

        vector_filename = (
            f"{image_id}-"
            f"{filtration_name}-"
            f"{vectorization_method}.npy"
        )
        vector_save_path = image_vector_output_dir / vector_filename

        # Saving occurs after the vectorization timer stops.
        np.save(vector_save_path, feature_vector)

        feature_rows.append(feature_vector)
        labels.append(label)
        image_names.append(image_path.name)

        manifest_records.append(
            {
                "Image_ID": image_id,
                "Original_Image_Name": image_path.name,
                "Filtration": filtration_name,
                "Vectorization_Method": vectorization_method,
                "Preprocessing_Version": preprocessing_version,
                "Label": label,
                "Class_Name": class_name,
                "Feature_Count": len(feature_vector),
                "Vector_Filename": vector_filename,
                "PH_Filename": ph_filename if save_recomputed_ph else "",
            }
        )

        timing_records.append(
            {
                "Image_Number": image_number,
                "Image_ID": image_id,
                "Original_Image_Name": image_path.name,
                "Filtration": filtration_name,
                "Vectorization_Method": vectorization_method,
                "Preprocessing_Version": preprocessing_version,
                "Class_Name": class_name,
                "Filtration_Time_Seconds": filtration_time,
                "Persistent_Homology_Time_Seconds": ph_time,
                "Vectorization_Time_Seconds": vectorization_time,
                "Total_Timed_Computation_Seconds": total_computation_time,
            }
        )

    X = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    image_names = np.asarray(image_names, dtype=str)

    manifest_df = pd.DataFrame(manifest_records)
    image_timings_df = pd.DataFrame(timing_records)

    return X, y, image_names, manifest_df, image_timings_df


# =====================================================================
# 7. CREATE IMAGE-STAGE TIMING SUMMARY
# =====================================================================

def create_image_timing_summary(image_timings_df):
    """Calculate timing summary statistics across all images."""
    stage_columns = {
        "Filtration": "Filtration_Time_Seconds",
        "Persistent Homology": "Persistent_Homology_Time_Seconds",
        "Vectorization": "Vectorization_Time_Seconds",
        "Total Timed Computation": "Total_Timed_Computation_Seconds",
    }

    summary_records = []

    for stage_name, column_name in stage_columns.items():
        stage_times = image_timings_df[column_name]

        summary_records.append(
            {
                "Stage": stage_name,
                "Number_Of_Images": stage_times.count(),
                "Mean_Time_Seconds": stage_times.mean(),
                "Standard_Deviation_Seconds": stage_times.std(),
                "Minimum_Time_Seconds": stage_times.min(),
                "Maximum_Time_Seconds": stage_times.max(),
                "Total_Time_Seconds": stage_times.sum(),
            }
        )

    return pd.DataFrame(summary_records)


# =====================================================================
# 8. TIME ONE SVM AND ONE NEURAL NETWORK RUN
# =====================================================================

def run_single_timed_ml_benchmark(
    X,
    y,
    output_dir,
    filtration_name,
    vectorization_method,
    preprocessing_version,
    test_size=0.20,
    split_seed=42,
    mlp_random_state=42,
):
    """
    Perform one train/test split and time each model once.

    Training and prediction are timed separately.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if X.shape[0] != len(y):
        raise ValueError(
            "The feature matrix and label vector contain different "
            "numbers of images."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "The feature matrix contains NaN or infinite values."
        )

    unique_classes, class_counts = np.unique(
        y,
        return_counts=True,
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "Machine learning requires at least two classes."
        )

    if np.any(class_counts < 2):
        raise ValueError(
            "Each class must contain at least two images for a "
            "stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=y,
    )

    print("\n============================================")
    print("SINGLE-RUN MACHINE-LEARNING TIMING")
    print("============================================")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"Features per image: {X.shape[1]}")
    print(f"Split seed: {split_seed}")

    models = [
        (
            "Linear SVM",
            SVC(
                kernel="linear",
                C=1.0,
            ),
        ),
        (
            "Neural Network (MLP)",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=mlp_random_state,
            ),
        ),
    ]

    model_records = []

    for model_name, classifier in models:
        model_pipeline = make_pipeline(
            StandardScaler(),
            classifier,
        )

        training_start = perf_counter()

        model_pipeline.fit(X_train, y_train)

        training_time = perf_counter() - training_start

        prediction_start = perf_counter()

        y_pred = model_pipeline.predict(X_test)

        prediction_time = perf_counter() - prediction_start
        total_model_time = training_time + prediction_time

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0,
        )

        model_records.append(
            {
                "Filtration": filtration_name,
                "Vectorization_Method": vectorization_method,
                "Preprocessing_Version": preprocessing_version,
                "Model": model_name,
                "Split_Seed": split_seed,
                "Training_Samples": len(y_train),
                "Testing_Samples": len(y_test),
                "Training_Time_Seconds": training_time,
                "Prediction_Time_Seconds": prediction_time,
                "Total_Model_Time_Seconds": total_model_time,
                "Accuracy": accuracy,
                "F1_Score": f1,
            }
        )

        print("\n--------------------------------------------")
        print(f"Model: {model_name}")
        print(f"Training time: {training_time:.6f} seconds")
        print(f"Prediction time: {prediction_time:.6f} seconds")
        print(f"Total model time: {total_model_time:.6f} seconds")
        print(f"Accuracy: {accuracy:.6f}")
        print(f"F1 Score: {f1:.6f}")

    model_timings_df = pd.DataFrame(model_records)
    model_timings_path = output_dir / "model_timings.csv"

    model_timings_df.to_csv(
        model_timings_path,
        index=False,
        float_format="%.9f",
    )

    print(f"\nModel timing table saved to:\n{model_timings_path}")

    return model_timings_df


# =====================================================================
# 9. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":
    script_start = perf_counter()

    ALL_IMAGES_DIR = Path(
        r"C:\Users\gabriel.garcia\OneDrive - Simpson College\Chloe Jamieson's files - IMAGES2.0\All Images"
    )

    
    PROCESSED_DIR = ALL_IMAGES_DIR / "preprocessed_imagesv2"

    
    TIMING_RESULTS_DIR = (
        ALL_IMAGES_DIR
        / "GabesTimingResults"
        / EXPERIMENT_FOLDER_NAME
        / VECTORIZATION_METHOD
    )

    PH_OUTPUT_DIR = TIMING_RESULTS_DIR / "Persistent_Homology"
    IMAGE_VECTOR_OUTPUT_DIR = TIMING_RESULTS_DIR / "Image_Vectors"
    DATASET_OUTPUT_DIR = TIMING_RESULTS_DIR / "Dataset"
    TABLE_OUTPUT_DIR = TIMING_RESULTS_DIR / "Tables"

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            "The V2 preprocessed-image folder was not found:\n"
            f"{PROCESSED_DIR}"
        )

    if SAVE_RECOMPUTED_PH:
        PH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    IMAGE_VECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n============================================")
    print("UPPER-STAR TIMING EXPERIMENT")
    print("============================================")
    print(f"V2 preprocessed images:\n{PROCESSED_DIR}")
    print(f"\nTiming results:\n{TIMING_RESULTS_DIR}")
    print("\nExisting saved PH results will not be loaded.")

    image_paths = sorted(PROCESSED_DIR.glob("*_processed.tif"))

    if not image_paths:
        raise FileNotFoundError(
            "Could not find any V2 preprocessed images in:\n"
            f"{PROCESSED_DIR}"
        )

    print(f"\nFound {len(image_paths)} V2 preprocessed images.")

    (
        X_topological_features,
        y_experimental_classes,
        image_names,
        manifest_df,
        image_timings_df,
    ) = build_and_time_upper_star_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        image_vector_output_dir=IMAGE_VECTOR_OUTPUT_DIR,
        filtration_name=FILTRATION_NAME,
        vectorization_method=VECTORIZATION_METHOD,
        preprocessing_version=PREPROCESSING_VERSION,
        n_bins=N_BINS,
        birth_range=BIRTH_RANGE,
        persistence_range=PERSISTENCE_RANGE,
        save_recomputed_ph=SAVE_RECOMPUTED_PH,
    )

    expected_features = 2 * N_BINS * N_BINS

    if X_topological_features.shape[1] != expected_features:
        raise ValueError(
            "Unexpected persistence-binning vector length. "
            f"Expected {expected_features}, but received "
            f"{X_topological_features.shape[1]}."
        )

    # Save the combined dataset.
    np.save(
        DATASET_OUTPUT_DIR / "combined_features.npy",
        X_topological_features,
    )
    np.save(
        DATASET_OUTPUT_DIR / "labels.npy",
        y_experimental_classes,
    )
    np.save(
        DATASET_OUTPUT_DIR / "image_names.npy",
        image_names,
    )

    # Save the manifest and image timing tables.
    manifest_path = TABLE_OUTPUT_DIR / "image_vector_manifest.csv"
    image_timings_path = TABLE_OUTPUT_DIR / "image_stage_timings.csv"
    image_summary_path = TABLE_OUTPUT_DIR / "image_timing_summary.csv"

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )
    image_timings_df.to_csv(
        image_timings_path,
        index=False,
        float_format="%.9f",
    )

    image_timing_summary_df = create_image_timing_summary(
        image_timings_df
    )
    image_timing_summary_df.to_csv(
        image_summary_path,
        index=False,
        float_format="%.9f",
    )

    print("\n============================================")
    print("IMAGE TIMING SUMMARY")
    print("============================================")
    print(image_timing_summary_df.to_string(index=False))
    print(f"\nPer-image timing table saved to:\n{image_timings_path}")
    print(f"\nImage timing summary saved to:\n{image_summary_path}")

    # Time one SVM and one Neural Network run.
    run_single_timed_ml_benchmark(
        X=X_topological_features,
        y=y_experimental_classes,
        output_dir=TABLE_OUTPUT_DIR,
        filtration_name=FILTRATION_NAME,
        vectorization_method=VECTORIZATION_METHOD,
        preprocessing_version=PREPROCESSING_VERSION,
        test_size=TEST_SIZE,
        split_seed=SPLIT_SEED,
        mlp_random_state=MLP_RANDOM_STATE,
    )

    script_total_time = perf_counter() - script_start

    script_runtime_path = TABLE_OUTPUT_DIR / "total_script_runtime.csv"
    pd.DataFrame(
        [
            {
                "Filtration": FILTRATION_NAME,
                "Vectorization_Method": VECTORIZATION_METHOD,
                "Preprocessing_Version": PREPROCESSING_VERSION,
                "Total_Script_Runtime_Seconds": script_total_time,
            }
        ]
    ).to_csv(
        script_runtime_path,
        index=False,
        float_format="%.9f",
    )

    print("\n============================================")
    print("TIMING EXPERIMENT COMPLETE")
    print("============================================")
    print(f"Total script runtime: {script_total_time:.6f} seconds")
    print(f"Total runtime table saved to:\n{script_runtime_path}")