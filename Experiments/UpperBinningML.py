# -*- coding: utf-8 -*-
"""
Upper-Star Persistence Binning Validation Experiment

Pipeline:
    1. Load V2 preprocessed microscopy images
    2. Load or compute upper-star persistent homology
    3. Apply persistence binning
    4. Save one 18-dimensional vector per image
    5. Create an image-vector manifest
    6. Save the combined machine-learning dataset
    7. Run 100 stratified train/test splits
    8. Train a Linear SVM and Neural Network during each run
    9. Save Accuracy and F1 Score for all runs
    10. Calculate the mean and standard deviation for each model

@author: Gabriel
"""

from pathlib import Path
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

N_RUNS = 100
TEST_SIZE = 0.20

# Keep the neural-network initialization constant so that the main
# difference between runs is the train/test split.
MLP_RANDOM_STATE = 42


# =====================================================================
# 2. UPPER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_upper_star(preprocessed_img):
    """
    Compute upper-star persistent homology from a grayscale image.

    Image intensities are inverted so that bright structures receive
    lower filtration values and enter the filtration earlier.
    """
    image = np.asarray(preprocessed_img, dtype=float)
    inverted_image = image.max() - image

    if hasattr(cripser, "compute_ph"):
        return cripser.compute_ph(inverted_image, maxdim=1)

    return cripser.computePH(inverted_image, maxdim=1)


# =====================================================================
# 3. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0),
):
    """
    Convert H0 and H1 diagrams into one persistence-binning vector.

    Each point is converted from (birth, death) to
    (birth, persistence), where persistence = death - birth.

    With n_bins=3:
        H0: 3 x 3 = 9 features
        H1: 3 x 3 = 9 features
        Total: 18 features
    """
    birth_bins = np.linspace(birth_range[0], birth_range[1], n_bins + 1)
    persistence_bins = np.linspace(
        persistence_range[0],
        persistence_range[1],
        n_bins + 1,
    )

    feature_blocks = []

    for diagram in persistence_diagrams:
        diagram = np.asarray(diagram, dtype=np.float64)

        if diagram.size == 0:
            feature_blocks.append(np.zeros(n_bins * n_bins, dtype=np.float64))
            continue

        finite_mask = (
            np.isfinite(diagram[:, 0])
            & np.isfinite(diagram[:, 1])
        )
        finite_diagram = diagram[finite_mask]

        if len(finite_diagram) == 0:
            feature_blocks.append(np.zeros(n_bins * n_bins, dtype=np.float64))
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
# 4. IMAGE INFORMATION HELPERS
# =====================================================================

def get_image_id(image_path):
    """
    Remove the file extension and trailing '_processed' suffix.

    Example:
        control_stub1_0001_processed.tif
        becomes
        control_stub1_0001
    """
    image_id = Path(image_path).stem

    if image_id.lower().endswith("_processed"):
        image_id = image_id[:-len("_processed")]

    return image_id


def get_label_from_filename(image_path):
    """
    Determine the class label from the image filename.

    Returns:
        (1, "Microgravity") for microgravity images
        (0, "Control") for control images
    """
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
# 5. BUILD AND SAVE THE VECTORIZED DATASET
# =====================================================================

def build_upper_star_dataset(
    image_paths,
    ph_output_dir,
    image_vector_output_dir,
    filtration_name,
    vectorization_method,
    preprocessing_version,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0),
):
    """
    Build the Upper Star persistence-binning dataset.

    For each image:
        1. Load the V2 preprocessed image
        2. Load or compute persistent homology
        3. Separate H0 and H1
        4. Apply persistence binning
        5. Determine the class label
        6. Save the individual vector
        7. Add the vector to the combined dataset
        8. Add one row to the vector manifest
    """
    ph_output_dir = Path(ph_output_dir)
    image_vector_output_dir = Path(image_vector_output_dir)

    image_vector_output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    labels = []
    image_names = []
    manifest_records = []

    for image_number, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)

        print("\n============================================")
        print(f"IMAGE {image_number} OF {len(image_paths)}")
        print(f"Processing: {image_path.name}")

        # Load and scale image to 0-255.
        image = img_as_float(io.imread(image_path, as_gray=True))

        if image.max() <= 1.0:
            image = image * 255.0

        image = np.asarray(image, dtype=np.float64)

        # Load or compute persistent homology.
        ph_save_path = (
            ph_output_dir
            / f"{image_path.stem}_upper_star_ph.npy"
        )

        if ph_save_path.exists():
            print("Loading previously saved V2 upper-star PH...")
            ph_upper = np.load(ph_save_path, allow_pickle=False)
        else:
            print("Saved PH was not found.")
            print("Computing upper-star persistent homology...")
            ph_upper = compute_upper_star(image)
            np.save(ph_save_path, ph_upper)
            print(f"Persistent homology saved to:\n{ph_save_path}")

        # Separate H0 and H1.
        persistence_0 = ph_upper[ph_upper[:, 0] == 0][:, 1:3]
        persistence_1 = ph_upper[ph_upper[:, 0] == 1][:, 1:3]

        print(f"H0 intervals: {len(persistence_0)}")
        print(f"H1 intervals: {len(persistence_1)}")

        # Create the persistence-binning vector.
        feature_vector = build_persistence_binning_vector(
            persistence_diagrams=[persistence_0, persistence_1],
            n_bins=n_bins,
            birth_range=birth_range,
            persistence_range=persistence_range,
        )

        print(f"Feature vector shape: {feature_vector.shape}")

        label, class_name = get_label_from_filename(image_path)
        image_id = get_image_id(image_path)

        # Save the vector using the requested naming format.
        vector_filename = (
            f"{image_id}-"
            f"{filtration_name}-"
            f"{vectorization_method}.npy"
        )
        vector_save_path = image_vector_output_dir / vector_filename

        np.save(vector_save_path, feature_vector)
        print(f"Individual image vector saved to:\n{vector_save_path}")

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
            }
        )

    X = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    image_names = np.asarray(image_names, dtype=str)
    manifest_df = pd.DataFrame(manifest_records)

    return X, y, image_names, manifest_df


# =====================================================================
# 6. RUN 100 MACHINE-LEARNING EXPERIMENTS
# =====================================================================

def run_repeated_ml_benchmark(
    X,
    y,
    output_dir,
    filtration_name,
    vectorization_method,
    preprocessing_version,
    n_runs=100,
    test_size=0.20,
    mlp_random_state=42,
):
    """
    Run repeated stratified train/test experiments.

    Each run uses a different split seed. The SVM and Neural Network
    use the same train/test split within a run. The Neural Network
    initialization remains constant.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_runs < 2:
        raise ValueError(
            "n_runs must be at least 2 to calculate a standard deviation."
        )

    if X.shape[0] != len(y):
        raise ValueError(
            "The feature matrix and label vector contain different "
            "numbers of images."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "The feature matrix contains NaN or infinite values."
        )

    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError(
            "Machine learning requires at least two classes."
        )

    if np.any(class_counts < 2):
        raise ValueError(
            "Each class must contain at least two images for a "
            "stratified train/test split."
        )

    print("\n============================================")
    print("REPEATED MACHINE-LEARNING VALIDATION")
    print("============================================")
    print(f"Total images: {X.shape[0]}")
    print(f"Features per image: {X.shape[1]}")
    print(f"Number of runs: {n_runs}")
    print(f"Testing proportion: {test_size}")

    all_run_records = []

    for run_number in range(1, n_runs + 1):
        split_seed = run_number - 1

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )

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

        for model_name, classifier in models:
            model_pipeline = make_pipeline(
                StandardScaler(),
                classifier,
            )

            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(
                y_test,
                y_pred,
                average="binary",
                zero_division=0,
            )

            all_run_records.append(
                {
                    "Run": run_number,
                    "Seed": split_seed,
                    "Filtration": filtration_name,
                    "Vectorization_Method": vectorization_method,
                    "Preprocessing_Version": preprocessing_version,
                    "Model": model_name,
                    "Training_Samples": len(y_train),
                    "Testing_Samples": len(y_test),
                    "Accuracy": accuracy,
                    "F1_Score": f1,
                }
            )

        if run_number == 1 or run_number % 10 == 0:
            print(f"Completed run {run_number} of {n_runs}")

    # Table 2: all model results from all runs.
    all_runs_df = pd.DataFrame(all_run_records)
    all_runs_path = output_dir / f"all_{n_runs}_runs.csv"

    all_runs_df.to_csv(
        all_runs_path,
        index=False,
        float_format="%.6f",
    )

    # Table 3: mean and sample standard deviation for each model.
    summary_df = (
        all_runs_df.groupby(
            [
                "Filtration",
                "Vectorization_Method",
                "Preprocessing_Version",
                "Model",
            ],
            as_index=False,
        )
        .agg(
            Runs=("Run", "count"),
            Mean_Accuracy=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            Mean_F1=("F1_Score", "mean"),
            F1_SD=("F1_Score", "std"),
        )
    )

    summary_path = output_dir / "summary_statistics.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        float_format="%.6f",
    )

    print("\n============================================")
    print("100-RUN VALIDATION COMPLETE")
    print("============================================")
    print(f"\nAll individual run results saved to:\n{all_runs_path}")
    print(f"\nSummary statistics saved to:\n{summary_path}")
    print("\nSUMMARY STATISTICS")
    print(summary_df.to_string(index=False))

    return all_runs_df, summary_df


# =====================================================================
# 7. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":
    ALL_IMAGES_DIR = Path(
        r"C:\Users\gabriel.garcia\OneDrive - Simpson College\Chloe Jamieson's files - IMAGES2.0\All Images"
    )

    # Confirm that this folder name exactly matches the actual V2 folder.
    PROCESSED_DIR = ALL_IMAGES_DIR / "preprocessed_imagesv2"

    # Existing V2 persistent-homology results.
    PH_OUTPUT_DIR = (
        ALL_IMAGES_DIR
        / "GabesResults"
        / EXPERIMENT_FOLDER_NAME
        / "Persistent_Homology"
    )

    # New validation outputs.
    VALIDATION_RESULTS_DIR = (
        ALL_IMAGES_DIR
        / "GabesValidationResults"
        / EXPERIMENT_FOLDER_NAME
        / VECTORIZATION_METHOD
    )

    IMAGE_VECTOR_OUTPUT_DIR = VALIDATION_RESULTS_DIR / "Image_Vectors"
    DATASET_OUTPUT_DIR = VALIDATION_RESULTS_DIR / "Dataset"
    TABLE_OUTPUT_DIR = VALIDATION_RESULTS_DIR / "Tables"

    # The PH folder should already exist. Do not silently create an
    # empty folder because of a path typo.
    if not PH_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            "The existing V2 persistent-homology folder was not found:\n"
            f"{PH_OUTPUT_DIR}"
        )

    IMAGE_VECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n============================================")
    print("EXPERIMENT FOLDERS")
    print("============================================")
    print(f"V2 preprocessed images:\n{PROCESSED_DIR}")
    print(f"\nExisting V2 persistent homology:\n{PH_OUTPUT_DIR}")
    print(f"\nNew validation results:\n{VALIDATION_RESULTS_DIR}")

    image_paths = sorted(PROCESSED_DIR.glob("*_processed.tif"))

    if not image_paths:
        raise FileNotFoundError(
            "Could not find any V2 preprocessed images in:\n"
            f"{PROCESSED_DIR}\n\n"
            "Check the spelling of the PROCESSED_DIR folder."
        )

    print(f"\nFound {len(image_paths)} V2 preprocessed images.")

    print("\n============================================")
    print("BUILDING UPPER-STAR V2 PERSISTENCE-BINNING DATASET")
    print("============================================")

    (
        X_topological_features,
        y_experimental_classes,
        image_names,
        manifest_df,
    ) = build_upper_star_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        image_vector_output_dir=IMAGE_VECTOR_OUTPUT_DIR,
        filtration_name=FILTRATION_NAME,
        vectorization_method=VECTORIZATION_METHOD,
        preprocessing_version=PREPROCESSING_VERSION,
        n_bins=N_BINS,
        birth_range=BIRTH_RANGE,
        persistence_range=PERSISTENCE_RANGE,
    )

    print("\n============================================")
    print("DATASET COMPLETE")
    print("============================================")
    print(f"Feature matrix shape: {X_topological_features.shape}")
    print(f"Label vector shape: {y_experimental_classes.shape}")
    print(
        "Control images:",
        np.sum(y_experimental_classes == 0),
    )
    print(
        "Microgravity images:",
        np.sum(y_experimental_classes == 1),
    )

    expected_features = 2 * N_BINS * N_BINS
    print(f"Expected features per image: {expected_features}")

    if X_topological_features.shape[1] != expected_features:
        raise ValueError(
            "Unexpected persistence-binning vector length. "
            f"Expected {expected_features}, but received "
            f"{X_topological_features.shape[1]}."
        )

    # Save the combined dataset.
    combined_features_path = DATASET_OUTPUT_DIR / "combined_features.npy"
    labels_path = DATASET_OUTPUT_DIR / "labels.npy"
    image_names_path = DATASET_OUTPUT_DIR / "image_names.npy"

    np.save(combined_features_path, X_topological_features)
    np.save(labels_path, y_experimental_classes)
    np.save(image_names_path, image_names)

    print("\n============================================")
    print("COMBINED DATASET SAVED")
    print("============================================")
    print(f"Features:\n{combined_features_path}")
    print(f"\nLabels:\n{labels_path}")
    print(f"\nImage names:\n{image_names_path}")

    # Table 1: image-vector manifest.
    manifest_path = TABLE_OUTPUT_DIR / "image_vector_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nImage-vector manifest saved to:\n{manifest_path}")

    # Tables 2 and 3: all 100 runs and summary statistics.
    run_repeated_ml_benchmark(
        X=X_topological_features,
        y=y_experimental_classes,
        output_dir=TABLE_OUTPUT_DIR,
        filtration_name=FILTRATION_NAME,
        vectorization_method=VECTORIZATION_METHOD,
        preprocessing_version=PREPROCESSING_VERSION,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        mlp_random_state=MLP_RANDOM_STATE,
    )