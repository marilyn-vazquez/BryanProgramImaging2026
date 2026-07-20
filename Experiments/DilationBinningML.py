# -*- coding: utf-8 -*-
"""
Dilation Filtration Persistence Binning Validation Experiment

Pipeline:
    1. Load V2 preprocessed microscopy images
    2. Convert each image to binary using a fixed threshold
    3. Apply dilation with increasing square structuring elements
    4. Load or compute persistent homology
    5. Apply persistence binning
    6. Save one 18-dimensional vector per image
    7. Create an image-vector manifest
    8. Save the combined machine-learning dataset
    9. Run 100 stratified train/test splits
    10. Train a Linear SVM and Neural Network during each run
    11. Save Accuracy and F1 Score for all runs
    12. Calculate the mean and standard deviation for each model

@author: Gabriel
"""

from pathlib import Path

import cripser
import numba as nb
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

MORPH_TYPE = "dilation"
FILTRATION_NAME = "Dilation"
VECTORIZATION_METHOD = "Persistence_Binning"
PREPROCESSING_VERSION = "V2"
EXPERIMENT_FOLDER_NAME = "Dilation_Preprocessed_V2"

THRESHOLD = 0.5
MAX_SE_LENGTH = 20
N_BINS = 3

BIRTH_RANGE = (0.0, float(MAX_SE_LENGTH))
PERSISTENCE_RANGE = (0.0, float(MAX_SE_LENGTH))

N_RUNS = 100
TEST_SIZE = 0.20

# Keep neural-network initialization constant so the main difference
# between runs is the train/test split.
MLP_RANDOM_STATE = 42


# =====================================================================
# 2. BINARY THRESHOLDING
# =====================================================================

def find(condition):
    """Return indices where a Boolean condition is True."""
    return np.nonzero(condition)


def biImg_by_threshold_leq(img, threshold):
    """
    Convert a grayscale image to binary.

    Pixels <= threshold become 0.
    Pixels > threshold become 1.
    """
    output_img = np.copy(img)
    output_img[find(img <= threshold)] = 0
    output_img[find(img > threshold)] = 1

    return output_img


# =====================================================================
# 3. MORPHOLOGICAL OPERATIONS
# =====================================================================

@nb.jit()
def erosion(
    input_np_array,
    input_list_of_points,
    minimal_pixel_value=0,
):
    """Perform morphological erosion on a binary image."""
    array_shape = np.shape(input_np_array)
    output_array = np.zeros(array_shape)

    for row in range(array_shape[0]):
        for column in range(array_shape[1]):
            if input_np_array[row, column] == minimal_pixel_value:
                output_array[row, column] = minimal_pixel_value
                continue

            relevant_pixel_values = []

            for point_number in range(len(input_list_of_points)):
                source_row = (
                    row
                    - input_list_of_points[point_number][1]
                )
                source_column = (
                    column
                    + input_list_of_points[point_number][0]
                )

                if (
                    source_row >= 0
                    and source_row < array_shape[0]
                    and source_column >= 0
                    and source_column < array_shape[1]
                ):
                    relevant_pixel_values.append(
                        input_np_array[source_row, source_column]
                    )

            output_array[row, column] = min(relevant_pixel_values)

    return output_array


@nb.jit()
def dilation(
    input_np_array,
    input_list_of_points,
    maximal_pixel_value=1,
):
    """Perform morphological dilation on a binary image."""
    array_shape = np.shape(input_np_array)
    output_array = np.zeros(array_shape)

    for row in range(array_shape[0]):
        for column in range(array_shape[1]):
            if input_np_array[row, column] == maximal_pixel_value:
                output_array[row, column] = maximal_pixel_value
                continue

            relevant_pixel_values = []

            for point_number in range(len(input_list_of_points)):
                source_row = (
                    row
                    + input_list_of_points[point_number][1]
                )
                source_column = (
                    column
                    - input_list_of_points[point_number][0]
                )

                if (
                    source_row >= 0
                    and source_row < array_shape[0]
                    and source_column >= 0
                    and source_column < array_shape[1]
                ):
                    relevant_pixel_values.append(
                        input_np_array[source_row, source_column]
                    )

            output_array[row, column] = max(relevant_pixel_values)

    return output_array


def closing(input_np_array, input_list_of_points):
    """Perform dilation followed by erosion."""
    dilated_image = dilation(
        input_np_array,
        input_list_of_points,
    )

    return erosion(
        dilated_image,
        input_list_of_points,
    )


# =====================================================================
# 4. STRUCTURING ELEMENTS
# =====================================================================

@nb.jit()
def get_rectangle_coordinates(input_np_array):
    """Generate coordinate offsets for a rectangular kernel."""
    array_shape = np.shape(input_np_array)
    output_list = []

    origin_row = int(array_shape[0] / 2)
    origin_column = int(array_shape[1] / 2)

    for row in range(array_shape[0]):
        for column in range(array_shape[1]):
            output_list.append(
                np.array(
                    [
                        origin_column - column,
                        origin_row - row,
                    ]
                )
            )

    return output_list


def get_square_SE_list(maximal_SE_lengths):
    """
    Generate square structuring elements from 2x2 through the
    requested maximum size.
    """
    kernel_list = []

    for size in range(2, maximal_SE_lengths + 1):
        kernel_list.append(
            get_rectangle_coordinates(
                np.zeros((size, size))
            )
        )

    return kernel_list


# =====================================================================
# 5. PERSISTENT HOMOLOGY
# =====================================================================

def persistence_of_img(img, maxdim=1):
    """
    Compute H0 and H1 persistence diagrams from a filtration image.
    """
    image = np.asarray(img, dtype=np.float64)

    if hasattr(cripser, "compute_ph"):
        ph = cripser.compute_ph(image, maxdim=maxdim)
    else:
        ph = cripser.computePH(image, maxdim=maxdim)

    persistence_0 = ph[ph[:, 0] == 0][:, 1:3]
    persistence_1 = ph[ph[:, 0] == 1][:, 1:3]

    return [persistence_0, persistence_1]


# =====================================================================
# 6. MORPHOLOGICAL FILTRATION
# =====================================================================

def persistence_of_morph_filtration(
    img,
    kernel_list,
    morph_type="dilation",
):
    """
    Apply the selected morphology operation at increasing scales,
    accumulate the results, and compute persistent homology.
    """
    filtration_image = np.zeros(np.shape(img)) + img

    for kernel in kernel_list:
        if morph_type == "closing":
            morphed_image = closing(
                input_np_array=img,
                input_list_of_points=kernel,
            )
        elif morph_type == "erosion":
            morphed_image = erosion(
                input_np_array=img,
                input_list_of_points=kernel,
            )
        elif morph_type == "dilation":
            morphed_image = dilation(
                input_np_array=img,
                input_list_of_points=kernel,
            )
        else:
            raise ValueError(
                "morph_type must be 'closing', 'erosion', or 'dilation'."
            )

        filtration_image = filtration_image + morphed_image

    return persistence_of_img(
        filtration_image,
        maxdim=1,
    )


# =====================================================================
# 7. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins=3,
    birth_range=(0.0, 20.0),
    persistence_range=(0.0, 20.0),
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
# 8. IMAGE INFORMATION HELPERS
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
# 9. BUILD AND SAVE THE VECTORIZED DATASET
# =====================================================================

def build_morphology_dataset(
    image_paths,
    ph_output_dir,
    image_vector_output_dir,
    morph_type,
    filtration_name,
    vectorization_method,
    preprocessing_version,
    threshold=0.5,
    max_se_length=20,
    n_bins=3,
):
    """
    Build the morphology persistence-binning dataset.

    For each image:
        1. Load the V2 preprocessed image
        2. Convert the image to binary
        3. Load or compute morphology persistent homology
        4. Apply persistence binning
        5. Determine the class label
        6. Save the individual vector
        7. Add one row to the vector manifest
    """
    ph_output_dir = Path(ph_output_dir)
    image_vector_output_dir = Path(image_vector_output_dir)

    image_vector_output_dir.mkdir(parents=True, exist_ok=True)

    kernel_list = get_square_SE_list(
        maximal_SE_lengths=max_se_length
    )

    print(f"Morphology type: {morph_type}")
    print(f"Binary threshold: {threshold}")
    print(f"Maximum structuring element size: {max_se_length}")
    print(f"Number of structuring elements: {len(kernel_list)}")

    feature_rows = []
    labels = []
    image_names = []
    manifest_records = []

    for image_number, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)

        print("\n============================================")
        print(f"IMAGE {image_number} OF {len(image_paths)}")
        print(f"Processing: {image_path.name}")

        image = img_as_float(
            io.imread(
                image_path,
                as_gray=True,
            )
        )

        binary_image = biImg_by_threshold_leq(
            img=image,
            threshold=threshold,
        )

        print(
            f"Foreground pixels: "
            f"{np.sum(binary_image == 1)}"
        )

        ph_save_path = (
            ph_output_dir
            / f"{image_path.stem}_{morph_type}_ph.npz"
        )

        if ph_save_path.exists():
            print(
                f"Loading previously saved V2 {morph_type} PH..."
            )

            with np.load(
                ph_save_path,
                allow_pickle=False,
            ) as saved_ph:
                persistence_0 = saved_ph["H0"]
                persistence_1 = saved_ph["H1"]

            persistence_diagrams = [
                persistence_0,
                persistence_1,
            ]
        else:
            print("Saved PH was not found.")
            print(
                f"Computing {morph_type} filtration and "
                "persistent homology..."
            )

            persistence_diagrams = persistence_of_morph_filtration(
                img=binary_image,
                kernel_list=kernel_list,
                morph_type=morph_type,
            )

            persistence_0 = persistence_diagrams[0]
            persistence_1 = persistence_diagrams[1]

            np.savez_compressed(
                ph_save_path,
                H0=persistence_0,
                H1=persistence_1,
            )

            print(f"Persistent homology saved to:\n{ph_save_path}")

        print(f"H0 intervals: {len(persistence_0)}")
        print(f"H1 intervals: {len(persistence_1)}")

        feature_vector = build_persistence_binning_vector(
            persistence_diagrams=persistence_diagrams,
            n_bins=n_bins,
            birth_range=(0.0, float(max_se_length)),
            persistence_range=(0.0, float(max_se_length)),
        )

        print(f"Feature vector shape: {feature_vector.shape}")

        label, class_name = get_label_from_filename(image_path)
        image_id = get_image_id(image_path)

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
                "Morphology_Type": morph_type,
                "Threshold": threshold,
                "Maximum_SE_Length": max_se_length,
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
# 10. RUN 100 MACHINE-LEARNING EXPERIMENTS
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

    Both models use the same split within each run. The neural-network
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
# 11. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":
    ALL_IMAGES_DIR = Path(
        r"C:\Users\gabriel.garcia\OneDrive - Simpson College\Chloe Jamieson's files - IMAGES2.0\All Images"
    )

    # Confirm that this name exactly matches the actual V2 image folder.
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
    print("BUILDING DILATION V2 PERSISTENCE-BINNING DATASET")
    print("============================================")

    (
        X_topological_features,
        y_experimental_classes,
        image_names,
        manifest_df,
    ) = build_morphology_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        image_vector_output_dir=IMAGE_VECTOR_OUTPUT_DIR,
        morph_type=MORPH_TYPE,
        filtration_name=FILTRATION_NAME,
        vectorization_method=VECTORIZATION_METHOD,
        preprocessing_version=PREPROCESSING_VERSION,
        threshold=THRESHOLD,
        max_se_length=MAX_SE_LENGTH,
        n_bins=N_BINS,
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