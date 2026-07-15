# -*- coding: utf-8 -*-
"""
Upper-Star Persistence Binning Machine Learning Experiment

Pipeline:
    1. Load preprocessed microscopy images
    2. Compute upper-star persistent homology
    3. Save persistent homology for each image
    4. Apply persistence binning
    5. Build an 18-dimensional feature vector for each image
    6. Save vectorization results
    7. Train a Linear SVM
    8. Train a Neural Network
    9. Calculate accuracy, F1 score, and confusion matrices
    10. Save classification results

@author: Gabriel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cripser

from skimage import io
from skimage.util import img_as_float

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =====================================================================
# 1. EXPERIMENT SETTINGS
# =====================================================================

# Number of bins along the birth and persistence axes
N_BINS = 3


# Images are scaled to 0-255 before persistent homology
BIRTH_RANGE = (
    0.0,
    255.0
)


PERSISTENCE_RANGE = (
    0.0,
    255.0
)


# =====================================================================
# 2. UPPER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_upper_star(
    preprocessed_img
):
    """
    Compute an upper-star persistence diagram from a
    preprocessed grayscale image.

    The image intensity values are inverted so that bright
    structures receive low filtration values and therefore
    enter the filtration earlier.

    Parameters
    ----------
    preprocessed_img : numpy.ndarray
        Preprocessed grayscale image.

    Returns
    -------
    numpy.ndarray
        Raw Cripser persistence diagram.
    """

    # ---------------------------------------------------------------
    # INVERT IMAGE INTENSITIES
    # ---------------------------------------------------------------

    inverted_img = (
        preprocessed_img.max()
        -
        preprocessed_img
    )


    # ---------------------------------------------------------------
    # COMPUTE UPPER-STAR PERSISTENT HOMOLOGY
    # ---------------------------------------------------------------

    if hasattr(
        cripser,
        "compute_ph"
    ):

        ph_upper = cripser.compute_ph(
            inverted_img.astype(float),
            maxdim=1
        )

    else:

        ph_upper = cripser.computePH(
            inverted_img.astype(float),
            maxdim=1
        )


    return ph_upper


# =====================================================================
# 3. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0)
):
    """
    Convert H0 and H1 persistence diagrams into one fixed-length
    persistence-binning feature vector.

    Each persistence point is transformed from:

        (birth, death)

    to:

        (birth, persistence)

    where:

        persistence = death - birth

    A two-dimensional histogram is constructed over the
    birth-persistence plane.

    Each persistence point contributes its persistence value as the
    weight of the bin containing that point.

    H0 and H1 are binned separately and then concatenated.

    Parameters
    ----------
    persistence_diagrams : list
        List containing:

            persistence_diagrams[0] = H0 diagram
            persistence_diagrams[1] = H1 diagram

        Each diagram contains:

            [birth, death]

    n_bins : int, optional
        Number of bins along each axis.
        Default is 3.

    birth_range : tuple, optional
        Minimum and maximum birth values used for binning.

    persistence_range : tuple, optional
        Minimum and maximum persistence values used for binning.

    Returns
    -------
    numpy.ndarray
        Fixed-length persistence-binning feature vector.

        With n_bins=3:

            H0 = 3 x 3 = 9 features
            H1 = 3 x 3 = 9 features

            Total = 18 features
    """

    # ---------------------------------------------------------------
    # CREATE BIN EDGES
    # ---------------------------------------------------------------

    birth_bins = np.linspace(
        birth_range[0],
        birth_range[1],
        n_bins + 1
    )


    persistence_bins = np.linspace(
        persistence_range[0],
        persistence_range[1],
        n_bins + 1
    )


    feature_blocks = []


    # ---------------------------------------------------------------
    # PROCESS H0 AND H1 SEPARATELY
    # ---------------------------------------------------------------

    for diagram in persistence_diagrams:

        pd = np.asarray(
            diagram,
            dtype=np.float64
        )


        # -----------------------------------------------------------
        # HANDLE EMPTY PERSISTENCE DIAGRAM
        # -----------------------------------------------------------

        if pd.size == 0:

            bin_matrix = np.zeros(
                (
                    n_bins,
                    n_bins
                ),
                dtype=np.float64
            )


            feature_blocks.append(
                bin_matrix.flatten()
            )


            continue


        # -----------------------------------------------------------
        # REMOVE INFINITE FEATURES
        # -----------------------------------------------------------

        finite_mask = (
            np.isfinite(
                pd[:, 0]
            )
            &
            np.isfinite(
                pd[:, 1]
            )
        )


        pd_finite = pd[
            finite_mask
        ]


        # -----------------------------------------------------------
        # HANDLE DIAGRAM WITH NO FINITE FEATURES
        # -----------------------------------------------------------

        if len(
            pd_finite
        ) == 0:

            bin_matrix = np.zeros(
                (
                    n_bins,
                    n_bins
                ),
                dtype=np.float64
            )


            feature_blocks.append(
                bin_matrix.flatten()
            )


            continue


        # -----------------------------------------------------------
        # CONVERT BIRTH-DEATH TO BIRTH-PERSISTENCE
        # -----------------------------------------------------------

        births = pd_finite[
            :,
            0
        ]


        deaths = pd_finite[
            :,
            1
        ]


        persistences = (
            deaths
            -
            births
        )


        # -----------------------------------------------------------
        # REMOVE INVALID NEGATIVE PERSISTENCE VALUES
        # -----------------------------------------------------------

        valid_mask = (
            persistences >= 0
        )


        births = births[
            valid_mask
        ]


        persistences = persistences[
            valid_mask
        ]


        # -----------------------------------------------------------
        # BUILD WEIGHTED 2D PERSISTENCE-BINNING GRID
        # -----------------------------------------------------------

        bin_matrix, _, _ = np.histogram2d(
            births,
            persistences,
            bins=[
                birth_bins,
                persistence_bins
            ],
            weights=persistences
        )


        # -----------------------------------------------------------
        # FLATTEN THE 2D GRID
        # -----------------------------------------------------------

        feature_blocks.append(
            bin_matrix.flatten()
        )


    # ---------------------------------------------------------------
    # CONCATENATE H0 AND H1
    # ---------------------------------------------------------------

    feature_vector = np.concatenate(
        feature_blocks
    )


    return feature_vector


# =====================================================================
# 4. BUILD UPPER-STAR PERSISTENCE-BINNING DATASET
# =====================================================================

def build_upper_star_dataset(
    image_paths,
    ph_output_dir,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0)
):
    """
    Build a machine-learning dataset from preprocessed images.

    For each image:

        1. Load the preprocessed image
        2. Scale intensities to 0-255
        3. Compute or load upper-star persistent homology
        4. Save persistent homology if newly computed
        5. Separate H0 and H1
        6. Apply persistence binning
        7. Store the resulting feature vector
        8. Assign the experimental class label

    Parameters
    ----------
    image_paths : list
        Paths to preprocessed microscopy images.

    ph_output_dir : pathlib.Path
        Folder used to save and load persistent homology results.

    n_bins : int, optional
        Number of persistence bins along each axis.

    birth_range : tuple, optional
        Birth-value range.

    persistence_range : tuple, optional
        Persistence-value range.

    Returns
    -------
    tuple
        X : numpy.ndarray
            Feature matrix.

        y : numpy.ndarray
            Class labels.

            0 = Control
            1 = Microgravity

        image_names : numpy.ndarray
            Image filenames.
    """

    # ---------------------------------------------------------------
    # MAKE SURE PH OUTPUT DIRECTORY EXISTS
    # ---------------------------------------------------------------

    ph_output_dir = Path(
        ph_output_dir
    )


    ph_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ---------------------------------------------------------------
    # CREATE DATASET STORAGE LISTS
    # ---------------------------------------------------------------

    X = []

    y = []

    image_names = []


    # ---------------------------------------------------------------
    # PROCESS EACH IMAGE
    # ---------------------------------------------------------------

    for image_number, path in enumerate(
        image_paths,
        start=1
    ):

        path = Path(
            path
        )


        print(
            "\n============================================"
        )


        print(
            f"IMAGE {image_number} OF {len(image_paths)}"
        )


        print(
            f"Processing: {path.name}"
        )


        # -----------------------------------------------------------
        # LOAD PREPROCESSED IMAGE
        # -----------------------------------------------------------

        img = img_as_float(
            io.imread(
                path,
                as_gray=True
            )
        )


        # -----------------------------------------------------------
        # SCALE IMAGE TO 0-255
        # -----------------------------------------------------------

        if img.max() <= 1.0:

            img = (
                img
                *
                255.0
            )


        img = np.asarray(
            img,
            dtype=np.float64
        )


        # -----------------------------------------------------------
        # DEFINE PERSISTENT HOMOLOGY SAVE PATH
        # -----------------------------------------------------------

        ph_save_path = (
            ph_output_dir
            /
            f"{path.stem}_upper_star_ph.npy"
        )


        # -----------------------------------------------------------
        # LOAD SAVED PH IF IT ALREADY EXISTS
        # -----------------------------------------------------------

        if ph_save_path.exists():

            print(
                "Loading previously saved upper-star "
                "persistent homology..."
            )


            ph_upper = np.load(
                ph_save_path
            )


            print(
                "Loaded from:"
            )


            print(
                ph_save_path
            )


        # -----------------------------------------------------------
        # OTHERWISE COMPUTE AND SAVE PH
        # -----------------------------------------------------------

        else:

            print(
                "Computing upper-star persistent homology..."
            )


            ph_upper = compute_upper_star(
                img
            )


            np.save(
                ph_save_path,
                ph_upper
            )


            print(
                "Persistent homology saved to:"
            )


            print(
                ph_save_path
            )


        print(
            "Raw persistence diagram shape:",
            ph_upper.shape
        )


        # -----------------------------------------------------------
        # SEPARATE H0 AND H1
        # -----------------------------------------------------------

        persistence_0 = ph_upper[
            ph_upper[:, 0] == 0
        ][
            :,
            1:3
        ]


        persistence_1 = ph_upper[
            ph_upper[:, 0] == 1
        ][
            :,
            1:3
        ]


        print(
            "H0 intervals:",
            len(
                persistence_0
            )
        )


        print(
            "H1 intervals:",
            len(
                persistence_1
            )
        )


        # -----------------------------------------------------------
        # APPLY PERSISTENCE BINNING
        # -----------------------------------------------------------

        print(
            "Applying persistence binning..."
        )


        feature_vector = build_persistence_binning_vector(
            persistence_diagrams=[
                persistence_0,
                persistence_1
            ],
            n_bins=n_bins,
            birth_range=birth_range,
            persistence_range=persistence_range
        )


        print(
            "Feature vector shape:",
            feature_vector.shape
        )


        X.append(
            feature_vector
        )


        # -----------------------------------------------------------
        # ASSIGN CLASS LABEL
        # -----------------------------------------------------------

        filename_lower = (
            path.name.lower()
        )


        if "microgravity" in filename_lower:

            label = 1


        elif "control" in filename_lower:

            label = 0


        else:

            raise ValueError(
                "Could not determine class label "
                f"from filename: {path.name}"
            )


        y.append(
            label
        )


        image_names.append(
            path.name
        )


    # ---------------------------------------------------------------
    # CONVERT TO NUMPY ARRAYS
    # ---------------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float64
    )


    y = np.asarray(
        y,
        dtype=int
    )


    image_names = np.asarray(
        image_names
    )


    return (
        X,
        y,
        image_names
    )


# =====================================================================
# 5. MACHINE LEARNING
# =====================================================================

def run_ml_benchmark(
    X,
    y,
    output_dir
):
    """
    Train and evaluate:

        1. Linear Support Vector Machine
        2. Multilayer Perceptron Neural Network

    Both models use standardized persistence-binning feature vectors.

    Accuracy, F1 score, and confusion matrices are calculated
    and saved.
    """

    # ---------------------------------------------------------------
    # MAKE SURE CLASSIFICATION OUTPUT DIRECTORY EXISTS
    # ---------------------------------------------------------------

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ---------------------------------------------------------------
    # DEFINE MODELS
    # ---------------------------------------------------------------

    models = [

        (
            "Linear SVM",

            SVC(
                kernel="linear",
                C=1.0,
                random_state=42
            )
        ),


        (
            "Neural Network (MLP)",

            MLPClassifier(
                hidden_layer_sizes=(
                    32,
                    16
                ),
                max_iter=1000,
                random_state=42
            )
        )
    ]


    # ---------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print(
        "\n============================================"
    )


    print(
        "MACHINE LEARNING DATASET"
    )


    print(
        "============================================"
    )


    print(
        "Training samples:",
        X_train.shape[0]
    )


    print(
        "Testing samples:",
        X_test.shape[0]
    )


    print(
        "Features per image:",
        X_train.shape[1]
    )


    metrics_records = []

    confusion_records = []


    # ---------------------------------------------------------------
    # TRAIN EACH MODEL
    # ---------------------------------------------------------------

    for (
        model_name,
        classifier
    ) in models:


        print(
            "\n============================================"
        )


        print(
            f"TRAINING: {model_name}"
        )


        print(
            "============================================"
        )


        # -----------------------------------------------------------
        # CREATE MODEL PIPELINE
        # -----------------------------------------------------------

        model_pipeline = make_pipeline(
            StandardScaler(),
            classifier
        )


        # -----------------------------------------------------------
        # TRAIN MODEL
        # -----------------------------------------------------------

        model_pipeline.fit(
            X_train,
            y_train
        )


        # -----------------------------------------------------------
        # MAKE PREDICTIONS
        # -----------------------------------------------------------

        y_pred = model_pipeline.predict(
            X_test
        )


        # -----------------------------------------------------------
        # CALCULATE ACCURACY
        # -----------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        # -----------------------------------------------------------
        # CALCULATE F1 SCORE
        # -----------------------------------------------------------

        f1 = f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )


        # -----------------------------------------------------------
        # CALCULATE CONFUSION MATRIX
        # -----------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[
                0,
                1
            ]
        )


        # -----------------------------------------------------------
        # PRINT RESULTS
        # -----------------------------------------------------------

        print(
            f"Accuracy: {accuracy:.4f}"
        )


        print(
            f"F1 Score: {f1:.4f}"
        )


        print(
            "Confusion Matrix:"
        )


        print(
            cm
        )


        # -----------------------------------------------------------
        # STORE RESULTS
        # -----------------------------------------------------------

        metrics_records.append(
            {
                "Model": model_name,

                "Accuracy": round(
                    accuracy,
                    4
                ),

                "F1-Score": round(
                    f1,
                    4
                ),

                "TN": cm[
                    0,
                    0
                ],

                "FP": cm[
                    0,
                    1
                ],

                "FN": cm[
                    1,
                    0
                ],

                "TP": cm[
                    1,
                    1
                ]
            }
        )


        confusion_records.append(
            (
                model_name,
                cm
            )
        )


    # ---------------------------------------------------------------
    # DISPLAY CONFUSION MATRICES
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10,
            4
        )
    )


    for ax, (
        model_name,
        cm
    ) in zip(
        axes,
        confusion_records
    ):


        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Control",
                "Microgravity"
            ]
        )


        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )


        ax.set_title(
            model_name
        )


    plt.tight_layout()


    # ---------------------------------------------------------------
    # SAVE CONFUSION MATRIX FIGURE
    # ---------------------------------------------------------------

    confusion_matrix_path = (
        output_dir
        /
        "upper_star_confusion_matrices.png"
    )


    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        "\nConfusion matrix figure saved to:"
    )


    print(
        confusion_matrix_path
    )


    plt.show()


    # ---------------------------------------------------------------
    # CREATE RESULTS TABLE
    # ---------------------------------------------------------------

    df_metrics = pd.DataFrame(
        metrics_records
    )


    print(
        "\n============================================"
    )


    print(
        "UPPER-STAR PERSISTENCE BINNING RESULTS"
    )


    print(
        "============================================"
    )


    print(
        df_metrics.to_string(
            index=False
        )
    )


    # ---------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------

    csv_path = (
        output_dir
        /
        "upper_star_persistence_binning_ml_metrics.csv"
    )


    df_metrics.to_csv(
        csv_path,
        index=False
    )


    print(
        "\nMetrics saved to:"
    )


    print(
        csv_path
    )


# =====================================================================
# 6. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # INPUT FOLDER: PREPROCESSED IMAGES
    # ---------------------------------------------------------------

    PROCESSED_DIR = Path(
        r"C:\Users\gabriel.garcia\OneDrive - Simpson College\Chloe Jamieson's files - IMAGES2.0\All Images\preprocessed_images"
    )


    # ---------------------------------------------------------------
    # BASE RESULTS FOLDER
    # ---------------------------------------------------------------

    BASE_RESULTS_DIR = Path(
        r"C:\Users\gabriel.garcia\OneDrive - Simpson College\Chloe Jamieson's files - IMAGES2.0\All Images\Results"
    )


    # ---------------------------------------------------------------
    # FILTRATION NAME
    # ---------------------------------------------------------------

    FILTRATION_NAME = "Upper_Star"


    # ---------------------------------------------------------------
    # FILTRATION-SPECIFIC RESULTS FOLDER
    # ---------------------------------------------------------------

    RESULTS_DIR = (
        BASE_RESULTS_DIR
        /
        FILTRATION_NAME
    )


    # ---------------------------------------------------------------
    # PERSISTENT HOMOLOGY OUTPUT FOLDER
    # ---------------------------------------------------------------

    PH_OUTPUT_DIR = (
        RESULTS_DIR
        /
        "Persistent_Homology"
    )


    # ---------------------------------------------------------------
    # VECTORIZATION OUTPUT FOLDER
    # ---------------------------------------------------------------

    VECTORIZATION_OUTPUT_DIR = (
        RESULTS_DIR
        /
        "Vectorization"
    )


    # ---------------------------------------------------------------
    # CLASSIFICATION OUTPUT FOLDER
    # ---------------------------------------------------------------

    CLASSIFICATION_OUTPUT_DIR = (
        RESULTS_DIR
        /
        "Classification"
    )


    # ---------------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # ---------------------------------------------------------------

    PH_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    VECTORIZATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    CLASSIFICATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ---------------------------------------------------------------
    # REPORT OUTPUT FOLDERS
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        "OUTPUT FOLDERS"
    )


    print(
        "============================================"
    )


    print(
        "Persistent Homology:"
    )


    print(
        PH_OUTPUT_DIR
    )


    print(
        "\nVectorization:"
    )


    print(
        VECTORIZATION_OUTPUT_DIR
    )


    print(
        "\nClassification:"
    )


    print(
        CLASSIFICATION_OUTPUT_DIR
    )


    # ---------------------------------------------------------------
    # FIND PREPROCESSED IMAGES
    # ---------------------------------------------------------------

    image_paths = sorted(
        PROCESSED_DIR.glob(
            "*_processed.tif"
        )
    )


    # ---------------------------------------------------------------
    # CHECK THAT IMAGES WERE FOUND
    # ---------------------------------------------------------------

    if not image_paths:

        raise FileNotFoundError(
            f"Could not find any preprocessed images in:\n"
            f"{PROCESSED_DIR}"
        )


    print(
        f"\nFound {len(image_paths)} "
        f"preprocessed images."
    )


    # ---------------------------------------------------------------
    # BUILD UPPER-STAR PERSISTENCE-BINNING DATASET
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        "BUILDING UPPER-STAR "
        "PERSISTENCE-BINNING DATASET"
    )


    print(
        "============================================"
    )


    (
        X_topological_features,
        y_experimental_classes,
        image_names
    ) = build_upper_star_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        n_bins=N_BINS,
        birth_range=BIRTH_RANGE,
        persistence_range=PERSISTENCE_RANGE
    )


    # ---------------------------------------------------------------
    # REPORT DATASET INFORMATION
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        "DATASET COMPLETE"
    )


    print(
        "============================================"
    )


    print(
        "Feature matrix shape:",
        X_topological_features.shape
    )


    print(
        "Label vector shape:",
        y_experimental_classes.shape
    )


    print(
        "Control images:",
        np.sum(
            y_experimental_classes == 0
        )
    )


    print(
        "Microgravity images:",
        np.sum(
            y_experimental_classes == 1
        )
    )


    # ---------------------------------------------------------------
    # VERIFY EXPECTED FEATURE VECTOR LENGTH
    # ---------------------------------------------------------------

    expected_features = (
        2
        *
        N_BINS
        *
        N_BINS
    )


    print(
        "Expected features per image:",
        expected_features
    )


    if (
        X_topological_features.shape[1]
        !=
        expected_features
    ):

        raise ValueError(
            "Unexpected persistence-binning vector length. "
            f"Expected {expected_features}, but received "
            f"{X_topological_features.shape[1]}."
        )


    # ---------------------------------------------------------------
    # DEFINE VECTORIZATION SAVE PATHS
    # ---------------------------------------------------------------

    features_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        "upper_star_18d_"
        "persistence_binning_features.npy"
    )


    labels_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        "upper_star_"
        "persistence_binning_labels.npy"
    )


    names_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        "upper_star_"
        "persistence_binning_names.npy"
    )


    # ---------------------------------------------------------------
    # SAVE FEATURE MATRIX
    # ---------------------------------------------------------------

    np.save(
        features_save_path,
        X_topological_features
    )


    # ---------------------------------------------------------------
    # SAVE LABELS
    # ---------------------------------------------------------------

    np.save(
        labels_save_path,
        y_experimental_classes
    )


    # ---------------------------------------------------------------
    # SAVE IMAGE NAMES
    # ---------------------------------------------------------------

    np.save(
        names_save_path,
        image_names
    )


    print(
        "\n============================================"
    )


    print(
        "VECTORIZATION RESULTS SAVED"
    )


    print(
        "============================================"
    )


    print(
        "Features:"
    )


    print(
        features_save_path
    )


    print(
        "\nLabels:"
    )


    print(
        labels_save_path
    )


    print(
        "\nImage names:"
    )


    print(
        names_save_path
    )


    # ---------------------------------------------------------------
    # RUN MACHINE LEARNING
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        "RUNNING LINEAR SVM AND NEURAL NETWORK"
    )


    print(
        "============================================"
    )


    run_ml_benchmark(
        X=X_topological_features,
        y=y_experimental_classes,
        output_dir=CLASSIFICATION_OUTPUT_DIR
    )