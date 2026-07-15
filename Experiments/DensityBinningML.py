# -*- coding: utf-8 -*-
"""
Density Filtration Persistence Binning Machine Learning Experiment

Pipeline:
    1. Load preprocessed microscopy images
    2. Convert each image to binary using Otsu thresholding
    3. Compute a density filtration using KDTree
    4. Compute persistent homology with Cripser
    5. Save persistent homology for each image
    6. Apply persistence binning
    7. Build an 18-dimensional feature vector for each image
    8. Save vectorization results
    9. Train a Linear SVM
    10. Train a Neural Network
    11. Calculate accuracy, F1 score, and confusion matrices
    12. Save classification results

@author: Gabriel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cripser

from skimage import io, filters
from skimage.util import img_as_float

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =====================================================================
# 1. EXPERIMENT SETTINGS
# =====================================================================

# Radius used for neighborhood density calculations
MAX_DIST = 5


# Number of bins along the birth and persistence axes
N_BINS = 3


# =====================================================================
# 2. DENSITY FILTRATION
# =====================================================================

def density_filtration(
    binary_image,
    max_dist=5
):
    """
    Generate a density-based filtration from a binary image.

    Local pixel density is calculated by counting foreground pixels
    within a specified radius using a KDTree.

    Dense regions receive lower filtration values and therefore
    enter the filtration earlier.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image containing foreground structures.

    max_dist : float, optional
        Radius used for neighborhood density calculations.
        Default is 5.

    Returns
    -------
    numpy.ndarray
        Density filtration image.
    """

    # ---------------------------------------------------------------
    # GET IMAGE DIMENSIONS
    # ---------------------------------------------------------------

    height, width = binary_image.shape


    # ---------------------------------------------------------------
    # FIND FOREGROUND PIXEL COORDINATES
    # ---------------------------------------------------------------

    points = np.argwhere(
        binary_image
    )


    # ---------------------------------------------------------------
    # MAKE SURE FOREGROUND PIXELS EXIST
    # ---------------------------------------------------------------

    if len(points) == 0:

        raise ValueError(
            "The binary image contains no foreground pixels. "
            "Density filtration cannot be computed."
        )


    # ---------------------------------------------------------------
    # BUILD KDTREE FROM FOREGROUND PIXELS
    # ---------------------------------------------------------------

    tree = KDTree(
        points,
        leaf_size=30,
        metric="euclidean"
    )


    # ---------------------------------------------------------------
    # CREATE COORDINATES FOR EVERY PIXEL
    # ---------------------------------------------------------------

    point_cloud = np.zeros(
        (
            height * width,
            2
        )
    )


    p = 0


    for i in range(height):

        for j in range(width):

            point_cloud[p, 0] = i

            point_cloud[p, 1] = j

            p += 1


    # ---------------------------------------------------------------
    # COUNT FOREGROUND NEIGHBORS
    # ---------------------------------------------------------------

    num_nbhs = tree.query_radius(
        point_cloud,
        r=max_dist,
        count_only=True
    )


    filt_func_vals = num_nbhs


    # ---------------------------------------------------------------
    # FIND MAXIMUM OBSERVED DENSITY
    # ---------------------------------------------------------------

    max_num_nbhs = filt_func_vals.max()


    # ---------------------------------------------------------------
    # CONVERT DENSITY INTO FILTRATION VALUES
    # ---------------------------------------------------------------

    # High density:
    #     high neighbor count
    #     becomes low filtration value
    #     enters earlier
    #
    # Low density:
    #     low neighbor count
    #     becomes high filtration value
    #     enters later

    filt_func_vals = (
        max_num_nbhs
        -
        filt_func_vals
    )


    # ---------------------------------------------------------------
    # RESHAPE BACK INTO IMAGE FORM
    # ---------------------------------------------------------------

    density_filt_img = filt_func_vals.reshape(
        height,
        width
    )


    return density_filt_img


# =====================================================================
# 3. DENSITY PERSISTENT HOMOLOGY
# =====================================================================

def compute_density_ph(
    binary_image,
    max_dist=5
):
    """
    Compute persistent homology using a density filtration.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary input image.

    max_dist : float, optional
        Neighborhood radius used for density calculations.
        Default is 5.

    Returns
    -------
    tuple
        density_img : numpy.ndarray
            Density filtration image.

        ph_density : numpy.ndarray
            Raw Cripser persistence diagram.
    """

    # ---------------------------------------------------------------
    # COMPUTE DENSITY FILTRATION
    # ---------------------------------------------------------------

    density_img = density_filtration(
        binary_image=binary_image,
        max_dist=max_dist
    )


    # ---------------------------------------------------------------
    # COMPUTE PERSISTENT HOMOLOGY
    # ---------------------------------------------------------------

    if hasattr(
        cripser,
        "compute_ph"
    ):

        ph_density = cripser.compute_ph(
            density_img.astype(
                np.float64
            ),
            maxdim=1
        )


    else:

        ph_density = cripser.computePH(
            density_img.astype(
                np.float64
            ),
            maxdim=1
        )


    return (
        density_img,
        ph_density
    )


# =====================================================================
# 4. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins,
    birth_range,
    persistence_range
):
    """
    Convert H0 and H1 persistence diagrams into one fixed-length
    persistence-binning feature vector.

    Each persistence point is transformed from:

        (birth, death)

    into:

        (birth, persistence)

    where:

        persistence = death - birth

    A two-dimensional histogram is constructed over the
    birth-persistence plane.

    Each persistence point contributes its persistence value as
    the weight of the bin containing that point.

    H0 and H1 are binned separately and then concatenated.

    Parameters
    ----------
    persistence_diagrams : list
        List containing:

            persistence_diagrams[0] = H0 diagram
            persistence_diagrams[1] = H1 diagram

        Each diagram contains:

            [birth, death]

    n_bins : int
        Number of bins along each axis.

    birth_range : tuple
        Minimum and maximum birth values.

    persistence_range : tuple
        Minimum and maximum persistence values.

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

        diagram = np.asarray(
            diagram,
            dtype=np.float64
        )


        # -----------------------------------------------------------
        # HANDLE EMPTY DIAGRAM
        # -----------------------------------------------------------

        if diagram.size == 0:

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
                diagram[:, 0]
            )
            &
            np.isfinite(
                diagram[:, 1]
            )
        )


        diagram_finite = diagram[
            finite_mask
        ]


        # -----------------------------------------------------------
        # HANDLE DIAGRAM WITH NO FINITE FEATURES
        # -----------------------------------------------------------

        if len(
            diagram_finite
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

        births = diagram_finite[
            :,
            0
        ]


        deaths = diagram_finite[
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
# 5. BUILD DENSITY PERSISTENCE-BINNING DATASET
# =====================================================================

def build_density_dataset(
    image_paths,
    ph_output_dir,
    max_dist=5,
    n_bins=3
):
    """
    Build a machine-learning dataset from preprocessed images.

    For each image:

        1. Load the preprocessed image
        2. Apply Otsu thresholding
        3. Create a binary image
        4. Compute or load density persistent homology
        5. Save persistent homology if newly computed
        6. Separate H0 and H1
        7. Apply persistence binning
        8. Store the resulting feature vector
        9. Assign the experimental class label

    Parameters
    ----------
    image_paths : list
        Paths to preprocessed microscopy images.

    ph_output_dir : pathlib.Path
        Folder used to save and load persistent homology results.

    max_dist : float, optional
        Radius used for density calculations.
        Default is 5.

    n_bins : int, optional
        Number of bins along each axis.
        Default is 3.

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
    # DETERMINE FIXED DENSITY FILTRATION RANGE
    # ---------------------------------------------------------------

    # Count the number of integer pixel coordinates that can exist
    # within the circular KDTree search radius.
    #
    # For max_dist = 5, this gives a maximum possible count of 81.

    coordinate_range = np.arange(
        -max_dist,
        max_dist + 1
    )


    row_offsets, column_offsets = np.meshgrid(
        coordinate_range,
        coordinate_range,
        indexing="ij"
    )


    max_density_value = np.sum(
        (
            row_offsets ** 2
            +
            column_offsets ** 2
        )
        <=
        max_dist ** 2
    )


    # ---------------------------------------------------------------
    # DEFINE FIXED RANGES SHARED BY EVERY IMAGE
    # ---------------------------------------------------------------

    birth_range = (
        0.0,
        float(
            max_density_value
        )
    )


    persistence_range = (
        0.0,
        float(
            max_density_value
        )
    )


    print(
        "Density neighborhood radius:",
        max_dist
    )


    print(
        "Maximum possible neighborhood count:",
        max_density_value
    )


    print(
        "Birth range:",
        birth_range
    )


    print(
        "Persistence range:",
        persistence_range
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

        img_grayscale = img_as_float(
            io.imread(
                path,
                as_gray=True
            )
        )


        # -----------------------------------------------------------
        # CHECK FOR CONSTANT IMAGE
        # -----------------------------------------------------------

        if (
            img_grayscale.min()
            ==
            img_grayscale.max()
        ):

            raise ValueError(
                f"Image contains only one intensity value: "
                f"{path.name}"
            )


        # -----------------------------------------------------------
        # APPLY OTSU THRESHOLDING
        # -----------------------------------------------------------

        threshold_value = filters.threshold_otsu(
            img_grayscale
        )


        binary_img = (
            img_grayscale
            >
            threshold_value
        )


        print(
            "Otsu threshold:",
            threshold_value
        )


        print(
            "Foreground pixels:",
            np.sum(
                binary_img
            )
        )


        # -----------------------------------------------------------
        # DEFINE PERSISTENT HOMOLOGY SAVE PATH
        # -----------------------------------------------------------

        ph_save_path = (
            ph_output_dir
            /
            f"{path.stem}_density_ph.npy"
        )


        # -----------------------------------------------------------
        # LOAD SAVED PH IF IT ALREADY EXISTS
        # -----------------------------------------------------------

        if ph_save_path.exists():

            print(
                "Loading previously saved density "
                "persistent homology..."
            )


            ph_density = np.load(
                ph_save_path
            )


            print(
                "Loaded from:"
            )


            print(
                ph_save_path
            )


        # -----------------------------------------------------------
        # OTHERWISE COMPUTE DENSITY FILTRATION AND SAVE PH
        # -----------------------------------------------------------

        else:

            print(
                "Computing density filtration and "
                "persistent homology..."
            )


            (
                density_img,
                ph_density
            ) = compute_density_ph(
                binary_image=binary_img,
                max_dist=max_dist
            )


            print(
                "Density filtration minimum:",
                density_img.min()
            )


            print(
                "Density filtration maximum:",
                density_img.max()
            )


            np.save(
                ph_save_path,
                ph_density
            )


            print(
                "Persistent homology saved to:"
            )


            print(
                ph_save_path
            )


        print(
            "Raw persistence diagram shape:",
            ph_density.shape
        )


        # -----------------------------------------------------------
        # SEPARATE H0 AND H1
        # -----------------------------------------------------------

        persistence_0 = ph_density[
            ph_density[:, 0] == 0
        ][
            :,
            1:3
        ]


        persistence_1 = ph_density[
            ph_density[:, 0] == 1
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
# 6. MACHINE LEARNING
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
        "density_confusion_matrices.png"
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
        "DENSITY PERSISTENCE BINNING RESULTS"
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
        "density_persistence_binning_ml_metrics.csv"
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
# 7. RUNNER CONTROLLER
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

    FILTRATION_NAME = "Density"


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
    # BUILD DENSITY PERSISTENCE-BINNING DATASET
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        "BUILDING DENSITY "
        "PERSISTENCE-BINNING DATASET"
    )


    print(
        "============================================"
    )


    (
        X_topological_features,
        y_experimental_classes,
        image_names
    ) = build_density_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        max_dist=MAX_DIST,
        n_bins=N_BINS
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
        "density_18d_"
        "persistence_binning_features.npy"
    )


    labels_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        "density_"
        "persistence_binning_labels.npy"
    )


    names_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        "density_"
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