# -*- coding: utf-8 -*-
"""
Erosion Filtration Persistence Binning Machine Learning Experiment

Pipeline:
    1. Load preprocessed microscopy images
    2. Convert each image to binary using a fixed threshold
    3. Apply an erosion filtration using increasing square
       structuring elements
    4. Compute persistent homology
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

import numba as nb
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

MORPH_TYPE = "erosion"


# ---------------------------------------------------------------
# BINARY THRESHOLD
# ---------------------------------------------------------------

# Pixels <= threshold become 0
# Pixels > threshold become 1

THRESHOLD = 0.5


# ---------------------------------------------------------------
# STRUCTURING ELEMENT SETTINGS
# ---------------------------------------------------------------

# Square structuring elements will increase from:
#
#     2 x 2
#     3 x 3
#     ...
#     MAX_SE_LENGTH x MAX_SE_LENGTH

MAX_SE_LENGTH = 20


# ---------------------------------------------------------------
# PERSISTENCE BINNING SETTINGS
# ---------------------------------------------------------------

# Number of bins along the birth and persistence axes

N_BINS = 3


# The accumulated morphology filtration begins with the original
# binary image and adds one morphed binary image for every
# structuring element from size 2 through MAX_SE_LENGTH.
#
# Therefore, with MAX_SE_LENGTH = 20, the filtration values
# can range from 0 to 20.

BIRTH_RANGE = (
    0.0,
    float(MAX_SE_LENGTH)
)


PERSISTENCE_RANGE = (
    0.0,
    float(MAX_SE_LENGTH)
)


# =====================================================================
# 2. BINARY THRESHOLDING
# =====================================================================

def find(
    condition
):
    """
    Find indices in an array satisfying a given condition.

    Parameters
    ----------
    condition : numpy.ndarray
        Boolean array indicating desired locations.

    Returns
    -------
    tuple
        Array indices where the condition is True.
    """

    res = np.nonzero(
        condition
    )


    return res


def biImg_by_threshold_leq(
    img,
    threshold
):
    """
    Convert an image into a binary image.

    Pixels with intensity values less than or equal to the threshold
    are assigned 0.

    Pixels with intensity values greater than the threshold
    are assigned 1.

    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image.

    threshold : float
        Intensity threshold value.

    Returns
    -------
    numpy.ndarray
        Binary image.
    """

    output_img = np.copy(
        img
    )


    idxs_0 = find(
        img <= threshold
    )


    idxs_1 = find(
        img > threshold
    )


    output_img[
        idxs_0
    ] = 0


    output_img[
        idxs_1
    ] = 1


    return output_img


# =====================================================================
# 3. MORPHOLOGICAL OPERATIONS
# =====================================================================

@nb.jit()
def erosion(
    input_np_array,
    input_list_of_points,
    minimal_pixel_value=0
):
    """
    Perform morphological erosion on a binary image.

    Erosion removes boundary pixels and reduces foreground structures.
    """

    input_np_array_shape = np.shape(
        input_np_array
    )


    output_np_array = np.zeros(
        input_np_array_shape
    )


    for i in range(
        input_np_array_shape[0]
    ):

        for j in range(
            input_np_array_shape[1]
        ):

            if (
                input_np_array[
                    i,
                    j
                ]
                ==
                minimal_pixel_value
            ):

                output_np_array[
                    i,
                    j
                ] = minimal_pixel_value

                continue


            relevant_pixel_values = []


            for k in range(
                len(
                    input_list_of_points
                )
            ):

                m = (
                    i
                    -
                    input_list_of_points[k][1]
                )


                n = (
                    j
                    +
                    input_list_of_points[k][0]
                )


                if (
                    (m >= 0)
                    and
                    (
                        m
                        <
                        input_np_array_shape[0]
                    )
                    and
                    (n >= 0)
                    and
                    (
                        n
                        <
                        input_np_array_shape[1]
                    )
                ):

                    relevant_pixel_values.append(
                        input_np_array[
                            m,
                            n
                        ]
                    )


            output_np_array[
                i,
                j
            ] = min(
                relevant_pixel_values
            )


    return output_np_array


@nb.jit()
def dilation(
    input_np_array,
    input_list_of_points,
    maximal_pixel_value=1
):
    """
    Perform morphological dilation on a binary image.

    Dilation expands foreground structures.
    """

    input_np_array_shape = np.shape(
        input_np_array
    )


    output_np_array = np.zeros(
        input_np_array_shape
    )


    for i in range(
        input_np_array_shape[0]
    ):

        for j in range(
            input_np_array_shape[1]
        ):

            if (
                input_np_array[
                    i,
                    j
                ]
                ==
                maximal_pixel_value
            ):

                output_np_array[
                    i,
                    j
                ] = maximal_pixel_value

                continue


            relevant_pixel_values = []


            for k in range(
                len(
                    input_list_of_points
                )
            ):

                m = (
                    i
                    +
                    input_list_of_points[k][1]
                )


                n = (
                    j
                    -
                    input_list_of_points[k][0]
                )


                if (
                    (m >= 0)
                    and
                    (
                        m
                        <
                        input_np_array_shape[0]
                    )
                    and
                    (n >= 0)
                    and
                    (
                        n
                        <
                        input_np_array_shape[1]
                    )
                ):

                    relevant_pixel_values.append(
                        input_np_array[
                            m,
                            n
                        ]
                    )


            output_np_array[
                i,
                j
            ] = max(
                relevant_pixel_values
            )


    return output_np_array


def closing(
    input_np_array,
    input_list_of_points
):
    """
    Perform morphological closing.

    Closing consists of:

        dilation
        followed by
        erosion

    Returns
    -------
    numpy.ndarray
        Closed binary image.
    """

    return erosion(

        dilation(
            input_np_array,
            input_list_of_points
        ),

        input_list_of_points
    )


# =====================================================================
# 4. STRUCTURING ELEMENTS
# =====================================================================

@nb.jit()
def get_rectangle_coordinates(
    input_np_array
):
    """
    Generate coordinate offsets for a rectangular structuring element.
    """

    input_np_array_shape = np.shape(
        input_np_array
    )


    output_list = []


    origin_i = int(
        input_np_array_shape[0]
        /
        2
    )


    origin_j = int(
        input_np_array_shape[1]
        /
        2
    )


    for i in range(
        input_np_array_shape[0]
    ):

        for j in range(
            input_np_array_shape[1]
        ):

            output_list.append(

                np.array(
                    [
                        origin_j - j,
                        origin_i - i
                    ]
                )

            )


    return output_list


def get_square_SE_list(
    maximal_SE_lengths
):
    """
    Generate square structuring elements of increasing size.

    The resulting list contains:

        2 x 2
        3 x 3
        ...
        maximal_SE_lengths x maximal_SE_lengths
    """

    kernel_list = []


    for i in range(
        2,
        maximal_SE_lengths + 1
    ):

        kernel_list.append(

            get_rectangle_coordinates(

                input_np_array=np.zeros(
                    (
                        i,
                        i
                    )
                )

            )

        )


    return kernel_list


# =====================================================================
# 5. PERSISTENT HOMOLOGY
# =====================================================================

def persistence_of_img(
    img,
    maxdim=1
):
    """
    Compute persistent homology of a filtration image.

    Returns
    -------
    list
        Two persistence diagrams:

            persistence_0 = H0
            persistence_1 = H1
    """

    img = np.asarray(
        img,
        dtype=np.float64
    )


    # ---------------------------------------------------------------
    # COMPUTE PERSISTENT HOMOLOGY
    # ---------------------------------------------------------------

    if hasattr(
        cripser,
        "compute_ph"
    ):

        ph = cripser.compute_ph(
            img,
            maxdim=maxdim
        )

    else:

        ph = cripser.computePH(
            img,
            maxdim=maxdim
        )


    # ---------------------------------------------------------------
    # SEPARATE H0 AND H1
    # ---------------------------------------------------------------

    persistence_0 = ph[
        ph[:, 0] == 0
    ][
        :,
        1:3
    ]


    persistence_1 = ph[
        ph[:, 0] == 1
    ][
        :,
        1:3
    ]


    return [
        persistence_0,
        persistence_1
    ]


# =====================================================================
# 6. MORPHOLOGICAL FILTRATION
# =====================================================================

def persistence_of_morph_filtration(
    img,
    kernel_list,
    morph_type="erosion"
):
    """
    Compute persistent homology from a morphological filtration.

    The selected morphological operation is applied using
    structuring elements of increasing size.

    Supported values:

        "closing"
        "erosion"
        "dilation"
    """

    # ---------------------------------------------------------------
    # INITIALIZE FILTRATION IMAGE
    # ---------------------------------------------------------------

    img_shape = np.shape(
        img
    )


    img_buff = np.zeros(
        img_shape
    )


    # Begin with the original binary image
    img_buff = (
        img_buff
        +
        img
    )


    # ---------------------------------------------------------------
    # APPLY INCREASING STRUCTURING ELEMENTS
    # ---------------------------------------------------------------

    for the_kernel in kernel_list:

        if morph_type == "closing":

            morphed_img = closing(
                input_np_array=img,
                input_list_of_points=the_kernel
            )


        elif morph_type == "erosion":

            morphed_img = erosion(
                input_np_array=img,
                input_list_of_points=the_kernel
            )


        elif morph_type == "dilation":

            morphed_img = dilation(
                input_np_array=img,
                input_list_of_points=the_kernel
            )


        else:

            raise ValueError(
                "morph_type must be "
                "'closing', 'erosion', or 'dilation'"
            )


        # Add this morphological scale
        # to the accumulated filtration image

        img_buff = (
            img_buff
            +
            morphed_img
        )


    # ---------------------------------------------------------------
    # COMPUTE PERSISTENT HOMOLOGY
    # ---------------------------------------------------------------

    persistence_diagrams = persistence_of_img(
        img_buff,
        maxdim=1
    )


    return persistence_diagrams


# =====================================================================
# 7. PERSISTENCE BINNING
# =====================================================================

def build_persistence_binning_vector(
    persistence_diagrams,
    n_bins=3,
    birth_range=(0.0, 20.0),
    persistence_range=(0.0, 20.0)
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

    H0 and H1 are binned separately and then concatenated.

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
        # HANDLE NO FINITE FEATURES
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
        # REMOVE INVALID NEGATIVE PERSISTENCE
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
        # BUILD WEIGHTED 2D BINNING GRID
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
# 8. BUILD MORPHOLOGY PERSISTENCE-BINNING DATASET
# =====================================================================

def build_morphology_dataset(
    image_paths,
    ph_output_dir,
    morph_type,
    threshold=0.5,
    max_se_length=20,
    n_bins=3
):
    """
    Build a machine-learning dataset from preprocessed images.

    For each image:

        1. Load the preprocessed image
        2. Convert the image to binary
        3. Compute or load the selected morphology PH
        4. Save persistent homology if newly computed
        5. Apply persistence binning
        6. Store the resulting feature vector
        7. Assign the experimental class label
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
    # CREATE STRUCTURING ELEMENTS ONCE
    # ---------------------------------------------------------------

    kernel_list = get_square_SE_list(
        maximal_SE_lengths=max_se_length
    )


    print(
        "Morphology type:",
        morph_type
    )


    print(
        "Binary threshold:",
        threshold
    )


    print(
        "Maximum structuring element size:",
        max_se_length
    )


    print(
        "Number of structuring elements:",
        len(
            kernel_list
        )
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
        # CONVERT TO BINARY IMAGE
        # -----------------------------------------------------------

        binary_img = biImg_by_threshold_leq(
            img=img_grayscale,
            threshold=threshold
        )


        print(
            "Foreground pixels:",
            np.sum(
                binary_img == 1
            )
        )


        # -----------------------------------------------------------
        # DEFINE PERSISTENT HOMOLOGY SAVE PATH
        # -----------------------------------------------------------

        ph_save_path = (
            ph_output_dir
            /
            f"{path.stem}_{morph_type}_ph.npz"
        )


        # -----------------------------------------------------------
        # LOAD SAVED PH IF IT ALREADY EXISTS
        # -----------------------------------------------------------

        if ph_save_path.exists():

            print(
                f"Loading previously saved {morph_type} "
                "persistent homology..."
            )


            with np.load(
                ph_save_path
            ) as saved_ph:

                persistence_0 = saved_ph[
                    "H0"
                ]


                persistence_1 = saved_ph[
                    "H1"
                ]


            persistence_diagrams = [
                persistence_0,
                persistence_1
            ]


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
                f"Computing {morph_type} filtration and "
                "persistent homology..."
            )


            persistence_diagrams = persistence_of_morph_filtration(
                img=binary_img,
                kernel_list=kernel_list,
                morph_type=morph_type
            )


            persistence_0 = persistence_diagrams[
                0
            ]


            persistence_1 = persistence_diagrams[
                1
            ]


            np.savez_compressed(
                ph_save_path,
                H0=persistence_0,
                H1=persistence_1
            )


            print(
                "Persistent homology saved to:"
            )


            print(
                ph_save_path
            )


        # -----------------------------------------------------------
        # REPORT PERSISTENCE DIAGRAM INFORMATION
        # -----------------------------------------------------------

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
            persistence_diagrams=persistence_diagrams,
            n_bins=n_bins,
            birth_range=(
                0.0,
                float(
                    max_se_length
                )
            ),
            persistence_range=(
                0.0,
                float(
                    max_se_length
                )
            )
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
# 9. MACHINE LEARNING
# =====================================================================

def run_ml_benchmark(
    X,
    y,
    output_dir,
    morph_type
):
    """
    Train and evaluate:

        1. Linear Support Vector Machine
        2. Multilayer Perceptron Neural Network

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
        # CALCULATE METRICS
        # -----------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        f1 = f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )


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
            f"{morph_type.capitalize()} - {model_name}"
        )


    plt.tight_layout()


    # ---------------------------------------------------------------
    # SAVE CONFUSION MATRIX FIGURE
    # ---------------------------------------------------------------

    confusion_matrix_path = (
        output_dir
        /
        f"{morph_type}_confusion_matrices.png"
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
        f"{morph_type.upper()} "
        "PERSISTENCE BINNING RESULTS"
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
        f"{morph_type}_"
        "persistence_binning_ml_metrics.csv"
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
# 10. RUNNER CONTROLLER
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

    FILTRATION_NAME = "Erosion"


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
    # BUILD EROSION PERSISTENCE-BINNING DATASET
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )


    print(
        f"BUILDING {MORPH_TYPE.upper()} "
        "PERSISTENCE-BINNING DATASET"
    )


    print(
        "============================================"
    )


    (
        X_topological_features,
        y_experimental_classes,
        image_names
    ) = build_morphology_dataset(
        image_paths=image_paths,
        ph_output_dir=PH_OUTPUT_DIR,
        morph_type=MORPH_TYPE,
        threshold=THRESHOLD,
        max_se_length=MAX_SE_LENGTH,
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
        f"{MORPH_TYPE}_18d_"
        "persistence_binning_features.npy"
    )


    labels_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        f"{MORPH_TYPE}_"
        "persistence_binning_labels.npy"
    )


    names_save_path = (
        VECTORIZATION_OUTPUT_DIR
        /
        f"{MORPH_TYPE}_"
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
        output_dir=CLASSIFICATION_OUTPUT_DIR,
        morph_type=MORPH_TYPE
    )