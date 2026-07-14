# -*- coding: utf-8 -*-
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

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

N_BINS = 3

BIRTH_RANGE = (
    0.0,
    255.0
)

PERSISTENCE_RANGE = (
    0.0,
    255.0
)


# =====================================================================
# 2. PERSISTENCE BINNING
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

    Each persistence diagram is transformed from:

        (birth, death)

    into:

        (birth, persistence)

    where:

        persistence = death - birth

    A two-dimensional histogram is then constructed over the
    birth-persistence plane.

    Each persistence point contributes its persistence value as the
    weight of the bin containing that point. Therefore, longer-lived
    topological features contribute more strongly than short-lived
    features.

    H0 and H1 are binned separately and then concatenated.

    Parameters
    ----------
    persistence_diagrams : list
        List containing two persistence diagrams:

            persistence_diagrams[0] = H0 diagram
            persistence_diagrams[1] = H1 diagram

        Each diagram must contain columns:

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

    # Create bin edges for birth values
    birth_bins = np.linspace(
        birth_range[0],
        birth_range[1],
        n_bins + 1
    )

    # Create bin edges for persistence values
    persistence_bins = np.linspace(
        persistence_range[0],
        persistence_range[1],
        n_bins + 1
    )

    # Store the flattened H0 and H1 bin matrices
    feature_blocks = []


    # ---------------------------------------------------------------
    # PROCESS H0 AND H1 SEPARATELY
    # ---------------------------------------------------------------

    for pd in persistence_diagrams:

        # Convert to NumPy array
        pd = np.asarray(
            pd,
            dtype=np.float64
        )


        # -----------------------------------------------------------
        # HANDLE EMPTY PERSISTENCE DIAGRAM
        # -----------------------------------------------------------

        if pd.size == 0:

            # Empty diagram produces an all-zero bin matrix
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
            np.isfinite(pd[:, 0])
            &
            np.isfinite(pd[:, 1])
        )

        pd_finite = pd[
            finite_mask
        ]


        # -----------------------------------------------------------
        # HANDLE DIAGRAM WITH NO FINITE FEATURES
        # -----------------------------------------------------------

        if len(pd_finite) == 0:

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


        # Remove any invalid negative persistence values
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

            # Longer-lived features contribute more weight
            weights=persistences
        )


        # Flatten the 2D bin matrix into a 1D vector
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
# 3. LOAD AND VECTORIZE SAVED LOWER-STAR DIAGRAMS
# =====================================================================

def vectorize_persistence_diagrams(
    diagram_paths,
    n_bins=3,
    birth_range=(0.0, 255.0),
    persistence_range=(0.0, 255.0)
):
    """
    Load saved lower-star persistence diagrams and convert each one
    into a persistence-binning feature vector.

    Parameters
    ----------
    diagram_paths : list
        Paths to saved lower-star persistence diagram .npy files.

    n_bins : int, optional
        Number of bins along each axis.

    birth_range : tuple, optional
        Birth-value range used for persistence binning.

    persistence_range : tuple, optional
        Persistence-value range used for persistence binning.

    Returns
    -------
    tuple
        X : numpy.ndarray
            Persistence-binning feature matrix.

        y : numpy.ndarray
            Experimental class labels.

            0 = Control
            1 = Microgravity

        image_names : numpy.ndarray
            Names of the persistence diagram files.
    """

    vectorized_features = []

    y_labels = []

    image_names = []


    # ---------------------------------------------------------------
    # PROCESS EACH SAVED PERSISTENCE DIAGRAM
    # ---------------------------------------------------------------

    for path in diagram_paths:

        path = Path(
            path
        )


        print(
            f"Vectorizing: {path.name}"
        )


        # -----------------------------------------------------------
        # LOAD RAW CRIPSER PERSISTENCE DIAGRAM
        # -----------------------------------------------------------

        ph = np.load(
            path
        )


        # Make sure the persistence diagram has the expected format
        if (
            ph.ndim != 2
            or
            ph.shape[1] < 3
        ):

            raise ValueError(
                f"Unexpected persistence diagram shape "
                f"for {path.name}: {ph.shape}"
            )


        # -----------------------------------------------------------
        # SEPARATE H0 AND H1
        # -----------------------------------------------------------

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


        persistence_diagrams = [
            persistence_0,
            persistence_1
        ]


        # -----------------------------------------------------------
        # BUILD PERSISTENCE-BINNING FEATURE VECTOR
        # -----------------------------------------------------------

        feature_vector = build_persistence_binning_vector(
            persistence_diagrams=persistence_diagrams,
            n_bins=n_bins,
            birth_range=birth_range,
            persistence_range=persistence_range
        )


        vectorized_features.append(
            feature_vector
        )


        # -----------------------------------------------------------
        # ASSIGN EXPERIMENTAL CLASS LABEL
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
                f"Could not determine Control or Microgravity "
                f"label from filename: {path.name}"
            )


        y_labels.append(
            label
        )


        image_names.append(
            path.name
        )


    # ---------------------------------------------------------------
    # CONVERT TO NUMPY ARRAYS
    # ---------------------------------------------------------------

    X = np.asarray(
        vectorized_features,
        dtype=np.float64
    )

    y = np.asarray(
        y_labels,
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
# 4. MACHINE LEARNING BENCHMARKS AND EVALUATION
# =====================================================================

def run_ml_benchmark(
    X_tda,
    y,
    output_dir,
    dataset_title="Lower-Star Persistence Binning"
):
    """
    Train and evaluate:

        1. Linear SVM
        2. RBF SVM
        3. Neural Network (MLP)

    Each model is trained using standardized persistence-binning
    feature vectors.

    Accuracy, F1 score, and confusion matrices are calculated.
    """

    # ---------------------------------------------------------------
    # DEFINE CLASSIFIERS
    # ---------------------------------------------------------------

    names = [
        "Linear SVM",
        "RBF SVM",
        "Neural Network (MLP)"
    ]


    classifiers = [

        SVC(
            kernel="linear",
            C=1.0,
            random_state=42
        ),

        SVC(
            kernel="rbf",
            gamma=2,
            C=1.0,
            random_state=42
        ),

        MLPClassifier(
            hidden_layer_sizes=(
                32,
                16
            ),
            max_iter=1000,
            random_state=42
        )
    ]


    # ---------------------------------------------------------------
    # SPLIT DATASET
    # ---------------------------------------------------------------

    (
        X_train_full,
        X_test_full,
        y_train,
        y_test
    ) = train_test_split(
        X_tda,
        y,
        test_size=0.4,
        random_state=42,

        # Preserve Control/Microgravity class proportions
        stratify=y
    )


    print(
        "\nTraining samples:",
        X_train_full.shape[0]
    )

    print(
        "Testing samples:",
        X_test_full.shape[0]
    )

    print(
        "Features per image:",
        X_train_full.shape[1]
    )


    # ---------------------------------------------------------------
    # PCA FOR 2D VISUALIZATION ONLY
    # ---------------------------------------------------------------

    # Standardize features before PCA visualization
    pca_scaler = StandardScaler()

    X_train_scaled = pca_scaler.fit_transform(
        X_train_full
    )

    X_test_scaled = pca_scaler.transform(
        X_test_full
    )


    pca = PCA(
        n_components=2,
        random_state=42
    )


    X_train_vis = pca.fit_transform(
        X_train_scaled
    )

    X_test_vis = pca.transform(
        X_test_scaled
    )


    print(
        "\nPCA explained variance:"
    )

    print(
        f"PC1: "
        f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
    )

    print(
        f"PC2: "
        f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
    )


    # ---------------------------------------------------------------
    # DEFINE PCA PLOT BOUNDARIES
    # ---------------------------------------------------------------

    x_min = (
        X_train_vis[:, 0].min()
        -
        1.0
    )

    x_max = (
        X_train_vis[:, 0].max()
        +
        1.0
    )

    y_min = (
        X_train_vis[:, 1].min()
        -
        1.0
    )

    y_max = (
        X_train_vis[:, 1].max()
        +
        1.0
    )


    # Plot color maps
    cm_standard = mcm.RdBu

    cm_bright = ListedColormap(
        [
            "#FF0000",
            "#0000FF"
        ]
    )


    # ---------------------------------------------------------------
    # CREATE PCA / DECISION BOUNDARY FIGURE
    # ---------------------------------------------------------------

    num_classifiers = len(
        classifiers
    )


    fig = plt.figure(
        figsize=(
            3 * num_classifiers + 3,
            4
        )
    )


    # ---------------------------------------------------------------
    # FIRST PANEL: PCA DATA DISTRIBUTION
    # ---------------------------------------------------------------

    ax = plt.subplot(
        1,
        num_classifiers + 1,
        1
    )


    ax.set_title(
        f"{dataset_title}\n(Data PCA)",
        fontsize=9,
        weight="bold"
    )


    # Training points
    ax.scatter(
        X_train_vis[:, 0],
        X_train_vis[:, 1],
        c=y_train,
        cmap=cm_bright,
        edgecolors="k",
        s=35
    )


    # Testing points
    ax.scatter(
        X_test_vis[:, 0],
        X_test_vis[:, 1],
        c=y_test,
        cmap=cm_bright,
        alpha=0.5,
        edgecolors="k",
        s=35
    )


    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.set_xticks(
        ()
    )

    ax.set_yticks(
        ()
    )


    # Store metrics for CSV output
    metrics_records = []

    # Store confusion matrices for plotting
    confusion_records = []


    # ---------------------------------------------------------------
    # TRAIN AND EVALUATE EACH CLASSIFIER
    # ---------------------------------------------------------------

    for idx, (
        name,
        clf
    ) in enumerate(
        zip(
            names,
            classifiers
        ),
        start=2
    ):

        print(
            "\n==================================="
        )

        print(
            f"TRAINING: {name}"
        )

        print(
            "==================================="
        )


        # -----------------------------------------------------------
        # CREATE MACHINE LEARNING PIPELINE
        # -----------------------------------------------------------

        model_pipeline = make_pipeline(
            StandardScaler(),
            clf
        )


        # -----------------------------------------------------------
        # TRAIN MODEL
        # -----------------------------------------------------------

        model_pipeline.fit(
            X_train_full,
            y_train
        )


        # -----------------------------------------------------------
        # MAKE PREDICTIONS
        # -----------------------------------------------------------

        y_pred = model_pipeline.predict(
            X_test_full
        )


        # -----------------------------------------------------------
        # CALCULATE METRICS
        # -----------------------------------------------------------

        acc = accuracy_score(
            y_test,
            y_pred
        )


        f1 = f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )


        cm_data = confusion_matrix(
            y_test,
            y_pred,
            labels=[
                0,
                1
            ]
        )


        print(
            f"Accuracy: {acc:.4f}"
        )

        print(
            f"F1 Score: {f1:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            cm_data
        )


        # -----------------------------------------------------------
        # SAVE METRICS
        # -----------------------------------------------------------

        metrics_records.append(
            {
                "Model": name,
                "Accuracy": round(
                    acc,
                    4
                ),
                "F1-Score": round(
                    f1,
                    4
                ),
                "TN": cm_data[
                    0,
                    0
                ],
                "FP": cm_data[
                    0,
                    1
                ],
                "FN": cm_data[
                    1,
                    0
                ],
                "TP": cm_data[
                    1,
                    1
                ]
            }
        )


        confusion_records.append(
            (
                name,
                cm_data
            )
        )


        # -----------------------------------------------------------
        # CREATE PCA DECISION BOUNDARY VISUALIZATION
        # -----------------------------------------------------------

        ax = plt.subplot(
            1,
            num_classifiers + 1,
            idx
        )


        # Create a separate copy of the classifier for
        # the two-dimensional PCA visualization
        vis_clf = copy.deepcopy(
            clf
        )


        vis_pipeline = make_pipeline(
            StandardScaler(),
            vis_clf
        )


        try:

            vis_pipeline.fit(
                X_train_vis,
                y_train
            )


            DecisionBoundaryDisplay.from_estimator(
                vis_pipeline,
                X_train_vis,
                cmap=cm_standard,
                alpha=0.8,
                ax=ax,
                eps=0.5
            )


        except Exception as error:

            print(
                f"Could not draw decision boundary "
                f"for {name}: {error}"
            )


        # Training points
        ax.scatter(
            X_train_vis[:, 0],
            X_train_vis[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k",
            s=25
        )


        # Testing points
        ax.scatter(
            X_test_vis[:, 0],
            X_test_vis[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.5,
            s=25
        )


        ax.set_xlim(
            x_min,
            x_max
        )

        ax.set_ylim(
            y_min,
            y_max
        )

        ax.set_xticks(
            ()
        )

        ax.set_yticks(
            ()
        )


        ax.set_title(
            name,
            fontsize=9,
            weight="bold"
        )


        metrics_str = (
            f"Acc: {acc:.2f}\n"
            f"F1: {f1:.2f}"
        )


        ax.text(
            x_max - 0.2,
            y_min
            +
            (
                0.35
                *
                (
                    y_max
                    -
                    y_min
                )
            ),
            metrics_str,
            size=9,
            horizontalalignment="right",
            weight="bold",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.7,
                edgecolor="none"
            )
        )


    # ---------------------------------------------------------------
    # SHOW PCA / DECISION BOUNDARY FIGURE
    # ---------------------------------------------------------------

    plt.tight_layout()

    plt.show()


    # ---------------------------------------------------------------
    # DISPLAY CONFUSION MATRICES
    # ---------------------------------------------------------------

    fig_cm, axes = plt.subplots(
        1,
        len(
            confusion_records
        ),
        figsize=(
            12,
            4
        )
    )


    for ax, (
        name,
        cm_data
    ) in zip(
        axes,
        confusion_records
    ):

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm_data,
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
            name
        )


    plt.tight_layout()

    plt.show()


    # ---------------------------------------------------------------
    # SAVE METRICS TABLE
    # ---------------------------------------------------------------

    df_metrics = pd.DataFrame(
        metrics_records
    )


    csv_path = (
        Path(
            output_dir
        )
        /
        "microgravity_lower_star_"
        "persistence_binning_ml_metrics.csv"
    )


    df_metrics.to_csv(
        csv_path,
        index=False
    )


    # ---------------------------------------------------------------
    # PRINT FINAL METRICS TABLE
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "MACHINE LEARNING BENCHMARK PERFORMANCE"
    )

    print(
        "============================================"
    )


    print(
        df_metrics.to_string(
            index=False
        )
    )


    print(
        f"\nEvaluation metrics table saved to:\n"
        f"{csv_path}"
    )


# =====================================================================
# 5. RUNNER CONTROLLER
# =====================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # DEFINE DIRECTORY
    # ---------------------------------------------------------------

    # This folder should contain the saved:
    #
    # *_lower_star_diagram.npy
    #
    # files created by the first lower-star PH script.

    PROCESSED_DIR = Path(
        r"C:\Users\chloe\OneDrive - Simpson College"
        r"\IMAGES2.0\All Images\preprocessed_images"
    )


    # ---------------------------------------------------------------
    # FIND SAVED LOWER-STAR PERSISTENCE DIAGRAMS
    # ---------------------------------------------------------------

    diagram_paths = sorted(
        PROCESSED_DIR.glob(
            "*_lower_star_diagram.npy"
        )
    )


    # ---------------------------------------------------------------
    # CHECK THAT PERSISTENCE DIAGRAMS WERE FOUND
    # ---------------------------------------------------------------

    if not diagram_paths:

        raise FileNotFoundError(
            "Could not find any lower-star persistence "
            "diagrams matching:\n"
            "*_lower_star_diagram.npy\n\n"
            f"Directory:\n{PROCESSED_DIR}\n\n"
            "Run the lower-star persistent homology "
            "extraction script first."
        )


    print(
        f"Found {len(diagram_paths)} "
        f"lower-star persistence diagrams."
    )


    # ---------------------------------------------------------------
    # PERSISTENCE BINNING
    # ---------------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "STARTING LOWER-STAR PERSISTENCE BINNING"
    )

    print(
        "============================================"
    )


    (
        X_topological_features,
        y_experimental_classes,
        image_names
    ) = vectorize_persistence_diagrams(
        diagram_paths=diagram_paths,
        n_bins=N_BINS,
        birth_range=BIRTH_RANGE,
        persistence_range=PERSISTENCE_RANGE
    )


    # ---------------------------------------------------------------
    # REPORT DATASET INFORMATION
    # ---------------------------------------------------------------

    print(
        "\nPersistence binning complete."
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
    # SAVE PERSISTENCE-BINNING DATASET
    # ---------------------------------------------------------------

    np.save(
        PROCESSED_DIR
        /
        "step2_lower_star_18d_"
        "persistence_binning_features.npy",
        X_topological_features
    )


    np.save(
        PROCESSED_DIR
        /
        "step2_lower_star_"
        "persistence_binning_labels.npy",
        y_experimental_classes
    )


    np.save(
        PROCESSED_DIR
        /
        "step2_lower_star_"
        "persistence_binning_names.npy",
        image_names
    )


    print(
        "\n18D persistence-binning arrays "
        "saved to disk."
    )


    # ---------------------------------------------------------------
    # RUN MACHINE LEARNING
    # ---------------------------------------------------------------

    if len(
        X_topological_features
    ) < 5:

        raise ValueError(
            "Not enough samples to perform "
            "the machine-learning split."
        )


    print(
        "\n============================================"
    )

    print(
        "RUNNING SUPPORT VECTOR MACHINES "
        "AND NEURAL NETWORK"
    )

    print(
        "============================================"
    )


    run_ml_benchmark(
        X_tda=X_topological_features,
        y=y_experimental_classes,
        output_dir=PROCESSED_DIR,
        dataset_title=(
            "Lower-Star Persistence Binning "
            "Machine Learning Experiment"
        )
    )

