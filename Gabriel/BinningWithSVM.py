# -*- coding: utf-8 -*-
"""
Lower-Star Persistence Binning with a Linear SVM 

This script: 
    1. Loads control and microgravity images from separate training and evaluation folders. 
    2. Preprocesses each image using cropping, Gaussian smoothing, and CLAHE. 
    3. Computes lower-star persistent homology with Cripser. 
    4. Separates the persistence output into H0 and H1 intervals. 
    5. Converts the intervals into persistence-binning feature vectors. 
    6. Standardizes the feature vectors. 
    7. Trains a Linear Support Vector Machine on the training images. 
    8. Evaluates the model using accuracy and macro F1 score. 
    9. Creates a confusion matrix and PCA visualizations. 
    10. Saves the vectors, datasets, predictions, metrics, and figures. 
"""
import csv
from itertools import combinations
from typing import Iterable
from pathlib import Path
import cripser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from skimage import exposure, filters, io
from skimage.util import img_as_float
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =============================================================================
# USER SETTINGS
# =============================================================================

IMAGE_DIR = Path(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images"
)

TRAINING_FOLDER = IMAGE_DIR / "Training"
EVALUATION_FOLDER = IMAGE_DIR / "Evaluation"

OUTPUT_DIR = IMAGE_DIR / "LowerStar_PersistenceBinning_SVM_Results"

IMAGE_PATTERN = "*.tif"

# Preprocessing
CROP_HEIGHT: int | None = 3850
GAUSSIAN_SIGMA = 0.5
CLAHE_KERNEL_SIZE = 256
CLAHE_CLIP_LIMIT = 0.015

# Persistence binning
N_BINS = 3
BIRTH_RANGE = (0.0, 1.0)
PERSISTENCE_RANGE = (0.0, 1.0)

# Linear SVM
SVM_C = 1.0
F1_AVERAGE = "macro"

# PCA and figures
PCA_COMPONENTS = 5
SHOW_PLOTS = True
SAVE_PLOTS = True
FIGURE_DPI = 300


# =============================================================================
# GENERAL HELPERS
# =============================================================================

CLASS_NAMES = {
    0: "Control",
    1: "Microgravity",
}


def print_section(title: str) -> None:
    """Print a consistent section heading in the console."""
    print(f"\n{title}")
    print("=" * len(title))


def create_output_folders(output_dir: Path) -> dict[str, Path]:
    """
    Create and return the output folders used by the experiment.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for all experiment outputs.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths for vector, dataset, table, and figure outputs.
    """
    folders = {
        "training_vectors": output_dir / "vectors" / "training",
        "evaluation_vectors": output_dir / "vectors" / "evaluation",
        "datasets": output_dir / "datasets",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def find_images(folder: Path, pattern: str = "*.tif") -> list[Path]:
    """
    Find image files in a folder.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to search.
    pattern : str, optional
        File pattern used for the search. Default is "*.tif".

    Returns
    -------
    list[pathlib.Path]
        Sorted image paths.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no matching images.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    image_paths = sorted(folder.glob(pattern))

    if not image_paths:
        raise FileNotFoundError(
            f"No images matching {pattern!r} were found in: {folder}"
        )

    return image_paths


def determine_class_from_filename(image_path: Path) -> tuple[int, str]:
    """
    Determine an image class from its filename.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to an image whose filename contains "control" or
        "microgravity".

    Returns
    -------
    tuple[int, str]
        Numeric label and readable class name.

    Raises
    ------
    ValueError
        If the filename does not contain a recognized class label.
    """
    image_name = image_path.name.lower()

    if "microgravity" in image_name:
        return 1, CLASS_NAMES[1]

    if "control" in image_name:
        return 0, CLASS_NAMES[0]

    raise ValueError(
        f"Could not determine the class of {image_path.name!r}. "
        'The filename must contain "control" or "microgravity".'
    )


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    image_path: Path,
    crop_height: int | None = 3850,
    sigma: float = 0.5,
    clahe_kernel_size: int = 256,
    clahe_clip_limit: float = 0.015,
) -> np.ndarray:
    """
    Load and preprocess one microscopy image.

    The image is converted to floating-point grayscale, optionally cropped,
    smoothed with a Gaussian filter, and enhanced with CLAHE.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the input image.
    crop_height : int or None, optional
        Number of rows retained from the top of the image. Set to None to
        keep the full image. Default is 3850.
    sigma : float, optional
        Gaussian smoothing standard deviation. Default is 0.5.
    clahe_kernel_size : int, optional
        CLAHE neighborhood size. Default is 256.
    clahe_clip_limit : float, optional
        CLAHE contrast-limiting value. Default is 0.015.

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale image with floating-point values.
    """
    image = img_as_float(io.imread(image_path, as_gray=True))

    if crop_height is not None:
        if crop_height <= 0:
            raise ValueError("crop_height must be positive or None.")
        image = image[:crop_height, :]

    smoothed = filters.gaussian(image, sigma=sigma)

    return exposure.equalize_adapthist(
        smoothed,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
    )


# =============================================================================
# LOWER-STAR PERSISTENT HOMOLOGY
# =============================================================================

def compute_lower_star(image: np.ndarray, max_dimension: int = 1) -> np.ndarray:
    """
    Compute a lower-star cubical persistence diagram.

    Lower image intensities enter the filtration first. Cripser returns one
    row per topological feature in the form:

        [dimension, birth, death, ...]

    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed grayscale filtration image.
    max_dimension : int, optional
        Maximum homology dimension to calculate. Default is 1.

    Returns
    -------
    numpy.ndarray
        Cripser persistence output containing H0 and H1 features.
    """
    filtration = np.asarray(image, dtype=float)

    if hasattr(cripser, "compute_ph"):
        return cripser.compute_ph(filtration, maxdim=max_dimension)

    return cripser.computePH(filtration, maxdim=max_dimension)


def separate_homology_dimensions(
    persistence_output: np.ndarray,
) -> list[np.ndarray]:
    """
    Separate Cripser output into H0 and H1 birth-death diagrams.

    Parameters
    ----------
    persistence_output : numpy.ndarray
        Complete Cripser persistence output.

    Returns
    -------
    list[numpy.ndarray]
        Two arrays: H0 birth-death pairs followed by H1 birth-death pairs.
    """
    persistence_output = np.asarray(persistence_output)

    h0 = persistence_output[persistence_output[:, 0] == 0][:, 1:3]
    h1 = persistence_output[persistence_output[:, 0] == 1][:, 1:3]

    return [h0, h1]


# =============================================================================
# PERSISTENCE-BINNING VECTORIZATION
# =============================================================================

def build_persistence_binning_vector(
    persistence_diagrams: Iterable[np.ndarray],
    n_bins: int = 3,
    birth_range: tuple[float, float] = (0.0, 1.0),
    persistence_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Convert H0 and H1 diagrams into one fixed-length feature vector.

    Each birth-death point is transformed into birth-persistence coordinates:

        persistence = death - birth

    A weighted two-dimensional histogram is then created. Persistence is used
    as the weight, so longer-lived features contribute more strongly.

    Parameters
    ----------
    persistence_diagrams : iterable of numpy.ndarray
        H0 and H1 birth-death diagrams.
    n_bins : int, optional
        Number of bins along each grid axis. Default is 3.
    birth_range : tuple[float, float], optional
        Minimum and maximum birth values represented by the grid.
    persistence_range : tuple[float, float], optional
        Minimum and maximum persistence values represented by the grid.

    Returns
    -------
    numpy.ndarray
        Concatenated H0 and H1 binning vector. Its length is:

            2 * n_bins**2

        With n_bins=3, the output contains 18 features.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    birth_bins = np.linspace(*birth_range, n_bins + 1)
    persistence_bins = np.linspace(*persistence_range, n_bins + 1)
    feature_vectors: list[np.ndarray] = []

    for diagram in persistence_diagrams:
        diagram = np.asarray(diagram, dtype=float)

        if diagram.size == 0:
            feature_vectors.append(np.zeros(n_bins * n_bins))
            continue

        finite_mask = np.isfinite(diagram[:, 0]) & np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]

        if finite_diagram.size == 0:
            feature_vectors.append(np.zeros(n_bins * n_bins))
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

        feature_vectors.append(bin_matrix.ravel())

    if not feature_vectors:
        raise ValueError("At least one persistence diagram is required.")

    return np.concatenate(feature_vectors)


# =============================================================================
# DATASET CREATION
# =============================================================================

def process_one_image(
    image_path: Path,
    vector_output_folder: Path,
) -> tuple[np.ndarray, int, str]:
    """
    Process one image and save its persistence-binning vector.

    Parameters
    ----------
    image_path : pathlib.Path
        Input TIFF image.
    vector_output_folder : pathlib.Path
        Folder where the resulting .npy vector is saved.

    Returns
    -------
    tuple[numpy.ndarray, int, str]
        Feature vector, numeric label, and original filename.
    """
    label, class_name = determine_class_from_filename(image_path)
    print(f"Processing: {image_path.name}")
    print(f"Class: {class_name}")

    image = preprocess_image(
        image_path=image_path,
        crop_height=CROP_HEIGHT,
        sigma=GAUSSIAN_SIGMA,
        clahe_kernel_size=CLAHE_KERNEL_SIZE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
    )

    persistence_output = compute_lower_star(image)
    h0, h1 = separate_homology_dimensions(persistence_output)

    print(f"H0 intervals: {len(h0)}")
    print(f"H1 intervals: {len(h1)}")

    vector = build_persistence_binning_vector(
        [h0, h1],
        n_bins=N_BINS,
        birth_range=BIRTH_RANGE,
        persistence_range=PERSISTENCE_RANGE,
    )

    vector_path = (
        vector_output_folder / f"{image_path.stem}_lowerstar_binning.npy"
    )
    np.save(vector_path, vector)

    print(f"Vector shape: {vector.shape}")
    print(f"Saved vector: {vector_path}")

    return vector, label, image_path.name


def build_dataset(
    image_paths: list[Path],
    dataset_name: str,
    vector_output_folder: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a collection of images into a machine-learning dataset.

    Images with unrecognized filename labels are reported and skipped.

    Parameters
    ----------
    image_paths : list[pathlib.Path]
        Images to process.
    dataset_name : str
        Readable dataset name used in console output.
    vector_output_folder : pathlib.Path
        Folder where individual vectors are saved.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Feature matrix, label array, and filename array.
    """
    print_section(f"BUILDING {dataset_name.upper()} DATASET")

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}]")

        try:
            vector, label, name = process_one_image(
                image_path,
                vector_output_folder,
            )
        except (OSError, ValueError) as error:
            print(f"Skipped {image_path.name}: {error}")
            continue

        vectors.append(vector)
        labels.append(label)
        names.append(name)

    if not vectors:
        raise RuntimeError(f"No usable images were found in the {dataset_name} set.")

    return (
        np.asarray(vectors, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(names, dtype=str),
    )


def validate_class_labels(labels: np.ndarray, dataset_name: str) -> None:
    """
    Confirm that a dataset contains both classification classes.

    Parameters
    ----------
    labels : numpy.ndarray
        Numeric class labels.
    dataset_name : str
        Dataset name used in an error message.

    Raises
    ------
    ValueError
        If fewer than two classes are present.
    """
    classes = np.unique(labels)

    if len(classes) < 2:
        raise ValueError(
            f"The {dataset_name} dataset contains only one class: {classes}. "
            "A classifier requires both control and microgravity examples."
        )


def print_dataset_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> None:
    """Print dataset dimensions and class counts."""
    print_section("DATASET SUMMARY")
    print(f"Training feature matrix: {X_train.shape}")
    print(f"Evaluation feature matrix: {X_eval.shape}")
    print(f"Training controls: {np.sum(y_train == 0)}")
    print(f"Training microgravity: {np.sum(y_train == 1)}")
    print(f"Evaluation controls: {np.sum(y_eval == 0)}")
    print(f"Evaluation microgravity: {np.sum(y_eval == 1)}")


def save_dataset(
    output_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    names: np.ndarray,
) -> None:
    """
    Save a feature matrix, labels, and filenames in one compressed file.
    """
    np.savez_compressed(
        output_path,
        X=features,
        y=labels,
        names=names,
    )


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[StandardScaler, SVC, np.ndarray]:
    """
    Standardize training features and fit a Linear SVM.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    y_train : numpy.ndarray
        Training labels.

    Returns
    -------
    tuple[StandardScaler, sklearn.svm.SVC, numpy.ndarray]
        Fitted scaler, fitted SVM, and scaled training features.
    """
    print_section("TRAINING LINEAR SUPPORT VECTOR MACHINE")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel="linear", C=SVM_C)
    model.fit(X_train_scaled, y_train)

    return scaler, model, X_train_scaled


def evaluate_model(
    scaler: StandardScaler,
    model: SVC,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scale evaluation features, predict labels, and calculate metrics.

    Returns
    -------
    tuple
        Scaled evaluation matrix, predicted labels, accuracy, and F1 score.
    """
    X_eval_scaled = scaler.transform(X_eval)
    y_pred = model.predict(X_eval_scaled)

    accuracy = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average=F1_AVERAGE)

    return X_eval_scaled, y_pred, accuracy, f1


def print_predictions(
    image_names: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Print the true and predicted class for each evaluation image."""
    print_section("EVALUATION IMAGE PREDICTIONS")

    for image_name, true_label, predicted_label in zip(
        image_names,
        true_labels,
        predicted_labels,
    ):
        print(f"\nImage: {image_name}")
        print(f"True class: {CLASS_NAMES[int(true_label)]}")
        print(f"Predicted class: {CLASS_NAMES[int(predicted_label)]}")


def save_metrics(
    output_path: Path,
    accuracy: float,
    f1: float,
    training_count: int,
    evaluation_count: int,
) -> None:
    """Save the primary experiment settings and model metrics to CSV."""
    rows = [
        ("filtration", "Lower star"),
        ("vectorization", "Persistence binning"),
        ("classifier", "Linear SVM"),
        ("n_bins", N_BINS),
        ("vector_length", 2 * N_BINS**2),
        ("svm_c", SVM_C),
        ("f1_average", F1_AVERAGE),
        ("training_images", training_count),
        ("evaluation_images", evaluation_count),
        ("accuracy", accuracy),
        ("f1_score", f1),
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["setting_or_metric", "value"])
        writer.writerows(rows)


def save_predictions(
    output_path: Path,
    image_names: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Save image-level evaluation predictions to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["image_name", "true_label", "true_class", "predicted_label",
             "predicted_class", "correct"]
        )

        for name, true_label, predicted_label in zip(
            image_names,
            true_labels,
            predicted_labels,
        ):
            writer.writerow(
                [
                    name,
                    int(true_label),
                    CLASS_NAMES[int(true_label)],
                    int(predicted_label),
                    CLASS_NAMES[int(predicted_label)],
                    bool(true_label == predicted_label),
                ]
            )


# =============================================================================
# FIGURES
# =============================================================================

def create_confusion_matrix(
    y_eval: np.ndarray,
    y_pred: np.ndarray,
    figure_path: Path,
) -> None:
    """Create, optionally save, and optionally display the confusion matrix."""
    display = ConfusionMatrixDisplay.from_predictions(
        y_eval,
        y_pred,
        display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    display.ax_.set_title("Lower-Star Persistence Binning: Linear SVM")
    plt.tight_layout()

    if SAVE_PLOTS:
        display.figure_.savefig(
            figure_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(display.figure_)


def create_pca_plots(
    X_train_scaled: np.ndarray,
    X_eval_scaled: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    y_train_pred: np.ndarray,
    y_eval_pred: np.ndarray,
    figure_folder: Path,
) -> None:
    """
    Create PCA plots for every unique pair of retained components.

    Marker shape represents the true class. Point color represents the SVM
    prediction. Training points are transparent and evaluation points are
    larger and opaque.
    """
    max_components = min(
        PCA_COMPONENTS,
        X_train_scaled.shape[0],
        X_train_scaled.shape[1],
    )

    if max_components < 2:
        print("PCA plots skipped: fewer than two components are available.")
        return

    print_section("RUNNING PRINCIPAL COMPONENT ANALYSIS")

    pca = PCA(n_components=max_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_eval_pca = pca.transform(X_eval_scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    colors = {
        0: "#FF1493",
        1: "#00BFFF",
    }
    markers = {
        0: "o",
        1: "s",
    }

    legend_items = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=9,
            label="True Control",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=9,
            label="True Microgravity",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colors[0],
            markeredgecolor="black",
            markersize=9,
            label="Predicted Control",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colors[1],
            markeredgecolor="black",
            markersize=9,
            label="Predicted Microgravity",
        ),
    ]

    for pc_x, pc_y in combinations(range(max_components), 2):
        component_x = pc_x + 1
        component_y = pc_y + 1
        print(f"Creating PC{component_x} vs PC{component_y}")

        figure, axis = plt.subplots(figsize=(10, 8))

        plot_pca_dataset(
            axis=axis,
            coordinates=X_train_pca,
            true_labels=y_train,
            predicted_labels=y_train_pred,
            pc_x=pc_x,
            pc_y=pc_y,
            colors=colors,
            markers=markers,
            alpha=0.4,
            point_size=60,
            edge_width=1,
        )

        plot_pca_dataset(
            axis=axis,
            coordinates=X_eval_pca,
            true_labels=y_eval,
            predicted_labels=y_eval_pred,
            pc_x=pc_x,
            pc_y=pc_y,
            colors=colors,
            markers=markers,
            alpha=1.0,
            point_size=140,
            edge_width=2,
        )

        axis.set_xlabel(
            f"Principal Component {component_x} "
            f"({explained_variance[pc_x]:.2f}% variance)"
        )
        axis.set_ylabel(
            f"Principal Component {component_y} "
            f"({explained_variance[pc_y]:.2f}% variance)"
        )
        axis.set_title(
            "Persistence Binning PCA Space\n"
            f"PC{component_x} vs PC{component_y}"
        )
        axis.legend(handles=legend_items, loc="best")
        axis.grid(True, linestyle="--", alpha=0.5)
        figure.tight_layout()

        if SAVE_PLOTS:
            figure.savefig(
                figure_folder / f"pca_pc{component_x}_vs_pc{component_y}.png",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(figure)


def plot_pca_dataset(
    axis: plt.Axes,
    coordinates: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    pc_x: int,
    pc_y: int,
    colors: dict[int, str],
    markers: dict[int, str],
    alpha: float,
    point_size: float,
    edge_width: float,
) -> None:
    """Plot one training or evaluation dataset in a PCA coordinate system."""
    for true_class in CLASS_NAMES:
        for predicted_class in CLASS_NAMES:
            indices = (
                (true_labels == true_class)
                & (predicted_labels == predicted_class)
            )

            if not np.any(indices):
                continue

            axis.scatter(
                coordinates[indices, pc_x],
                coordinates[indices, pc_y],
                color=colors[predicted_class],
                marker=markers[true_class],
                alpha=alpha,
                edgecolors="black",
                linewidths=edge_width,
                s=point_size,
            )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """Run the complete lower-star persistence-binning SVM experiment."""
    print_section("LOWER-STAR PERSISTENCE BINNING WITH LINEAR SVM")

    folders = create_output_folders(OUTPUT_DIR)
    training_images = find_images(TRAINING_FOLDER, IMAGE_PATTERN)
    evaluation_images = find_images(EVALUATION_FOLDER, IMAGE_PATTERN)

    print(f"Training images found: {len(training_images)}")
    print(f"Evaluation images found: {len(evaluation_images)}")
    print(f"Output directory: {OUTPUT_DIR}")

    X_train, y_train, names_train = build_dataset(
        training_images,
        dataset_name="training",
        vector_output_folder=folders["training_vectors"],
    )
    X_eval, y_eval, names_eval = build_dataset(
        evaluation_images,
        dataset_name="evaluation",
        vector_output_folder=folders["evaluation_vectors"],
    )

    validate_class_labels(y_train, "training")
    validate_class_labels(y_eval, "evaluation")
    print_dataset_summary(X_train, y_train, X_eval, y_eval)

    save_dataset(
        folders["datasets"] / "training_dataset.npz",
        X_train,
        y_train,
        names_train,
    )
    save_dataset(
        folders["datasets"] / "evaluation_dataset.npz",
        X_eval,
        y_eval,
        names_eval,
    )

    scaler, svm, X_train_scaled = train_linear_svm(X_train, y_train)
    X_eval_scaled, y_pred, accuracy, f1 = evaluate_model(
        scaler,
        svm,
        X_eval,
        y_eval,
    )
    y_train_pred = svm.predict(X_train_scaled)

    print_section("SVM EVALUATION RESULTS")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1 score: {f1 * 100:.2f}%")
    print_predictions(names_eval, y_eval, y_pred)

    save_metrics(
        folders["tables"] / "svm_metrics.csv",
        accuracy,
        f1,
        training_count=len(y_train),
        evaluation_count=len(y_eval),
    )
    save_predictions(
        folders["tables"] / "evaluation_predictions.csv",
        names_eval,
        y_eval,
        y_pred,
    )

    create_confusion_matrix(
        y_eval,
        y_pred,
        folders["figures"] / "confusion_matrix.png",
    )
    create_pca_plots(
        X_train_scaled,
        X_eval_scaled,
        y_train,
        y_eval,
        y_train_pred,
        y_pred,
        folders["figures"],
    )

    print_section("EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()