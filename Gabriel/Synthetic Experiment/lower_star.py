# -*- coding: utf-8 -*-
"""
Lower-Star Persistent Homology: Control vs Microgravity

This script:
    1. Loads one control image and one microgravity image.
    2. Scales both grayscale images to 0-255.
    3. Creates visualizations of the lower-star filtration.
    4. Computes lower-star persistent homology with Cripser.
    5. Separates H0 and H1 intervals.
    6. Plots the filtration stages and persistence diagram.
    7. Saves the raw PH output, thresholds, and figures.

For a lower-star filtration, pixels with lower intensity values enter first.

At threshold t, the filtration contains every pixel satisfying:

    image intensity <= t

Persistent homology is computed on the complete grayscale image, not on
the displayed threshold screenshots.
"""

from pathlib import Path

import cripser
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.util import img_as_float


# =====================================================================
# 1. USER SETTINGS
# =====================================================================

CONTROL_IMAGE = Path(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images\fake_control001.png"
)

MICROGRAVITY_IMAGE = Path(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images\fake_micro_001.png"
)

MAX_HOMOLOGY_DIMENSION = 1

OUTPUT_DIRECTORY = (
    CONTROL_IMAGE.parent
    / "Persistent_Homology_Visualizationsv2"
    / "Lower_Star"
)

# ---------------------------------------------------------------------
# FILTRATION VISUALIZATION SETTINGS
# ---------------------------------------------------------------------

# Number of threshold stages displayed for each image.
NUMBER_OF_FILTRATION_STAGES = 6

CUSTOM_THRESHOLD_VALUES = [0, 50, 100, 150, 200, 255]

# Resolution used when saving figures.
FIGURE_DPI = 300

# =====================================================================
# 2. LOAD IMAGE LIKE THE LOWER-STAR EXPERIMENT
# =====================================================================

def load_lower_star_image(image_path):
    """
    Load one grayscale image and scale it to 0-255 when needed.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            "Could not find the input image:\n"
            f"{image_path}"
        )

    image = img_as_float(
        io.imread(image_path, as_gray=True)
    )

    # img_as_float usually produces values between 0 and 1.
    # Convert those values to the 0-255 scale.
    if image.max() <= 1.0:
        image = image * 255.0

    return np.asarray(image, dtype=np.float64)

# =====================================================================
# 3. SELECT FILTRATION THRESHOLDS
# =====================================================================

def get_lower_star_thresholds(
    image,
    number_of_stages=6,
    custom_thresholds= CUSTOM_THRESHOLD_VALUES,
):
    """
    Select threshold values for displaying the lower-star filtration.

    When custom_thresholds is None, the values are evenly spaced between
    the minimum and maximum image intensities.
    """
    image = np.asarray(image, dtype=np.float64)

    image_minimum = float(np.min(image))
    image_maximum = float(np.max(image))

    if custom_thresholds is not None:
        thresholds = np.asarray(
            custom_thresholds,
            dtype=np.float64,
        )

        if thresholds.ndim != 1 or len(thresholds) == 0:
            raise ValueError(
                "CUSTOM_THRESHOLD_VALUES must be a nonempty "
                "one-dimensional list."
            )

        # Sort the custom thresholds and remove duplicate values.
        thresholds = np.unique(thresholds)

        return thresholds

    if number_of_stages < 2:
        raise ValueError(
            "NUMBER_OF_FILTRATION_STAGES must be at least 2."
        )

    # Use thresholds across the actual intensity range of the image.
    thresholds = np.linspace(
        image_minimum,
        image_maximum,
        number_of_stages,
    )

    return thresholds


# =====================================================================
# 4. CREATE ONE LOWER-STAR FILTRATION STAGE
# =====================================================================

def create_lower_star_stage(image, threshold):
    """
    Create one visual lower-star filtration stage.

    Pixels satisfying image <= threshold have entered the filtration.
    Pixels that have not entered yet are displayed as white.

    Returns
    -------
    displayed_image : numpy.ndarray
        Image used for the visualization.

    included_pixels : numpy.ndarray
        Boolean mask indicating which pixels have entered.
    """
    image = np.asarray(image, dtype=np.float64)

    included_pixels = image <= threshold

    # Begin with a completely white background.
    displayed_image = np.full_like(
        image,
        fill_value=255.0,
        dtype=np.float64,
    )

    # Reveal only pixels that have entered the filtration.
    displayed_image[included_pixels] = image[included_pixels]

    return displayed_image, included_pixels


# =====================================================================
# 5. COMPUTE LOWER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_lower_star_ph(image, maxdim=1):
    """
    Compute lower-star persistent homology.

    Lower-valued pixels enter the filtration first.
    """
    image = np.asarray(image, dtype=np.float64)

    if hasattr(cripser, "compute_ph"):
        try:
            return cripser.compute_ph(
                image,
                filtration="V",
                maxdim=maxdim,
            )
        except TypeError:
            return cripser.compute_ph(
                image,
                maxdim=maxdim,
            )

    ph = np.asarray(
        cripser.computePH(
            image,
            maxdim=maxdim,
        ),
        dtype=np.float64,
    )

    # Older computePH versions may use DBL_MAX for an essential death.
    if ph.ndim == 2 and ph.shape[1] >= 3:
        ph[ph[:, 2] > 1.0e300, 2] = np.inf

    return ph


# =====================================================================
# 6. SEPARATE H0 AND H1
# =====================================================================

def separate_persistence_diagrams(ph):
    """
    Extract birth-death pairs for H0 and H1.
    """
    ph = np.asarray(ph, dtype=np.float64)

    if ph.ndim != 2 or ph.shape[1] < 3:
        raise ValueError(
            "Unexpected Cripser output. Expected columns beginning "
            "with dimension, birth, and death."
        )

    h0 = ph[ph[:, 0] == 0][:, 1:3]
    h1 = ph[ph[:, 0] == 1][:, 1:3]

    return h0, h1


# =====================================================================
# 7. PRINT SUMMARY
# =====================================================================

def print_ph_summary(label, ph, h0, h1):
    """
    Print interval counts and longest finite lifetimes.
    """
    finite_h0 = (
        h0[np.isfinite(h0[:, 1])]
        if len(h0)
        else h0
    )

    finite_h1 = (
        h1[np.isfinite(h1[:, 1])]
        if len(h1)
        else h1
    )

    print("\n============================================")
    print(f"{label.upper()} LOWER-STAR PERSISTENT HOMOLOGY")
    print("============================================")
    print(f"Raw Cripser output shape: {ph.shape}")
    print(f"H0 intervals: {len(h0)}")
    print(f"Finite H0 intervals: {len(finite_h0)}")
    print(f"H1 intervals: {len(h1)}")
    print(f"Finite H1 intervals: {len(finite_h1)}")

    if len(finite_h0):
        h0_lifetimes = (
            finite_h0[:, 1] - finite_h0[:, 0]
        )

        print(
            "Longest finite H0 lifetime: "
            f"{np.max(h0_lifetimes):.3f}"
        )

    if len(finite_h1):
        h1_lifetimes = (
            finite_h1[:, 1] - finite_h1[:, 0]
        )

        print(
            "Longest finite H1 lifetime: "
            f"{np.max(h1_lifetimes):.3f}"
        )


# =====================================================================
# 8. PLOT ONE PERSISTENCE DIAGRAM
# =====================================================================

def plot_persistence_diagram(
    label,
    h0,
    h1,
    output_path=None,
):
    """
    Plot H0 and H1 using Cripser's plot_diagrams helper.
    """
    if not hasattr(cripser, "plot_diagrams"):
        raise AttributeError(
            "This Cripser installation does not contain "
            "plot_diagrams().\n"
            "Update it with: pip install -U cripser"
        )

    figure, axis = plt.subplots(
        figsize=(8, 8),
        constrained_layout=True,
    )

    cripser.plot_diagrams(
        [h0, h1],
        labels=[r"$H_0$", r"$H_1$"],
        ax=axis,
        title=f"{label} Lower-Star Persistence Diagram",
        legend=True,
        diagonal=True,
        marker_size=28,
        alpha=0.80,
        show=False,
    )

    figure.suptitle(
        "Points farther above the diagonal have longer persistence",
        fontsize=12,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            f"\n{label} persistence diagram saved to:\n"
            f"{output_path}"
        )

    return figure


# =====================================================================
# 9. PLOT FILTRATION STAGES AND PERSISTENCE DIAGRAM
# =====================================================================

def plot_filtration_and_persistence(
    label,
    image,
    thresholds,
    h0,
    h1,
    output_path=None,
):
    """
    Plot the lower-star filtration stages across the top and the
    persistence diagram underneath.

    The final threshold stage contains the entire original image.
    """
    if not hasattr(cripser, "plot_diagrams"):
        raise AttributeError(
            "This Cripser installation does not contain "
            "plot_diagrams().\n"
            "Update it with: pip install -U cripser"
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    number_of_thresholds = len(thresholds)

    # Make the figure wider when more filtration stages are requested.
    figure_width = max(
        14,
        3.0 * number_of_thresholds,
    )

    figure = plt.figure(
        figsize=(figure_width, 10),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=number_of_thresholds,
        height_ratios=[1.0, 1.35],
    )

    # -------------------------------------------------------------
    # Top row: filtration threshold stages
    # -------------------------------------------------------------

    for stage_index, threshold in enumerate(thresholds):
        axis = figure.add_subplot(
            grid[0, stage_index]
        )

        filtration_stage, included_pixels = (
            create_lower_star_stage(
                image=image,
                threshold=threshold,
            )
        )

        percentage_included = (
            100.0 * np.mean(included_pixels)
        )

        axis.imshow(
            filtration_stage,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axis.set_title(
            f"Threshold ≤ {threshold:.1f}\n"
            f"{percentage_included:.1f}% of pixels included",
            fontsize=11,
        )

        axis.axis("off")

    # -------------------------------------------------------------
    # Bottom row: persistence diagram
    # -------------------------------------------------------------

    persistence_axis = figure.add_subplot(
        grid[1, :]
    )

    cripser.plot_diagrams(
        [h0, h1],
        labels=[r"$H_0$", r"$H_1$"],
        ax=persistence_axis,
        title=f"{label} Lower-Star Persistence Diagram",
        legend=True,
        diagonal=True,
        marker_size=28,
        alpha=0.80,
        show=False,
    )

    figure.suptitle(
        f"{label}: Lower-Star Filtration and Persistent Homology\n"
        "Darker pixels enter first as the threshold increases",
        fontsize=18,
    )

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            f"\n{label} filtration visualization saved to:\n"
            f"{output_path}"
        )

    return figure


# =====================================================================
# 10. PROCESS ONE IMAGE
# =====================================================================

def process_lower_star_image(
    label,
    image_path,
    output_directory,
):
    """
    Load one image, visualize its filtration, compute PH, and save
    the resulting data and figures.
    """
    image_path = Path(image_path)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = load_lower_star_image(image_path)

    print("\n============================================")
    print(f"PROCESSING {label.upper()} IMAGE")
    print("============================================")
    print(f"Image: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Minimum intensity: {image.min():.3f}")
    print(f"Maximum intensity: {image.max():.3f}")

    # -------------------------------------------------------------
    # Choose the threshold values to display
    # -------------------------------------------------------------

    thresholds = get_lower_star_thresholds(
        image=image,
        number_of_stages=NUMBER_OF_FILTRATION_STAGES,
        custom_thresholds=CUSTOM_THRESHOLD_VALUES,
    )

    print("\nFiltration thresholds:")
    for threshold in thresholds:
        pixels_included = np.mean(
            image <= threshold
        )

        print(
            f"  Threshold {threshold:8.3f}: "
            f"{100.0 * pixels_included:6.2f}% included"
        )

    # -------------------------------------------------------------
    # Compute persistent homology
    # -------------------------------------------------------------

    ph = compute_lower_star_ph(
        image=image,
        maxdim=MAX_HOMOLOGY_DIMENSION,
    )

    h0, h1 = separate_persistence_diagrams(ph)

    print_ph_summary(
        label=label,
        ph=ph,
        h0=h0,
        h1=h1,
    )

    file_label = (
        label.lower()
        .replace(" ", "_")
    )

    # -------------------------------------------------------------
    # Save raw persistent-homology output
    # -------------------------------------------------------------

    ph_output_path = (
        output_directory
        / f"lower_star_{file_label}_ph.npy"
    )

    np.save(
        ph_output_path,
        ph,
    )

    print(
        f"\n{label} raw PH saved to:\n"
        f"{ph_output_path}"
    )

    # -------------------------------------------------------------
    # Save the thresholds used in the visualization
    # -------------------------------------------------------------

    threshold_output_path = (
        output_directory
        / f"lower_star_{file_label}_thresholds.npy"
    )

    np.save(
        threshold_output_path,
        thresholds,
    )

    print(
        f"\n{label} threshold values saved to:\n"
        f"{threshold_output_path}"
    )

    # -------------------------------------------------------------
    # Save the standalone persistence diagram
    # -------------------------------------------------------------

    diagram_output_path = (
        output_directory
        / (
            f"lower_star_{file_label}_"
            "persistence_diagram.png"
        )
    )

    diagram_figure = plot_persistence_diagram(
        label=label,
        h0=h0,
        h1=h1,
        output_path=diagram_output_path,
    )

    # The persistence diagram also appears in the combined figure,
    # so close the standalone version after saving it. This prevents
    # too many figure windows from opening.
    plt.close(diagram_figure)

    # -------------------------------------------------------------
    # Save the combined filtration and persistence figure
    # -------------------------------------------------------------

    combined_output_path = (
        output_directory
        / (
            f"lower_star_{file_label}_"
            "filtration_and_persistence.png"
        )
    )

    combined_figure = plot_filtration_and_persistence(
        label=label,
        image=image,
        thresholds=thresholds,
        h0=h0,
        h1=h1,
        output_path=combined_output_path,
    )

    return {
        "label": label,
        "image": image,
        "thresholds": thresholds,
        "ph": ph,
        "H0": h0,
        "H1": h1,
        "ph_path": ph_output_path,
        "threshold_path": threshold_output_path,
        "diagram_path": diagram_output_path,
        "combined_figure_path": combined_output_path,
        "combined_figure": combined_figure,
    }


# =====================================================================
# 11. RUNNER
# =====================================================================

if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    control_results = process_lower_star_image(
        label="Control",
        image_path=CONTROL_IMAGE,
        output_directory=OUTPUT_DIRECTORY,
    )

    microgravity_results = process_lower_star_image(
        label="Microgravity",
        image_path=MICROGRAVITY_IMAGE,
        output_directory=OUTPUT_DIRECTORY,
    )

    print("\n============================================")
    print("LOWER-STAR PERSISTENT HOMOLOGY COMPLETE")
    print("============================================")
    print(f"Results folder:\n{OUTPUT_DIRECTORY}")

    # Displays one combined figure for the control image and one
    # combined figure for the microgravity image.
    plt.show()