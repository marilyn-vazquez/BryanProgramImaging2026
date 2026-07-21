# -*- coding: utf-8 -*-
"""
Dilation Filtration Comparison: Control vs Microgravity

This script uses the same dilation setup as the experiment:
    1. Load each grayscale image in the range 0-1.
    2. Convert each image to binary using threshold 0.5.
    3. Dilate the ORIGINAL binary image using square structuring elements
       from 2x2 through 20x20.
    4. Accumulate the original binary image and all dilation outputs.
    5. Compare control and microgravity using the SAME settings.

Three figures are saved:
    1. Individual dilation outputs.
    2. Accumulated dilation filtration stages.
    3. Lower-star sublevel sets of the final accumulated image.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from skimage import io
from skimage.util import img_as_float


# =====================================================================
# 1. USER SETTINGS
# =====================================================================

# Replace these with the paths to your two synthetic images.
CONTROL_IMAGE = Path(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images\fake_control001.png"
)

MICROGRAVITY_IMAGE = Path(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images\fake_micro_001.png"
)

THRESHOLD = 0.5
MAX_SE_LENGTH = 20

# Structuring-element sizes displayed in the comparison.
SNAPSHOT_SIZES = (2, 5, 10, 15, 20)

# Thresholds for the final accumulated filtration image.
FINAL_SUBLEVEL_THRESHOLDS = (0, 5, 10, 15, 20)

# You can replace this with any folder where you want the figures saved.
OUTPUT_DIRECTORY = CONTROL_IMAGE.parent / "Filtration_Visualizations"


# =====================================================================
# 2. BINARY THRESHOLDING
# =====================================================================

def find(condition):
    """Return indices where a Boolean condition is True."""
    return np.nonzero(condition)


def biImg_by_threshold_leq(img, threshold):
    """
    Match the experiment's fixed binary thresholding rule.

    Pixels <= threshold become 0.
    Pixels > threshold become 1.
    """
    output_img = np.copy(img)
    output_img[find(img <= threshold)] = 0
    output_img[find(img > threshold)] = 1

    return output_img


# =====================================================================
# 3. DILATION AND STRUCTURING ELEMENTS
# =====================================================================

@nb.jit()
def dilation(
    input_np_array,
    input_list_of_points,
    maximal_pixel_value=1,
):
    """Perform the same binary dilation used in the experiment."""
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


@nb.jit()
def get_rectangle_coordinates(input_np_array):
    """Generate coordinate offsets for one rectangular kernel."""
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
    """Generate square structuring elements from 2x2 to the maximum size."""
    kernel_list = []

    for size in range(2, maximal_SE_lengths + 1):
        kernel_list.append(
            get_rectangle_coordinates(
                np.zeros((size, size))
            )
        )

    return kernel_list


# =====================================================================
# 4. LOAD IMAGE
# =====================================================================

def load_dilation_image(image_path):
    """Load one grayscale image in the same 0-1 scale as the experiment."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            "Could not find the input image:\n"
            f"{image_path}"
        )

    image = img_as_float(
        io.imread(
            image_path,
            as_gray=True,
        )
    )

    return np.asarray(image, dtype=np.float64)


# =====================================================================
# 5. BUILD DILATION DATA
# =====================================================================

def build_dilation_visualization_data(
    binary_image,
    max_se_length=20,
    snapshot_sizes=(2, 5, 10, 15, 20),
):
    """
    Apply each square structuring element to the original binary image.

    Every dilation is computed from the original binary image, matching
    the experiment code.
    """
    requested_sizes = set(snapshot_sizes)

    if not requested_sizes:
        raise ValueError("At least one snapshot size must be supplied.")

    if min(requested_sizes) < 2:
        raise ValueError("Structuring-element sizes must be at least 2.")

    if max(requested_sizes) > max_se_length:
        raise ValueError(
            "A snapshot size is larger than MAX_SE_LENGTH."
        )

    kernel_list = get_square_SE_list(max_se_length)

    # Start with the original binary image, just like the experiment.
    filtration_image = np.zeros(np.shape(binary_image)) + binary_image

    dilation_snapshots = {}
    accumulated_snapshots = {}

    for size, kernel in enumerate(kernel_list, start=2):
        morphed_image = dilation(
            input_np_array=binary_image,
            input_list_of_points=kernel,
        )

        filtration_image = filtration_image + morphed_image

        if size in requested_sizes:
            dilation_snapshots[size] = morphed_image.copy()
            accumulated_snapshots[size] = filtration_image.copy()

    return (
        dilation_snapshots,
        accumulated_snapshots,
        filtration_image,
    )


# =====================================================================
# 6. PLOTTING
# =====================================================================

def plot_individual_dilation_comparison(records, output_path=None):
    """
    Compare original, binary, and individual dilation outputs.

    Row 1: Control
    Row 2: Microgravity
    """
    sizes = list(SNAPSHOT_SIZES)
    number_of_columns = 2 + len(sizes)

    figure, axes = plt.subplots(
        nrows=len(records),
        ncols=number_of_columns,
        figsize=(3.0 * number_of_columns, 7.0),
        squeeze=False,
    )

    for row, record in enumerate(records):
        label = record["label"]

        axes[row, 0].imshow(record["original"], cmap="gray")
        axes[row, 0].set_title("Original Image")
        axes[row, 0].set_ylabel(f"{label}\ny-coordinate")

        axes[row, 1].imshow(
            record["binary"],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title(
            f"Binary Image\nThreshold = {THRESHOLD}"
        )

        for column, size in enumerate(sizes, start=2):
            axes[row, column].imshow(
                record["dilation_snapshots"][size],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].set_title(
                f"Dilation\n{size} x {size} SE"
            )

        for axis in axes[row]:
            axis.set_xlabel("x-coordinate")

    figure.suptitle(
        "Dilation Comparison at Increasing Structuring-Element Sizes",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nIndividual dilation comparison saved to:\n{output_path}")

    return figure


def plot_accumulated_comparison(records, output_path=None):
    """
    Compare the accumulated filtration images.

    Each accumulation includes the original binary image and every
    dilation from 2x2 through the displayed structuring-element size.
    """
    sizes = list(SNAPSHOT_SIZES)
    number_of_columns = 1 + len(sizes)

    figure, axes = plt.subplots(
        nrows=len(records),
        ncols=number_of_columns,
        figsize=(3.1 * number_of_columns, 7.0),
        squeeze=False,
    )

    displayed_image = None

    for row, record in enumerate(records):
        label = record["label"]

        displayed_image = axes[row, 0].imshow(
            record["binary"],
            cmap="gray",
            vmin=0,
            vmax=MAX_SE_LENGTH,
        )
        axes[row, 0].set_title("Starting Binary Image")
        axes[row, 0].set_ylabel(f"{label}\ny-coordinate")

        for column, size in enumerate(sizes, start=1):
            displayed_image = axes[row, column].imshow(
                record["accumulated_snapshots"][size],
                cmap="gray",
                vmin=0,
                vmax=MAX_SE_LENGTH,
            )
            axes[row, column].set_title(
                f"Accumulated Through\n{size} x {size} SE"
            )

        for axis in axes[row]:
            axis.set_xlabel("x-coordinate")

    colorbar = figure.colorbar(
        displayed_image,
        ax=axes,
        shrink=0.75,
        pad=0.02,
    )
    colorbar.set_label("Accumulated filtration value")

    figure.suptitle(
        "Accumulated Dilation Filtration Comparison",
        fontsize=16,
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.92,
        bottom=0.10,
        top=0.86,
        wspace=0.28,
        hspace=0.30,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nAccumulated comparison saved to:\n{output_path}")

    return figure


def plot_final_sublevel_comparison(records, output_path=None):
    """
    Compare lower-star sublevel sets of the final accumulated images.
    """
    thresholds = tuple(sorted(FINAL_SUBLEVEL_THRESHOLDS))
    number_of_columns = len(thresholds)

    figure, axes = plt.subplots(
        nrows=len(records),
        ncols=number_of_columns,
        figsize=(3.3 * number_of_columns, 7.0),
        squeeze=False,
    )

    for row, record in enumerate(records):
        label = record["label"]
        final_image = record["final_filtration"]

        for column, threshold in enumerate(thresholds):
            included_mask = final_image <= threshold

            display_image = np.ones_like(final_image, dtype=float)
            display_image[included_mask] = 0.0

            axes[row, column].imshow(
                display_image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].set_title(
                f"Value <= {threshold}\n"
                f"{100.0 * np.mean(included_mask):.1f}% included"
            )
            axes[row, column].set_xlabel("x-coordinate")

            if column == 0:
                axes[row, column].set_ylabel(
                    f"{label}\ny-coordinate"
                )

    figure.suptitle(
        "Lower-Star Sublevel Sets of the Final Dilation Filtration",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nFinal sublevel comparison saved to:\n{output_path}")

    return figure


# =====================================================================
# 7. RUNNER
# =====================================================================

if __name__ == "__main__":
    input_records = [
        {
            "label": "Control",
            "path": CONTROL_IMAGE,
        },
        {
            "label": "Microgravity",
            "path": MICROGRAVITY_IMAGE,
        },
    ]

    processed_records = []

    print("\n============================================")
    print("DILATION CONTROL VS MICROGRAVITY")
    print("============================================")
    print(f"Binary threshold: {THRESHOLD}")
    print(f"Maximum structuring-element size: {MAX_SE_LENGTH}")
    print(f"Displayed sizes: {SNAPSHOT_SIZES}")

    for record in input_records:
        original_image = load_dilation_image(record["path"])

        binary_image = biImg_by_threshold_leq(
            img=original_image,
            threshold=THRESHOLD,
        )

        (
            dilation_snapshots,
            accumulated_snapshots,
            final_filtration_image,
        ) = build_dilation_visualization_data(
            binary_image=binary_image,
            max_se_length=MAX_SE_LENGTH,
            snapshot_sizes=SNAPSHOT_SIZES,
        )

        processed_records.append(
            {
                "label": record["label"],
                "path": record["path"],
                "original": original_image,
                "binary": binary_image,
                "dilation_snapshots": dilation_snapshots,
                "accumulated_snapshots": accumulated_snapshots,
                "final_filtration": final_filtration_image,
            }
        )

        print(f"\n{record['label']} image: {record['path'].name}")
        print(f"Shape: {original_image.shape}")
        print(f"Foreground pixels: {np.sum(binary_image == 1)}")
        print(
            "Final filtration range: "
            f"{final_filtration_image.min():.3f} to "
            f"{final_filtration_image.max():.3f}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    plot_individual_dilation_comparison(
        records=processed_records,
        output_path=(
            OUTPUT_DIRECTORY
            / "dilation_control_vs_microgravity_individual.png"
        ),
    )

    plot_accumulated_comparison(
        records=processed_records,
        output_path=(
            OUTPUT_DIRECTORY
            / "dilation_control_vs_microgravity_accumulated.png"
        ),
    )

    plot_final_sublevel_comparison(
        records=processed_records,
        output_path=(
            OUTPUT_DIRECTORY
            / "dilation_control_vs_microgravity_sublevels.png"
        ),
    )

    plt.show()