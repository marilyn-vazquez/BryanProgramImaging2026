# -*- coding: utf-8 -*-
"""
Lower-Star Filtration Comparison: Control vs Microgravity

This script uses the same lower-star setup as the experiment:
    1. Load each grayscale image.
    2. Convert it to floating point.
    3. Scale values to 0-255 when needed.
    4. Display lower-star sublevel sets f(x) <= threshold.

The control and microgravity images use the SAME thresholds so their
filtration behavior can be compared fairly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

# Use the same thresholds for both images.
THRESHOLDS = (25, 50, 100, 150, 200, 255)

# Included pixels are shown in black to resemble the example figure.
INCLUDED_PIXELS_ARE_BLACK = True

# You can replace this with any folder where you want the figure saved.
OUTPUT_DIRECTORY = CONTROL_IMAGE.parent / "Filtration_Visualizations"
OUTPUT_FILENAME = "lower_star_control_vs_microgravity.png"


# =====================================================================
# 2. LOAD IMAGE EXACTLY LIKE THE EXPERIMENT
# =====================================================================

def load_lower_star_image(image_path):
    """
    Load one grayscale image and scale it to 0-255 when needed.

    This matches the image-loading behavior in the lower-star experiment.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            "Could not find the input image:\n"
            f"{image_path}"
        )

    image = img_as_float(io.imread(image_path, as_gray=True))

    if image.max() <= 1.0:
        image = image * 255.0

    return np.asarray(image, dtype=np.float64)


# =====================================================================
# 3. CREATE LOWER-STAR STAGES
# =====================================================================

def create_lower_star_stages(image, thresholds):
    """
    Create lower-star sublevel sets.

    At threshold t, a pixel is included when:
        image[row, column] <= t
    """
    thresholds = tuple(sorted(thresholds))

    if len(thresholds) == 0:
        raise ValueError("At least one threshold must be supplied.")

    return {
        threshold: image <= threshold
        for threshold in thresholds
    }


# =====================================================================
# 4. PLOT CONTROL AND MICROGRAVITY TOGETHER
# =====================================================================

def plot_lower_star_comparison(
    image_records,
    thresholds,
    output_path=None,
    included_pixels_are_black=True,
):
    """
    Plot one row for the control image and one row for the microgravity image.

    Each row contains:
        original image, followed by all lower-star threshold stages.
    """
    thresholds = tuple(sorted(thresholds))
    number_of_columns = 1 + len(thresholds)

    figure, axes = plt.subplots(
        nrows=len(image_records),
        ncols=number_of_columns,
        figsize=(3.25 * number_of_columns, 7.0),
        squeeze=False,
    )

    for row, record in enumerate(image_records):
        label = record["label"]
        image = record["image"]
        stages = record["stages"]

        # Original image.
        original_axis = axes[row, 0]
        original_axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        original_axis.set_title("Original Image")
        original_axis.set_ylabel(
            f"{label}\ny-coordinate",
            fontsize=11,
        )
        original_axis.set_xlabel("x-coordinate")

        # Lower-star threshold stages.
        for column, threshold in enumerate(thresholds, start=1):
            axis = axes[row, column]
            included_mask = stages[threshold]

            if included_pixels_are_black:
                display_image = np.ones_like(image, dtype=float)
                display_image[included_mask] = 0.0
            else:
                display_image = np.zeros_like(image, dtype=float)
                display_image[included_mask] = 1.0

            axis.imshow(
                display_image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            included_percent = 100.0 * np.mean(included_mask)

            axis.set_title(
                f"Threshold = {threshold}\n"
                f"{included_percent:.1f}% included",
                fontsize=10,
            )
            axis.set_xlabel("x-coordinate")

    figure.suptitle(
        "Lower-Star Filtration Comparison\n"
        "Pixels enter when intensity is less than or equal to the threshold",
        fontsize=16,
    )

    figure.tight_layout(rect=[0, 0, 1, 0.92])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"\nComparison figure saved to:\n{output_path}")

    return figure


# =====================================================================
# 5. RUNNER
# =====================================================================

if __name__ == "__main__":
    control_image = load_lower_star_image(CONTROL_IMAGE)
    microgravity_image = load_lower_star_image(MICROGRAVITY_IMAGE)

    control_stages = create_lower_star_stages(
        image=control_image,
        thresholds=THRESHOLDS,
    )

    microgravity_stages = create_lower_star_stages(
        image=microgravity_image,
        thresholds=THRESHOLDS,
    )

    image_records = [
        {
            "label": "Control",
            "path": CONTROL_IMAGE,
            "image": control_image,
            "stages": control_stages,
        },
        {
            "label": "Microgravity",
            "path": MICROGRAVITY_IMAGE,
            "image": microgravity_image,
            "stages": microgravity_stages,
        },
    ]

    print("\n============================================")
    print("LOWER-STAR CONTROL VS MICROGRAVITY")
    print("============================================")
    print(f"Thresholds: {THRESHOLDS}")

    for record in image_records:
        image = record["image"]
        print(f"\n{record['label']} image: {record['path'].name}")
        print(f"Shape: {image.shape}")
        print(f"Minimum intensity: {image.min():.3f}")
        print(f"Maximum intensity: {image.max():.3f}")

    output_path = OUTPUT_DIRECTORY / OUTPUT_FILENAME

    plot_lower_star_comparison(
        image_records=image_records,
        thresholds=THRESHOLDS,
        output_path=output_path,
        included_pixels_are_black=INCLUDED_PIXELS_ARE_BLACK,
    )

    plt.show()