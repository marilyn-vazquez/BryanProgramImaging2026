# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cripser
from persim import PersLandscapeApprox
from persim.landscapes import plot_landscape_simple


# ------------------------------------------------------------
# 1. Load and crop image
# ------------------------------------------------------------

def load_grayscale_image(image_path, pixels_to_remove=296):
    """
    Loads an image, converts it to grayscale, removes the bottom scale bar,
    and returns the image as a float NumPy array.
    """
    img = Image.open(image_path).convert("L")

    width, height = img.size
    cropped_img = img.crop((0, 0, width, height - pixels_to_remove))

    img_array = np.array(cropped_img).astype(float)

    return img_array


# ------------------------------------------------------------
# 2. Compute persistent homology with cripser
# ------------------------------------------------------------

def compute_persistence(img_array):
    """
    Computes cubical persistent homology using cripser.
    This is lower-star filtration on the image.
    """
    ph = cripser.computePH(img_array)
    return ph


# ------------------------------------------------------------
# 3. Convert cripser output into persim format and filter points
# ------------------------------------------------------------

def cripser_output_to_persim_dgms(
    ph,
    maxdim=1,
    min_persistence=5.0,
    max_pairs=1000
):
    """
    Converts cripser output into the format persim expects.

    Persim wants:
        dgms[0] = H0 birth-death pairs
        dgms[1] = H1 birth-death pairs

    This function also filters the diagram so Persim does not
    try to build landscapes from hundreds of thousands of points.
    """

    dgms = []

    for dim in range(maxdim + 1):

        # Select birth-death pairs for this homology dimension
        dgm = ph[ph[:, 0] == dim][:, 1:3].astype(float)

        # Remove infinite values
        dgm = dgm[np.isfinite(dgm).all(axis=1)]

        original_count = len(dgm)

        if len(dgm) > 0:

            # Compute persistence = death - birth
            persistence = dgm[:, 1] - dgm[:, 0]

            # Keep only features above the persistence threshold
            dgm = dgm[persistence > min_persistence]

            # Recompute persistence after filtering
            persistence = dgm[:, 1] - dgm[:, 0]

            # Keep only the most persistent features
            if max_pairs is not None and len(dgm) > max_pairs:
                order = np.argsort(persistence)[::-1]
                dgm = dgm[order[:max_pairs]]

        filtered_count = len(dgm)

        print(f"H{dim} original points: {original_count}")
        print(f"H{dim} points after filtering: {filtered_count}")

        dgms.append(dgm)

    return dgms


# ------------------------------------------------------------
# 4. Create persistence landscape vector
# ------------------------------------------------------------

def landscape_vector(
    dgms,
    hom_deg,
    start=0,
    stop=255,
    num_steps=100,
    num_layers=3
):
    """
    Converts one persistence diagram into a fixed-length vector
    using persistence landscapes.

    Example:
        num_layers = 3
        num_steps = 100

    Then one homology dimension gives:
        3 * 100 = 300 features
    """

    # If there are no features in this dimension, return zeros
    if hom_deg >= len(dgms) or len(dgms[hom_deg]) == 0:
        return np.zeros(num_layers * num_steps)

    # Compute approximate persistence landscape
    pl = PersLandscapeApprox(
        dgms=dgms,
        hom_deg=hom_deg,
        start=start,
        stop=stop,
        num_steps=num_steps
    )

    # Sampled landscape values
    values = pl.values

    # Force every image to have the same number of landscape layers
    fixed_values = np.zeros((num_layers, num_steps))

    layers_to_copy = min(num_layers, values.shape[0])
    fixed_values[:layers_to_copy, :] = values[:layers_to_copy, :]

    # Flatten into one long vector
    return fixed_values.ravel()


def full_landscape_feature_vector(
    dgms,
    start=0,
    stop=255,
    num_steps=100,
    num_layers=3
):
    """
    Creates one feature vector using both H0 and H1 landscapes.

    H0 vector length = num_layers * num_steps
    H1 vector length = num_layers * num_steps

    Total length = 2 * num_layers * num_steps
    """

    h0_vec = landscape_vector(
        dgms,
        hom_deg=0,
        start=start,
        stop=stop,
        num_steps=num_steps,
        num_layers=num_layers
    )

    h1_vec = landscape_vector(
        dgms,
        hom_deg=1,
        start=start,
        stop=stop,
        num_steps=num_steps,
        num_layers=num_layers
    )

    feature_vec = np.concatenate([h0_vec, h1_vec])

    return feature_vec


# ------------------------------------------------------------
# 5. Plot persistence landscape
# ------------------------------------------------------------

def plot_landscape(dgms, hom_deg, label, start=0, stop=255, num_steps=100):
    if hom_deg >= len(dgms) or len(dgms[hom_deg]) == 0:
        print(f"No H{hom_deg} features found for {label}")
        return

    pl = PersLandscapeApprox(
        dgms=dgms,
        hom_deg=hom_deg,
        start=start,
        stop=stop,
        num_steps=num_steps
    )

    plt.figure(figsize=(7, 5))
    ax = plot_landscape_simple(pl)
    ax.set_title(f"H{hom_deg} Persistence Landscape: {label}")

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. Main experiment on four images
# ------------------------------------------------------------

image_paths = [
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Control_Stub4_0000 .tif",
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\control_stub1_3-2-26_0008.tif",
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\microgravity_stub1_3-4-26_0001.tif",
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\microgravity_stub1_3-4-26_0002.tif",
]

image_labels = [
    "Control 1",
    "Control 2",
    "Microgravity 1",
    "Microgravity 2",
]


# ------------------------------------------------------------
# 7. Settings
# ------------------------------------------------------------

# For lower-star grayscale filtration on 0-255 images
START = 0
STOP = 255

# Smaller landscape settings for testing
NUM_STEPS = 100
NUM_LAYERS = 3

# Filtering settings
MIN_PERSISTENCE = 5.0
MAX_PAIRS = 1000

# Plot settings
PLOT_PD = True
PLOT_H0_LANDSCAPE = False
PLOT_H1_LANDSCAPE = True


# ------------------------------------------------------------
# 8. Run the experiment
# ------------------------------------------------------------

all_vectors = []

for image_path, label in zip(image_paths, image_labels):

    print("\n--------------------------------------------")
    print("Processing:", label)
    print("--------------------------------------------")

    # Load image
    img_array = load_grayscale_image(image_path)

    print("Image shape:", img_array.shape)
    print("Image min:", img_array.min())
    print("Image max:", img_array.max())

    # Compute persistent homology
    print("Computing persistent homology with cripser...")
    ph = compute_persistence(img_array)

    # Plot persistence diagram using cripser
    if PLOT_PD:
        print("Plotting persistence diagram...")
        cripser.plot_diagrams(ph)
        plt.title(f"Persistence Diagram: {label}")
        plt.show()

    # Convert and filter for Persim
    print("Converting and filtering diagram for Persim...")

    dgms = cripser_output_to_persim_dgms(
        ph,
        maxdim=1,
        min_persistence=MIN_PERSISTENCE,
        max_pairs=MAX_PAIRS
    )

    # Plot H0 landscape only if requested
    if PLOT_H0_LANDSCAPE:
        print("Plotting H0 persistence landscape...")
        plot_landscape(
            dgms,
            hom_deg=0,
            label=label,
            start=START,
            stop=STOP,
            num_steps=NUM_STEPS
        )

    # Plot H1 landscape only if requested
    if PLOT_H1_LANDSCAPE:
        print("Plotting H1 persistence landscape...")
        plot_landscape(
            dgms,
            hom_deg=1,
            label=label,
            start=START,
            stop=STOP,
            num_steps=NUM_STEPS
        )

    # Create full fixed-length vector
    print("Creating landscape feature vector...")

    feature_vec = full_landscape_feature_vector(
        dgms,
        start=START,
        stop=STOP,
        num_steps=NUM_STEPS,
        num_layers=NUM_LAYERS
    )

    print("Feature vector length:", len(feature_vec))

    all_vectors.append(feature_vec)


# ------------------------------------------------------------
# 9. Final feature matrix
# ------------------------------------------------------------

X = np.vstack(all_vectors)

print("\n============================================")
print("Final feature matrix shape:")
print(X.shape)
print("Rows = images")
print("Columns = persistence landscape features")
print("============================================")



