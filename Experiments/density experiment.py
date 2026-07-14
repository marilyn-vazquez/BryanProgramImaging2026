# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:51:17 2026

@author: chloe
"""

import os
from pathlib import Path
import numpy as np
import cripser as cr
from scipy.spatial import KDTree
from skimage import io, filters
from skimage.util import img_as_float

# =====================================================================
# 1. DENSITY FILTRATION & TDA FUNCTIONS
# =====================================================================

def density_filtration(binary_image, max_dist=5):
    """
    Generate a density-based filtration from a binary image.

    Local pixel density is calculated by counting foreground pixels
    within a specified radius. Dense regions receive lower filtration
    values and appear earlier.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image containing foreground structures.
    max_dist : float, optional
        Radius used for neighborhood density calculations. Default at 5.

    Returns
    -------
    numpy.ndarray
        Density filtration image.
    """
    height, width = binary_image.shape
    # Find foreground pixel coordinates
    points = np.argwhere(binary_image)

    # Build neighborhood tree
    tree = KDTree(points, leaf_size=30, metric="euclidean")
   
    point_cloud = np.zeros((height * width, 2))
   
    p = 0
    for i in range(height):
        for j in range(width):
            point_cloud[p, 0] = i
            point_cloud[p, 1] = j
            p += 1

    # Coordinates for every pixel
    num_nbhs = tree.query_radius(
        point_cloud,
        r=max_dist,
        count_only=True
    )

    filt_func_vals = num_nbhs

    # Maximum number of pixels possible in neighborhood
    max_num_nbhs = filt_func_vals.max()

    # Convert density into filtration values
    filt_func_vals = max_num_nbhs - filt_func_vals

    # Reshape back into image
    density_filt_img = filt_func_vals.reshape(height, width)

    return density_filt_img


def compute_density_ph(binary_image, max_dist=5):
    """
    Compute persistent homology using density filtration.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary input image.
    max_dist : float, optional
        Neighborhood radius for density calculation. Default is 5

    Returns
    -------
    tuple
        Density filtration image and persistence diagram.
    """
    density_img = density_filtration(binary_image, max_dist)

    # Adapt dynamically to whichever syntax your cripser build expects
    if hasattr(cr, "computePH"):
        ph_density = cr.computePH(density_img.astype(np.float64))
    else:
        ph_density = cr.compute_ph(density_img.astype(np.float64))

    return density_img, ph_density

# =====================================================================
# 2. RUNNER CONTROLLER
# =====================================================================

if __name__ == '__main__':
    # Directly point to the folder containing your preprocessed images
    PROCESSED_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
   
    # Gather processed target images directly from the directory
    processed_paths = sorted(list(PROCESSED_DIR.glob('*_processed.tif')))
   
    if not processed_paths:
        raise FileNotFoundError(f"Could not find any processed images in: {PROCESSED_DIR}.")
       
    print(f"Found {len(processed_paths)} preprocessed images to process.")

    print("\n=== Starting Isolated Density Homology Extraction ===")
    
    for path in processed_paths:
        path = Path(path)
        print(f"Processing Density TDA for: {path.name}")
        
        # 1. Load preprocessed image as grayscale float
        img_grayscale = img_as_float(io.imread(path, as_gray=True))
        
        # 2. Binarize using an automated Otsu global threshold
        thresh_val = filters.threshold_otsu(img_grayscale)
        binary_img = img_grayscale > thresh_val
        
        # 3. Compute the custom density filtration and persistent homology diagram
        density_img, ph_diagram = compute_density_ph(binary_img, max_dist=5)
        
        # 4. Save the raw persistence diagram matrix [dim, birth, death]
        save_path = PROCESSED_DIR / f"{path.stem}_density_diagram.npy"
        np.save(save_path, ph_diagram)
        
    print(f"\n✅ All density persistence diagrams saved to: {PROCESSED_DIR}")