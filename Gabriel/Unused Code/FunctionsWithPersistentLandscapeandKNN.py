# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import pandas as pd
import cripser
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
from skimage import io, exposure, filters
from skimage.util import img_as_float
from persim import PersLandscapeApprox
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

# -------------------------------------------------------------------
# PRE-PROCESSING
# -------------------------------------------------------------------

def preprocess_single(image_input, sigma=0.5, clip_limit=0.015):
    """
    Preprocess a single microscopy image.

    This function loads an image, converts it to grayscale floating-point
    format, removes non-biological regions, smooths noise using a Gaussian
    filter, and enhances local contrast using CLAHE.

    Parameters
    ----------
    image_input : str, pathlib.Path, or numpy.ndarray
        Input image file path or image array.
    sigma : float, optional
        Standard deviation for Gaussian smoothing. Larger values produce
        stronger smoothing. Default value is 0.5
    clip_limit : float, optional
        Contrast limiting parameter for CLAHE enhancement. Default value is 0.015

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale image with enhanced contrast.

    Notes
    -----
    The image is cropped to remove the microscope information bar before
    further processing.

    Example
    -------
  >>> processed_img = preprocess_single("cilia_image.png", sigma=0.5, clip_limit=0.015)
    >>> print(type(processed_img), processed_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    # prepares a raw microscopy image before applying filtration methods. The goal is to remove noise improve contrast, and standardize images. 
    # Load image as grayscale if it's a path
    img = img_as_float(io.imread(image_input, as_gray=True)) if isinstance(image_input, (str, Path)) else img_as_float(image_input)
    # Reads image as grayscale, converts into floating point pixel values between 0 and 1
    # Crop information bar, ensures only biological regions are analyzed
    img = img[:3850, :]
   
    # Smooth image, applies gaussian blur, reduces pixel-level noise. 
    img_smoothed = filters.gaussian(img, sigma=sigma)
   
    # Apply CLAHE, adaptive histogram equalization, improves contrast. 
    img_clahe = exposure.equalize_adapthist(img_smoothed, kernel_size=256, clip_limit=clip_limit)
   # returns img_clache, normalized and enhanced grayscale image ready for filtration
    return img_clahe

def process_batch(image_paths, reference_path, out_dir, sigma=0.5, clip_limit=0.015):
    """
    Apply preprocessing to a collection of microscopy images.

    Each image is processed using Gaussian smoothing and CLAHE.
    Histogram matching is then applied so all images have comparable
    intensity distributions relative to a reference image.

    Parameters
    ----------
    image_paths : list
        Collection of image file paths to process.
    reference_path : str or pathlib.Path
        Reference image used for histogram normalization.
    out_dir : str or pathlib.Path
        Directory where processed images are saved.
    sigma : float, optional
        Gaussian smoothing parameter.
    clip_limit : float, optional
        CLAHE contrast enhancement parameter.

    Returns
    -------
    int
        Number of successfully processed images.

    Notes
    -----
    Images are saved as 32-bit floating point TIFF files to preserve
    intensity precision.

    Example
    ------
    >>> processed = process_batch(["img1.png", "img2.png"], "ref.png", "./output")
    >>> print(processed)
    """
    # Processes muultiple images using the same pipeline and standardizes their appearance relative to reference image. 
    #Ensure the output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare reference image (at full resolution)
    print("Preparing reference image...")
    ref_ready = preprocess_single(reference_path, sigma=sigma, clip_limit=clip_limit)
   
    processed_files_count = 0
   
    # Process all other images in batch
    for path in image_paths:
        path = Path(path)
        print(f"Processing: {path.name}")
       
        # Check if this image is the reference image
        if path == Path(reference_path):
            img_ready = ref_ready
        else:
            # Apply smoothing and CLAHE
            img_clahe = preprocess_single(path, sigma=sigma, clip_limit=clip_limit)
           
            # Globally match histogram to reference image at full resolution
            img_ready = exposure.match_histograms(img_clahe, ref_ready)
                       
        # Define save path and save immediately as a high-precision 32-bit float TIFF
        save_path = out_dir / f"{path.stem}_processed.tif"
        io.imsave(save_path, img_ready, check_contrast=False)
       
        processed_files_count += 1
       
    return processed_files_count
# saves standard TIFF files, returns the number of processed images
# -------------------------------------------------------------------
# MORPHOLOGY FILTRATION
# -------------------------------------------------------------------

# Basic tool for finding indices in a NumPy array
def find(condition):
    res = np.nonzero(condition)
    return res
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
    Example
    ------
    >>> idxs = find(img > 0.5)
    >>> print(type(idxs), len(idxs))
    <class 'tuple'> 2
    """
# helper function for finding pixel locations satisfy a condition. 

# Thresholding operation: pixels <= threshold become 0, pixels > threshold become 1
def biImg_by_threshold_leq(img, threshold):
    """
    Convert an image into a binary image using a lower threshold.

    Pixels with intensity values less than or equal to the threshold
    are assigned 0, while pixels above the threshold are assigned 1.

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
    Example
    -------
    >>> binary_img = biImg_by_threshold_leq(img, threshold=0.5)
    >>> print(binary_img.min(), binary_img.max())
    0.0 1.0
    """
    output_img = np.copy(img)

    idxs_0 = find(img <= threshold)
    idxs_1 = find(img > threshold)

    output_img[idxs_0] = 0
    output_img[idxs_1] = 1

    return output_img


# Thresholding operation: pixels >= threshold become 0, pixels < threshold become 1
def biImg_by_threshold_geq(img, threshold):
    """
    Convert an image into a binary image using an upper threshold.

    Pixels greater than or equal to the threshold are assigned 0,
    while pixels below the threshold are assigned 1.

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

    Example
    -------
    >>> binary_img = biImg_by_threshold_geq(img, threshold=0.5)
    >>> print(binary_img.min(), binary_img.max())
    0.0 1.0
    """
    output_img = np.copy(img)

    idxs_0 = find(img >= threshold)
    idxs_1 = find(img < threshold)

    output_img[idxs_0] = 0
    output_img[idxs_1] = 1

    return output_img

   
@nb.jit()
def erosion(input_np_array,
            input_list_of_points,
            minimal_pixel_value=0):
    """
    Perform morphological erosion on a binary image.

    Erosion removes boundary pixels and reduces foreground structures.
    The output value of each pixel is determined by the minimum value
    within the structuring element neighborhood.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinate offsets.
    minimal_pixel_value : int, optional
        Pixel value representing background. Default at 0. 

    Returns
    -------
    numpy.ndarray
        Eroded binary image.

    Example
    -------
    >>> eroded_img = erosion(binary_img, square_kernels[0])
    >>> print(type(eroded_img), eroded_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    input_np_array_shape = np.shape(input_np_array)
    output_np_array = np.zeros(input_np_array_shape)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            if input_np_array[i, j] == minimal_pixel_value:
                output_np_array[i, j] = minimal_pixel_value
                continue

            relevant_pixel_values = []

            for k in range(len(input_list_of_points)):

                m = i - input_list_of_points[k][1]
                n = j + input_list_of_points[k][0]

                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])

            output_np_array[i, j] = min(relevant_pixel_values)

    return output_np_array


@nb.jit()
def dilation(input_np_array,
             input_list_of_points,
             maximal_pixel_value=1):
    """
    Perform morphological dilation on a binary image.

    Dilation expands foreground regions by assigning each pixel the
    maximum value found within the structuring element neighborhood.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinate offsets.
    maximal_pixel_value : int, optional
        Pixel value representing foreground. Default of 1. 

    Returns
    -------
    numpy.ndarray
        Dilated binary image.

    Example
    -------
    >>> dilated_img = dilation(binary_img, square_kernels[0])
    >>> print(type(dilated_img), dilated_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    input_np_array_shape = np.shape(input_np_array)
    output_np_array = np.zeros(input_np_array_shape)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            if input_np_array[i, j] == maximal_pixel_value:
                output_np_array[i, j] = maximal_pixel_value
                continue

            relevant_pixel_values = []

            for k in range(len(input_list_of_points)):

                m = i + input_list_of_points[k][1]
                n = j - input_list_of_points[k][0]

                if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
                    relevant_pixel_values.append(input_np_array[m, n])

            output_np_array[i, j] = max(relevant_pixel_values)

    return output_np_array


def opening(input_np_array,
               input_list_of_points):
    """
    Perform morphological opening.

    Opening consists of erosion followed by dilation. It removes small
    objects and noise while preserving larger structures.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinates.

    Returns
    -------
    numpy.ndarray
        Opened image.

   Example
   -------
   >>> opened_img = opening(binary_img, square_kernels[0])
    >>> print(type(opened_img), opened_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    return dilation(
        erosion(input_np_array, input_list_of_points),
        input_list_of_points
    )


def closing(input_np_array,
            input_list_of_points):
    """
    Perform morphological closing.

    Closing consists of dilation followed by erosion. It fills small
    holes and connects nearby foreground regions.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Binary input image.
    input_list_of_points : list
        Structuring element coordinates.

    Returns
    -------
    numpy.ndarray
        Closed image.

    Example
    -------
    >>> closed_img = closing(binary_img, square_kernels[0])
    >>> print(type(closed_img), closed_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    return erosion(
        dilation(input_np_array, input_list_of_points),
        input_list_of_points
    )

@nb.jit()
def get_rectangle_coordinates(input_np_array):
    """
    Generate coordinate offsets for a rectangular structuring element.

    This function creates a list of relative pixel coordinates centered
    around the origin of an input array. These coordinates define the
    neighborhood used by morphological operations such as erosion and
    dilation.

    Parameters
    ----------
    input_np_array : numpy.ndarray
        Array representing the desired size of the rectangular
        structuring element.

    Returns
    -------
    list
        List of numpy arrays containing coordinate offsets relative
        to the center pixel.

    Notes
    -----
    The center of the structuring element is treated as (0,0). For
    example, a 3x3 structuring element produces coordinates covering
    the entire neighborhood around the central pixel.

    Example
    -------
    A 3x3 structuring element generates offsets:

        [-1, -1]  [0, -1]  [1, -1]
        [-1,  0]  [0,  0]  [1,  0]
        [-1,  1]  [0,  1]  [1,  1]
    """
    input_np_array_shape = np.shape(input_np_array)
    output_list = []

    origin_i = int(input_np_array_shape[0] / 2)
    origin_j = int(input_np_array_shape[1] / 2)

    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):

            output_list.append(np.array([origin_j - j, origin_i - i]))

    return output_list


def get_horizontal_SE_list(maximal_SE_lengths):
    """
    Generate horizontal line structuring elements of increasing size.

    This function creates a collection of horizontal structuring elements
    with lengths ranging from 2 pixels up to the specified maximum length.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum size of the horizontal structuring element.

    Returns
    -------
    list
        List of horizontal structuring elements. Each element contains
        coordinate offsets representing pixels along a horizontal line.

    Notes
    -----
    These structuring elements are used in morphological operations to
    analyze horizontally oriented structures at different spatial scales.

    Example
    -------
    >>> h_kernels = get_horizontal_SE_list(maximal_SE_lengths=5)
    >>> print(type(h_kernels), len(h_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):

        buffer_list = []
        buffer_list.append(np.array([0, 0]))

        counter = int(1)
        left_counter = int(-1)
        right_counter = int(1)

        while len(buffer_list) < i:

            if counter % 2 == 1:
                buffer_list.append(np.array([right_counter, 0]))
                counter = counter + 1
                right_counter = right_counter + 1

            else:
                buffer_list.append(np.array([left_counter, 0]))
                counter = counter + 1
                left_counter = left_counter - 1

        kernel_list.append(buffer_list)

    return kernel_list


def get_vertical_SE_list(maximal_SE_lengths):
    """
    Generate vertical line structuring elements of increasing size.

    Creates vertical structuring elements ranging from length 2 to the
    specified maximum length.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum size of the vertical structuring element.

    Returns
    -------
    list
        List of vertical structuring elements represented as coordinate
        offsets.

    Notes
    -----
    These kernels allow morphological operations to examine vertically
    oriented image structures across multiple spatial scales.

    Example
    -------
    >>> v_kernels = get_vertical_SE_list(maximal_SE_lengths=5)
    >>> print(type(v_kernels), len(v_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):

        buffer_list = []
        buffer_list.append(np.array([0, 0]))

        counter = int(1)
        left_counter = int(-1)
        right_counter = int(1)

        while len(buffer_list) < i:

            if counter % 2 == 1:
                buffer_list.append(np.array([0, right_counter]))
                counter = counter + 1
                right_counter = right_counter + 1

            else:
                buffer_list.append(np.array([0, left_counter]))
                counter = counter + 1
                left_counter = left_counter - 1

        kernel_list.append(buffer_list)

    return kernel_list


def get_square_SE_list(maximal_SE_lengths):
    """
    Generate square structuring elements of increasing size.

    This function creates square neighborhoods used for morphological
    operations. Each square size produces a different spatial scale
    for filtration.

    Parameters
    ----------
    maximal_SE_lengths : int
        Maximum side length of the square structuring element.

    Returns
    -------
    list
        Collection of square structuring elements represented by
        coordinate offsets.

    Notes
    -----
    Increasing structuring element size allows morphological
    filtrations to capture features at progressively larger scales.

    Example
    -------
    >>> square_kernels = get_square_SE_list(maximal_SE_lengths=5)
    >>> print(type(square_kernels), len(square_kernels))
    <class 'list'> 4
    """
    kernel_list = []

    for i in range(2, maximal_SE_lengths + 1):
        kernel_list.append(
            get_rectangle_coordinates(input_np_array=np.zeros((i, i)))
        )

    return kernel_list

def img_to_1d_array(img):
    """
    Convert a two-dimensional image array into a one-dimensional array.

    Persistent homology libraries often require image data to be stored
    as a flattened vector. This function converts each row of the image
    into a single ordered array.

    Parameters
    ----------
    img : numpy.ndarray
        Two-dimensional image array.

    Returns
    -------
    numpy.ndarray
        Flattened one-dimensional representation of the image.

    Notes
    -----
    Pixel ordering is preserved by concatenating rows in reverse order
    relative to the input image.

    Example
    -------
    >>> flattened = flatten_img_reverse_rows(img)
    >>> print(flattened.ndim, flattened.shape[0] == img.size)
    1 True
    """
    result = []
    img_shape = np.shape(img)

    for i in range(img_shape[0]):
        result = list(img[i, :]) + result

    return np.array(result)


def persistence_of_img(img, maxdim=1):
    """
    Compute persistent homology of an image filtration.

    Calculates persistence diagrams describing the birth and death
    of topological features.

    Parameters
    ----------
    img : numpy.ndarray
        Input filtration image.
    maxdim : int, optional
        Maximum homology dimension calculated.

    Returns
    -------
    list
        Two persistence diagrams:
        - H0: connected components
        - H1: loops and holes

    Example
    -------
    >>> h0_h1_diagrams = compute_persistence(img, maxdim=1)
    >>> print(type(h0_h1_diagrams), len(h0_h1_diagrams))
    <class 'list'> 2
    """
    img = np.asarray(img, dtype=np.float64)

    if hasattr(cripser, "compute_ph"):
        ph = cripser.compute_ph(img, maxdim=maxdim)
    else:
        ph = cripser.computePH(img, maxdim=maxdim)

    persistence_0 = ph[ph[:, 0] == 0][:, 1:3]
    persistence_1 = ph[ph[:, 0] == 1][:, 1:3]

    return [persistence_0, persistence_1]

def persistence_of_morph_filtration(img,
                                    kernel_list,
                                    morph_type='closing'):
    """
    Compute persistent homology from a morphological filtration.

    Multiple morphological operations are applied using structuring
    elements of increasing size. The resulting filtration captures
    how image structures change across spatial scales.

    Parameters
    ----------
    img : numpy.ndarray
        Input binary image.
    kernel_list : list
        Collection of structuring elements.
    morph_type : str, optional
        Morphological operation:
        'opening', 'closing', 'erosion', or 'dilation'.

    Returns
    -------
    list
        Persistence diagrams for H0 and H1 features.

    Example
    -------
    >>> h0_h1_diagrams = compute_morphological_persistence(binary_img, square_kernels, morph_type='opening')
    >>> print(type(h0_h1_diagrams), len(h0_h1_diagrams))
    <class 'list'> 2
    """
    img_shape = np.shape(img)
    img_buff = np.zeros(img_shape)
    img_buff = img_buff + img

    for the_kernel in kernel_list:

        if morph_type == 'opening':
            morphed_img = opening(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'closing':
            morphed_img = closing(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'erosion':
            morphed_img = erosion(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        elif morph_type == 'dilation':
            morphed_img = dilation(
                input_np_array=img,
                input_list_of_points=the_kernel
            )

        else:
            break

        img_buff = img_buff + morphed_img


    PDs = persistence_of_img(img_buff)

    return PDs

# --------------------------------------------------------------
# UPPER / LOWER STAR FILTRATION
# --------------------------------------------------------------

def compute_upper_star(cropped):
    cropped_inv = cropped.max() - cropped
    ph_upper = cripser.compute_ph(
       cropped_inv.astype(float),
       maxdim=1
    )
    """
    Compute an upper-star filtration persistence diagram.

    The image intensity is inverted so bright structures appear earlier
    in the filtration.

    Parameters
    ----------
    cropped : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    tuple
        Persistence diagram and inverted filtration image.
    
    Example
    -------
    >>> diagram, inverted_img = compute_upper_star_persistence(cropped)
    >>> print(type(diagram), type(inverted_img))
    <class 'tuple'> <class 'numpy.ndarray'>
    """
    return ph_upper, cropped_inv

def compute_lower_star(cropped):
    ph = cripser.compute_ph(
        cropped.astype(float),
        maxdim=1
    )
    """
    Compute a lower-star filtration persistence diagram.

    Uses original image intensities so darker structures appear earlier
    in the filtration.

    Parameters
    ----------
    cropped : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    numpy.ndarray
        Persistence diagram.
        
    Example 
    -------
    >>> diagram = compute_lower_star_persistence(cropped)
    >>> print(type(diagram), diagram.ndim)
    <class 'numpy.ndarray'> 2
    """
    return ph


# --------------------------------------------------------------
# DENSITY FILTRATION
# --------------------------------------------------------------


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
    
    Example
    -------
    >>> filtration_img = compute_density_filtration(binary_img, max_dist=5.0)
    >>> print(type(filtration_img), filtration_img.shape)
    <class 'numpy.ndarray'> (3850, 4000)
    """
    height, width = binary_image.shape
    # Find foreground pixel coordinates
    points = np.argwhere(binary_image)

    # Build neighborhood tree
    tree = KDTree(points, leaf_size = 30, metric="euclidean")
   
    point_cloud = np.zeros((height*width,2))
   
    p=0
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
    density_filtration = filt_func_vals.reshape(height, width)

    return density_filtration


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
    
    Example
    -------
    >>> filtration_img, diagram = compute_density_persistence(binary_img, max_dist=5.0)
    >>> print(type(filtration_img), type(diagram))
    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    """
    density_img = density_filtration(binary_image, max_dist)

    ph_density = cripser.compute_ph(
        density_img.astype(float),
        maxdim=1
    )

    return density_img, ph_density

# -------------------------------------------------------------------
# PERSISTENCE LANDSCAPE VECTORIZATION
# -------------------------------------------------------------------

def cripser_ph_to_dgms(ph_array, maxdim=1):
    """
    Converts raw Cripser output into Persim format.

    Cripser output:
        dim, birth, death, ...

    Persim format:
        dgms[0] = H0 birth-death pairs
        dgms[1] = H1 birth-death pairs
    """

    dgms = []

    for dim in range(maxdim + 1):
        dgm = ph_array[ph_array[:, 0] == dim][:, 1:3].astype(float)
        dgms.append(dgm)

    return dgms


def filter_dgm_for_landscape(dgm, min_persistence=0.02):
    """
    Filters one persistence diagram before computing the landscape.
    """

    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))

    dgm = np.asarray(dgm, dtype=float)

    # Remove infinite or NaN values
    dgm = dgm[np.isfinite(dgm).all(axis=1)]

    if len(dgm) == 0:
        return np.empty((0, 2))

    # Keep only valid birth-death pairs
    dgm = dgm[dgm[:, 1] > dgm[:, 0]]

    if len(dgm) == 0:
        return np.empty((0, 2))

    # Compute persistence
    persistence = dgm[:, 1] - dgm[:, 0]

    # Remove low-persistence features
    dgm = dgm[persistence > min_persistence]

    return dgm


def filter_dgms_for_landscape(dgms, min_persistence=0.02):
    """
    Filters both H0 and H1 diagrams.
    """

    filtered_dgms = []

    for dim, dgm in enumerate(dgms):

        filtered = filter_dgm_for_landscape(
            dgm,
            min_persistence=min_persistence
        )

        print(f"H{dim} intervals after filtering:", len(filtered))

        filtered_dgms.append(filtered)

    return filtered_dgms


def landscape_vector(
    dgms,
    hom_deg,
    num_steps=100,
    num_layers=3
):
    """
    Converts one persistence diagram into a fixed-length landscape vector.

    Output length:
        num_layers * num_steps
    """

    if hom_deg >= len(dgms) or len(dgms[hom_deg]) == 0:
        return np.zeros(num_layers * num_steps)

    # Create approximate persistence landscape
    pl = PersLandscapeApprox(
        dgms=dgms,
        hom_deg=hom_deg,
        num_steps=num_steps
    )

    # This is the actual vectorized landscape data
    values = pl.values

    # Force a fixed number of layers
    fixed_values = np.zeros((num_layers, num_steps))

    layers_to_copy = min(num_layers, values.shape[0])
    fixed_values[:layers_to_copy, :] = values[:layers_to_copy, :]

    # Flatten into one vector
    return fixed_values.ravel()


def full_landscape_feature_vector(
    dgms,
    num_steps=100,
    num_layers=3
):
    """
    Creates one fixed-length vector using both H0 and H1 landscapes.

    H0 vector length:
        num_layers * num_steps

    H1 vector length:
        num_layers * num_steps

    Total:
        2 * num_layers * num_steps
    """

    h0_vec = landscape_vector(
        dgms,
        hom_deg=0,
        num_steps=num_steps,
        num_layers=num_layers
    )

    h1_vec = landscape_vector(
        dgms,
        hom_deg=1,
        num_steps=num_steps,
        num_layers=num_layers
    )

    return np.concatenate([h0_vec, h1_vec])


def persistence_landscape_from_ph(
    ph_array,
    min_persistence=0.02,
    num_steps=100,
    num_layers=3
):
    """
    Converts raw Cripser output directly into a
    persistence landscape feature vector.
    """

    dgms = cripser_ph_to_dgms(
        ph_array,
        maxdim=1
    )

    dgms_filtered = filter_dgms_for_landscape(
        dgms,
        min_persistence=min_persistence
    )

    feature_vec = full_landscape_feature_vector(
        dgms_filtered,
        num_steps=num_steps,
        num_layers=num_layers
    )

    return feature_vec

# -------------------------------------------------------------------
# IMAGE TO LANDSCAPE VECTOR
# -------------------------------------------------------------------

def image_to_landscape_vector(
    image_path,
    min_persistence=0.02,
    num_steps=100,
    num_layers=3
):
    """
    Convert one microscopy image into a persistence
    landscape feature vector.

    This initial classifier uses lower-star filtration.
    """

    # Preprocess image
    img = preprocess_single(
        image_path
    )

    # Compute lower-star persistent homology
    ph_lower = compute_lower_star(
        img
    )

    # Convert persistence into landscape vector
    feature_vec = (
        persistence_landscape_from_ph(
            ph_lower,
            min_persistence=min_persistence,
            num_steps=num_steps,
            num_layers=num_layers
        )
    )

    return feature_vec

# -------------------------------------------------------------------
# BUILD MACHINE LEARNING DATASET
# -------------------------------------------------------------------

def build_landscape_dataset(
    image_dir,
    min_persistence=0.02,
    num_steps=100,
    num_layers=3
):
    """
    Convert Control and Microgravity microscopy images
    into a machine learning dataset.

    Labels
    ------
    0 = Control
    1 = Microgravity

    Returns
    -------
    X : numpy.ndarray
        Feature matrix.

    y : numpy.ndarray
        Class labels.

    image_names : list
        Image names corresponding to rows of X.
    """

    image_dir = Path(
        image_dir
    )

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{image_dir}"
        )

    X = []
    y = []
    image_names = []

    valid_extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    }

    # Search folder and subfolders for images
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in valid_extensions
            and "_processed" not in path.stem.lower()
        )
    )

    print(
        "Images found:",
        len(image_paths)
    )

    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        # Use folder and filename to identify class
        path_text = str(
            image_path
        ).lower()

        if "control" in path_text:

            label = 0

        elif "microgravity" in path_text:

            label = 1

        else:

            print(
                f"\nSkipping {image_path.name}: "
                f"class could not be determined."
            )

            continue

        print(
            "\n==================================="
        )

        print(
            f"Image {image_number} "
            f"of {len(image_paths)}"
        )

        print(
            "Processing:",
            image_path.name
        )

        if label == 0:

            print(
                "Class: Control"
            )

        else:

            print(
                "Class: Microgravity"
            )

        try:

            feature_vec = (
                image_to_landscape_vector(
                    image_path,
                    min_persistence=min_persistence,
                    num_steps=num_steps,
                    num_layers=num_layers
                )
            )

            X.append(
                feature_vec
            )

            y.append(
                label
            )

            image_names.append(
                image_path.name
            )

            print(
                "Landscape vector shape:",
                feature_vec.shape
            )

        except Exception as error:

            print(
                f"Could not process "
                f"{image_path.name}"
            )

            print(
                "Error:",
                error
            )

    if len(X) == 0:

        raise ValueError(
            "No images were successfully "
            "converted into feature vectors."
        )

    # Stack landscape vectors into feature matrix
    X = np.vstack(
        X
    )

    # Convert labels into NumPy array
    y = np.asarray(
        y
    )

    return (
        X,
        y,
        image_names
    )

# -------------------------------------------------------------------
# K-NEAREST NEIGHBORS CLASSIFICATION
# -------------------------------------------------------------------

def train_knn_classifier(
    X,
    y,
    image_names,
    n_neighbors=5,
    test_size=0.2,
    random_state=42
):
    """
    Train and evaluate a K-Nearest Neighbors classifier.

    Parameters
    ----------
    X : numpy.ndarray
        Persistence landscape feature matrix.

    y : numpy.ndarray
        Class labels.

    image_names : list
        Image names corresponding to X.

    n_neighbors : int, optional
        Number of nearest neighbors.

    test_size : float, optional
        Proportion of images used for testing.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    KNeighborsClassifier
        Trained KNN classifier.
    """

    # Count images in each class
    classes, class_counts = np.unique(
        y,
        return_counts=True
    )

    print(
        "\nCLASS COUNTS"
    )

    print(
        "-----------------------------------"
    )

    for class_label, count in zip(
        classes,
        class_counts
    ):

        if class_label == 0:

            class_name = "Control"

        else:

            class_name = "Microgravity"

        print(
            f"{class_name}: {count}"
        )

    # Make sure both classes exist
    if len(classes) < 2:

        raise ValueError(
            "KNN classification requires both "
            "Control and Microgravity images."
        )

    # Make sure stratified split is possible
    if class_counts.min() < 2:

        raise ValueError(
            "Each class needs at least two images "
            "for a stratified train/test split."
        )

    # Split feature vectors into training and testing data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        names_train,
        names_test
    ) = train_test_split(
        X,
        y,
        image_names,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(
        "\nTRAIN / TEST INFORMATION"
    )

    print(
        "-----------------------------------"
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

    # Make sure k is not larger than training set
    if n_neighbors > X_train.shape[0]:

        raise ValueError(
            f"n_neighbors={n_neighbors} is larger "
            f"than the number of training images "
            f"({X_train.shape[0]})."
        )

    # Create KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    # Train KNN
    knn.fit(
        X_train,
        y_train
    )

    # Predict test image labels
    y_pred = knn.predict(
        X_test
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Calculate weighted F1 score
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(
        "\nKNN RESULTS"
    )

    print(
        "-----------------------------------"
    )

    print(
        "Number of neighbors:",
        n_neighbors
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Weighted F1 Score:",
        f1
    )

    label_names = {
        0: "Control",
        1: "Microgravity"
    }

    print(
        "\nINDIVIDUAL TEST PREDICTIONS"
    )

    print(
        "-----------------------------------"
    )

    for (
        image_name,
        true_label,
        predicted_label
    ) in zip(
        names_test,
        y_test,
        y_pred
    ):

        print(
            "\nImage:",
            image_name
        )

        print(
            "Actual:",
            label_names[true_label]
        )

        print(
            "Predicted:",
            label_names[predicted_label]
        )

        if true_label == predicted_label:

            print(
                "Result: Correct"
            )

        else:

            print(
                "Result: Incorrect"
            )

    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[
            0,
            1
        ],
        display_labels=[
            "Control",
            "Microgravity"
        ]
    )

    plt.title(
        "KNN Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    return knn

def plot_pca_components(
    pca_output,
    pca,
    x_component=1,
    y_component=2
):
    """
    Plot any two of the first five PCA components.

    Parameters
    ----------
    pca_output : pandas.DataFrame
        Table containing image PCA coordinates.

    pca : sklearn.decomposition.PCA
        Fitted PCA model.

    x_component : int
        PCA component used for the x-axis.

    y_component : int
        PCA component used for the y-axis.
    """

    # Validate requested components
    if x_component < 1 or x_component > 5:
        raise ValueError(
            "x_component must be between 1 and 5."
        )

    if y_component < 1 or y_component > 5:
        raise ValueError(
            "y_component must be between 1 and 5."
        )

    if x_component == y_component:
        raise ValueError(
            "Choose two different PCA components."
        )


    x_column = f"PC{x_component}"

    y_column = f"PC{y_component}"


    colors = {
        "Control": "#1f77b4",
        "Microgravity": "#d62728"
    }


    markers = {
        "Control": "o",
        "Microgravity": "s"
    }


    plt.figure(
        figsize=(11, 8)
    )


    # Plot by true and predicted label
    for true_label in [
        "Control",
        "Microgravity"
    ]:

        for predicted_label in [
            "Control",
            "Microgravity"
        ]:

            mask = (
                (
                    pca_output["True_Label"]
                    == true_label
                )
                &
                (
                    pca_output["Predicted_Label"]
                    == predicted_label
                )
            )


            if mask.sum() > 0:

                label = (
                    f"True {true_label} | "
                    f"Predicted {predicted_label}"
                )


                plt.scatter(
                    pca_output.loc[
                        mask,
                        x_column
                    ],

                    pca_output.loc[
                        mask,
                        y_column
                    ],

                    c=colors[
                        predicted_label
                    ],

                    marker=markers[
                        true_label
                    ],

                    s=100,

                    edgecolors="black",

                    alpha=0.8,

                    label=label
                )


    x_variance = (
        pca.explained_variance_ratio_[
            x_component - 1
        ]
        * 100
    )


    y_variance = (
        pca.explained_variance_ratio_[
            y_component - 1
        ]
        * 100
    )


    plt.title(
        f"{x_column} vs {y_column} of "
        f"Persistence Landscape Vectors"
    )


    plt.xlabel(
        f"{x_column} "
        f"({x_variance:.1f}%)"
    )


    plt.ylabel(
        f"{y_column} "
        f"({y_variance:.1f}%)"
    )


    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )


    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    plt.tight_layout()

    plt.show()

# -------------------------------------------------------------------
# RUN LOWER-STAR + LANDSCAPE + KNN PIPELINE
# -------------------------------------------------------------------

if __name__ == "__main__":

    # Folder containing Control and Microgravity images
    image_dir = Path(
        r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Images"
    )

    print(
        "BUILDING PERSISTENCE LANDSCAPE DATASET"
    )

    print(
        "==================================="
    )

    # Convert every image into a persistence landscape vector
    X, y, image_names = build_landscape_dataset(
        image_dir,
        min_persistence=0.005,
        num_steps=100,
        num_layers=3
    )

    print(
        "\nDATASET INFORMATION"
    )

    print(
        "==================================="
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "Label vector shape:",
        y.shape
    )

    print(
        "Number of Control images:",
        np.sum(y == 0)
    )

    print(
        "Number of Microgravity images:",
        np.sum(y == 1)
    )

    print(
        "\nTRAINING KNN CLASSIFIER"
    )

    print(
        "==================================="
    )

    # Train and evaluate KNN
    knn = train_knn_classifier(
        X,
        y,
        image_names,
        n_neighbors=5,
        test_size=0.2,
        random_state=42
    )

    # -------------------------------------------------------------------
# PCA VISUALIZATION SPACE
# -------------------------------------------------------------------

    print(
    "\nGENERATING PCA VISUALIZATION SPACE"
    )

    print(
    "==================================="
    )

    # Scale persistence landscape features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
        )

    # Compute the first five principal components
    pca = PCA(
        n_components=5
        )

    X_pca = pca.fit_transform(
        X_scaled
        )

    # Get KNN predictions for all images
    y_pred = knn.predict(
        X
        )

    # Convert numeric labels to class names
    label_names = {
        0: "Control",
        1: "Microgravity"
        }


    # Create PCA output table
    pca_output = pd.DataFrame({
        "Image": image_names,

        "True_Label": [
            label_names[label]
            for label in y
            ],
        
        "Predicted_Label": [
            label_names[label]
            for label in y_pred
            ],

        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2],
        "PC4": X_pca[:, 3],
        "PC5": X_pca[:, 4]
    })


    # Save PCA coordinates
    pca_output.to_csv(
        "PCA_First_Five_Components.csv",
        index=False
    )
    
    # Save explained variance for each PCA component
    variance_output = pd.DataFrame({
        "Component": [
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5"
        ],

        "Explained_Variance": (
            pca.explained_variance_ratio_
            * 100
        )
    })


    variance_output.to_csv(
        "PCA_Explained_Variance.csv",
        index=False
    )
    
    print(
        "\nPCA OUTPUT SAVED"
    )
    
    print(
        "PCA_First_Five_Components.csv"
    )
    
    
    # Print explained variance
    print(
        "\nPCA EXPLAINED VARIANCE"
    )
    
    print(
        "==================================="
    )
    
    for component_number, variance in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):
    
        print(
            f"PC{component_number}: "
            f"{variance * 100:.2f}%"
        )
    
    
    print(
        "\nTotal variance explained by "
        "first five components:"
    )
    
    print(
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    # Plot selected PCA components
    plot_pca_components(
        pca_output,
        pca,
        x_component=1,
        y_component=2
    )
    
# Rerun, save first five components so that no matter which comgination we want to see, i can just pull up the output and plot those too