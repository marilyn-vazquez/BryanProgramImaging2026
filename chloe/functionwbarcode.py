# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:58:13 2026

@author: chloe
"""

import os
from pathlib import Path
import cv2
import numpy as np
import numba as nb
import cripser
import gudhi as gd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage import io, exposure, filters
from skimage.util import img_as_float

# Silence OpenCV metadata logging warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

# -------------------------------------------------------------------
# PRE-PROCESSING
# -------------------------------------------------------------------

def preprocess_single(image_input, sigma=0.5, clip_limit=0.015):
    """Preprocess a single microscopy image."""
    if isinstance(image_input, (str, Path)):
        img = img_as_float(io.imread(image_input, as_gray=True))
    else:
        img = img_as_float(image_input)
        
    img = img[:3850, :]
    img_smoothed = filters.gaussian(img, sigma=sigma)
    img_clahe = exposure.equalize_adapthist(img_smoothed, kernel_size=256, clip_limit=clip_limit)
    return img_clahe


def process_batch(image_paths, reference_path, out_dir, sigma=0.5, clip_limit=0.015):
    """Apply preprocessing to a collection of microscopy images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing reference image...")
    ref_ready = preprocess_single(reference_path, sigma=sigma, clip_limit=clip_limit)
   
    processed_files_count = 0
   
    for path in image_paths:
        path = Path(path)
        print(f"Processing Image Adjustment: {path.name}")
       
        if path == Path(reference_path):
            img_ready = ref_ready
        else:
            img_clahe = preprocess_single(path, sigma=sigma, clip_limit=clip_limit)
            img_ready = exposure.match_histograms(img_clahe, ref_ready)
                       
        save_path = out_dir / f"{path.stem}_processed.tif"
        io.imsave(save_path, img_ready, check_contrast=False)
        processed_files_count += 1
       
    return processed_files_count

# -------------------------------------------------------------------
# MORPHOLOGY
# -------------------------------------------------------------------

def find(condition):
    res = np.nonzero(condition)
    return res


def biImg_by_threshold_leq(img, threshold):
    output_img = np.copy(img)
    idxs_0 = find(img <= threshold)
    idxs_1 = find(img > threshold)
    output_img[idxs_0] = 0
    output_img[idxs_1] = 1
    return output_img


def biImg_by_threshold_geq(img, threshold):
    output_img = np.copy(img)
    idxs_0 = find(img >= threshold)
    idxs_1 = find(img < threshold)
    output_img[idxs_0] = 0
    output_img[idxs_1] = 1
    return output_img

   
@nb.jit(nopython=True)
def erosion(input_np_array, input_list_of_points, minimal_pixel_value=0):
    input_np_array_shape = input_np_array.shape
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


@nb.jit(nopython=True)
def dilation(input_np_array, input_list_of_points, maximal_pixel_value=1):
    input_np_array_shape = input_np_array.shape
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


def closing(input_np_array, input_list_of_points):
    return erosion(dilation(input_np_array, input_list_of_points), input_list_of_points)


@nb.jit()
def get_rectangle_coordinates(input_np_array):
    input_np_array_shape = np.shape(input_np_array)
    output_list = []
    origin_i = int(input_np_array_shape[0] / 2)
    origin_j = int(input_np_array_shape[1] / 2)
    for i in range(input_np_array_shape[0]):
        for j in range(input_np_array_shape[1]):
            output_list.append(np.array([origin_j - j, origin_i - i]))
    return output_list


def get_horizontal_SE_list(maximal_SE_lengths):
    kernel_list = []
    for i in range(2, maximal_SE_lengths + 1):
        buffer_list = [np.array([0, 0])]
        counter, left_counter, right_counter = 1, -1, 1
        while len(buffer_list) < i:
            if counter % 2 == 1:
                buffer_list.append(np.array([right_counter, 0]))
                right_counter += 1
            else:
                buffer_list.append(np.array([left_counter, 0]))
                left_counter -= 1
            counter += 1
        kernel_list.append(buffer_list)
    return kernel_list


def get_vertical_SE_list(maximal_SE_lengths):
    kernel_list = []
    for i in range(2, maximal_SE_lengths + 1):
        buffer_list = [np.array([0, 0])]
        counter, left_counter, right_counter = 1, -1, 1
        while len(buffer_list) < i:
            if counter % 2 == 1:
                buffer_list.append(np.array([0, right_counter]))
                right_counter += 1
            else:
                buffer_list.append(np.array([0, left_counter]))
                left_counter -= 1
            counter += 1
        kernel_list.append(buffer_list)
    return kernel_list


def get_square_SE_list(maximal_SE_lengths):
    kernel_list = []
    for i in range(2, maximal_SE_lengths + 1):
        kernel_list.append(get_rectangle_coordinates(input_np_array=np.zeros((i, i))))
    return kernel_list


def img_to_1d_array(img):
    result = []
    img_shape = np.shape(img)
    for i in range(img_shape[0]):
        result = list(img[i, :]) + result
    return np.array(result)


def persistence_of_img(img, maxdim=1):
    img = np.asarray(img, dtype=np.float64)
    if hasattr(cripser, "compute_ph"):
        ph = cripser.compute_ph(img, maxdim=maxdim)
    else:
        ph = cripser.computePH(img, maxdim=maxdim)
    persistence_0 = ph[ph[:, 0] == 0][:, 1:3]
    persistence_1 = ph[ph[:, 0] == 1][:, 1:3]
    return [persistence_0, persistence_1]


def persistence_of_morph_filtration(img, kernel_list, morph_type='closing'):
    img_shape = np.shape(img)
    img_buff = np.zeros(img_shape) + img

    for the_kernel in kernel_list:
        if morph_type == 'closing':
            morphed_img = closing(input_np_array=img, input_list_of_points=the_kernel)
        elif morph_type == 'erosion':
            morphed_img = erosion(input_np_array=img, input_list_of_points=the_kernel)
        elif morph_type == 'dilation':
            morphed_img = dilation(input_np_array=img, input_list_of_points=the_kernel)
        else:
            break
        img_buff = img_buff + morphed_img

    return persistence_of_img(img_buff)

# --------------------------------------------------------------
# UPPER / LOWER STAR
# --------------------------------------------------------------

def compute_upper_star(cropped):
    cropped_inv = cropped.max() - cropped
    ph_upper = cripser.compute_ph(cropped_inv.astype(float), maxdim=1)
    return ph_upper, cropped_inv


def compute_lower_star(cropped):
    ph = cripser.compute_ph(cropped.astype(float), maxdim=1)
    return ph

# --------------------------------------------------------------
# DENSITY FILTRATION
# --------------------------------------------------------------

def density_filtration(binary_image, max_dist=5):
    height, width = binary_image.shape
    points = np.argwhere(binary_image)
    tree = KDTree(points, leaf_size=30, metric="euclidean")
   
    point_cloud = np.zeros((height*width, 2))
    p = 0
    for i in range(height):
        for j in range(width):
            point_cloud[p, 0] = i
            point_cloud[p, 1] = j
            p += 1

    num_nbhs = tree.query_radius(point_cloud, r=max_dist, count_only=True)
    filt_func_vals = num_nbhs
    max_num_nbhs = filt_func_vals.max()
    filt_func_vals = max_num_nbhs - filt_func_vals
    return filt_func_vals.reshape(height, width)


def compute_density_ph(binary_image, max_dist=5):
    density_img = density_filtration(binary_image, max_dist)
    ph_density = cripser.compute_ph(density_img.astype(float), maxdim=1)
    return density_img, ph_density

# -------------------------------------------------------------------
# VECTORIZATION HELPER
# -------------------------------------------------------------------

def get_barcode_stats(intervals):
    """Computes a standardized vector of 10 statistical features from persistence intervals."""
    if intervals is None or len(intervals) == 0 or not isinstance(intervals, np.ndarray) or len(intervals.shape) < 2:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    try:
        bc_av0, bc_av1 = np.mean(intervals, axis=0)
        bc_std0, bc_std1 = np.std(intervals, axis=0)
        bc_med0, bc_med1 = np.median(intervals, axis=0)
        
        diff_barcode = np.abs(intervals[:, 1] - intervals[:, 0])
        bc_lengthAverage = np.mean(diff_barcode)
        bc_lengthSTD = np.std(diff_barcode)
        bc_lengthMedian = np.median(diff_barcode)
        bc_count = len(diff_barcode)

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                              bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count])
    except Exception:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    return bar_stats


def extract_cripser_dims(ph_array):
    """Helper to separate a raw Cripser output array into distinct dim0 and dim1 arrays."""
    if ph_array is None or len(ph_array) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    dim0 = ph_array[ph_array[:, 0] == 0][:, 1:3]
    dim1 = ph_array[ph_array[:, 0] == 1][:, 1:3]
    return dim0, dim1

# -------------------------------------------------------------------
# MASTER EXECUTION PIPELINE
# -------------------------------------------------------------------

if __name__ == '__main__':
    images = [
        r"C:\Users\chloe\.spyder-py3\microgravity_stub1_3-4-26_0001.tif",
        r"C:\Users\chloe\.spyder-py3\microgravity_stub1_3-4-26_0002.tif",
        r"C:\Users\chloe\.spyder-py3\control_stub1_3-2-26_0008.tif",
        r"C:\Users\chloe\.spyder-py3\Control_Stub4_0000.tif"
    ]
    
    preprocessed_folder = 'Preprocessed_Images'
    outputFolder = 'All_Filtrations_Vectorized'
    os.makedirs(outputFolder, exist_ok=True)
    
    # Step 1: Preprocessing & Equalization
    print("Executing global image transformations...")
    process_batch(image_paths=images, reference_path=images[2], out_dir=preprocessed_folder)
    
    processed_images = [Path(preprocessed_folder) / f"{Path(p).stem}_processed.tif" for p in images]
    square_kernels = get_square_SE_list(maximal_SE_lengths=5)
    
    # Step 2: Loop through every image and evaluate every filtration engine
    print("\nStarting multi-filtration processing loop...")
    for img_path in processed_images:
        base_name = img_path.stem
        print(f"\n🔄 Extracting all filtrations for: {base_name}")
        
        img = cv2.imread(str(img_path), 0)
        if img is None:
            print(f"   ❌ Error loading {img_path.name}")
            continue
            
        # Standard dynamic global threshold for Morphology/Density
        binary_img = biImg_by_threshold_leq(img, threshold=127)
        
        # --- FILTRATION 1: GUDHI Cubical Complex ---
        cubical_complex = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
        cubical_complex.compute_persistence()
        g_d0 = np.array(cubical_complex.persistence_intervals_in_dimension(0))
        g_d1 = np.array(cubical_complex.persistence_intervals_in_dimension(1))
        # Strip infinity before stats
        g_d0 = g_d0[np.isfinite(g_d0[:, 1])] if len(g_d0) > 0 else np.empty((0,2))
        g_d1 = g_d1[np.isfinite(g_d1[:, 1])] if len(g_d1) > 0 else np.empty((0,2))
        
        np.save(f'{outputFolder}/{base_name}_cubical_dim0.npy', get_barcode_stats(g_d0))
        np.save(f'{outputFolder}/{base_name}_cubical_dim1.npy', get_barcode_stats(g_d1))
        
        # --- FILTRATION 2: Lower Star (Cripser) ---
        ph_lower = compute_lower_star(img)
        l_d0, l_d1 = extract_cripser_dims(ph_lower)
        np.save(f'{outputFolder}/{base_name}_lowerstar_dim0.npy', get_barcode_stats(l_d0))
        np.save(f'{outputFolder}/{base_name}_lowerstar_dim1.npy', get_barcode_stats(l_d1))
        
        # --- FILTRATION 3: Upper Star (Cripser) ---
        ph_upper, _ = compute_upper_star(img)
        u_d0, u_d1 = extract_cripser_dims(ph_upper)
        np.save(f'{outputFolder}/{base_name}_upperstar_dim0.npy', get_barcode_stats(u_d0))
        np.save(f'{outputFolder}/{base_name}_upperstar_dim1.npy', get_barcode_stats(u_d1))

        # --- FILTRATION 4: Density Filtration ---
        _, ph_density = compute_density_ph(binary_img, max_dist=5)
        dns_d0, dns_d1 = extract_cripser_dims(ph_density)
        np.save(f'{outputFolder}/{base_name}_density_dim0.npy', get_barcode_stats(dns_d0))
        np.save(f'{outputFolder}/{base_name}_density_dim1.npy', get_barcode_stats(dns_d1))

        # --- FILTRATIONS 5-7: Morphological Pipelines (Closing, Erosion, Dilation) ---
        for morph_mode in ['closing', 'erosion', 'dilation']:
            m_d0, m_d1 = persistence_of_morph_filtration(binary_img, square_kernels, morph_type=morph_mode)
            np.save(f'{outputFolder}/{base_name}_morph_{morph_mode}_dim0.npy', get_barcode_stats(m_d0))
            np.save(f'{outputFolder}/{base_name}_morph_{morph_mode}_dim1.npy', get_barcode_stats(m_d1))
            
        print(f"   ✅ Saved all feature vectors successfully.")

    print(f"\nPipeline finished! View your statistical collections inside: '{outputFolder}/'")