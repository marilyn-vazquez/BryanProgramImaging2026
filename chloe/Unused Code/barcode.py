# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:58:13 2026

@author: chloe
"""


import os
import cv2
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

# Silence OpenCV metadata logging warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

# Function to compute persistent statistics features safely
def get_barcode_stats(intervals):
    """Computes a standardized vector of 10 statistical features from persistence intervals.

    This function takes raw birth-death pairs from a topological dimension and
    extracts summary metrics detailing their distributions and absolute lifetimes. 
    It features rigorous safety checks to gracefully handle empty persistence diagrams 
    without throwing exceptions.

    Args:
        intervals (numpy.ndarray): An Nx2 array where each row represents a 
            persistent feature's [birth_time, death_time].

    Returns:
        numpy.ndarray: A 1D array of length 10 containing the following metrics:
            [0] Mean Birth Time          [5] Median Death Time
            [1] Mean Death Time          [6] Mean Bar Length
            [2] Std Dev of Birth Times   [7] Std Dev of Bar Lengths
            [3] Std Dev of Death Times   [8] Median Bar Length
            [4] Median Birth Time        [9] Total Bar Count (Feature Count)
    """
    
    # Safety Check: If intervals is empty, None, or un-subscriptable
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
        # Fallback if any unexpected math error occurs
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    return bar_stats

# Function to safely compute cubical persistence diagrams and plot barcodes
def get_cubical_features(img, title_name):
    """Computes cubical persistence diagrams and generates barcode visualizations.

    Constructs a Gudhi Cubical Complex directly from a 2D digital image using pixel 
    intensities as top-dimensional filtration values. It extracts finite persistence
    intervals for 0D and 1D features and renders the collective persistence barcode.

    Args:
        img (numpy.ndarray): A 2D array representing the grayscale input image.
        title_name (str): The name of the file or image to display on the plot title.

    Returns:
        tuple: Contains two elements:
            - dim0 (numpy.ndarray): Nx2 array of finite persistent features in Dimension 0.
            - dim1 (numpy.ndarray): Mx2 array of finite persistent features in Dimension 1.
    """
    cubical_complex = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
    persistence_matrix = cubical_complex.compute_persistence()
    
    # 1. Safely handle Dimension 0
    dim0 = cubical_complex.persistence_intervals_in_dimension(0)
    if dim0 is not None and len(dim0) > 0:
        dim0 = np.array(dim0)
        dim0 = dim0[np.isfinite(dim0[:, 1])] if len(dim0.shape) == 2 else np.empty((0, 2))
    else:
        dim0 = np.empty((0, 2))
    
    # 2. Safely handle Dimension 1
    dim1 = cubical_complex.persistence_intervals_in_dimension(1)
    if dim1 is not None and len(dim1) > 0:
        dim1 = np.array(dim1)
        dim1 = dim1[np.isfinite(dim1[:, 1])] if len(dim1.shape) == 2 else np.empty((0, 2))
    else:
        dim1 = np.empty((0, 2))
    
    # 3. Generate and display the Barcode Plot safely
    try:
        if len(persistence_matrix) > 0:
            plt.figure(figsize=(10, 4))
            gd.plot_persistence_barcode(persistence_matrix)
            plt.title(f"Persistence Barcode - {title_name}")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"   ⚠️ Could not generate plot for {title_name}: {e}")
    
    return dim0, dim1

def process_images_cubical(image_paths):
    """Executes the full TDA extraction loop across a collection of file paths.

    Iterates through a list of system file paths, handles grayscale disk reading via 
    OpenCV, and coordinates calls to the feature extraction and statistical summary 
    generators.

    Args:
        image_paths (list of str): A list containing absolute or relative system 
            strings pointing to target .tif or image files.

    Returns:
        list of list: A collection of records containing structural information for 
            each successfully read file, formatted as:
            `[[base_name_string, dim0_stats_array, dim1_stats_array], ...]`
    """
    results = []
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        print(f"Processing & Plotting: {base_name}")
        img = cv2.imread(img_path, 0)  # Read as grayscale
        
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue
            
        dim0_barcode, dim1_barcode = get_cubical_features(img, base_name)
        
        dim0_stats = get_barcode_stats(dim0_barcode)
        dim1_stats = get_barcode_stats(dim1_barcode)
        
        results.append([base_name, dim0_stats, dim1_stats])
    return results


if __name__ == '__main__':
    # Define your 4 specific image paths
    images = [
        r"C:\Users\chloe\.spyder-py3\microgravity_stub1_3-4-26_0001.tif",
        r"C:\Users\chloe\.spyder-py3\microgravity_stub1_3-4-26_0002.tif",
        r"C:\Users\chloe\.spyder-py3\control_stub1_3-2-26_0008.tif",
        r"C:\Users\chloe\.spyder-py3\Control_Stub4_0000.tif"
    ]
    
    print("Starting topological feature extraction and plotting...")
    results = process_images_cubical(images)
    
    # Save the binary feature arrays to a clean folder destination
    outputFolder = 'CubicalFeatures_Visualized'
    os.makedirs(outputFolder, exist_ok=True)
    
    for res in results:
        file_name, d0, d1 = res
        base_name = os.path.splitext(file_name)[0]
        
        np.save(f'{outputFolder}/{base_name}_dim0_stats.npy', d0)
        np.save(f'{outputFolder}/{base_name}_dim1_stats.npy', d1)
        
    print(f"\nExtraction complete! Binary results saved in '{outputFolder}/'\n")

    # Read and print text summary tables to the terminal window
    stat_names = [
        "1. Average Birth Time",
        "2. Average Death Time",
        "3. Standard Deviation of Birth Times",
        "4. Standard Deviation of Death Times",
        "5. Median of Birth Times",
        "6. Median of Death Times",
        "7. Average Length of Bars",
        "8. Standard Deviation of Bar Lengths",
        "9. Median of Bar Lengths",
        "10. Total Number of Bars (Count)"
    ]

    print("=" * 60)
    print("       SUMMARY STATISTICS FOR EXTRACTED IMAGES")
    print("=" * 60)

    if os.path.exists(outputFolder):
        files = [f for f in os.listdir(outputFolder) if f.endswith('.npy')]
        for file_name in sorted(files):
            full_path = os.path.join(outputFolder, file_name)
            data = np.load(full_path)
            
            print(f"\n📄 FILE: {file_name}")
            print("-" * 50)
            
            for name, value in zip(stat_names, data):
                if "Count" in name:
                    print(f"{name:<40} : {int(value)}")
                else:
                    print(f"{name:<40} : {value:.4f}")
            print("-" * 50)

    print("\n" + "=" * 60)