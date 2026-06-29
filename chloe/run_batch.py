# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:04:57 2026

@author: chloe
"""

from pathlib import Path
import functions as fn

def main():
    input_dir = Path(r"C:\Users\chloe\Images\control_stub1_3-2-26_0008.tif")
    output_dir = Path(r"C:\Users\chloe\Images\control_stub1_3-2-26_0021.tif")
    
    # Common image extensions to look for
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    
    # -------------------------------------------------------------------
    # DIRECTORY SETUP
    # -------------------------------------------------------------------
    if not input_dir.exists():
        print(f"Creating input directory at: {input_dir.resolve()}")
        input_dir.mkdir(parents=True, exist_ok=True)
        print("Please drop your images into the 'input_images' folder and rerun the script.")
        return

    # Gather all valid image paths from the input directory
    all_images = [
        p for p in input_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if not all_images:
        print(f"No valid images found in {input_dir.resolve()}")
        return

    reference_image = all_images[0]
    print(f"Using reference image for histogram matching: {reference_image.name}\n")

    # -------------------------------------------------------------------
    # PHASE 1: PRE-PROCESSING & HISTOGRAM MATCHING
    # -------------------------------------------------------------------
    print("--- Starting Batch Pre-Processing ---")
    processed_count = fn.process_batch(
        image_paths=all_images,
        reference_path=reference_image,
        out_dir=output_dir,
        sigma=0.5,
        clip_limit=0.015
    )
    print(f"Successfully pre-processed {processed_count} images.\n")

    # -------------------------------------------------------------------
    # PHASE 2: ADVANCED FILTRATIONS & TOPOLOGICAL DATA ANALYSIS (TDA)
    # -------------------------------------------------------------------
    print("--- Starting Morphology, Star, and Density Filtrations ---")
    
    # Grab the high-precision tiff images we just outputted
    processed_images = list(output_dir.glob("*_processed.tif"))
    
    # Generate structured element (SE) kernels for morphological filtrations
    square_kernels = fn.get_square_SE_list(maximal_SE_lengths=5)

    for img_path in processed_images:
        print(f"\nAnalyzing Topological Features for: {img_path.name}")
        
        # 1. Load the pre-processed image
        from skimage import io
        img = io.imread(img_path, as_gray=True)
        
        # 2. Upper and Lower Star Filtrations
        print(" -> Computing Star Filtrations...")
        ph_lower = fn.compute_lower_star(img)
        ph_upper, img_inv = fn.compute_upper_star(img)
        
       # 3. Morphological Filtrations (All Variants)
        print(" -> Computing Morphological Filtrations...")
        
        # Erosion
        pd_erosion = fn.persistence_of_morph_filtration(
            img=img, kernel_list=square_kernels, morph_type='erosion'
        )
        # Dilation
        pd_dilation = fn.persistence_of_morph_filtration(
            img=img, kernel_list=square_kernels, morph_type='dilation'
        )
        # Closing
        pd_closing = fn.persistence_of_morph_filtration(
            img=img, kernel_list=square_kernels, morph_type='closing'
        )
        
        # 4. Density Filtration (Requires a binary representation)
        print(" -> Computing Density Filtration...")
        # Generates a quick thresholded binary image for the density tree evaluation
        binary_img = fn.biImg_by_threshold_leq(img, threshold=0.5) 
        density_img, ph_density = fn.compute_density_ph(binary_img, max_dist=5)
        
        # Example tracking summary for checking array outputs
        print(f"    ✓ Lower Star PH Shape:  {ph_lower.shape}")
        print(f"    ✓ Upper Star PH Shape:  {ph_upper.shape}")
        print(f"    ✓ Erosion PDs Count:    {len(pd_erosion)}")
        print(f"    ✓ Dilation PDs Count:   {len(pd_dilation)}")
        print(f"    ✓ Closing PDs Count:    {len(pd_closing)}")
        print(f"    ✓ Density PH Shape:     {ph_density.shape}")

    print("\nAll batch operations completed successfully!")

if __name__ == "__main__":
    main()
