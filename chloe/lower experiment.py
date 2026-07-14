import os
from pathlib import Path
import numpy as np
import cripser as cr
from skimage import io
from skimage.util import img_as_float

# =====================================================================
# 1. CRIPSER LOWER-STAR PERSISTENT HOMOLOGY
# =====================================================================

def compute_lower_star_ph(images_paths, output_dir):
    """
    Computes raw lower-star persistent homology via Cripser for a list of images
    and saves each raw persistence diagram matrix as a separate .npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for path in images_paths:
        path = Path(path)
        print(f"Computing Lower-Star PH for: {path.name}")
        
        # Load the preprocessed image
        img = img_as_float(io.imread(path, as_gray=True))
        
        # Scale to standard 0-255 domain for numerical stability in Cripser
        if img.max() <= 1.0:
            img = img * 255.0
            
        img_input = np.asarray(img, dtype=np.float64)
        
        # Compute Persistent Homology (Lower-star cubical filtration)
        ph_diagram = cr.computePH(img_input) if hasattr(cr, "computePH") else cr.compute_ph(img_input)
        
        # Save the raw persistence diagram matrix [dim, birth, death] for this specific image
        save_path = output_dir / f"{path.stem}_lower_star_diagram.npy"
        np.save(save_path, ph_diagram)
        
    print(f"\n✅ All lower-star persistence diagrams saved to: {output_dir}")

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

    # Execute the isolated lower-star TDA computation
    print("\n=== Starting Isolated Lower-Star Homology Extraction ===")
    compute_lower_star_ph(images_paths=processed_paths, output_dir=PROCESSED_DIR)