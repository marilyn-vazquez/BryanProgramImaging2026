from pathlib import Path
from skimage import io, exposure, filters
from skimage.util import img_as_float

# -------------------------------------------------------------------
# 1. PRE-PROCESSING FUNCTIONS
# -------------------------------------------------------------------

def preprocess_single(image_input, sigma=0.5, clip_limit=0.015):
    """
    Preprocess a single microscopy image: convert to float, crop
    the info bar, smooth with Gaussian blur, and apply CLAHE.
    """
    # Load image as grayscale float between 0 and 1
    if isinstance(image_input, (str, Path)):
        img = img_as_float(io.imread(image_input, as_gray=True))
    else:
        img = img_as_float(image_input)
       
    # Crop the microscope information bar (adjust coordinate if needed)
    img_cropped = img[:-300, :]
   
    # Apply gaussian blur to reduce noise
    img_smoothed = filters.gaussian(img_cropped, sigma=sigma)
   
    # Apply CLAHE to enhance local contrast
    img_clahe = exposure.equalize_adapthist(img_smoothed, kernel_size=256, clip_limit=clip_limit)
   
    return img_clahe


def process_batch(image_paths, reference_path, out_dir, sigma=0.5, clip_limit=0.015):
    """
    Apply preprocessing to a collection of images and match their histograms
    to a reference image for uniform contrast across the batch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing baseline reference image...")
    ref_ready = preprocess_single(reference_path, sigma=sigma, clip_limit=clip_limit)
   
    processed_files_count = 0
   
    for path in image_paths:
        path = Path(path)
        print(f"Processing: {path.name}")
       
        if path == Path(reference_path):
            img_ready = ref_ready
        else:
            # Preprocess and then globally match histogram to reference
            img_clahe = preprocess_single(path, sigma=sigma, clip_limit=clip_limit)
            img_ready = exposure.match_histograms(img_clahe, ref_ready)
                       
        # Save the preprocessed image as a floating point TIFF
        save_path = out_dir / f"{path.stem}_processed.tif"
        io.imsave(save_path, img_ready, check_contrast=False)
       
        processed_files_count += 1
       
    return processed_files_count


# -------------------------------------------------------------------
# 2. BATCH EXECUTION RUNNER
# -------------------------------------------------------------------

if __name__ == '__main__':
    # Define paths
    IMAGE_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images")
    PROCESSED_DIR = IMAGE_DIR / "preprocessed_images"  # New folder for output
   
    # Gather target images
    IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(IMAGE_DIR.glob(ext))
        image_paths.extend(IMAGE_DIR.glob(ext.upper()))

    # Clean and sort the path list
    image_paths = sorted(list(set(image_paths)))
   
    if not image_paths:
        raise FileNotFoundError(f"Could not find any images in: {IMAGE_DIR}")
       
    print(f"Found {len(image_paths)} images to process.")
   
    # Select the reference image for histogram matching
    reference_image = (IMAGE_DIR / "control_stub1_3-2-26_0046(1).tif")
    print(f"Using {reference_image.name} as the master baseline reference.\n")
   
    print("=== Starting Image Pre-processing ===")
    count = process_batch(
        image_paths=image_paths,
        reference_path=reference_image,
        out_dir=PROCESSED_DIR,
        sigma=0.5,
        clip_limit=0.015
    )
   
    print(f"\nPreprocessing complete! {count} standardized images are saved in:\n{PROCESSED_DIR}")