import os
import glob
from pathlib import Path
import numpy as np
import numba as nb
import cripser
import matplotlib.pyplot as plt
from skimage import io, exposure, filters
from skimage.util import img_as_float
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# Silence OpenCV metadata logging warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
plt.ion()

# -------------------------------------------------------------------
# 1. PRE-PROCESSING (EXACT MATCH)
# -------------------------------------------------------------------

def preprocess_single(image_input, sigma=0.5, clip_limit=0.015):
    """Preprocess a single microscopy image uniformly."""
    if isinstance(image_input, (str, Path)):
        img = img_as_float(io.imread(image_input, as_gray=True))
    else:
        img = img_as_float(image_input)
        
    img = img[:3850, :]
    img_smoothed = filters.gaussian(img, sigma=sigma)
    img_clahe = exposure.equalize_adapthist(img_smoothed, kernel_size=256, clip_limit=clip_limit)
    return img_clahe


def process_batch(image_paths, reference_path, out_dir, sigma=0.5, clip_limit=0.015):
    """Apply preprocessing uniformly across the entire discovered batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing batch baseline reference image...")
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

# --------------------------------------------------------------
# 2. LOWER STAR FILTRATION (CRIPSER)
# --------------------------------------------------------------

def compute_lower_star(cropped):
    """Compute lower-star persistence using Cripser."""
    ph = cripser.compute_ph(cropped.astype(float), maxdim=1)
    return ph

# -------------------------------------------------------------------
# 3. VECTORIZATION HELPER
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
# 4. CUSTOM SINGLE-PLOT PCA WITH ACCURACY & F1 METRICS
# -------------------------------------------------------------------

def plot_5component_pca_scatter(X_features, y_labels, train_mask, eval_mask, test_acc, test_f1, pc_x=1, pc_y=2, title="Lower-Star PCA Dimensionality Reduction"):
    """
    Projects onto a single clean PCA plot using manual train/evaluation selection masks
    and adjustable component index parameters.
    """
    X_train, y_train = X_features[train_mask], y_labels[train_mask]
    X_eval, y_eval = X_features[eval_mask], y_labels[eval_mask]
    
    # Scale features using the designated training distribution baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    
    # Run standard PCA mapping down to 5 principal components
    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_eval_pca = pca.transform(X_eval_scaled)
    
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Map 1-based user input down to 0-based Python array indices
    idx_x = int(pc_x) - 1
    idx_y = int(pc_y) - 1
    
    if max(idx_x, idx_y) > 4 or min(idx_x, idx_y) < 0:
        raise ValueError("Selected components must be integers chosen strictly between 1 and 5.")
        
    plt.figure(figsize=(9, 7))
    
    # --- PLOT TRAINING DATA (True Labels) ---
    plt.scatter(
        X_train_pca[y_train == 0, idx_x], X_train_pca[y_train == 0, idx_y],
        color='#FF1493', marker='o', label='Train: Control (True)', edgecolors='k', s=80
    )
    plt.scatter(
        X_train_pca[y_train == 1, idx_x], X_train_pca[y_train == 1, idx_y],
        color='#00BFFF', marker='o', label='Train: Microgravity (True)', edgecolors='k', s=80
    )
    
    # --- PLOT EVALUATION DATA (Predicted Labels) ---
    plt.scatter(
        X_eval_pca[y_eval == 0, idx_x], X_eval_pca[y_eval == 0, idx_y],
        color='#FF1493', marker='^', label='Eval: Control (Predicted)', edgecolors='k', s=90, alpha=0.5
    )
    plt.scatter(
        X_eval_pca[y_eval == 1, idx_x], X_eval_pca[y_eval == 1, idx_y],
        color='#00BFFF', marker='^', label='Eval: Microgravity (Predicted)', edgecolors='k', s=90, alpha=0.5
    )
    
    # Display metrics layout panel
    metrics_text = f"NN Evaluation Accuracy: {test_acc * 100:.2f}%\nNN Evaluation F1 Score: {test_f1 * 100:.2f}%"
    plt.text(
        0.05, 0.92, metrics_text,
        transform=plt.gca().transAxes, fontsize=11, weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
    )
    
    plt.xlabel(f"Principal Component {pc_x} ({var_exp[idx_x]:.2f}%)", weight="bold")
    plt.ylabel(f"Principal Component {pc_y} ({var_exp[idx_y]:.2f}%)", weight="bold")
    plt.title(title, fontsize=12, weight="bold", pad=15)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 5. MASTER CONTROL BLOCK (SEPARATE DIRECTORY PIPELINE)
# -------------------------------------------------------------------

if __name__ == '__main__':
    # Explicit folder paths provided
    train_folder = r"C:\Users\chloe\Downloads\Images (1)\Images\Training"
    eval_folder = r"C:\Users\chloe\Downloads\Images (1)\Images\Evaluation"
    
    preprocessed_folder = 'Preprocessed_Images'
    outputFolder = 'LowerStar_Filtrations_Vectorized'
    os.makedirs(outputFolder, exist_ok=True)
    
    # Gather files from both locations
    train_images = glob.glob(os.path.join(train_folder, "*.tif"))
    eval_images = glob.glob(os.path.join(eval_folder, "*.tif"))
    all_raw_images = train_images + eval_images

    if not all_raw_images:
        print(f"❌ Error: No .tif images detected in either folder path.")
    else:
        # Step 1: Uniform Preprocessing & Equalization (Using first train image as baseline)
        print(f"Executing batch preprocessing transformations across all {len(all_raw_images)} files...")
        process_batch(image_paths=all_raw_images, reference_path=all_raw_images[0], out_dir=preprocessed_folder)
        
        # Helper function to extract features from a specific folder slice
        def process_directory_subset(image_paths_subset, subset_name):
            features_list = []
            labels_list = []
            names_list = []
            
            print(f"\n🔄 Starting lower-star processing for the {subset_name} set...")
            for orig_path in image_paths_subset:
                proc_img_path = Path(preprocessed_folder) / f"{Path(orig_path).stem}_processed.tif"
                base_name = proc_img_path.stem
                
                try:
                    img = io.imread(str(proc_img_path))
                    img = np.asarray(img, dtype=np.float64)
                except Exception as e:
                    print(f"   ❌ Skipping file: {base_name} | Error: {e}")
                    continue
                    
                # Determine class: 0 for Control, 1 for Microgravity
                label = 1 if "microgravity" in base_name.lower() else 0
                labels_list.append(label)
                names_list.append(Path(orig_path).name)
                
                # --- LOWER STAR FILTRATION ---
                ph_lower = compute_lower_star(img)
                l_d0, _ = extract_cripser_dims(ph_lower)
                
                # --- VECTORIZE ---
                lowerstar_vector = get_barcode_stats(l_d0)
                np.save(os.path.join(outputFolder, f'{base_name}_lowerstar_dim0.npy'), lowerstar_vector)
                features_list.append(lowerstar_vector)
                
            return np.array(features_list), np.array(labels_list), names_list

        # Step 2: Process training set files vs evaluation set files separately
        X_train, y_train, names_train = process_directory_subset(train_images, "TRAINING")
        X_eval, y_eval, names_eval = process_directory_subset(eval_images, "EVALUATION")
        
        # Re-combine into global arrays exclusively for drawing unified PCA space coordinates
        X_all_combined = np.vstack([X_train, X_eval])
        y_all_combined = np.concatenate([y_train, y_eval])
        
        # Build strict Boolean masks so the plotting function knows where they belong
        train_mask = np.zeros(len(y_all_combined), dtype=bool)
        train_mask[:len(y_train)] = True
        eval_mask = ~train_mask

        # Step 3: Run Neural Network Pipeline
        if len(X_train) > 0 and len(X_eval) > 0:
            try:
                print("\nTRAINING NEURAL NETWORK")
                print("===================================")
                print("Training samples:", X_train.shape[0])
                print("Testing samples:", X_eval.shape[0])
                print("Features per image:", X_train.shape[1])
                
                # Create neural network pipeline with standard scaling built-in
                mlp = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42
                    )
                )
                
                # Train neural network
                mlp.fit(X_train, y_train)
                
                # Predict evaluation image labels
                y_pred = mlp.predict(X_eval)
                
                # Calculate metrics
                final_acc = accuracy_score(y_eval, y_pred)
                final_f1 = f1_score(y_eval, y_pred, average="weighted")
                
                print("\nNEURAL NETWORK RESULTS")
                print("===================================")
                print("Accuracy:", final_acc)
                print("Weighted F1 Score:", final_f1)
                
                # Convert numeric labels to class names
                label_names = {0: "Control", 1: "Microgravity"}
                
                print("\nINDIVIDUAL TEST PREDICTIONS")
                print("===================================")
                for image_name, true_label, predicted_label in zip(names_eval, y_eval, y_pred):
                    print(f"\nImage: {image_name}")
                    print(f"Actual: {label_names[true_label]}")
                    print(f"Predicted: {label_names[predicted_label]}")
                    print("Result: Correct" if true_label == predicted_label else "Result: Incorrect")
                
                # Display confusion matrix
                ConfusionMatrixDisplay.from_predictions(
                    y_eval, y_pred,
                    labels=[0, 1],
                    display_labels=["Control", "Microgravity"]
                )
                plt.title("Neural Network Confusion Matrix")
                plt.tight_layout()
                plt.show()
                
                # Step 4: Launch Customizable 5-Component PCA Alignment Scatter Plot
                print("\n📊 Launching customizable 5-Component PCA alignment scatter plot...")
                
                # CHOOSE YOUR COMPONENTS HERE (Pick any numbers 1 through 5)
                component_for_x = 1
                component_for_y = 2
                
                plot_5component_pca_scatter(
                    X_features=X_all_combined, 
                    y_labels=y_all_combined, 
                    train_mask=train_mask,
                    eval_mask=eval_mask,
                    test_acc=final_acc, 
                    test_f1=final_f1, 
                    pc_x=component_for_x, 
                    pc_y=component_for_y,
                    title="Lower-Star Filtration Space (Training vs Evaluation Folders)"
                )
                
            except ValueError as val_err:
                print(f"\n❌ Execution Stopped: {val_err}")
        else:
            print("\n❌ Error: Verify that both your training and evaluation directories contain valid image files.")