import numpy as np
from scipy.ndimage import gaussian_filter

def compute_persistence_image(pd, n_bins=20, sigma=0.5, birth_range=(0.0, 1.0), persistence_range=(0.0, 1.0)):
    """
    Creates a true Persistence Image from a diagram using pure NumPy and SciPy.
    Transforms [Birth, Death] -> [Birth, Persistence] space.
    Giotto cannot be used on Python 3.13, so this is the replacement function.
    """
    if pd.size == 0:
        return np.zeros((n_bins, n_bins))
    
    births = pd[:, 0]
    persistences = pd[:, 1] - pd[:, 0]  # Persistence = Death - Birth
    
    # Linear weight function (features with high persistence get higher weight)
    weights = persistences
    
    # Set up empty 2D grid boundaries
    b_grid = np.linspace(birth_range[0], birth_range[1], n_bins + 1)
    p_grid = np.linspace(persistence_range[0], persistence_range[1], n_bins + 1)
    
    # Bin the points into a 2D histogram grid weighted by persistence
    img_matrix, _, _ = np.histogram2d(
        births, 
        persistences, 
        bins=[b_grid, p_grid], 
        weights=weights
    )
    
    # Apply the 2D Gaussian smoothing over the grid (transposed for correct row/col orientation)
    persistence_image = gaussian_filter(img_matrix.T, sigma=sigma)
    
    return persistence_image

import matplotlib.pyplot as plt
import cripser
import functions as fn # use full functions file

img_ready = fn.preprocess_single("IMAGE PATH HERE")


print("Computing topological persistence...")
ph = cripser.compute_ph(img_ready.astype(float), maxdim=1)

# Separate H0 and H1
ph_0 = ph[ph[:, 0] == 0][:, 1:3]
ph_1 = ph[ph[:, 0] == 1][:, 1:3]

# Generate true Persistence Images cleanly
print("Vectorizing into Persistence Images...")
h0_matrix = compute_persistence_image(ph_0, n_bins=20, sigma=0.8)
h1_matrix = compute_persistence_image(ph_1, n_bins=20, sigma=0.8)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_ready, cmap='gray')
axes[0].set_title("Input Image")

im1 = axes[1].imshow(h0_matrix, cmap='jet', origin='lower')
axes[1].set_title("Native Persistence Image ($H_0$)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(h1_matrix, cmap='jet', origin='lower')
axes[2].set_title("Native Persistence Image ($H_1$)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# Flattening to an ML feature vector
flat_vector = np.concatenate([h0_matrix.flatten(), h1_matrix.flatten()])
print(f"Machine learning ready vector shape: {flat_vector.shape}")
