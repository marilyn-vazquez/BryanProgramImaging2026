import numpy as np
import matplotlib.pyplot as plt
import cripser
import functions as fn

def compute_persistence_image(pd, n_bins=20, sigma=0.05, birth_range=(0.0, 1.0), persistence_range=(0.0, 1.0)):
    """
    Creates a true continuous Persistence Image from a diagram using pure NumPy.
    Transforms [Birth, Death] -> [Birth, Persistence] space.
    Each point explicitly projects its own continuous 2D Gaussian surface onto the grid.
    """
    if pd.size == 0:
        return np.zeros((n_bins, n_bins))
    
    finite_mask = np.isfinite(pd[:, 1])
    pd = pd[finite_mask]
    
    births = pd[:, 0]
    persistences = pd[:, 1] - pd[:, 0]  # Persistence = Death - Birth
    weights = persistences              # Linear weight function
    
    # 1. Create the continuous pixel grid centers
    b_centers = np.linspace(birth_range[0], birth_range[1], n_bins)
    p_centers = np.linspace(persistence_range[0], persistence_range[1], n_bins)
    B, P = np.meshgrid(b_centers, p_centers, indexing='ij')
    
    persistence_image = np.zeros((n_bins, n_bins))
    
    # 2. Accumulate continuous Gaussian surfaces for every point
    for b, p, w in zip(births, persistences, weights):
        # Continuous 2D Gaussian evaluation centered exactly at (b, p)
        gaussian = np.exp(-((B - b)**2 + (P - p)**2) / (2 * sigma**2))
        persistence_image += w * gaussian
        
    # Transpose to match your original row/col orientation for plotting
    return persistence_image.T

# Preprocess the raw image
img_ready = fn.preprocess_single("C:/Users/ezrad/Downloads/Control_Stub4_0000.tif")

print("Computing topological persistence...")
ph = cripser.compute_ph(img_ready.astype(float), maxdim=1)

# Separate H0 and H1
ph_0 = ph[ph[:, 0] == 0][:, 1:3]
ph_1 = ph[ph[:, 0] == 1][:, 1:3]

# Print the actual data ranges to determine perfect grid boundaries
print(f"H0 - Max Birth: {ph_0[:, 0].max():.4f}, Max Persistence: {(ph_0[:, 1] - ph_0[:, 0]).max():.4f}")
print(f"H1 - Max Birth: {ph_1[:, 0].max():.4f}, Max Persistence: {(ph_1[:, 1] - ph_1[:, 0]).max():.4f}")

# Generate true Persistence Images via continuous surface projection
print("Vectorizing into Persistence Images...")
h0_matrix = compute_persistence_image(ph_0, n_bins=50, sigma=0.015, birth_range=(0.0, 1.0), persistence_range=(0.0, 1.0))
h1_matrix = compute_persistence_image(ph_1, n_bins=50, sigma=0.015, birth_range=(0.0, 1.0), persistence_range=(0.0, 1.0))

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_ready, cmap='gray')
axes[0].set_title("Input Image")

im1 = axes[1].imshow(h0_matrix, cmap='jet', origin='lower')
axes[1].set_title("True Persistence Image ($H_0$)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(h1_matrix, cmap='jet', origin='lower')
axes[2].set_title("True Persistence Image ($H_1$)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# Flattening the continuous matrices to a 1D ML feature vector
flat_vector = np.concatenate([h0_matrix.flatten(), h1_matrix.flatten()])
print(f"Machine learning ready vector shape: {flat_vector.shape}")
