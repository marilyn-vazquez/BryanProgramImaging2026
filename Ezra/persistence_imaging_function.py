def vectorize_via_cripser_image(ph_output, dim=1, pixel_size=0.05):
    """
    Extracts features for a specific dimension from raw cubical ripser output,
    removes infinite features, and converts it to a flattened Persistence Image.
    """
    # 1. Filter out only the features matching the desired target dimension
    target_dg = ph_output[ph_output[:, 0] == dim]
    
    # 2. Check if the diagram is empty
    if target_dg.size == 0:
        return np.zeros(400) # Fallback vector size
        
    # 3. Clean out infinite persistence values
    finite_mask = np.isfinite(target_dg).all(axis=1)
    cleaned_dg = target_dg[finite_mask]
    
    if cleaned_dg.size == 0:
        return np.zeros(400)

    # 4. Generate the True Persistence Image using cripser's native utility
    p_image = cripser.persistence_image(cleaned_dg)
    
    # 5. Flatten the 2D grid matrix into a 1D vector suitable for Machine Learning
    return p_image.flatten()
