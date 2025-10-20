import numpy as np
import matplotlib.pyplot as plt

def compute_mask(proj, sigma=0.01, chan_vese_iters=500):
    """
    Compute the inside and outside mask for a given projection using Gaussian smoothing, Chan-Vese segmentation,
    and binary dilation of the inverted mask.
    """
    from skimage.segmentation import morphological_chan_vese
    from scipy.ndimage import gaussian_filter, binary_dilation
    from skimage.filters import threshold_otsu
    # Smooth and normalize
    proj_smoothed = gaussian_filter(proj, sigma=sigma)
    proj_norm = (proj_smoothed - np.min(proj_smoothed)) / (np.max(proj_smoothed) - np.min(proj_smoothed) + 1e-8)
    # Otsu's threshold for initialization
    otsu_thresh = threshold_otsu(proj_norm)
    init_ls = proj_norm > otsu_thresh
    inside_mask = morphological_chan_vese(image=proj_norm, num_iter=chan_vese_iters, init_level_set=init_ls, smoothing=1, lambda1=1, lambda2=10)
    inside_mask = inside_mask.astype(bool)
    outside_mask = ~inside_mask.astype(bool)
    return inside_mask, outside_mask

def save_projections(projections, filename_prefix="projection", save_dir="outputs/tmp_figs/prjections"):
    """
    Save projection images to files.
    Parameters:
        projections : ndarray
            Array of projection images with shape (num_projections, height, width)
        filename_prefix : str
            Prefix for the saved filenames
        save_dir : str
            Directory to save the images
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    for i, proj in enumerate(projections):
        filename = f"{save_dir}/{filename_prefix}_{i:03d}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(proj, cmap='gray')
        plt.title(f'Projection {i}')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved projection {i} to {filename}")

def projections_correlation(projections, compute_mask_fn=compute_mask, visualize=False, save_dir="outputs/tmp_figs/projections"):
    """
    Measure the normalized inner product (cosine similarity) between two projection images.
    Parameters:
        projections : ndarray
            Array of shape (2, H, W) containing two projection images to compare.
        compute_mask_fn : function
            Function to compute inside/outside masks for a projection
        visualize : bool
            If True, save processed projections and masks as images for debugging/visualization.
        save_dir : str
            Directory to save images if visualize is True.
    Returns:
        corr : float
            Normalized inner product (cosine similarity) between the two projection images.
    """

    def preprocess(proj, idx=None):
        inside_mask, outside_mask = compute_mask_fn(proj)
        proj = proj.copy()
        if np.any(inside_mask):
            proj[inside_mask] -= np.mean(proj[inside_mask])
        if np.any(outside_mask):
            proj[outside_mask] -= np.mean(proj[outside_mask])
        
        if visualize:
            import os
            import matplotlib.pyplot as plt
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(8, 8))
            im = plt.imshow(proj, cmap='gray')
            plt.title(f'Projection {idx} (means removed)')
            plt.axis('off')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.savefig(f"{save_dir}/projection{idx}_means_removed.png", dpi=150, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8, 8))
            mask_im = plt.imshow(inside_mask.astype(float), cmap='gray')
            plt.title(f'Mask {idx} (Chan-Vese)')
            plt.axis('off')
            plt.colorbar(mask_im, fraction=0.046, pad=0.04)
            plt.savefig(f"{save_dir}/mask{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        return proj.flatten()

    proj1_flat = preprocess(projections[0], idx=1)
    proj2_flat = preprocess(projections[1], idx=2)
    numerator = np.dot(proj1_flat, proj2_flat)
    denom = (np.sqrt(np.dot(proj1_flat, proj1_flat)) * np.sqrt(np.dot(proj2_flat, proj2_flat)))
    corr = numerator / denom
    return corr
