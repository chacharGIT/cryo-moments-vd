from tqdm import tqdm
import numpy as np
import torch
import torch
from aspire.image import Image
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
import matplotlib.pyplot as plt
from src.utils.distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution


class VolumeDistributionModel:
    """
    A class containing all the data necessary to solve the Cryo-EM problem.

    This class encapsulates a volume, a set of rotations (quadrature rule),
    and a probability distribution over those rotations. It provides methods
    to generate noisy projections of the volume.

    Attributes:
    -----------
    volume : Volume
        The 3D volume to project
    rotations : Rotation
        A set of rotations in SO(3) (quadrature rule)
    distribution : ndarray
        The probability distribution associated with each rotation
    distribution_metadata : dict, optional
        Dictionary describing how the distribution was generated.
        The required keys depend on the generation method. Supported formats:

        - For "s2_delta_mixture":
            {
                "type": "s2_delta_mixture",
                "s2_points": ndarray of shape (K, 3),
                "s2_weights": ndarray of shape (K,)
            }

        - For "vmf_mixture":
            {
                "type": "vmf_mixture",
                "means": ndarray of shape (K, 3),
                "kappas": ndarray of shape (K,),
                "weights": ndarray of shape (K,)
            }
    """

    def __init__(self, volume: Volume, rotations: Rotation, distribution: np.ndarray, 
                 distribution_metadata=None):
        self.volume = volume
        self.rotations = rotations
        self.distribution = self.normalize_distribution(distribution)
        # Store general distribution metadata (must be a dict with a 'type' key)
        if distribution_metadata is not None:
            if not isinstance(distribution_metadata, dict):
                raise ValueError("distribution_metadata must be a dictionary if provided.")
            if 'type' not in distribution_metadata:
                raise ValueError("distribution_metadata must contain a 'type' key.")
        self.distribution_metadata = distribution_metadata
    @staticmethod
    def compute_mask(proj, sigma=0.01, chan_vese_iters=500):
        """
        Compute the inside and outside mask for a given projection using Gaussian smoothing, Chan-Vese segmentation,
        and binary dilation of the inverted mask.
        """
        from skimage.segmentation import morphological_chan_vese
        from scipy.ndimage import gaussian_filter, binary_dilation
        import numpy as np
        # Smooth and normalize
        proj_smoothed = gaussian_filter(proj, sigma=sigma)
        proj_norm = (proj_smoothed - np.min(proj_smoothed)) / (np.max(proj_smoothed) - np.min(proj_smoothed) + 1e-8)
        # Otsu's threshold for initialization
        from skimage.filters import threshold_otsu
        otsu_thresh = threshold_otsu(proj_norm)
        init_ls = proj_norm > otsu_thresh
        inside_mask = morphological_chan_vese(image=proj_norm, num_iter=chan_vese_iters, init_level_set=init_ls, smoothing=1, lambda1=1, lambda2=10)
        inside_mask = inside_mask.astype(bool)
        outside_mask = ~inside_mask.astype(bool)
        return inside_mask, outside_mask
    @staticmethod
    def normalize_distribution(distribution):
        """
        Normalize a probability distribution to sum to 1 (if not already normalized).
        """
        distribution = np.asarray(distribution)
        total = np.sum(distribution)
        if np.abs(total - 1.0) > 1e-10:
            return distribution / total
        return distribution

    def generate_noisy_projections(self, num_projections, sigma):
        """
        Generate multiple noisy projections of the volume from randomly sampled rotations.
        """
        # Sample rotations based on the provided distribution
        rotation_indices = np.random.choice(len(self.rotations), size=num_projections, p=self.distribution)
        sampled_rotations = self.rotations[rotation_indices]
        
        # Project the volume using the sampled rotations (batch projection)
        projections = self.volume.project(sampled_rotations).asnumpy()
        
        # Add Gaussian white noise with mean 0 and standard deviation sigma to all projections at once
        noise = np.random.normal(0, sigma, projections.shape)
        noisy_projections = projections + noise
        
        return noisy_projections, sampled_rotations
    
    def first_analytical_moment(self, device=None):
        """
        Calculate the first analytical moment
        
        Returns:
        --------
        first_moment : ndarray of shape (L, L)
            The computed first analytical moment.
        """
        L = self.volume.resolution
        # Batch project all rotations
        projections = self.volume.project(self.rotations).asnumpy().copy()  # shape: (N, L, L), ensure writeable
        weights = self.distribution
        # Move to torch and device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        projections_t = torch.from_numpy(projections).to(device=device, dtype=torch.float32)
        weights_t = torch.from_numpy(weights).to(device=device, dtype=torch.float32)
        # Weighted sum
        weighted_projections = projections_t * weights_t[:, None, None]
        first_moment = torch.sum(weighted_projections, dim=0)
        return first_moment.cpu().numpy()


    def second_analytical_moment(self, batch_size=10, show_progress=False, device=None, dtype=torch.float32):
        """
        Calculate the second analytical moment in batches to avoid memory blowup.

        Parameters:
        -----------
        batch_size : int
            Number of projections to process per batch.
        show_progress : bool
            Whether to show a progress bar.
        device : str, optional
            Device to use for computation ('cuda' or 'cpu'). If None, uses GPU if available.
        dtype : torch.dtype
            Data type for computation (e.g., torch.float32 or torch.float64).

        Returns:
        --------
        second_moment : ndarray of shape (L, L, L, L)
            The computed second analytical moment.
        """
        L = self.volume.resolution
        N = len(self.rotations)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        second_moment = torch.zeros((L, L, L, L), dtype=dtype, device=device)
        projections = self.volume.project(self.rotations).asnumpy().copy()  # shape: (N, L, L), ensure writeable
        projections_t = torch.from_numpy(projections).to(device=device, dtype=dtype)
        weights_t = torch.from_numpy(self.distribution).to(device=device, dtype=dtype)

        iterator = range(0, N, batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=(N + batch_size - 1) // batch_size, desc="Second Moment", leave=False)
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, N)
            batch_projs = projections_t[start_idx:end_idx]  # (B, L, L)
            batch_weights = weights_t[start_idx:end_idx]  # (B,)
            # Compute weighted outer products for this batch
            # einsum: bij,bkl->bijkl
            outer = torch.einsum('bij,bkl->bijkl', batch_projs, batch_projs)
            weighted_outer = batch_weights[:, None, None, None, None] * outer
            second_moment += torch.sum(weighted_outer, dim=0)
        return second_moment.cpu().numpy()
        
    def save_projections(self, projections, filename_prefix="projection", save_dir="tmp_figs"):
        """
        Save projection images to files.
        
        Parameters:
        -----------
        projections : ndarray
            Array of projection images with shape (num_projections, height, width)
        filename_prefix : str
            Prefix for the saved filenames
        save_dir : str
            Directory to save the images
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each projection
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

    def projections_correlation(self, rotation1=None, rotation2=None, return_rotations=False):
        """
        Measure the normalized inner product (cosine similarity) between two projections of the volume for different directions.

        Parameters:
        -----------
        rotation1 : Rotation or None
            First rotation. If None, a random rotation is sampled from the distribution.
        rotation2 : Rotation or None
            Second rotation. If None, a random rotation is sampled from the distribution.

        Returns:
        --------
        corr : float
            Normalized inner product (cosine similarity) between the two projection images.
        """
        if (rotation1 is None) or (rotation2 is None):
            idx1 = np.random.choice(len(self.rotations), p=self.distribution)
            rotation1 = self.rotations[idx1]
            idx2 = np.random.choice(len(self.rotations), p=self.distribution)
            rotation2 = self.rotations[idx2]

        # Project the volume for each rotation
        proj1 = self.volume.project(rotation1).asnumpy().squeeze()
        proj2 = self.volume.project(rotation2).asnumpy().squeeze()
    
        inside_mask1, outside_mask1 = self.compute_mask(proj1)
        proj1 = proj1.copy()
        if np.any(inside_mask1):
            mean_inside1 = np.mean(proj1[inside_mask1])
            proj1[inside_mask1] = proj1[inside_mask1] - mean_inside1
        if np.any(outside_mask1):
            mean_outside1 = np.mean(proj1[outside_mask1])
            proj1[outside_mask1] = proj1[outside_mask1] - mean_outside1
        inside_mask2, outside_mask2 = self.compute_mask(proj2)
        proj2 = proj2.copy()
        if np.any(inside_mask2):
            mean_inside2 = np.mean(proj2[inside_mask2])
            proj2[inside_mask2] = proj2[inside_mask2] - mean_inside2
        if np.any(outside_mask2):
            mean_outside2 = np.mean(proj2[outside_mask2])
            proj2[outside_mask2] = proj2[outside_mask2] - mean_outside2
            
        # Save processed projections and masks as images for debugging/visualization
        import os
        import matplotlib.pyplot as plt
        save_dir = "outputs/tmp_figs"
        os.makedirs(save_dir, exist_ok=True)
        # Save processed projections
        plt.figure(figsize=(8, 8))
        im1 = plt.imshow(proj1, cmap='gray')
        plt.title('Projection 1 (both means removed)')
        plt.axis('off')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.savefig(f"{save_dir}/projection1_means_removed.png", dpi=150, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 8))
        im2 = plt.imshow(proj2, cmap='gray')
        plt.title('Projection 2 (both means removed)')
        plt.axis('off')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.savefig(f"{save_dir}/projection2_means_removed.png", dpi=150, bbox_inches='tight')
        plt.close()
        # Save masks
        plt.figure(figsize=(8, 8))
        im3 = plt.imshow(inside_mask1.astype(float), cmap='gray')
        plt.title('Mask 1 (Chan-Vese)')
        plt.axis('off')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        plt.savefig(f"{save_dir}/mask1.png", dpi=150, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 8))
        im4 = plt.imshow(inside_mask2.astype(float), cmap='gray')
        plt.title('Mask 2 (Chan-Vese)')
        plt.axis('off')
        plt.colorbar(im4, fraction=0.046, pad=0.04)
        plt.savefig(f"{save_dir}/mask2.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Flatten the projections
        proj1_flat = proj1.flatten()
        proj2_flat = proj2.flatten()


        # Compute normalized inner product (cosine similarity)
        numerator = np.dot(proj1_flat, proj2_flat)
        denom = (np.sqrt(np.dot(proj1_flat, proj1_flat)) * np.sqrt(np.dot(proj2_flat, proj2_flat)))
        corr = numerator / denom

        if return_rotations:
            return corr, rotation1, rotation2
        else:
            return corr
    

if __name__ == "__main__":

    from src.utils.distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution
    from src.utils.volume_generation_functions import white_noise_on_unit_ball
    from aspire.volume import Volume
    from config.config import settings

    # Load volume and ensure mean zero
    from aspire.downloader import emdb_2660
    vol_ds = emdb_2660().downsample(settings.data_generation.downsample_size)
    L = vol_ds.resolution

    # Use VMF mixture for the distribution
    from src.utils.von_mises_fisher_distributions import (
        generate_random_von_mises_fisher_parameters,
        so3_distribution_from_von_mises_mixture
    )
    # Get VMF parameters from config
    vmf_cfg = settings.data_generation.von_mises_fisher
    num_vmf = vmf_cfg.num_distributions
    kappa_range = tuple(vmf_cfg.kappa_range)
    fibonacci_n = vmf_cfg.fibonacci_spiral_n
    num_in_plane = vmf_cfg.num_in_plane_rotations

    # Generate S2 quadrature points
    from src.utils.distribution_generation_functions import fibonacci_sphere_points
    quadrature_points = fibonacci_sphere_points(n=fibonacci_n)

    # Generate VMF mixture parameters
    mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
        num_vmf, kappa_range=kappa_range
    )

    # Create SO(3) distribution and S2 weights from VMF mixture
    rotations, rotation_weights = so3_distribution_from_von_mises_mixture(
        quadrature_points, mu_directions, kappa_values, mixture_weights, num_in_plane
    )
    vdm = VolumeDistributionModel(vol_ds, rotations, rotation_weights, distribution_metadata={
        'type': 'vmf_mixture',
        'means': mu_directions,
        'kappas': kappa_values,
        'weights': mixture_weights
    })

    # Print the correlation and geodesic distance for multiple pairs of random projections
    from src.utils.rotation_utils import geodesic_distance_SO3
    num_trials = 1  # Number of random pairs to sample per run
    for i in range(num_trials):
        corr, rot1, rot2 = vdm.projections_correlation(return_rotations=True)
        print(f"[{i+1}/{num_trials}] Correlation between two random projections: {corr:.6f}")
        dist = geodesic_distance_SO3(rot1, rot2)
        print(f"[{i+1}/{num_trials}] Geodesic distance between the two rotations: {dist:.6f} radians\n")
