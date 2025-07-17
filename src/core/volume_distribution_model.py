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


    def second_analytical_moment(self, batch_size=10, show_progress=False, device=None):
        """
        Calculate the second analytical moment in batches to avoid memory blowup.

        Parameters:
        -----------
        batch_size : int
            Number of projections to process per batch.

        Returns:
        --------
        second_moment : ndarray of shape (L, L, L, L)
            The computed second analytical moment.
        """
        L = self.volume.resolution
        N = len(self.rotations)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        second_moment = torch.zeros((L, L, L, L), dtype=torch.float32, device=device)
        projections = self.volume.project(self.rotations).asnumpy().copy()  # shape: (N, L, L), ensure writeable
        projections_t = torch.from_numpy(projections).to(device=device, dtype=torch.float32)
        weights_t = torch.from_numpy(self.distribution).to(device=device, dtype=torch.float32)

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
    

if __name__ == "__main__":
    from aspire.downloader import emdb_2660
    from src.utils.distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution
    from config.config import settings

    # Get a sample volume
    vol_ds = emdb_2660().downsample(settings.data_generation.downsample_size)
    L = vol_ds.resolution

    # Generate random S2 points with non-uniform weights using the new function
    s2_coords, s2_weights = generate_weighted_random_s2_points(num_points=settings.data_generation.s2_delta_mixture.num_s2_points)
    print(f"Generated {len(s2_coords)} random S2 points with non-uniform weights:")
    print(f"S2 coordinates (phi, theta):\n{s2_coords}")
    print(f"S2 weights: {s2_weights}")
    print(f"Weights sum: {np.sum(s2_weights):.6f}")

    # Create in-plane invariant distribution with in-plane rotations using the weighted S2 points
    rotations, rotation_weights = create_in_plane_invariant_distribution(
        s2_coords, s2_weights, num_in_plane_rotations=settings.data_generation.s2_delta_mixture.num_in_plane_rotations, is_s2_uniform=False
    )
    vdm = VolumeDistributionModel(vol_ds, rotations, rotation_weights, distribution_metadata={
        'type': 's2_delta_mixture',
        's2_points': s2_coords,
        's2_weights': s2_weights
    })

    # Generate a single noisy projection
    projection, rotation = vdm.generate_noisy_projections(num_projections=1, sigma=settings.data_generation.noisy.sigma)
    print(f"Generated a single projection with shape {projection[0].shape}")

    # Calculate analytical moments using the quadrature
    A_1 = vdm.first_analytical_moment()
    print("Analytical first moment shape:", A_1.shape)

    A_2 = vdm.second_analytical_moment()
    print("Analytical second moment shape:", A_2.shape)
