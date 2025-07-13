from operator import le
import numpy as np
from aspire.image import Image
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
import matplotlib.pyplot as plt


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
    
    def first_analytical_moment(self):
        """
        Calculate the first analytical moment by summing weighted projections.
        """
        # Get resolution from volume
        L = self.volume.resolution
        
        # Initialize the first moment
        first_moment = np.zeros((L, L), dtype=self.volume.dtype)
        
        # For each rotation in the quadrature
        for i, rotation in enumerate(self.rotations):
            # Project the volume using the rotation
            projection = self.volume.project(rotation).asnumpy()
            
            # Add the weighted projection to the first moment
            # Extract the first (and only) image from the projection
            first_moment += self.distribution[i] * projection[0]
    
        return first_moment


    def second_analytical_moment(self):
        """
        Calculate the second analytical moment by summing outer products of weighted projections.
        """        
        # Get resolution from volume
        L = self.volume.resolution
        
        # Initialize the second moment
        second_moment = np.zeros((L, L, L, L), dtype=self.volume.dtype)
        
        # For each rotation in the quadrature
        for i, rotation in enumerate(self.rotations):
            # Project the volume using the rotation
            projection = self.volume.project(rotation).asnumpy()
            proj_data = projection[0]
            
            # Compute the outer product of the projection with itself
            # (i,j) ⊗ (k,l) → (i,j,k,l)
            outer_product = np.einsum('ij,kl->ijkl', proj_data, proj_data)
            
            # Add the weighted outer product to the second moment
            second_moment += self.distribution[i] * outer_product
        
        return second_moment
    
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
    from distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution
    
    # Get a sample volume
    vol_ds = emdb_2660().downsample(64)
    L = vol_ds.resolution
    
    # Generate random S2 points with non-uniform weights using the new function
    s2_coords, s2_weights = generate_weighted_random_s2_points(num_points=3)
    print(f"Generated {len(s2_coords)} random S2 points with non-uniform weights:")
    print(f"S2 coordinates (phi, theta):\n{s2_coords}")
    print(f"S2 weights: {s2_weights}")
    print(f"Weights sum: {np.sum(s2_weights):.6f}")
    
    # Create in-plane invariant distribution with 8 in-plane rotations using the weighted S2 points
    num_in_plane = 8
    rotations, rotation_weights = create_in_plane_invariant_distribution(
        s2_coords, s2_weights, num_in_plane_rotations=num_in_plane, is_s2_uniform=False
    )
    vdm = VolumeDistributionModel(vol_ds, rotations, rotation_weights)

    # Generate a single noisy projection
    projection, rotation = vdm.generate_noisy_projections(num_projections=1, sigma=0.05)
    print(f"Generated a single projection with shape {projection[0].shape}")

        
    # Calculate analytical moments using the quadrature
    A_1 = vdm.first_analytical_moment()
    print("Analytical first moment shape:", A_1.shape)
    
    A_2 = vdm.second_analytical_moment()
    print("Analytical second moment shape:", A_2.shape)
