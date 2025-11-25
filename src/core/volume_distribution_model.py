from tqdm import tqdm
import numpy as np
import torch
import torch
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from config.config import settings
from src.utils.matrix_generation_functions import generate_gaussian_matrix
from src.utils.polar_transform import cartesian_to_polar, polar_to_cartesian

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
    rotations : Rotation or ndarray
        A set of rotations in SO(3) (quadrature rule), or S2 quadrature points if in_plane_invariant_distribution is True.
    distribution : ndarray
        The probability distribution associated with each rotation (SO(3)) or S2 point, depending on
        the value of in_plane_invariant_distribution.
    in_plane_invariant_distribution : bool
        If True, the distribution is over S2 points (in-plane invariant). If False, it is over SO(3) rotations.
    distribution_metadata : dict, optional
        Dictionary describing how the distribution was generated.
        The required keys depend on the generation method. Supported formats:

        - For "s2_delta_mixture":
            {
                "type": "s2_delta_mixture",
                "s2_points": ndarray of shape (K, 3),
                "weights": ndarray of shape (K,)
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
                 distribution_metadata=None, fourier_domain=False, in_plane_invariant_distribution=False):
        self.volume = volume
        self.rotations = rotations
        self.distribution = self.normalize_distribution(distribution)
        self.in_plane_invariant_distribution = in_plane_invariant_distribution
        self.fourier_domain = fourier_domain
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
    
    def sample_rotations(self, num_rotations=1, s2_points=None):
        """
        Sample random SO(3) rotations according to the model's distribution.
        For in-plane invariant (S2) case, can specify S2 points or sample them from the distribution.
        For each S2 point, samples in-plane angles (psi).

        Parameters
        ----------
        num_rotations : int, optional
            Number of rotations to sample per S2 point. Default is 1.
            If s2_points is None, this is the total number of S2 points to sample.
            If s2_points is provided, this is the number of psi angles per S2 point.
        s2_points : ndarray or None, optional
            Array of S2 points (shape (k, 3)). For each, sample SO(3) rotation using phi, theta from S2 and random psi.
            If None, sample S2 points from distribution and random psi for each.

        Returns
        -------
        rotations : ndarray
            Array of sampled rotations (SO(3) points).
        """
        if not self.in_plane_invariant_distribution:
            if s2_points is not None:
                    import warnings
                    warnings.warn("s2_points is set but in_plane_invariant_distribution is False. Ignoring s2_points.")
                    
            return np.random.choice(len(self.rotations), size=num_rotations, p=self.distribution)
        else:
            if s2_points is not None:
                s2_points = np.asarray(s2_points)
                num_s2_points = s2_points.shape[0]
            else:
                idx = np.random.choice(len(self.rotations), size=num_rotations, p=self.distribution)
                s2_points = self.rotations[idx]
                num_s2_points = num_rotations
                num_rotations = 1

            # Convert S2 points to spherical coordinates (phi, theta)
            x, y, z = s2_points[:, 0], s2_points[:, 1], s2_points[:, 2]
            theta = np.arccos(z)
            phi = np.arctan2(y, x)
            
            # Sample psi angles
            if num_rotations == 1:
                # Single psi per S2 point
                psi = np.random.uniform(0, 2 * np.pi, size=num_s2_points)
            else:
                # Multiple psi per S2 point: first random, others evenly spaced
                psi_list = []
                for i in range(num_s2_points):
                    first_psi = np.random.uniform(0, 2 * np.pi)
                    psi_for_point = (first_psi + np.arange(num_rotations) * 2 * np.pi / num_rotations) % (2 * np.pi)
                    psi_list.append(psi_for_point)
                psi = np.concatenate(psi_list)
                # Replicate phi and theta for each psi
                phi = np.repeat(phi, num_rotations)
                theta = np.repeat(theta, num_rotations)
            
            # Euler angles: (phi, theta, psi)
            euler_angles = np.stack([phi, theta, psi], axis=1)
            rotations = Rotation.from_euler(euler_angles, dtype=np.float32)
            return rotations
        
    def generate_projections(self, rotations_to_project=None, num_projections=None, sigma=0, return_used_rotations=False):
        """
        Generate multiple noisy projections of the volume from specified or randomly sampled rotations.

        Parameters
        ----------
        rotations_to_project : ndarray or None, optional
            For non in-plane invariant: Array of SO(3) rotations to use for projection.
            For in-plane invariant: Array of S2 points. If num_projections is also provided, 
            generates num_projections rotations per S2 point with evenly spaced psi angles.
        num_projections : int or None, optional
            For non in-plane invariant: Number of projections to generate (samples SO(3) rotations).
            For in-plane invariant: If rotations_to_project is None, samples this many S2 points.
            If rotations_to_project is provided, samples this many psi angles per S2 point.
        sigma : float, optional
            Standard deviation of Gaussian noise to add to each projection. Default is 0 (no noise).
        return_used_rotations : bool, optional
            If True and random rotations are sampled, return the used rotations along with projections.
            Ignored if input rotations are provided.

        Returns
        -------
        projections : ndarray
            Array of projections.
        used_rotations : ndarray (optional - if return_used_rotations is True)
            Array of rotations used for projection.
        """
        if rotations_to_project is None and num_projections is None:
            raise ValueError("One of 'rotations_to_project' or 'num_projections' parameters must be provided.")
        
        if self.in_plane_invariant_distribution:
            # For in-plane invariant case, rotations_to_project are S2 points
            if rotations_to_project is not None and num_projections is not None:
                # Both provided: generate num_projections psi angles per S2 point
                rotations_to_project = self.sample_rotations(num_rotations=num_projections, s2_points=rotations_to_project)
            elif rotations_to_project is not None:
                # Only S2 points provided: one random psi per S2 point
                rotations_to_project = self.sample_rotations(s2_points=rotations_to_project)
            else:
                # Only num_projections provided: sample S2 points and one psi each
                rotations_to_project = self.sample_rotations(num_rotations=num_projections)
        else:
            # Non-invariant case
            if num_projections is not None and rotations_to_project is None:
                rotations_to_project = self.sample_rotations(num_rotations=num_projections)
        
        projections = self.volume.project(rotations_to_project).asnumpy()
        noise = np.random.normal(0, sigma, projections.shape)
        projections = projections + noise
        if self.fourier_domain:
            projections = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(projections, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))   
        if not return_used_rotations:
            return projections
        else:
            return projections, rotations_to_project
    
    def first_analytical_moment(self, num_projections_per_s2_point=1,
                                 n_theta=settings.data_generation.cartesian_to_polar_n_theta, dtype=torch.float64):
        """
        Calculate the first analytical moment.
        For in-plane invariant distributions, averages the polar representation of projections over theta,
        then computes a weighted average over S2 points, and reconstructs a radial, angularly symmetric image.

        Parameters
        ----------
        num_projections_per_s2 : int
            For in-plane invariant distributions, number of projections (psi angles) to sample per S2 point.
            Ignored for non in-plane invariant distributions.
        n_theta : int
            Number of angular samples for polar transform.

        Returns
        -------
        first_moment : ndarray of shape (L, L)
            The computed first analytical moment.
        """
        weights = self.distribution
        if self.fourier_domain:
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        if not self.in_plane_invariant_distribution:
            if settings.device.use_cuda:
                device = f"cuda:{settings.device.cuda_device}"
            else:
                device = "cpu"
            projections = self.generate_projections(rotations_to_project=self.rotations)
            weights_t = torch.from_numpy(weights).to(device=device, dtype=dtype)
            projections_t = torch.from_numpy(projections).to(device=device, dtype=dtype)
            weighted_projections = projections_t * weights_t[:, None, None]
            first_moment = torch.sum(weighted_projections, dim=0)
            return first_moment.cpu().numpy()
        else:
            # Generate projections with multiple psi angles per S2 point
            projections = self.generate_projections(
                rotations_to_project=self.rotations, 
                num_projections=num_projections_per_s2_point
            )
            # projections shape: (N * num_projections_per_s2, H, W) where N = len(self.rotations)
            N_s2 = len(self.rotations)
            N_total, H, W = projections.shape
            
            # Convert each projection to polar and average over theta
            radial_profiles = []
            for i in range(N_total):
                polar_img, r_vals, _ = cartesian_to_polar(projections[i], n_theta=n_theta)
                radial_profile = np.mean(polar_img, axis=1)  # average over theta
                radial_profiles.append(radial_profile)
            radial_profiles = np.stack(radial_profiles, axis=0)  # shape (N_total, n_r)
            
            # Reshape to group by S2 point and average over psi angles
            radial_profiles_grouped = radial_profiles.reshape(N_s2, num_projections_per_s2_point, -1)
            radial_profiles_avg = np.mean(radial_profiles_grouped, axis=1)  # shape (N_s2, n_r)
            
            # Weighted average over S2 points
            weighted_radial = np.average(radial_profiles_avg, axis=0, weights=weights)
            
            # Build polar image: repeat weighted_radial for all theta
            polar_image = np.tile(weighted_radial[:, None], (1, n_theta))
            first_moment = polar_to_cartesian(polar_image, r_vals, n_theta, output_shape=(H, W))
            return first_moment

    def second_analytical_moment(self, batch_size=10, show_progress=False, dtype=torch.float64):
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
        if settings.device.use_cuda:
            device = f"cuda:{settings.device.cuda_device}"
        else:
            device = "cpu"
        if self.fourier_domain:
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        second_moment = torch.zeros((L, L, L, L), dtype=dtype, device=device)
        projections = self.generate_projections(self.rotations)
        # projections = self.volume.project(self.rotations).asnumpy().copy()  # shape: (N, L, L), ensure writeable
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
            if self.fourier_domain:
                outer = torch.einsum('bij,bkl->bijkl', batch_projs, batch_projs.conj())
            else:
                outer = torch.einsum('bij,bkl->bijkl', batch_projs, batch_projs)
            weighted_outer = batch_weights[:, None, None, None, None] * outer
            second_moment += torch.sum(weighted_outer, dim=0)
        return second_moment.cpu().numpy()
    
    def third_analytical_moment_sketch(self, sketch_size, device=None, dtype=torch.float32):
        """
        Calculate the third analytical moment using structured tensor sketching (see Eq. 46 in the referenced method).
        This method avoids memory blowup by sketching the unfolded third moment tensor using two Gaussian matrices.

        In plane invariant case, NOT SUPPORTED YET.
        Parameters:
        -----------
        sketch_size : int
            Number of columns in the Gaussian sketch matrices (s). It is required that s << d^2 for efficient sketching.
        device : str or torch.device, optional
            Device for computation.
        dtype : torch.dtype, optional
            Data type for sketch matrices.

        Returns:
        --------
        sketch : torch.Tensor, shape (d, s)
            Sketched third analytical moment.
        """
        # Prepare projections and weights as in first/second moment
        L = self.volume.resolution
        d = L * L
        projections = self.volume.project(self.rotations).asnumpy().copy()  # (N, L, L)
        weights = self.distribution
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        projections_t = torch.from_numpy(projections).to(device=device, dtype=dtype)
        weights_t = torch.from_numpy(weights).to(device=device, dtype=dtype)
        N = projections_t.shape[0]
        # Flatten each projection to (N, d)
        I_hat = projections_t.view(N, -1)
        # Apply weights
        I_hat = I_hat * weights_t[:, None]
        # Generate sketch matrices
        G1 = generate_gaussian_matrix((d, sketch_size), device=device, dtype=dtype)
        G2 = generate_gaussian_matrix((d, sketch_size), device=device, dtype=dtype)
        v1 = I_hat @ G1  # (N, s)
        v2 = I_hat @ G2  # (N, s)
        sketch = torch.einsum('nd,ns->nds', I_hat, v1 * v2)  # (N, d, s)
        sketch = sketch.mean(dim=0)  # (d, s)
        return sketch

if __name__ == "__main__":

    from aspire.volume import Volume
    from aspire.utils.rotation import Rotation
    from config.config import settings
    from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture

    # Load volume and ensure mean zero
    from src.data.emdb_downloader import load_aspire_volume
    downsample_size = settings.data_generation.downsample_size

    emdb_id = "emd_47031"
    emdb_path = f"/data/shachar/emdb_downloads/{emdb_id}.map.gz"
    save_dir = f"outputs/spectral_analysis/{emdb_id}"
    
    # Generate VDM using the generator
    print("\n1. Loading volume and generating VDM...")
    volume = load_aspire_volume(emdb_path, downsample_size=settings.data_generation.downsample_size)
    vol_ds = volume.downsample(downsample_size)
    L = vol_ds.resolution

    # Get VMF parameters from config
    vmf_cfg = settings.data_generation.von_mises_fisher

    # Generate S2 quadrature points
    from src.utils.distribution_generation_functions import fibonacci_sphere_points
    quadrature_points = fibonacci_sphere_points(n=vmf_cfg.fibonacci_spiral_n)

    # Generate VMF mixture parameters
    mu, kappa, weights = generate_random_vmf_parameters(
        vmf_cfg.num_distributions, kappa_start=vmf_cfg.kappa_start, kappa_mean=vmf_cfg.kappa_mean
    )
    s2_distribution = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)

    from src.utils.distribution_generation_functions import generate_weighted_random_s2_points
    import numpy as np
    # quadrature_points, s2_distribution = generate_weighted_random_s2_points(1)
    
    vdm1 = VolumeDistributionModel(vol_ds, rotations=quadrature_points, distribution=s2_distribution,
                                   distribution_metadata={
                                        'type': 'vmf_mixture',
                                        'means': mu,
                                        'kappas': kappa,
                                        'weights': weights
                                    }, in_plane_invariant_distribution=True)
    
    # Sample random rotations from S2 points
    # first_moment_1 = vdm1.first_analytical_moment(num_projections_per_s2_point=300, n_theta=350)
    #first_moment_1 = vdm1.first_analytical_moment(num_projections_per_s2_point=64, n_theta=256)
    #first_moment_2 = vdm1.first_analytical_moment(num_projections_per_s2_point=64, n_theta=256)
    from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
    diff = 0
    diff_second = 0
    so3_rotations, so3_weights = create_in_plane_invariant_distribution(quadrature_points, s2_distribution, 
                                                                        num_in_plane_rotations=200)
    vdm = VolumeDistributionModel(vol_ds, rotations=so3_rotations, distribution=so3_weights,
                                    distribution_metadata={
                                        'type': 'vmf_mixture',
                                        'means': mu,
                                        'kappas': kappa,
                                        'weights': weights
                                    }, in_plane_invariant_distribution=False)
    first_moment_1 = vdm.first_analytical_moment()
    second_moment_1 = vdm.second_analytical_moment(batch_size=50, show_progress=True)
    so3_rotations, so3_weights = create_in_plane_invariant_distribution(quadrature_points, s2_distribution, 
                                                                        num_in_plane_rotations=200)
    vdm = VolumeDistributionModel(vol_ds, rotations=so3_rotations, distribution=so3_weights,
                                    distribution_metadata={
                                        'type': 'vmf_mixture',
                                        'means': mu,
                                        'kappas': kappa,
                                        'weights': weights
                                    }, in_plane_invariant_distribution=False)
    first_moment_2 = vdm.first_analytical_moment()
    second_moment_2 = vdm.second_analytical_moment(batch_size=50, show_progress=True)
    diff += np.linalg.norm(first_moment_1 - first_moment_2)
    diff_second += np.linalg.norm(second_moment_1 - second_moment_2)
    # --- Compare results ---
    print(f"Relative difference norm between first moments: {diff/(np.linalg.norm(first_moment_1)):.4e}")
    print(f"Relative difference norm between second moments: {diff_second/(np.linalg.norm(second_moment_1)):.4e}")