import numpy as np
from scipy.special import ive
from config import settings
from distribution_generation_functions import cartesian_to_spherical, create_in_plane_invariant_distribution

def von_mises_fisher_normalization_constant(kappa):
    """
    Calculate the normalization constant for von-Mises Fisher distribution on S^(2).
    
    Parameters:
    -----------
    kappa : float or ndarray
        Concentration parameter(s)
    
    Returns:
    --------
    C : float or ndarray
        Normalization constant(s)
    """
    # For S2 (d=3), the normalization constant is  kappa / (4*pi * sinh(kappa))
    
    kappa = np.asarray(kappa)
    
    # Handle kappa = 0 case (uniform distribution)
    result = np.zeros_like(kappa, dtype=float)
    nonzero_mask = kappa != 0
    
    if np.any(nonzero_mask):
        kappa_nz = kappa[nonzero_mask]
        # Use sinh for numerical stability
        result[nonzero_mask] = kappa_nz / (4 * np.pi * np.sinh(kappa_nz))
    
    # For kappa = 0, the normalization constant is 1/(4*pi) (uniform on S2)
    result[~nonzero_mask] = 1.0 / (4 * np.pi)
    
    return result if result.shape else float(result)


def von_mises_fisher_pdf(x, mu, kappa):
    """
    Evaluate the von-Mises Fisher probability density function on S2.
    
    Parameters:
    -----------
    x : ndarray
        Points on S2 in Cartesian coordinates, shape (n_points, 3)
    mu : ndarray
        Mean direction(s) on S2, shape (3,) or (n_distributions, 3)
    kappa : float or ndarray
        Concentration parameter(s), shape () or (n_distributions,)
    
    Returns:
    --------
    pdf_values : ndarray
        PDF values at each point for each distribution
        Shape (n_points,) if single distribution, (n_points, n_distributions) if multiple
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    kappa = np.asarray(kappa)
    
    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Ensure mu is 2D
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    
    # Calculate dot products: x · mu
    dot_products = np.dot(x, mu.T)  # Shape: (n_points, n_distributions)
    
    # Get normalization constants
    C = von_mises_fisher_normalization_constant(kappa)
    
    # Calculate PDF values
    if np.isscalar(kappa):
        pdf_values = C * np.exp(kappa * dot_products.flatten())
    else:
        pdf_values = C * np.exp(kappa * dot_products)
    
    return pdf_values


def generate_random_von_mises_fisher_parameters(num_distributions, kappa_range):
    """
    Generate random parameters for von-Mises Fisher distributions including mixture weights.
    
    Parameters:
    -----------
    num_distributions : int
        Number of von-Mises Fisher distributions to generate parameters for
    kappa_range : tuple
        Range for concentration parameters (min_kappa, max_kappa)
    
    Returns:
    --------
    mu_directions : ndarray
        Random mean directions on S2, shape (num_distributions, 3)
    kappa_values : ndarray
        Random concentration parameters, shape (num_distributions,)
    mixture_weights : ndarray
        Random mixture weights that sum to 1, shape (num_distributions,)
    """
    # Generate random mean directions by sampling from 3D Gaussian and normalizing
    mu_directions = np.random.randn(num_distributions, 3)
    mu_directions = mu_directions / np.linalg.norm(mu_directions, axis=1, keepdims=True)
    
    # Generate random concentration parameters
    kappa_min, kappa_max = kappa_range
    kappa_values = np.random.uniform(kappa_min, kappa_max, num_distributions)
    
    # Generate random mixture weights that sum to 1
    mixture_weights = np.random.uniform(0, 1, num_distributions)
    mixture_weights = mixture_weights / np.sum(mixture_weights)
    
    return mu_directions, kappa_values, mixture_weights


def evaluate_von_mises_fisher_mixture(quadrature_points, mu_directions, kappa_values, mixture_weights):
    """
    Evaluate a linear combination (mixture) of von-Mises Fisher distributions at quadrature points.
    
    Parameters:
    -----------
    quadrature_points : ndarray
        Points on S2 in Cartesian coordinates, shape (n_points, 3)
    mu_directions : ndarray
        Mean directions for each distribution, shape (num_distributions, 3)
    kappa_values : ndarray
        Concentration parameters, shape (num_distributions,)
    mixture_weights : ndarray
        Weights for linear combination, shape (num_distributions,)
    
    Returns:
    --------
    mixture_pdf : ndarray
        PDF values of the mixture at each quadrature point, shape (n_points,)
    """
    quadrature_points = np.asarray(quadrature_points)
    mu_directions = np.asarray(mu_directions)
    kappa_values = np.asarray(kappa_values)
    mixture_weights = np.asarray(mixture_weights)
    
    # Ensure quadrature_points is 2D
    if quadrature_points.ndim == 1:
        quadrature_points = quadrature_points.reshape(1, -1)
    
    # Evaluate each von-Mises Fisher distribution
    pdf_values = von_mises_fisher_pdf(quadrature_points, mu_directions, kappa_values)
    
    # If single distribution, reshape for consistency
    if pdf_values.ndim == 1 and len(mixture_weights) > 1:
        pdf_values = pdf_values.reshape(-1, 1)
    
    # Calculate weighted sum
    if pdf_values.ndim == 1:
        mixture_pdf = mixture_weights[0] * pdf_values
    else:
        mixture_pdf = np.dot(pdf_values, mixture_weights)
    
    return mixture_pdf

def so3_distribution_from_von_mises_mixture(quadrature_points, mu_directions, kappa_values, mixture_weights, num_in_plane_rotations):
    """
    Given S2 quadrature points, von Mises mixture parameters, and in-plane rotations, return SO(3) rotations and weights, and S2 weights.
    """
    s2_pdf_values = evaluate_von_mises_fisher_mixture(
        quadrature_points, mu_directions, kappa_values, mixture_weights
    )
    s2_weights = s2_pdf_values / np.sum(s2_pdf_values)
    s2_spherical_coords = cartesian_to_spherical(quadrature_points)
    rotations, distribution = create_in_plane_invariant_distribution(
        s2_spherical_coords, s2_weights, num_in_plane_rotations=num_in_plane_rotations, is_s2_uniform=False
    )
    return rotations, distribution

if __name__ == "__main__":
    from distribution_generation_functions import fibonacci_sphere_points
    from volume_distribution_model import VolumeDistributionModel
    from aspire.downloader import emdb_2660
    
    print("von-Mises Fisher Distribution Pipeline")

    # Generate S2 quadrature points
    quadrature_points = fibonacci_sphere_points(n=settings.von_mises_fisher.fibonacci_spiral_n)
    print(f"Generated {len(quadrature_points)} S2 quadrature points")
    
    # Generate von-Mises Fisher parameters
    num_vmf = settings.von_mises_fisher.num_distributions
    mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
        num_vmf, kappa_range=tuple(settings.von_mises_fisher.kappa_range)
    )
    
    # Create SO(3) distribution and S2 weights from von Mises mixture
    rotations, rotation_weights, s2_weights = so3_distribution_from_von_mises_mixture(
        quadrature_points, mu_directions, kappa_values, mixture_weights, settings.von_mises_fisher.num_in_plane_rotations
    )

    print(f"Created {num_vmf} von-Mises Fisher mixture, kappa range: [{np.min(kappa_values):.2f}, {np.max(kappa_values):.2f}]")
    print(f"Created SO(3) distribution: {len(rotations)} rotations")

    # Create VDM object using downsample size from config
    vol_ds = emdb_2660().downsample(settings.data_generation.downsample_size)
    vdm = VolumeDistributionModel(vol_ds, rotations, rotation_weights, quadrature_points, s2_weights)

    print(f"VDM created: {vdm.volume.resolution}³ volume, {len(vdm.rotations)} rotations")

    # Test functionality
    projections, _ = vdm.generate_noisy_projections(3, sigma=0.05)
    first_moment = vdm.first_analytical_moment()
    print(f"Generated {len(projections)} projections, first moment shape: {first_moment.shape}")
