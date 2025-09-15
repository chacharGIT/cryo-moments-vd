import numpy as np
from config.config import settings
from src.utils.distribution_generation_functions import cartesian_to_spherical, create_in_plane_invariant_distribution

# --- PyTorch GPU-native vMF functions ---
import torch

def von_mises_fisher_normalization_constant_torch(kappa, kappa_clamp_max=None):
    """
    PyTorch version: Calculate normalization constant for vMF on S^2.
    kappa: tensor (...,)
    Returns: tensor (...,)
    """
    if kappa_clamp_max is None:
        kappa_clamp_max = float(settings.data_generation.von_mises_fisher.kappa_clamp_max)
    kappa = torch.clamp(kappa, 0, kappa_clamp_max)
    # Avoid division by zero for kappa=0
    four_pi = 4 * torch.pi
    sinh_kappa = torch.sinh(kappa)
    # For kappa=0, sinh(0)=0, so set normalization to 1/(4pi)
    norm = torch.where(kappa == 0, 1.0 / four_pi, kappa / (four_pi * sinh_kappa))
    return norm

def von_mises_fisher_pdf_torch(x, mu, kappa, kappa_clamp_max=None):
    """
    PyTorch version: Evaluate vMF PDF on S^2.
    x: (..., 3)
    mu: (..., 3) or (n_distributions, 3)
    kappa: (...,) or (n_distributions,)
    Returns: (..., n_distributions) or (...,) if single distribution
    """
    if kappa_clamp_max is None:
        kappa_clamp_max = float(settings.data_generation.von_mises_fisher.kappa_clamp_max)
    x = torch.as_tensor(x)
    mu = torch.as_tensor(mu)
    kappa = torch.as_tensor(kappa)
    # Ensure 2D
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    # (n_points, 3) @ (3, n_distributions) -> (n_points, n_distributions)
    dot_products = torch.matmul(x, mu.T)
    kappa = torch.clamp(kappa, 0, kappa_clamp_max)
    # Broadcast kappa if needed
    if kappa.ndim == 0:
        kappa = kappa.unsqueeze(0)
    exp_arg = kappa * dot_products
    exp_arg_max = torch.amax(exp_arg, dim=0, keepdim=True)
    exp_stable = torch.exp(exp_arg - exp_arg_max)
    C = von_mises_fisher_normalization_constant_torch(kappa, kappa_clamp_max)
    pdf_values = C * exp_stable * torch.exp(exp_arg_max)
    # Check for NaNs or infs
    if not torch.all(torch.isfinite(pdf_values)):
        raise RuntimeError("NaN or Inf encountered in von_mises_fisher_pdf_torch. Check kappa and input values.")
    return pdf_values.squeeze()

def generate_random_von_mises_fisher_parameters_torch(num_distributions, kappa_start, kappa_mean, device=None, dtype=None):
    """
    PyTorch version: Generate random vMF parameters and mixture weights.
    Returns: mu_directions (num_distributions,3), kappa_values (num_distributions,), mixture_weights (num_distributions,)
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    # Random mean directions
    mu = torch.randn(num_distributions, 3, device=device, dtype=dtype)
    mu = mu / mu.norm(dim=1, keepdim=True)
    # Translated exponential for kappa
    scale = kappa_mean - kappa_start
    kappa = kappa_start + torch.distributions.Exponential(1.0/scale).sample((num_distributions,)).to(device=device, dtype=dtype)
    # Mixture weights
    weights = torch.rand(num_distributions, device=device, dtype=dtype)
    weights = weights / weights.sum()
    return mu, kappa, weights

def evaluate_von_mises_fisher_mixture_torch(quadrature_points, mu_directions, kappa_values, mixture_weights, kappa_clamp_max=None):
    """
    PyTorch version: Evaluate vMF mixture at quadrature points.
    quadrature_points: (n_points, 3)
    mu_directions: (n_distributions, 3)
    kappa_values: (n_distributions,)
    mixture_weights: (n_distributions,)
    Returns: (n_points,)
    """
    quadrature_points = torch.as_tensor(quadrature_points)
    mu_directions = torch.as_tensor(mu_directions)
    kappa_values = torch.as_tensor(kappa_values)
    mixture_weights = torch.as_tensor(mixture_weights)
    pdf_values = von_mises_fisher_pdf_torch(quadrature_points, mu_directions, kappa_values, kappa_clamp_max)
    # pdf_values: (n_points, n_distributions)
    if pdf_values.ndim == 1 and mixture_weights.numel() > 1:
        pdf_values = pdf_values.unsqueeze(1)
    if pdf_values.ndim == 1:
        mixture_pdf = mixture_weights[0] * pdf_values
    else:
        mixture_pdf = torch.matmul(pdf_values, mixture_weights)
    return mixture_pdf

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
    # For S2, the normalization constant is  kappa / (4*pi * sinh(kappa))
    
    kappa = np.asarray(kappa)
    # Clamp kappa to avoid overflow in sinh
    kappa = np.clip(kappa, 0, settings.data_generation.von_mises_fisher.kappa_clamp_max)
    result = kappa / (4 * np.pi * np.sinh(kappa))
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
    
    # Ensure x,mu are 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    
    dot_products = np.dot(x, mu.T)  # Shape: (n_points, n_distributions)

    # Clamp kappa to avoid overflow in exp, using config
    kappa = np.clip(kappa, 0, settings.data_generation.von_mises_fisher.kappa_clamp_max)
    # Get normalization constants
    C = von_mises_fisher_normalization_constant(kappa)

    # Use numerically stable exponentials
    # Subtract max for stability
    exp_arg = kappa * dot_products
    exp_arg_max = np.max(exp_arg, axis=0, keepdims=True)
    exp_stable = np.exp(exp_arg - exp_arg_max)
    if np.isscalar(kappa):
        pdf_values = C * exp_stable.flatten() * np.exp(exp_arg_max.flatten())
    else:
        pdf_values = C * exp_stable * np.exp(exp_arg_max)

    # Check for NaNs or infs
    if np.any(~np.isfinite(pdf_values)):
        raise RuntimeError("NaN or Inf encountered in von_mises_fisher_pdf. Check kappa and input values.")

    return pdf_values


def generate_random_von_mises_fisher_parameters(num_distributions, kappa_start, kappa_mean):
    """
    Generate random parameters for von-Mises Fisher distributions including mixture weights.
    kappa is sampled from a translated exponential distribution:
        kappa = kappa_start + Exp(kappa_mean - kappa_start)
    so that the mean of kappa is exactly kappa_mean.
    Parameters:
    -----------
    num_distributions : int
        Number of von-Mises Fisher distributions to generate parameters for
    kappa_start : float
        Starting value for kappa (translation of exponential)
    kappa_mean : float
        Desired mean of the sampled kappa values
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
    # Sample kappa from translated exponential so that mean is kappa_mean
    # kappa = kappa_start + Exp(kappa_mean - kappa_start)
    scale = kappa_mean - kappa_start
    kappa_values = kappa_start + np.random.exponential(scale=scale, size=num_distributions)
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
    from src.utils.distribution_generation_functions import fibonacci_sphere_points
    from src.core.volume_distribution_model import VolumeDistributionModel
    from aspire.downloader import emdb_2660
    
    print("von-Mises Fisher Distribution Pipeline")

    # Generate S2 quadrature points
    quadrature_points = fibonacci_sphere_points(n=settings.data_generation.von_mises_fisher.fibonacci_spiral_n)
    print(f"Generated {len(quadrature_points)} S2 quadrature points")
    
    # Generate von-Mises Fisher parameters
    num_vmf = settings.data_generation.von_mises_fisher.num_distributions
    mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
        num_vmf, kappa_start=settings.data_generation.von_mises_fisher.kappa_start, 
        kappa_mean=settings.data_generation.von_mises_fisher.kappa_mean
    )
    
    # Create SO(3) distribution and S2 weights from von Mises mixture
    rotations, rotation_weights, s2_weights = so3_distribution_from_von_mises_mixture(
        quadrature_points, mu_directions, kappa_values, mixture_weights, settings.data_generation.von_mises_fisher.num_in_plane_rotations
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
