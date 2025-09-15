import torch
from config.config import settings

def von_mises_fisher_normalization_constant(kappa, kappa_clamp_max=None):
    """
    Calculate the normalization constant for von-Mises Fisher distribution on S^(2).
    
    Parameters:
    -----------
    kappa : float or torch.Tensor
        Concentration parameter(s)
    
    Returns:
    --------
    C : float or torch.Tensor
        Normalization constant(s)
    """
    kappa_clamp_max = float(settings.data_generation.von_mises_fisher.kappa_clamp_max)
    kappa = torch.as_tensor(kappa, dtype=torch.float64)
    kappa = torch.clamp(kappa, 0, kappa_clamp_max)
    norm = kappa / (4 * torch.pi * (torch.sinh(kappa) + 1e-12))
    return norm

def von_mises_fisher_pdf(x, mu, kappa):
    """
    Evaluate the von-Mises Fisher probability density function on S2.
    
    Parameters:
    -----------
    x : torch.Tensor
        Points on S2 in Cartesian coordinates, shape (n_points, 3)
    mu : torch.Tensor
        Mean direction(s) on S2, shape (3,) or (n_distributions, 3)
    kappa : float or torch.Tensor
        Concentration parameter(s), shape () or (n_distributions,)
    
    Returns:
    --------
    pdf_values : torch.Tensor
        PDF values at each point for each distribution
        Shape (n_points,) if single distribution, (n_points, n_distributions) if multiple
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    mu = torch.as_tensor(mu, dtype=torch.float64)
    kappa = torch.as_tensor(kappa, dtype=torch.float64)
    # Ensure x,mu are 2D
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    dot_products = torch.matmul(x, mu.T)  # Shape: (n_points, n_distributions)
    kappa_clamp_max = float(settings.data_generation.von_mises_fisher.kappa_clamp_max)
    kappa = torch.clamp(kappa, 0, kappa_clamp_max)
    C = von_mises_fisher_normalization_constant(kappa)
    # Use numerically stable exponentials
    # Subtract max for stability
    exp_arg = kappa * dot_products
    exp_arg_max = torch.amax(exp_arg, dim=0, keepdim=True)
    exp_stable = torch.exp(exp_arg - exp_arg_max)
    if C.ndim == 0:
        C = C.unsqueeze(0)
    if exp_stable.ndim == 1:
        exp_stable = exp_stable.unsqueeze(-1)
    if exp_arg_max.ndim == 1:
        exp_arg_max = exp_arg_max.unsqueeze(0)
    pdf_values = C * exp_stable * torch.exp(exp_arg_max)
    # Check for NaNs or infs
    if not torch.all(torch.isfinite(pdf_values)):
        print("[DEBUG] NaN/Inf in vMF PDF: kappa=", kappa)
        print("[DEBUG] norm=", C)
        print("[DEBUG] exp_arg=", exp_arg)
        print("[DEBUG] exp_stable=", exp_stable)
        raise RuntimeError("NaN or Inf encountered in von_mises_fisher_pdf. Check kappa and input values.")
    return pdf_values.squeeze()

def generate_random_von_mises_fisher_parameters(num_distributions, kappa_start, kappa_mean):
    """
    Generate random parameters for von-Mises Fisher distributions including mixture weights (PyTorch version).
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
    device : torch.device (from config)
        Device for returned tensors (always taken from config)
    dtype : torch.dtype (always torch.float32)
        Data type for returned tensors (always float32)
    Returns:
    --------
    mu_directions : torch.Tensor
        Random mean directions on S2, shape (num_distributions, 3)
    kappa_values : torch.Tensor
        Random concentration parameters, shape (num_distributions,)
    mixture_weights : torch.Tensor
        Random mixture weights that sum to 1, shape (num_distributions,)
    """
    device = torch.device(f'cuda:{settings.device.cuda_device}' if
                           settings.device.use_cuda and torch.cuda.is_available() else 'cpu')    
    dtype = torch.float64
    # Generate random mean directions by sampling from 3D Gaussian and normalizing
    mu_directions = torch.randn(num_distributions, 3, device=device, dtype=dtype)
    mu_directions = mu_directions / mu_directions.norm(dim=1, keepdim=True)
    # Sample kappa from translated exponential so that mean is kappa_mean
    scale = kappa_mean - kappa_start
    kappa_values = kappa_start + torch.distributions.Exponential(1.0/scale).sample((num_distributions,)).to(device=device, dtype=dtype)
    # Generate random mixture weights that sum to 1
    mixture_weights = torch.rand(num_distributions, device=device, dtype=dtype)
    mixture_weights = mixture_weights / mixture_weights.sum()
    return mu_directions, kappa_values, mixture_weights

def evaluate_von_mises_fisher_mixture(quadrature_points, mu_directions, kappa_values, mixture_weights):
    """
    Evaluate a linear combination (mixture) of von-Mises Fisher distributions at quadrature points (PyTorch version).
    
    Parameters:
    -----------
    quadrature_points : torch.Tensor
        Points on S2 in Cartesian coordinates, shape (n_points, 3)
    mu_directions : torch.Tensor
        Mean directions for each distribution, shape (num_distributions, 3)
    kappa_values : torch.Tensor
        Concentration parameters, shape (num_distributions,)
    mixture_weights : torch.Tensor
        Weights for linear combination, shape (num_distributions,)
    
    Returns:
    --------
    mixture_pdf : torch.Tensor
        PDF values of the mixture at each quadrature point, shape (n_points,)
    """
    quadrature_points = torch.as_tensor(quadrature_points)
    mu_directions = torch.as_tensor(mu_directions)
    kappa_values = torch.as_tensor(kappa_values)
    mixture_weights = torch.as_tensor(mixture_weights)
    # Ensure quadrature_points is 2D
    if quadrature_points.ndim == 1:
        quadrature_points = quadrature_points.unsqueeze(0)
    # Evaluate each von-Mises Fisher distribution
    pdf_values = von_mises_fisher_pdf(quadrature_points, mu_directions, kappa_values)
    # If single distribution, reshape for consistency
    if pdf_values.ndim == 1 and mixture_weights.numel() > 1:
        pdf_values = pdf_values.unsqueeze(1)
    # Calculate weighted sum
    if pdf_values.ndim == 1:
        mixture_pdf = mixture_weights[0] * pdf_values
    else:
        mixture_pdf = torch.matmul(pdf_values, mixture_weights)
    return mixture_pdf

def so3_distribution_from_von_mises_mixture(quadrature_points, mu_directions, kappa_values,
                                             mixture_weights, num_in_plane_rotations):
    """
    Given S2 quadrature points, von Mises mixture parameters, and in-plane rotations, return SO(3) rotations and weights, and S2 weights (PyTorch version).
    
    Parameters:
    -----------
    quadrature_points : torch.Tensor
        Points on S2 in Cartesian coordinates, shape (n_points, 3)
    mu_directions : torch.Tensor
        Mean directions for each distribution, shape (num_distributions, 3)
    kappa_values : torch.Tensor
        Concentration parameters, shape (num_distributions,)
    mixture_weights : torch.Tensor
        Mixture weights, shape (num_distributions,)
    num_in_plane_rotations : int
        Number of in-plane rotations for SO(3) construction
    
    Returns:
    --------
    rotations : torch.Tensor or np.ndarray
        SO(3) rotations (format depends on downstream code)
    distribution : torch.Tensor or np.ndarray
        SO(3) distribution weights (format depends on downstream code)
    """
    from src.utils.distribution_generation_functions import cartesian_to_spherical, create_in_plane_invariant_distribution
    # Ensure all inputs are torch tensors
    quadrature_points = torch.as_tensor(quadrature_points)
    mu_directions = torch.as_tensor(mu_directions)
    kappa_values = torch.as_tensor(kappa_values)
    mixture_weights = torch.as_tensor(mixture_weights)

    s2_pdf_values = evaluate_von_mises_fisher_mixture(
        quadrature_points, mu_directions, kappa_values, mixture_weights
    )
    s2_weights = s2_pdf_values / s2_pdf_values.sum()
    # Convert to numpy for downstream functions
    quadrature_points_np = quadrature_points.cpu().numpy()
    s2_weights_np = s2_weights.cpu().numpy()
    s2_spherical_coords = cartesian_to_spherical(quadrature_points_np)
    rotations, distribution = create_in_plane_invariant_distribution(
        s2_spherical_coords, s2_weights_np, num_in_plane_rotations=num_in_plane_rotations, is_s2_uniform=False
    )
    return rotations, distribution, s2_weights


if __name__ == "__main__":
    from src.utils.distribution_generation_functions import fibonacci_sphere_points
    from src.core.volume_distribution_model import VolumeDistributionModel
    from aspire.downloader import emdb_2660
    import numpy as np

    print("von-Mises Fisher Distribution Pipeline (PyTorch)")

    # Generate S2 quadrature points (still numpy)
    quadrature_points = fibonacci_sphere_points(n=settings.data_generation.von_mises_fisher.fibonacci_spiral_n)
    print(f"Generated {len(quadrature_points)} S2 quadrature points")

    # Generate von-Mises Fisher parameters (torch)
    num_vmf = settings.data_generation.von_mises_fisher.num_distributions
    mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
        num_vmf, kappa_start=settings.data_generation.von_mises_fisher.kappa_start,
        kappa_mean=settings.data_generation.von_mises_fisher.kappa_mean
    )

    # Convert quadrature_points to torch
    quadrature_points_torch = torch.tensor(quadrature_points, dtype=torch.float32, device=mu_directions.device)

    # Create SO(3) distribution and S2 weights from von Mises mixture (torch)
    rotations, rotation_weights, s2_weights = so3_distribution_from_von_mises_mixture(
        quadrature_points_torch, mu_directions, kappa_values, mixture_weights, settings.data_generation.von_mises_fisher.num_in_plane_rotations
    )

    print(f"Created {num_vmf} von-Mises Fisher mixture, kappa range: ["
          f"{kappa_values.min().item():.2f}, {kappa_values.max().item():.2f}]")
    print(f"Created SO(3) distribution: {len(rotations)} rotations")
