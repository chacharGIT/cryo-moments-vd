
import math
import torch
import numpy as np
from e3nn.o3 import spherical_harmonics

from src.utils.von_mises_fisher_distributions import generate_random_von_mises_fisher_parameters, evaluate_von_mises_fisher_mixture
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from config.config import settings

def fourier_encoding(x, num_frequencies):
    """
    General Fourier feature encoding.
    Args:
        x: Tensor of shape [batch] or [batch, 1], input values (float or int)
        num_frequencies: int, number of Fourier frequencies
    Returns:
        Tensor of shape [batch, num_frequencies * 2] (sin/cos interleaved)
    """
    x = x.unsqueeze(-1).float()  # [batch, 1]
    freqs = torch.arange(num_frequencies, device=x.device).float() * math.pi
    args = x * freqs  # [batch, num_frequencies]
    sin = torch.sin(args)
    cos = torch.cos(args)
    # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
    out = torch.stack((sin, cos), dim=-1)  # [batch, num_frequencies, 2]
    out = out.view(x.shape[0], -1)  # [batch, num_frequencies*2]
    return out

def spherical_harmonic_encoding(points, max_degree):
    """
    Compute real spherical harmonics for S² points using e3nn (batch-friendly).
    Args:
        points: torch.Tensor of shape [batch, n_points, 3] or [n_points, 3] (unit vectors)
        max_degree: int, maximum degree l (inclusive)
    Returns:
        torch.Tensor of shape [batch, n_points, num_harmonics] (real-valued)
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)  # [1, n_points, 3]
    points = torch.as_tensor(points, dtype=torch.float64, device=points.device)
    points = points / points.norm(dim=-1, keepdim=True)
    sph = spherical_harmonics(list(range(1, max_degree + 1)), points.reshape(-1, 3), normalize=True, normalization='component')
    sph = sph.view(points.shape[0], points.shape[1], -1)  # [batch, n_points, num_harmonics]
    return sph

def build_network_input(points, t, func_data, time_enc_len, sph_enc_len):
    """
    Build the input tensor for the Perceiver network for a batch of samples.
    Args:
        points: [batch, n_points, 3] - S² points for each sample in the batch
        t: [batch] or [batch, 1] - diffusion times for each sample
        func_data: [batch, n_points, d] - function values for each sample
        time_enc_len: int - number of Fourier encoding frequencies
        sph_enc_len: int - max degree for spherical harmonics encoding
    Returns:
        [batch, n_points, time_enc_len*2 + sph_enc_len*4 + d]
    """
    # Ensure batch dimension
    if points.dim() == 2:
        points = points.unsqueeze(0)  # [1, n_points, 3]
    if func_data.dim() == 2:
        func_data = func_data.unsqueeze(-1)  # [batch, n_points, 1]
    batch, n_points, _ = points.shape
    # Time encoding: [batch, time_enc_len*2] -> [batch, n_points, time_enc_len*2]
    t = t if t.dim() > 1 else t.unsqueeze(-1)  # [batch, 1]
    t_enc = fourier_encoding(t.squeeze(-1), time_enc_len).unsqueeze(1).expand(-1, n_points, -1)
    # Spherical encoding: [batch, n_points, sph_enc_dim]
    sph_enc = spherical_harmonic_encoding(points, sph_enc_len)
    # If sph_enc batch size is 1 but t_enc and func_data have batch > 1, repeat sph_enc
    if sph_enc.shape[0] == 1 and t_enc.shape[0] > 1:
        sph_enc = sph_enc.repeat(t_enc.shape[0], 1, 1)
    # Concatenate all features
    net_input = torch.cat([t_enc, sph_enc, func_data], dim=-1)
    return net_input


# --- Shared helper for vMF mixture generation on S² ---
def generate_vmf_mixture_on_s2(batch_size=1):
    """
    Generate a batch of mixtures of von Mises-Fisher distributions on S2 and evaluate the mixtures on sampled points.
    Uses config values for all parameters.
    Args:
        batch_size: int, number of mixtures to generate in the batch
    Returns:
        points: np.ndarray, shape (batch_size, n_quadrature, 3), points on S2
        mixture_pdf: np.ndarray, shape (batch_size, n_quadrature), mixture evaluated at points
        mixture_params: list of dicts, each containing 'mu_directions', 'kappa_values', 'mixture_weights'
    """
    vmf_cfg = settings.data_generation.von_mises_fisher
    num_components = vmf_cfg.num_distributions
    kappa_start = vmf_cfg.kappa_start
    kappa_mean = vmf_cfg.kappa_mean
    n_quadrature = vmf_cfg.fibonacci_spiral_n

    points_batch = []
    pdf_batch = []
    params_batch = []
    for _ in range(batch_size):
        mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
            num_components, kappa_start, kappa_mean
        )
        points = fibonacci_sphere_points(n_quadrature)  # numpy array [n_quadrature, 3]
        mixture_pdf = evaluate_von_mises_fisher_mixture(
            points, mu_directions, kappa_values, mixture_weights
        )  # mixture_pdf: numpy array [n_quadrature]
        points_batch.append(points)
        pdf_batch.append(mixture_pdf)
        params_batch.append({
            'mu_directions': mu_directions,
            'kappa_values': kappa_values,
            'mixture_weights': mixture_weights
        })
    points_batch = np.stack(points_batch, axis=0)  # [batch, n_quadrature, 3]
    pdf_batch = np.stack(pdf_batch, axis=0)        # [batch, n_quadrature]
    return points_batch, pdf_batch, params_batch


if __name__ == "__main__":
    print("Testing direct calls to generate_vmf_mixture_on_s2...")
    for i in range(3):
        points, values, mixture_params = generate_vmf_mixture_on_s2()
        print(f"Call {i+1}: points shape {points.shape}, values shape {values.shape}")
    print("Test complete.")

    # Test fourier_encoding
    import torch
    x = torch.linspace(0, 1, 5)
    fe = fourier_encoding(x, 4)
    print("\nFourier encoding test:")
    print("Input:", x)
    print("Encoded shape:", fe.shape)
    print("Encoded:", fe)

    # Test spherical_harmonic_encoding (real harmonics)
    batch = 2
    n_points = 3
    max_degree = 4
    pts = torch.randn(batch, n_points, 3)
    sph = spherical_harmonic_encoding(pts, max_degree)
    print("\nSpherical harmonic encoding test (real):")
    print("Input shape:", pts.shape)
    print("Output shape:", sph.shape)
    print("First point, first harmonics:", sph[0,0,:(max_degree+1)**2])
