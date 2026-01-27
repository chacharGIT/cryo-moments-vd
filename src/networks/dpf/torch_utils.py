import torch

from config.config import settings

def linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    Linear beta schedule as in Nochil & Dhariwal 2021 (DDPM).
    Returns a tensor of betas for each timestep.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s):
    """
    Cosine beta schedule as in Improved DDPM (Nochil & Dhariwal 2021).
    Returns a tensor of betas for each timestep.
    """
    import numpy as np
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(np.clip(betas, 0, 0.999)).float()


def rotate_s2_function_interpolated(grid_points, rho, R):
    """
    Rotate a scalar function on S^2 by R using nearest neighbour exponential kernel interpolation.

    Parameters
    ----------
    grid_points : torch.Tensor, shape (N, 3)
        S^2 grid points n_i (assumed unit-norm), dtype float.
    rho : torch.Tensor, shape (B, N)
        Batched scalar values rho_b(n_i) on the grid for each batch b.
    R : torch.Tensor, shape (3, 3)
        Rotation matrix in SO(3).

    Returns
    -------
    rho_rot : torch.Tensor, shape (B, N)
        Approximated values (rho_b ◦ R^T)(m_j) on the original grid.
    """
    k = settings.dpf.output_rotation_interpolation.num_k
    alphas = settings.dpf.output_rotation_interpolation.alphas

    if grid_points.ndim != 2 or grid_points.shape[1] != 3:
        raise ValueError(f"grid_points must have shape (N, 3), got {grid_points.shape}")
    N = grid_points.shape[0]
    if rho.ndim != 2 or rho.shape[1] != N:
        raise ValueError(f"rho must have shape (B, N), got {rho.shape} for N={grid_points.shape[0]}")

    device = grid_points.device
    dtype = grid_points.dtype
    rho = rho.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)

    N = grid_points.shape[0]
    k = min(k, N)

    # x_j = R^T @ x_j; with row vectors this is x^T_j @ R
    eval_points = grid_points @ R  # (N, 3)

    # Pointwise dot-products between eval points and grid points
    sims = grid_points @ eval_points.t()  # (N, N)

    # k nearest neighbours per column j (largest dot-products)
    vals, idx = torch.topk(sims, k=k, dim=0)  # vals, idx: (k, N)

    # Batched functions: rho shape (B, N)
    B = rho.shape[0]
    rho_rot_list = []

    # Expand rho and indices for gather once
    idx_exp = idx.unsqueeze(0).expand(B, -1, -1)        # (B, k, N)
    rho_exp = rho.unsqueeze(1).expand(-1, k, -1)        # (B, k, N)
    rho_neighbors = torch.gather(rho_exp, 2, idx_exp)   # (B, k, N)

    for alpha in alphas:
        # Exponential kernel weights and normalization over neighbours i
        weights = torch.exp(alpha * vals)                       # (k, N)
        weights_sum = weights.sum(dim=0, keepdim=True) + 1e-13
        weights = weights / weights_sum                         # (k, N)
        weights_b = weights.unsqueeze(0)                        # (1, k, N) -> broadcasts over B

        # Interpolated values at each m_j for this alpha
        rho_rot_alpha = (weights_b * rho_neighbors).sum(dim=1)  # (B, N)
        rho_rot_list.append(rho_rot_alpha)

    rho_rot = torch.stack(rho_rot_list, dim=0).mean(dim=0)  # (B, N)

    return rho_rot
