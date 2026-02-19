import torch
import numpy as np
from aspire.utils.rotation import Rotation

from config.config import settings
from src.utils.distribution_generation_functions import cartesian_to_spherical, spherical_to_cartesian
from src.utils.von_mises_fisher_distributions import evaluate_vmf_mixture

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

def fit_vmf_mu(eval_points, values):
    """
    Fit the best mu for a vMF given values on eval_points.

    Parameters
    ----------
    eval_points : torch.Tensor, shape (N, 3)
        Points on S^2 where the vMF is evaluated (assumed unit-norm).
    values : torch.Tensor, shape (N,)
        Values at eval_points corresponding to the vMF (e.g., from a mixture of vMFs).
    Returns
    -------
    mu_hat : torch.Tensor, shape (3,)
        Estimated mean direction for the vMF that best fits the given values.
    """
    weighted_sum = (values[:, None] * eval_points).sum(dim=0)
    mu_hat = weighted_sum / (weighted_sum.norm() + 1e-13)
    return mu_hat

def rotate_s2_function_interpolated(grid_points, rho, R, mus, kappas, weights):
    """
    Rotate in plane invariant euler angle distribution by R,
      using nearest neighbour exponential kernel interpolation.

    Parameters
    ----------
    grid_points : torch.Tensor, shape (N, 3)
        S^2 grid points n_i (assumed unit-norm), dtype float.
    rho : torch.Tensor, shape (B, N)
        Batched scalar values rho_b(n_i) on the grid for each batch b.
    R : torch.Tensor, shape (3, 3)
        Rotation matrix in SO(3).
    mus : torch.Tensor, shape (B, num_distributions, 3)
        Mean directions for von Mises-Fisher distributions.
    kappas : torch.Tensor, shape (B, num_distributions)
        Concentration parameters for von Mises-Fisher distributions.
    weights : torch.Tensor, shape (B, num_distributions)
        Weights for von Mises-Fisher distributions.

    Returns
    -------
    rho_rot : torch.Tensor, shape (B, N)
        Approximated values (rho_b ◦ R^T)(m_j) on the original grid.
    """
    k = settings.dpf.output_rotation_interpolation.num_k
    alphas = settings.dpf.output_rotation_interpolation.alphas

    if isinstance(grid_points, torch.Tensor):
        grid_points_np = grid_points.cpu().numpy()
    else:
        grid_points_np = grid_points
    if grid_points_np.ndim != 2 or grid_points_np.shape[1] != 3:
        raise ValueError(f"grid_points must have shape (N, 3), got {grid_points.shape}")
    N = grid_points_np.shape[0]
    if rho.ndim != 2 or rho.shape[1] != N:
        raise ValueError(f"rho must have shape (B, N), got {rho.shape} for N={grid_points.shape[0]}")

    device = grid_points.device
    dtype = grid_points.dtype
    rho = rho.to(device=device, dtype=dtype)
    rho_normalization = rho.sum(dim=1, keepdim=True)
    B, num_distributions, _ = mus.shape
    N = grid_points.shape[0]
    k = min(k, N)

    # Convert grid points to eval_points using left rotation on euler angles
    #  with invariance to third euler angle.
    spherical_grid_points = cartesian_to_spherical(grid_points_np)  # shape (N, 2)
    phi = spherical_grid_points[:, 0]
    theta = spherical_grid_points[:, 1]
    euler_angles = np.stack([phi, theta, np.zeros_like(phi)], axis=1)  # (N, 3)
    rots = Rotation.from_euler(euler_angles, dtype=np.float64) # (N, 3, 3)
    rotated_matrices = np.matmul(R[None, :, :], rots.matrices)  # (N, 3, 3)
    rotated_eulers = Rotation(rotated_matrices).angles  # (N, 3)
    a = rotated_eulers[:, 0]
    b = rotated_eulers[:, 1]
    eval_points = spherical_to_cartesian(np.stack([a, b], axis=1))
    eval_points_torch = torch.from_numpy(eval_points).to(device=device, dtype=dtype)

    # For each batch/component, fit new mu on eval_points
    new_mus = torch.zeros_like(mus)
    for b in range(B):
        for i in range(num_distributions):
            # Evaluate vMF on grid points for this component
            vmf_values = evaluate_vmf_mixture(
                grid_points_np,  # (N, 3)
                mus[b, i:i+1].detach().cpu().numpy().astype(np.float64),      # (1, 3)
                kappas[b, i:i+1].detach().cpu().numpy().astype(np.float64),   # (1,)
                weights[b, i:i+1].detach().cpu().numpy().astype(np.float64)   # (1,)
            )  # (N,)
            # Fit new mu for this component on eval points
            vmf_values = torch.from_numpy(vmf_values).to(device=device, dtype=dtype)
            mu_hat = fit_vmf_mu(eval_points_torch, vmf_values)
            new_mus[b, i] = mu_hat

    # Build vMF mixture on eval_points with new mus (loop over batch; numpy impl is not batched)
    approx_eval_vmf_list = []
    for b in range(B):
        approx_b = evaluate_vmf_mixture(
            eval_points,    # (N, 3) numpy
            new_mus[b].detach().cpu().numpy().astype(np.float64),     # (num_distributions, 3) numpy
            kappas[b].detach().cpu().numpy().astype(np.float64),      # (num_distributions,) numpy
            weights[b].detach().cpu().numpy().astype(np.float64),     # (num_distributions,) numpy
        )  # (N,) numpy
        approx_b = approx_b / (approx_b.sum() + 1e-13) * rho_normalization[b].item()  # Normalize to have same sum as original rho
        approx_eval_vmf_list.append(torch.from_numpy(approx_b).to(device=device, dtype=dtype))
    approx_eval_vmf = torch.stack(approx_eval_vmf_list, dim=0)  # (B, N)
    diff = rho - approx_eval_vmf # (B, N)
    # Pointwise dot-products between eval points and grid points
    sims = eval_points_torch @ grid_points.t()  # (N, N)
    # k nearest neighbours per column j (largest dot-products)
    vals, idx = torch.topk(sims, k=k, dim=0)  # vals, idx: (k, N)

    # Batched functions: rho shape (B, N)
    B = rho.shape[0]
    diff_interpolated_list = []

    # Expand rho and indices for gather once
    idx_exp = idx.unsqueeze(0).expand(B, -1, -1)        # (B, k, N)
    diff_exp = diff.unsqueeze(1).expand(-1, k, -1)        # (B, k, N)
    diff_neighbors = torch.gather(diff_exp, 2, idx_exp)   # (B, k, N)

    for alpha in alphas:
        # Exponential kernel weights and normalization over neighbours i
        w = torch.exp(alpha * vals)  # (k, N)
        w = w / (w.sum(dim=0, keepdim=True) + 1e-13)
        diff_interpolated_list.append((w.unsqueeze(0) * diff_neighbors).sum(dim=1))  # (B, N)

    diff_grid_interpolated = torch.stack(diff_interpolated_list, dim=0).mean(dim=0)  # (B, N)

     # Analytic approx on grid_points using new_mus ----
    approx_grid_vmf_list = []
    for b in range(B):
        approx_grid_b = evaluate_vmf_mixture(
            grid_points_np,    # (N, 3) numpy
            new_mus[b].detach().cpu().numpy().astype(np.float64),     # (num_distributions, 3) numpy
            kappas[b].detach().cpu().numpy().astype(np.float64),      # (num_distributions,) numpy
            weights[b].detach().cpu().numpy().astype(np.float64),     # (num_distributions,) numpy
        )  # (N,) numpy
        approx_grid_b = approx_grid_b / (approx_grid_b.sum() + 1e-13) * rho_normalization[b].item()  # Normalize to have same sum as original rho
        approx_grid_vmf_list.append(torch.from_numpy(approx_grid_b).to(device=device, dtype=dtype))
    approx_grid_vmf = torch.stack(approx_grid_vmf_list, dim=0)  # (B, N)
    
    rho_rot_grid = approx_grid_vmf + diff_grid_interpolated
    rho_rot_grid = torch.clamp(rho_rot_grid, min=0)
    rho_rot_grid = rho_rot_grid / (rho_rot_grid.sum(dim=1, keepdim=True) + 1e-13) * rho_normalization
    """
    from src.inference.sample_from_dpf import plot_s2_comparison
    plot_s2_comparison(
        grid_points, 
        plot_dict={"rho on grid_points": rho[0],
                    "rho_rot on grid_points": rho_rot_grid[0],
                    "approx_grid_vmf on grid_points": approx_grid_vmf[0],
                    "diff_grid_interpolated on grid_points": diff_grid_interpolated[0]},
        save_path='outputs/tmp_figs/rotated_s2_function_comparison.png')

    # Plot on eval_points
    plot_s2_comparison(
        eval_points_torch,
          plot_dict={"rho_rot on eval_points": rho[0]},
          save_path='outputs/tmp_figs/rotated_s2_function_on_eval_points.png')
    """
    return rho_rot_grid