import matplotlib.pyplot as plt
import torch
import numpy as np
from config.config import settings
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.sample_generation import build_network_input, generate_vmf_mixture_on_s2
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule, beta_schedule
# DPM-Solver++ integration
from packages.dpm_solver.dpm_solver_pytorch import model_wrapper, DPM_Solver

def diffusion_inference_process(model, points, x_t_init, t_start=1.0, langevin_step_size=1e-2, initial_langevin_steps=3):
    """
    Start with a few Langevin steps, then alternate blocks of DPM-Solver++ (order 3, steps 3) and Langevin steps.

    Args:
        model: Trained diffusion model (score network)
        points: [1, n_points, 3] torch.Tensor (S2 points)
        x_t_init: Initial image data, shape [1, n_points]
        t_start: Starting timestep for the reverse process (default 1.0).
        langevin_step_size: Step size for Langevin update.
        initial_langevin_steps: Number of initial Langevin steps before alternating (default 3).

    Returns:
        x: [1, n_points] torch.Tensor, final solution at t=0
    """
    device = torch.device(f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda and torch.cuda.is_available() else "cpu")
    # Accept float64 input for high-precision integration
    x = x_t_init.to(device=device, dtype=torch.float64)
    langevin_steps = settings.dpf.inference_langevin_steps
    solver_steps = settings.dpf.inference_solver_timesteps
    t_start = min(t_start, 1 - 1e-3)
    # Langevin: times from t_start to inference_langevin_t_end
    langevin_t_start = t_start
    langevin_t_end = settings.dpf.inference_langevin_t_end
    langevin_t_vals = torch.linspace(langevin_t_start, langevin_t_end, langevin_steps, device=device)
    # DPM-Solver++: times from inference_langevin_t_end to 1e-3
    solver_t_start = langevin_t_end
    solver_t_end = 1e-3
    solver_t_vals = torch.linspace(solver_t_start, solver_t_end, solver_steps+1, device=device)

    def score_model_wrapper(x, t):
        # Cast x and t to float32 for model, keep float64 for integration
        x_model = x.to(torch.float32)
        t_model = t.to(torch.float32)
        context_encoding = build_network_input(
            points, t_model, x_model,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        context_encoding = context_encoding.to(model.parameters().__next__().dtype)
        query_encoding = context_encoding
        with torch.no_grad():
            score = model(context=context_encoding, queries=query_encoding).squeeze(-1)
        # Cast score back to float64 for solver
        return score.to(torch.float64)

    def alpha_fn(t):
        return torch.sqrt(cosine_signal_scaling_schedule(t))
    def sigma_fn(t):
        return torch.sqrt(1 - cosine_signal_scaling_schedule(t))
    
    class CustomCosineSchedule:
        def __init__(self):
            self.T = 1.0
            self.schedule = 'cosine'
        def marginal_alpha(self, t):
            return alpha_fn(t)
        def marginal_std(self, t):
            return sigma_fn(t)
        def marginal_log_mean_coeff(self, t):
            return torch.log(alpha_fn(t))
        def marginal_lambda(self, t):
            return torch.log(alpha_fn(t)) - torch.log(sigma_fn(t))
    noise_schedule = CustomCosineSchedule()
    model_fn_ode = model_wrapper(score_model_wrapper, noise_schedule, model_type="score")
    dpm_solver_ode = DPM_Solver(model_fn_ode, noise_schedule, algorithm_type="dpmsolver++")

    x_curr = x
    # Initial Langevin steps
    for i in range(langevin_steps):
        t_langevin = langevin_t_vals[i]
        score = score_model_wrapper(x_curr, t_langevin)
        noise = torch.randn_like(x_curr) * torch.sqrt(torch.tensor(langevin_step_size, dtype=x_curr.dtype, device=device))
        x_curr = x_curr + 0.5 * langevin_step_size * score + noise
    # DPM-Solver++ for all remaining steps in one call
    t0 = solver_t_vals[0]
    tN = solver_t_vals[-1]
    x_curr = dpm_solver_ode.sample(x_curr, steps=solver_steps, t_start=t0.item(), t_end=tN.item(), order=min(3, solver_steps), skip_type='time_uniform', method='multistep')
    return x_curr

def plot_s2_comparison(points, plot_dict, t=None):
    """
    Visualizes multiple S2 functions on the sphere, split by hemisphere.

    Args:
        points (torch.Tensor or np.ndarray): S2 coordinates, shape [batch, n_points, 3] or [n_points, 3].
            Only the first batch is plotted if batch dimension is present.
        plot_dict (dict): Dictionary mapping plot titles to arrays/tensors of shape [batch, n_points] or [n_points].
            Each entry is visualized as a separate column.
        t (float, optional): Diffusion time step, shown in the figure title if provided.

    The function splits the sphere into two hemispheres (z>0 and z<=0) and plots each function for both hemispheres.
    Color represents function value at each point. Results are saved to 'outputs/tmp_figs/inference_comparison.png'.
    """
    # Accept points as [batch, n_points, 3] or [n_points, 3]
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if points.ndim == 3:
        points = points[0]
    arrs = []
    titles = []
    for k, v in plot_dict.items():
        arrs.append(v[0].detach().cpu().numpy())
        titles.append(k)
    n_cols = len(arrs)
    mask_top = points[:, 2] > 0
    mask_bottom = points[:, 2] <= 0
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))
    if t is not None:
        fig.suptitle(f"t = {float(t):.4f}", fontsize=20)
    for row, mask in enumerate([mask_top, mask_bottom]):
        for col, (arr, title) in enumerate(zip(arrs, titles)):
            sc = axes[row, col].scatter(points[mask, 0], points[mask, 1], c=arr[mask], cmap='viridis', alpha=0.8)
            axes[row, col].set_title(title + (" (z>0)" if row==0 else " (z<0)"))
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            axes[row, col].set_aspect('equal')
            plt.colorbar(sc, ax=axes[row, col])
    plt.tight_layout()
    plt.savefig('outputs/tmp_figs/inference_comparison.png')

if __name__ == "__main__":
    # Use CUDA if available and allowed by config
    device = torch.device(f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda and torch.cuda.is_available() else "cpu")
    model = S2ScoreNetwork().to(device)

    checkpoint = torch.load("./outputs/model_parameter_files/dpf_test_5_epoch_170.pth", map_location=device)
    # Selection: plot single timestep or run full diffusion inference
    mode = "diffusion" # "single" or "diffusion"
    t_value = 1 # Starting time for diffusion inference/timestep for single mode (between 0 and 1, inference default = 1.0)
    use_zarr_example = False

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(torch.float64)
    model.eval()

    # Prepare S2 points
    n_points = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    from src.utils.distribution_generation_functions import fibonacci_sphere_points
    points = fibonacci_sphere_points(n_points)
    points = torch.from_numpy(points).float().unsqueeze(0).to(device)

    # Common function generation code
    batch = 1
    n_points = points.shape[1]
    if use_zarr_example:
        import zarr
        import os
        import random
        zarr_path = os.path.join(settings.data_generation.zarr.save_dir, "vmf_mixtures_evaluations.zarr")
        z = zarr.open(zarr_path, mode='r')
        num_examples = z['func_data'].shape[0]
        idx = random.randint(0, num_examples - 1)
        func_data = torch.from_numpy(z['func_data'][idx:idx+1]).float().to(device)
    else:
        _, func_data, _ = generate_vmf_mixture_on_s2(batch_size=batch)
        func_data = torch.from_numpy(func_data).float().to(device)
    func_data = func_data / (func_data.std(dim=1, keepdim=True))

    if mode == 'single':
        t = torch.full((batch,), t_value, device=device)
        x_t = q_sample(func_data, t)
        context_encoding = build_network_input(
            points, t, x_t,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        context_encoding = context_encoding.to(torch.float64)
        query_encoding = context_encoding.to(torch.float64)
        with torch.no_grad():
            pred_score = model(context=context_encoding, queries=query_encoding)
        scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
        x0_est = (x_t + pred_score.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)
        true_score = -(x_t - torch.sqrt(scaling_t) * func_data) / (1 - scaling_t)
        print(f'true_score stats: min={true_score.min().item():.4g}, max={true_score.max().item():.4g}, mean={true_score.mean().item():.4g}, std={true_score.std().item():.4g}')
        print(f'pred_score stats: min={pred_score.min().item():.4g}, max={pred_score.max().item():.4g}, mean={pred_score.mean().item():.4g}, std={pred_score.std().item():.4g}')
        plot_s2_comparison(points, {
            "x_0 (clean)": func_data,
            "x_t (noised)": x_t,
            "x_0 est (network)": x0_est,
            "true score": true_score,
            "pred score": pred_score.squeeze(-1)
        }, t=t_value)
    elif mode == 'diffusion':
        t = torch.full((batch,), t_value, device=device)
        initial_image = q_sample(func_data, t)
        x0_est = diffusion_inference_process(model, points, initial_image, t_start=t_value)
        plot_s2_comparison(points, {
            "x_0 (clean)": func_data,
            f"Initial image (t={t_value})": initial_image,
            "x_0 est (diffusion)": x0_est
        }, t=None)
    else:
        print("Invalid mode. Please choose 'single' or 'diffusion'.")
