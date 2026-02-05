import matplotlib.pyplot as plt
import torch
import numpy as np
import zarr
import random
from config.config import settings
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.sample_generation import build_network_input, generate_vmf_mixture_on_s2
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule, beta_schedule
from src.networks.dpf.conditional_moment_encoder import CryoMomentsConditionalEncoder

# DPM-Solver++ integration
from packages.dpm_solver.dpm_solver_pytorch import model_wrapper, DPM_Solver

def diffusion_inference_process(model, points, x_t_init, t_start=1.0, langevin_step_size=1e-2,
                                cond_feat=None, use_guidance=False):
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
    solver_steps = settings.dpf.inference_solver_steps
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
            if cond_feat is not None:
                cond_score = model(context=context_encoding, queries=query_encoding, cond_feat=cond_feat).squeeze(-1)
            if cond_feat is None or use_guidance:
                score = model(context=context_encoding, queries=query_encoding).squeeze(-1)
        if cond_feat is not None and use_guidance:
            w = 1 # Guidance weight
            score = score + w * (cond_score - score)
        elif cond_feat is not None and not use_guidance:
            score = cond_score
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

def plot_s2_comparison(points, plot_dict, t=None, save_path=None):
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
    if save_path is None:
        save_path = 'outputs/tmp_figs/inference_comparison.png'
    # Accept points as [batch, n_points, 3] or [n_points, 3]
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if points.ndim == 3:
        points = points[0]
    arrs = []
    titles = []
    for k, v in plot_dict.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if v.ndim == 2 and v.shape[0] == 1:
            arrs.append(v[0])
        else:
            arrs.append(v)
        titles.append(k)
    n_cols = len(arrs)
    mask_top = points[:, 2] > 0
    mask_bottom = points[:, 2] <= 0
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
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
    plt.savefig(save_path)

if __name__ == "__main__":
    # Use CUDA if available and allowed by config
    device = torch.device(f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda and torch.cuda.is_available() else "cpu")
    model = S2ScoreNetwork().to(device)

    checkpoint = torch.load("./outputs/model_parameter_files/dpf_cond_test_2_epoch_19_batch_1600.pth", map_location=device)
    # Selection: plot single timestep or run full diffusion inference
    mode = "diffusion" # "single" or "diffusion"
    t_value = 0.88 # Starting time for diffusion inference/timestep for single mode (between 0 and 1, inference default = 1.0)
    use_conditional = True
    plot_conditional_vs_unconditional_comparison = True
    use_guidance = True
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
    if use_conditional:
        zarr_path = "/data/shachar/zarr_files/emdb_vmf_top_eigen.zarr"
        z = zarr.open(zarr_path, mode='r')
        num_examples = z['distribution_evaluations'].shape[0]
        idx = random.randint(0, num_examples - 1)
        func_data = torch.from_numpy(z['distribution_evaluations'][idx:idx+1]).float().to(device)
        eigen_images = torch.from_numpy(z['eigen_images'][idx:idx+1]).float().to(device)
        eigen_values = torch.from_numpy(z['eigen_values'][idx:idx+1]).float().to(device)
        first_moments = torch.from_numpy(z['first_moments'][idx:idx+1]).float().to(device)
        # Instantiate conditional encoder
        D = eigen_images.shape[1]
        cond_encoder = CryoMomentsConditionalEncoder(
            output_dim=settings.dpf.perceiver.latent_dim,
            num_queries_eig=settings.dpf.conditional_moment_encoder.num_queries_eig,
            unet_out_channels=settings.dpf.conditional_moment_encoder.first_moment_unet_out_channels,
            D=D
        ).to(device)
        cond_feat = cond_encoder(eigen_images.transpose(1, 2), first_moments, eigen_values)
    else:
        if use_zarr_example:
            import os
            zarr_path = os.path.join(settings.data_generation.zarr.save_dir, "vmf_mixtures_evaluations.zarr")
            z = zarr.open(zarr_path, mode='r')
            num_examples = z['func_data'].shape[0]
            idx = random.randint(0, num_examples - 1)
            func_data = torch.from_numpy(z['func_data'][idx:idx+1]).float().to(device)
        else:
            _, func_data, _ = generate_vmf_mixture_on_s2(batch_size=batch)
            func_data = torch.from_numpy(func_data).float().to(device)
        cond_feat = None
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
            # cond_feat is None if not using conditional model
            pred_score = model(context=context_encoding, queries=query_encoding, cond_feat=cond_feat)
        scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
        x0_est = (x_t + pred_score.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)
        true_score = -(x_t - torch.sqrt(scaling_t) * func_data) / (1 - scaling_t)
        if plot_conditional_vs_unconditional_comparison:
            with torch.no_grad():
                    pred_score_uncond = model(context=context_encoding, queries=query_encoding, cond_feat=None)
            x0_est_uncond = (x_t + pred_score_uncond.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)
            print("Conditional vs True difference (MSE):", torch.mean((func_data - x0_est) ** 2).item())
            print("Unconditional vs True difference (MSE):", torch.mean((func_data - x0_est_uncond) ** 2).item())
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                "x_t (noised)": x_t,
                "x_0 est (network, conditional)": x0_est,
                "x_0 est (network, unconditional)": x0_est_uncond
            }, t=t_value)
        else:        
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                "x_t (noised)": x_t,
                "x_0 est (network)": x0_est,
            }, t=t_value)
        
            
    elif mode == 'diffusion':
        t = torch.full((batch,), t_value, device=device)
        initial_image = q_sample(func_data, t)

        # cond_feat is None if not using conditional model
        x0_est = diffusion_inference_process(model, points, initial_image, t_start=t_value, cond_feat=cond_feat,
                                              use_guidance=use_guidance)
        if plot_conditional_vs_unconditional_comparison:
            x0_est_uncond = diffusion_inference_process(model, points, initial_image, t_start=t_value, cond_feat=None,
                                                         use_guidance=use_guidance)
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                f"Initial image (t={t_value})": initial_image,
                "x_0 est (diffusion, conditional)": x0_est,
                "x_0 est (diffusion, unconditional)": x0_est_uncond
            }, t=None)
            print("Conditional vs True difference (MSE):", torch.mean((func_data - x0_est) ** 2).item())
            print("Unconditional vs True difference (MSE):", torch.mean((func_data - x0_est_uncond) ** 2).item())
        else:
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                f"Initial image (t={t_value})": initial_image,
                "x_0 est (diffusion)": x0_est
            }, t=None)
    else:
        print("Invalid mode. Please choose 'single' or 'diffusion'.")
