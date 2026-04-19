import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np
import zarr
import random
from tqdm import tqdm
from aspire.utils.rotation import Rotation
from packages.dpm_solver.dpm_solver_pytorch import model_wrapper, DPM_Solver

from config.config import settings
from src.utils.distribution_generation_functions import fibonacci_sphere_points, s2_points_to_in_plane_euler_angles
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.sample_generation import build_network_input, generate_vmf_mixture_on_s2
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule, beta_schedule
from src.networks.dpf.conditional_moment_encoder import CryoMomentsConditionalEncoder
from src.networks.dpf.torch_utils import rotate_s2_function_interpolated
from src.networks.dpf.loss import dpf_score_matching_loss
from src.training.train_dpf import load_filtered_state_dict_compat, debug_compare_items

def best_sign_rotation_target_for_predscore(points, func_base, R_inv, mus, 
                                            kappas, weights, x_t, scaling_t, pred_score):
    """
    Pick the best of the 4 sign-ambiguity rotations by comparing candidate true-scores to pred_score (batch_size=1).

    Parameters:
        points (torch.Tensor): S2 points, shape [N, 3].
        func_base (torch.Tensor): Unrotated clean function, shape [1, N].
        R_inv (np.ndarray): Base inverse rotation matrix, shape [3, 3].
        mus: Mixture means passed to rotate_s2_function_interpolated.
        kappas: Mixture kappas passed to rotate_s2_function_interpolated.
        weights: Mixture weights passed to rotate_s2_function_interpolated.
        x_t (torch.Tensor): Noisy sample generated from candidate 0, shape [1, N].
        scaling_t (torch.Tensor): Alpha(t) scaling broadcastable to x_t, shape [1, 1] or [1, ...].
        pred_score (torch.Tensor): Model score prediction, shape [1, N] or [1, N, 1].

    Returns:
        best_func (torch.Tensor): Selected rotated batch function, shape [1, N].
        best_true_score (torch.Tensor): True score computed from (x_t, best_func), shape [1, N].
        best_loss (torch.Tensor): Scalar loss value for the selected candidate.
        best_idx (int): Index in {0,1,2,3} of the chosen sign configuration.
    """
    sign_configs = [
    (1,  1,  1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    ]

    # pred_score could be [1, N, 1] or [1, N]
    if pred_score.dim() == 3 and pred_score.shape[-1] == 1:
        pred_score = pred_score.squeeze(-1)

    func_candidates = []
    true_score_candidates = []
    losses = []
    for signs in sign_configs:
        S = np.diag(signs).astype(np.float32)
        R_inv_variant = S @ R_inv

        bf = rotate_s2_function_interpolated(points, func_base, R_inv_variant, mus, kappas, weights)
        bf = bf / (bf.std(dim=1, keepdim=True) + 1e-8)

        ts = -(x_t - torch.sqrt(scaling_t) * bf) / (1 - scaling_t)

        func_candidates.append(bf)
        true_score_candidates.append(ts)
        losses.append(dpf_score_matching_loss(pred_score, ts))  # scalar

    losses = torch.stack(losses)  # [4]
    best_idx = int(torch.argmin(losses).item())
    return func_candidates[best_idx], true_score_candidates[best_idx], losses[best_idx], best_idx

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

    def score_model_wrapper(x, t, cond_feat=cond_feat):
        # Cast x and t to float32 for model, keep float64 for integration
        x_model = x.to(torch.float32)
        t_model = t.to(torch.float32)
        context_encoding = build_network_input(
            points, t_model, x_model,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        model_dtype = next(model.parameters()).dtype
        context_encoding = context_encoding.to(dtype=model_dtype)
        query_encoding = context_encoding
        cond_feat = None if cond_feat is None else cond_feat.to(dtype=model_dtype)

        with torch.no_grad():
            if cond_feat is not None:
                cond_score = model(context=context_encoding, queries=query_encoding, cond_feat=cond_feat).squeeze(-1)
            if cond_feat is None or use_guidance:
                score = model(context=context_encoding, queries=query_encoding).squeeze(-1)
        if cond_feat is not None and use_guidance:
            w = 0.3 # Guidance weight
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

# Checkpoint key adjustment (training uses DDP, inference does not)
def _adjust_checkpoint_keys(ckpt_state, model_state):
    if all(k.startswith('module.') for k in model_state.keys()) and not all(k.startswith('module.') for k in ckpt_state.keys()):
        ckpt_state = {'module.' + k if not k.startswith('module.') else k: v for k, v in ckpt_state.items()}
    elif not all(k.startswith('module.') for k in model_state.keys()) and all(k.startswith('module.') for k in ckpt_state.keys()):
        ckpt_state = {k.replace('module.', '', 1): v for k, v in ckpt_state.items()}
    return ckpt_state

def best_rotated_func_data(x0_est, func_data, mus, kappas, weights, points, num_in_plane_rotations):
    """
    Find the rotation of `func_data` that best matches `x0_est` in MSE.

    Args:
        x0_est (torch.Tensor): Estimated clean signal, shape [1, N] or [1, N, 1].
        func_data (torch.Tensor): Reference clean signal to rotate, shape [1, N].
        mus: Mean directions for `rotate_s2_function_interpolated`.
        kappas: Concentration parameters for `rotate_s2_function_interpolated`.
        weights: Mixture weights for `rotate_s2_function_interpolated`.
        points (torch.Tensor or np.ndarray): Sphere points, shape [N, 3].
        num_in_plane_rotations (int): Number of in-plane rotations per input point.

    Returns:
        best_func_data_rot (torch.Tensor): Rotated `func_data` with lowest MSE to `x0_est`.
        best_loss (torch.Tensor): Best MSE value.
        best_est_rot (np.ndarray): Rotation matrix of the best candidate.
    """
    if x0_est.dim() == 3 and x0_est.shape[-1] == 1:
        x0_est = x0_est.squeeze(-1)

    euler_angles = s2_points_to_in_plane_euler_angles(
        fibonacci_sphere_points(150),
        num_in_plane_rotations=num_in_plane_rotations,
        random_start=True,
    )
    rotations = Rotation.from_euler(euler_angles, dtype=np.float32)

    best_loss = None
    best_idx = None
    best_est_rot = None
    for i, R in enumerate(tqdm(rotations.matrices, desc="Determining best rotation")):
        func_data_rot = rotate_s2_function_interpolated(
                points, func_data, R, mus, kappas, weights, use_residual_interpolation=False
            )
        loss = torch.mean((x0_est - func_data_rot) ** 2)

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_est_rot = R
            best_func_data_rot = func_data_rot

    return best_func_data_rot, best_loss, best_est_rot

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

    def _minmax(vals):
        vals = vals[np.isfinite(vals)]
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax > 0 and vmin < 0:
            vmin = 0
        return vmin, vmax

    mask_top = points[:, 2] > 0
    mask_bottom = points[:, 2] <= 0
    top_vals = np.concatenate([a[mask_top].reshape(-1) for a in arrs], axis=0)
    bot_vals = np.concatenate([a[mask_bottom].reshape(-1) for a in arrs], axis=0)
    vmin_top, vmax_top = _minmax(top_vals)
    vmin_bot, vmax_bot = _minmax(bot_vals)

    gamma=0.5
    norm_top = mcolors.PowerNorm(gamma=gamma, vmin=vmin_top, vmax=vmax_top)
    norm_bot = mcolors.PowerNorm(gamma=gamma, vmin=vmin_bot, vmax=vmax_bot)

    n_cols = len(arrs)
    mask_top = points[:, 2] > 0
    mask_bottom = points[:, 2] <= 0
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    if t is not None:
        fig.suptitle(f"t = {float(t):.4f}", fontsize=20)

    for row, (mask, norm) in enumerate([(mask_top, norm_top), (mask_bottom, norm_bot)]):
        for col, (arr, title) in enumerate(zip(arrs, titles)):
            ax = axes[row, col]
            sc = ax.scatter(points[mask, 0], points[mask, 1], c=arr[mask],
                            cmap='viridis', alpha=0.8, norm=norm)

            ax.set_title(f"{title}\n" + ("(z>0)" if row == 0 else "(z<0)"), fontsize=15)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', labelsize=20)

            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(save_path)

def main(return_mse=True):
    # Selection: plot single timestep or run full diffusion inference
    mode = "diffusion" # "single" or "diffusion"
    t_value = 1 # Starting time for diffusion inference/timestep for single mode (between 0 and 1, inference default = 1.0)
    plot_conditional_vs_unconditional_comparison = False
    use_guidance = True
    use_zarr_example = False
    compare_best_rotation = True

    device = torch.device(f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda and torch.cuda.is_available() else "cpu")
    use_conditional = settings.dpf.conditional_moment_encoder.use_encoder
    separate_fourier_modes = settings.data_generation.separate_fourier_modes
    batch_size = 1

    score_model = S2ScoreNetwork().to(device)
    if use_conditional:
        emdb_volumes_rotations = np.load(
                settings.data_generation.emdb.volume_rotations_path,
                allow_pickle=True
            ).item()
        if separate_fourier_modes:
            from src.data.emdb_vmf_separated_subspace_moments_dataloader import create_subspace_moments_dataloader
            #from src.networks.dpf.conditional_separated_moment_encoder import CryoMomentsConditionalEncoder
            #cond_encoder = CryoMomentsConditionalEncoder().to(device)
            from src.networks.dpf.simple_conditional_moment_encoder import DictComplexToBert
            cond_encoder = DictComplexToBert().to(device)
            zarr_path = settings.data_generation.zarr.separated_modes_data_save_path
            dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=batch_size,
                                                                shuffle=True, debug=True, mode='train', device=device, single_volume_id="emd_19777")        
    checkpoint = torch.load("./outputs/model_parameter_files/dpf_cond_simple_3_batch_2400.pth", map_location=device, weights_only=False)
    load_filtered_state_dict_compat(score_model, checkpoint["model_state_dict"], verbose=True)
    score_model = score_model.to(torch.float32)
    score_model.eval()
    if use_conditional and ("cond_encoder_state_dict" in checkpoint):
        load_filtered_state_dict_compat(cond_encoder, checkpoint["cond_encoder_state_dict"], verbose=True)
        cond_encoder = cond_encoder
        cond_encoder.eval()

    # Prepare S2 points
    n_points = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    points = fibonacci_sphere_points(n_points)
    points = torch.from_numpy(points).float().to(device)

    if use_conditional:
        if separate_fourier_modes:
            batch = next(iter(dataloader))
            batch_func_base = batch['distribution_evaluations'].float().to(device)
            batch_func_base = batch_func_base / (batch_func_base.std(dim=1, keepdim=True) + 1e-8)
            volume_id = batch["volume_id"][0]
            R_vol_np = emdb_volumes_rotations[volume_id]["rotation"].astype(np.float32)
            R_inv = R_vol_np.T
            mus = batch['s2_distribution_means']
            kappas = batch['s2_distribution_kappas']
            weights = batch['s2_distribution_weights']
            func_data = rotate_s2_function_interpolated(
                points, batch_func_base, R_inv, mus, kappas, weights
            )
            cond_feat = cond_encoder(
                batch['eigen_radial_by_m'],
                batch['eigen_values_by_m'],
                batch['first_moments_radial'],
                batch['mask_by_m']
            )
        else:
            zarr_path = "/data/shachar/zarr_files/emdb_vmf_top_eigen.zarr"
            z = zarr.open(zarr_path, mode='r')
            num_examples = z['distribution_evaluations'].shape[0]
            idx = random.randint(0, num_examples - 1)
            func_data = torch.from_numpy(z['distribution_evaluations'][idx:idx+1]).float().to(device)
            eigen_images = torch.from_numpy(z['eigen_images'][idx:idx+1]).float().to(device)
            eigen_values = torch.from_numpy(z['eigen_values'][idx:idx+1]).float().to(device)
            first_moments = torch.from_numpy(z['first_moments'][idx:idx+1]).float().to(device)
            mus = torch.from_numpy(z['s2_distribution_means'][idx:idx+1]).float().to(device)
            kappas = torch.from_numpy(z['s2_distribution_kappas'][idx:idx+1]).float().to(device)
            weights = torch.from_numpy(z['s2_distribution_weights'][idx:idx+1]).float().to(device)
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
            _, func_data, _ = generate_vmf_mixture_on_s2(batch_size=batch_size)
            func_data = torch.from_numpy(func_data).float().to(device)
        cond_feat = None
    func_data = func_data / (func_data.std(dim=1, keepdim=True) + 1e-8)
    if mode == 'single':
        t = torch.full((batch_size,), t_value, device=device)
        x_t = q_sample(func_data, t)
        context_encoding = build_network_input(
            points, t, x_t,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        model_dtype = next(score_model.parameters()).dtype
        context_encoding = context_encoding.to(dtype=model_dtype)
        query_encoding = context_encoding
        if cond_feat is not None:
            cond_feat = cond_feat.to(dtype=model_dtype)

        with torch.no_grad():
            # cond_feat is None if not using conditional model
            pred_score = score_model(context=context_encoding, queries=query_encoding, cond_feat=cond_feat)
        scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
        x0_est = (x_t + pred_score.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)
        true_score = -(x_t - torch.sqrt(scaling_t) * func_data) / (1 - scaling_t)
        debug_compare_items({'pred_score': pred_score, 'true_score': true_score}, 'conditional')
        if plot_conditional_vs_unconditional_comparison:
            with torch.no_grad():
                    pred_score_uncond = score_model(context=context_encoding, queries=query_encoding, cond_feat=None)
            x0_est_uncond = (x_t + pred_score_uncond.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)
            debug_compare_items({'pred_score': pred_score_uncond, 'true_score': true_score}, 'unconditional')
            debug_compare_items({'x0_est': x0_est, 'func_data': func_data}, 'conditional')
            print("Conditional vs True difference (MSE):", torch.mean((func_data - x0_est) ** 2).item())
            print("Unconditional vs True difference (MSE):", torch.mean((func_data - x0_est_uncond) ** 2).item())
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                "x_t (noised)": x_t,
                "x_0 estimate - conditional": x0_est,
                "x_0 estimate - unconditional": x0_est_uncond,
                #"true score": true_score,
                #"pred score (conditional)": pred_score.squeeze(-1),
                #"score difference (conditional)": (true_score - pred_score.squeeze(-1)),
                #"pred score (unconditional)": pred_score_uncond.squeeze(-1),
                #"score difference (unconditional)": (true_score - pred_score_uncond.squeeze(-1)),
            }, t=t_value)
        else:        
            plot_s2_comparison(points, {
                "x_0 (clean)": func_data,
                "x_t (noised)": x_t,
                "x_0 reconstruction - network": x0_est,
            }, t=t_value)
        
            
    elif mode == 'diffusion':
        t = torch.full((batch_size,), t_value, device=device)
        initial_image = q_sample(func_data, t)

        # cond_feat is None if not using conditional model
        x0_est = diffusion_inference_process(score_model, points, initial_image, t_start=t_value, cond_feat=cond_feat,
                                              use_guidance=use_guidance)

        if compare_best_rotation:
            best_func_data_rot, best_loss, best_est_rot = best_rotated_func_data(
                x0_est, func_data, mus, kappas, weights, points, num_in_plane_rotations=8
            )
            func_data = best_func_data_rot
            print(f"[INFO] Best conditional rotation MSE={best_loss.item():.6f}")
            if return_mse == True:
                return best_loss.item()

        if plot_conditional_vs_unconditional_comparison:
            x0_est_uncond = diffusion_inference_process(score_model, points, initial_image, t_start=t_value, cond_feat=None,
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

if __name__ == "__main__":
    diffs = []
    for i in tqdm(range(1)):
        diff_mse = main(return_mse=False)
        diffs.append(diff_mse)

    diffs = np.array(diffs, dtype=np.float64)
    print(f"Average MSE over {len(diffs)} runs: {diffs.mean():.6f}")
    print(f"Std MSE over {len(diffs)} runs: {diffs.std():.6f}")

