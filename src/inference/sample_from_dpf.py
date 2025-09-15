import matplotlib.pyplot as plt
import torch
import numpy as np
from config.config import settings
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.sample_generation import build_network_input, generate_vmf_mixture_on_s2
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule

# Plotting: compare func_data (x_0), x_t, and x0_est for both hemispheres

def plot_s2_comparison(points, plot_dict):
    """
    Plots S2 data for both hemispheres for any number of function arrays.
    Args:
        points: [batch, n_points, 3] torch.Tensor
        plot_dict: dict of {plot_title: array/tensor [batch, n_points]}
    """
    points = points[0].cpu().numpy()
    arrs = []
    titles = []
    for k, v in plot_dict.items():
        arrs.append(v[0].cpu().numpy())
        titles.append(k)
    n_cols = len(arrs)
    mask_top = points[:, 2] > 0
    mask_bottom = points[:, 2] <= 0
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))
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
    plt.show()

if __name__ == "__main__":
    # Use CUDA if available and allowed by config
    device = torch.device(f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda and torch.cuda.is_available() else "cpu")
    model = S2ScoreNetwork().to(device)
    checkpoint = torch.load("./outputs/model_parameter_files/dpf_test_4_epoch_255.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.float64)
    model.eval()

    # Use the same S2/vMF mixture generation as in training
    batch = 1
    points_np, func_data_np, _ = generate_vmf_mixture_on_s2(batch_size=batch)
    points = torch.from_numpy(points_np).float().to(device)  # [batch, n_points, 3]
    func_data = torch.from_numpy(func_data_np).float().to(device)  # [batch, n_points]
    n_points = points.shape[1]

    # Normalize func_data to unit variance per function, preserve mean (for positive functions)
    func_data = func_data / (func_data.std(dim=1, keepdim=True))

    # Choose a specific time t for inference
    t_value = 0.05
    t = torch.full((batch,), t_value, device=device)

    # Generate x_t using the forward diffusion schedule and func_data
    x_t = q_sample(func_data, t)

    # Build context/query encodings (as in training)
    context_encoding = build_network_input(
        points, t, x_t,
        time_enc_len=settings.dpf.time_encoding_len,
        sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
    )
    fourier_dim = settings.dpf.time_encoding_len * 2
    sph_dim = (settings.dpf.pos_encoding_max_harmonic_degree + 1) ** 2
    query_encoding = context_encoding[..., fourier_dim:fourier_dim + sph_dim]

    # Ensure correct dtype for model (float64)
    context_encoding = context_encoding.to(torch.float64)
    query_encoding = query_encoding.to(torch.float64)
    with torch.no_grad():
        pred_score = model(context=context_encoding, queries=query_encoding)

    print('x_t shape:', x_t.shape)
    print('pred_score shape:', pred_score.shape)
    print('Device used:', device)

    # Estimate x_0 from x_t and predicted score (reverse of score matching)
    scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
    # Noise = -score*std_t
    x0_est = (x_t + pred_score.squeeze(-1) * (1 - scaling_t)) / torch.sqrt(scaling_t)

    # Compute the true score (as in training) and print stats for comparison
    # Score = -noise/sqrt(1 - scaling_t)
    true_score = -(x_t - torch.sqrt(scaling_t) * func_data) / (1 - scaling_t)
    print(f'true_score stats: min={true_score.min().item():.4g}, max={true_score.max().item():.4g}, mean={true_score.mean().item():.4g}, std={true_score.std().item():.4g}')
    print(f'pred_score stats: min={pred_score.min().item():.4g}, max={pred_score.max().item():.4g}, mean={pred_score.mean().item():.4g}, std={pred_score.std().item():.4g}')

    plot_s2_comparison(points, {
        "x_0 (clean)": func_data,
        "x_t (noised)": x_t,
        "x_0 est (network)": x0_est,
        "true score": true_score,
        "pred score": pred_score.squeeze(-1)
    })
