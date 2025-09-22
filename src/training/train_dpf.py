import torch
from tqdm import tqdm
import os
import zarr
from config.config import settings
from src.networks.dpf.sample_generation import build_network_input
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule
from src.networks.dpf.loss import dpf_score_matching_loss

def train():
    if settings.device.use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{settings.device.cuda_device}')
    else:
        device = torch.device('cpu')


    # Model
    model = S2ScoreNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.training.learning_rate)

    # Load model parameters
    
    ckpt_path = settings.model.checkpoint.load_path
    if ckpt_path:
        ckpt_path = os.path.expanduser(ckpt_path)
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model parameters from {ckpt_path} (strict=False)")
        else:
            print(f"Model parameters file not found at {ckpt_path}")

    zarr_path = os.path.join(settings.data_generation.zarr.save_dir, "vmf_mixtures_evaluations.zarr")
    z = zarr.open(zarr_path, mode='r')
    func_data = torch.from_numpy(z['func_data'][:]).float().to(device)
    kappa = torch.from_numpy(z['kappa'][:]).float().to(device)
    mu = torch.from_numpy(z['mu_directions'][:]).float().to(device)
    weights = torch.from_numpy(z['mixture_weights'][:]).float().to(device)
    num_examples = func_data.shape[0]
    quadrature_n = func_data.shape[1]
    points = fibonacci_sphere_points(quadrature_n)
    points = torch.from_numpy(points).float().to(device)  # [n_points, 3]
    # Split indices for reproducible train/val split
    indices = torch.randperm(num_examples, generator=torch.Generator().manual_seed(42))
    train_size = int(settings.training.train_val_split * num_examples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    batch_size = settings.training.batch_size
    for epoch in range(settings.training.num_epochs):
        running_loss = 0.0
        num_batches = 0
        # Shuffle train indices each epoch if desired (optional)
        perm = train_indices[torch.randperm(train_size)]
        for batch_idx in tqdm(range(0, train_size, batch_size), desc=f"Epoch {epoch+1}", leave=False):
            idx = perm[batch_idx:batch_idx+batch_size]
            batch_func = func_data[idx]
            batch_kappa = kappa[idx]
            batch_mu = mu[idx]
            batch_weights = weights[idx]
            batch_func = batch_func / (batch_func.std(dim=1, keepdim=True))
            current_batch_size = batch_func.shape[0]
            t = torch.rand(current_batch_size, device=batch_func.device)
            x_t = q_sample(batch_func, t) # [batch, n_points]
            context_encoding = build_network_input(
                points, t, x_t,
                time_enc_len=settings.dpf.time_encoding_len,
                sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
            )
            query_encoding = context_encoding
            # Ensure model inputs are float32
            context_encoding = context_encoding.to(dtype=torch.float32)
            query_encoding = query_encoding.to(dtype=torch.float32)
            pred_score = model(context=context_encoding, queries=query_encoding)
            scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
            # Score = -noise/sqrt(1 - scaling_t)
            true_score = -(x_t - torch.sqrt(scaling_t) * batch_func) /(1 - scaling_t)
            # Squeeze pred_score if it has an extra singleton dimension
            if pred_score.dim() == true_score.dim() + 1 and pred_score.shape[-1] == 1:
                pred_score = pred_score.squeeze(-1)
            loss = dpf_score_matching_loss(pred_score, true_score, scale_invariant=True, variance_matching=False,
                                           correlation_matching=False, third_cumulant_matching=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{settings.training.num_epochs} | Loss: {avg_loss:.6f}")
        print(f"[DEBUG] pred_score stats (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            pred_score.min().item(), pred_score.max().item(), pred_score.mean().item(), pred_score.std().item()))
        print(f"[DEBUG] true_score stats (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            true_score.min().item(), true_score.max().item(), true_score.mean().item(), true_score.std().item()))
        diff = true_score - pred_score
        print(f"[DEBUG] score diff (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            diff.min().item(), diff.max().item(), diff.mean().item(), diff.std().item()))
        print(f"func_scale value: {model.perceiver.func_scale.item()}")
        if (epoch + 1) % settings.training.epochs_per_checkpoint == 0 or (epoch + 1) == settings.training.num_epochs:
            checkpoint_path = settings.model.checkpoint.save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    train()
