from matplotlib.pylab import float32
import torch
import numpy as np
from tqdm import tqdm
import os
import zarr
import threading
import queue

from config.config import settings
from src.data.emdb_downloader import load_aspire_volume
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.networks.dpf.sample_generation import build_network_input
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule
from src.networks.dpf.loss import dpf_score_matching_loss, partial_moment_loss
from src.networks.dpf.torch_utils import rotate_s2_function_interpolated

def train():
    # torch.autograd.set_detect_anomaly(True)
    if settings.device.use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{settings.device.cuda_device}')
    else:
        device = torch.device('cpu')

    batch_size = settings.training.batch_size
    use_conditional = settings.dpf.conditional_moment_encoder.use_encoder
    separate_fourier_modes = settings.data_generation.separate_fourier_modes
    score_model = S2ScoreNetwork().to(device)
    quadrature_n = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    points = fibonacci_sphere_points(quadrature_n)
    points = torch.from_numpy(points).float().to(device)  # [n_points, 3]

    if use_conditional:
        # Load per-volume rotations
        emdb_volumes_rotations_np = np.load(
            settings.data_generation.emdb.volume_rotations_path,
            allow_pickle=True
        ).item()
        emdb_volumes_rotations = {
            key: torch.from_numpy(value).to(device)
            if isinstance(value, np.ndarray) else value
            for key, value in emdb_volumes_rotations_np.items()
        }
        # Define sign configurations for possible per volume rotations
        sign_configs = np.array(
            [
                [ 1.0,  1.0,  1.0],   # R
                [ 1.0, -1.0, -1.0],   # [R1, -R2, -R3]
                [-1.0, -1.0,  1.0],   # [-R1, -R2, R3]
                [-1.0,  1.0, -1.0],   # [-R1, R2, -R3]
            ],
            dtype=np.float32  # or np.float64 if you prefer
        )
    
    if not use_conditional:
        zarr_path = os.path.join(settings.data_generation.zarr.save_dir, "vmf_mixtures_evaluations.zarr")
        z = zarr.open(zarr_path, mode='r')
        func_data = torch.from_numpy(z['func_data'][:]).float().to(device)
        kappa = torch.from_numpy(z['kappa'][:]).float().to(device)
        mu = torch.from_numpy(z['mu_directions'][:]).float().to(device)
        weights = torch.from_numpy(z['mixture_weights'][:]).float().to(device)
        num_examples = func_data.shape[0]
        # Split indices for reproducible train/val split
        indices = torch.randperm(num_examples, generator=torch.Generator().manual_seed(42))
        train_size = int(settings.training.train_val_split * num_examples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        print("[INFO] Using standard (non-conditional) model for training.")
        optim_params = score_model.parameters()
    else:
        batch_log_interval = settings.dpf.conditional_moment_encoder.batch_log_interval

        if separate_fourier_modes:
            from src.data.emdb_vmf_separated_subspace_moments_dataloader import create_subspace_moments_dataloader
            from src.networks.dpf.conditional_separated_moment_encoder import CryoMomentsConditionalEncoder
            zarr_path = settings.data_generation.zarr.separated_modes_data_save_path
            cond_encoder = CryoMomentsConditionalEncoder().to(device)
            print("[INFO] Using conditional SEPARATED moment encoder for training.")
        else:
            from src.data.emdb_vmf_subspace_moments_dataloader import create_subspace_moments_dataloader
            from src.networks.dpf.conditional_moment_encoder import CryoMomentsConditionalEncoder
            zarr_path = settings.data_generation.zarr.full_data_save_path
            z = zarr.open(zarr_path, mode='r')
            num_examples = z['distribution_evaluations'].shape[0]
            dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=1, shuffle=True)
            sample_batch = next(iter(dataloader))
            D = sample_batch['eigen_images'].shape[1]  # [B, D, N]
            cond_encoder = CryoMomentsConditionalEncoder(
                output_dim= settings.dpf.perceiver.latent_dim,
                num_queries_eig= settings.dpf.conditional_moment_encoder.num_queries_eig,
                unet_out_channels= settings.dpf.conditional_moment_encoder.first_moment_unet_out_channels,
                D=D
            ).to(device)
            print("[INFO] Using conditional FULL moment encoder for training.")

    # Load model parameters
    ckpt_path = settings.model.checkpoint.load_path
    if ckpt_path:
        ckpt_path = os.path.expanduser(ckpt_path)
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_state = checkpoint['model_state_dict']
            score_model_state = score_model.state_dict()
            # keep only keys that exist in current model AND have the same shape
            score_model_filtered_state = {}
            for k, v in ckpt_state.items():
                if k in score_model_state and v.shape == score_model_state[k].shape:
                    score_model_filtered_state[k] = v
                else:
                    print(f"[CKPT] skipping key due to shape mismatch or missing: {k} "
                        f"ckpt={tuple(v.shape)} "
                        f"model={tuple(score_model_state.get(k, torch.empty(0)).shape)}")
            score_model_state.update(score_model_filtered_state)
            score_model.load_state_dict(score_model_state, strict=False)
            if use_conditional and 'cond_encoder_state_dict' in checkpoint:
                cond_model_filtered_state = {}
                cond_ckpt_state = checkpoint['cond_encoder_state_dict']
                cond_model_state = cond_encoder.state_dict()
                for k, v in cond_ckpt_state.items():
                    if k in cond_model_state and v.shape == cond_model_state[k].shape:
                        cond_model_filtered_state[k] = v
                    else:
                        print(f"[CKPT] skipping cond_encoder key due to shape mismatch or missing: {k} "
                            f"ckpt={tuple(v.shape)} "
                            f"model={tuple(cond_model_state.get(k, torch.empty(0)).shape)}")
                cond_model_state.update(cond_model_filtered_state)
                cond_encoder.load_state_dict(cond_model_state, strict=False)

                # Freeze all model weights only for conditional training
                for param in score_model.parameters():
                    param.requires_grad = False
                for param in score_model.perceiver.cond_cross_attn_layers.parameters():
                    param.requires_grad = True
                for param in score_model.perceiver.cond_scales.parameters():
                    param.requires_grad = True
                optim_params = (
                    [p for p in score_model.parameters() if p.requires_grad] +
                    list(cond_encoder.parameters())
                )
            else:
                optim_params = [p for p in score_model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(optim_params, lr=settings.training.learning_rate)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model parameters from {ckpt_path} (strict=False)")
        else:
            print(f"Model parameters file not found at {ckpt_path}")

    def batch_provider(batch_size=batch_size):
        if not use_conditional:
            for batch_idx in range(0, train_size, batch_size):
                idx = train_indices[batch_idx:batch_idx+batch_size]
                batch = {
                    'func_data': func_data[idx],
                    'kappa': kappa[idx],
                    'mu': mu[idx],
                    'weights': weights[idx],
                }
                yield batch
        else:
            dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=batch_size,
                                                             shuffle=True, debug=True, mode='train')
            for batch in dataloader:
                # Remap keys to match unconditional convention
                mapped_batch = {
                    'func_data': batch['distribution_evaluations'],
                    'kappa': batch['s2_distribution_kappas'],
                    'mu': batch['s2_distribution_means'],
                    'weights': batch['s2_distribution_weights'],
                    'volume_id': batch['volume_id'],                  
                }
                if separate_fourier_modes:
                    mapped_batch.update({
                        'eigen_values_by_m': batch['eigen_values_by_m'],
                        'eigen_radial_by_m': batch['eigen_radial_by_m'],
                        'mask_by_m': batch['mask_by_m'],
                        'first_moments_radial': batch['first_moments_radial'],
                    })
                else:
                    mapped_batch.update({
                        'eigen_images': batch['eigen_images'],
                        'first_moments': batch['first_moments'],
                        'eigen_values': batch['eigen_values'],
                    })
                yield mapped_batch
                
    # Shared batch training function for both conditional and unconditional cases
    def train_on_batch(batch, model, cond_encoder=None, device=None, points=None, optimizer=None):
        if batch is None:
            return None, None, None
        batch_func = batch['func_data']
        batch_func = batch_func / (batch_func.std(dim=1, keepdim=True))
        current_batch_size = batch_func.shape[0]
        t = torch.rand(current_batch_size, device=batch_func.device)
        if use_conditional:
            volume_id = batch["volume_id"][0]
            R_vol_np = emdb_volumes_rotations[volume_id]["rotation"].astype(np.float32)
            R_inv = R_vol_np.T
            batch_func_old = batch_func.clone()
            batch_func = rotate_s2_function_interpolated(
                points, batch_func, R_inv
            )
        x_t = q_sample(batch_func, t)
        context_encoding = build_network_input(
            points, t, x_t,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        context_encoding = context_encoding.to(dtype=torch.float32)
        query_encoding = context_encoding
        if use_conditional:
            if separate_fourier_modes:
                conditional_encoding = cond_encoder(
                    batch['eigen_radial_by_m'],
                    batch['eigen_values_by_m'],
                    batch['first_moments_radial'],
                    batch['mask_by_m']
                )
            else:
                conditional_encoding = cond_encoder(
                    batch['eigen_images'].transpose(1, 2),
                    batch['first_moments'],
                    batch['eigen_values']
                )
            if conditional_encoding is None:
                return None, None, None  # Early return on failed encoder
            pred_score = model(context=context_encoding, queries=query_encoding, cond_feat=conditional_encoding)
        else:
            pred_score = model(context=context_encoding, queries=query_encoding)
            
        scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
        scaling_t = scaling_t.clamp(min=1e-6, max=1.0 - 1e-6)
        true_score = -(x_t - torch.sqrt(scaling_t) * batch_func) / (1 - scaling_t)
        if pred_score.dim() == true_score.dim() + 1 and pred_score.shape[-1] == 1:
            pred_score = pred_score.squeeze(-1)

        # Rotate predicted score on S^2 using the per-volume rotation (conditional case)
        if use_conditional:
            candidate_targets = []
            candidate_losses = []

            for signs in sign_configs:
                S = np.diag(signs)
                # (R * S)^{-1} = S * R^{-1} for diagonal S with ±1
                R_inv_variant = S @ R_inv
                rotated_true_score = rotate_s2_function_interpolated(
                    points, true_score.detach(), R_inv_variant
                )
                candidate_targets.append(rotated_true_score)
                candidate_losses.append(dpf_score_matching_loss(pred_score, rotated_true_score))
            candidate_losses = torch.stack(candidate_losses)  # shape [4]
            loss, best_idx = torch.min(candidate_losses, dim=0)
            final_true_score = candidate_targets[best_idx.item()]
        else:
            final_true_score = true_score
            loss = dpf_score_matching_loss(pred_score, final_true_score)

        aspire_volume = load_aspire_volume(
            settings.data_generation.emdb.download_folder + "/" + batch['volume_id'][0] + ".map.gz",
            downsample_size=settings.data_generation.downsample_size,
        )
        """
        from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
        from src.core.volume_distribution_model import VolumeDistributionModel
        so3_rotations, so3_weights = create_in_plane_invariant_distribution(points.cpu().numpy(), batch_func_old[0].cpu().numpy(), 
                                                                        num_in_plane_rotations=128)
        V1 =  VolumeDistributionModel(
            aspire_volume, rotations=so3_rotations, distribution=so3_weights, in_plane_invariant_distribution=False)
        m11 = V1.first_analytical_moment()
        m21 = V1.second_analytical_moment(batch_size=50, show_progress=True)
        so3_rotations, so3_weights = create_in_plane_invariant_distribution(points.cpu().numpy(), batch_func[0].cpu().numpy(), 
                                                                        num_in_plane_rotations=128)
        so3_rotations = np.matmul(R_inv.T[None, :, :], so3_rotations)
        v2 =  VolumeDistributionModel(
            aspire_volume, rotations=so3_rotations, distribution=so3_weights, in_plane_invariant_distribution=False)
        m12 = v2.first_analytical_moment()
        m22 = v2.second_analytical_moment(batch_size=50, show_progress=True)
        print(np.linalg.norm(m11-m12), np.linalg.norm(m21-m22))
        raise Exception("Debugging moment mismatch")
        """
        pred_distribution = ((1 - scaling_t) * pred_score + x_t)/torch.sqrt(scaling_t)
        loss += partial_moment_loss(
            volume=aspire_volume,
            back_rotation=R_inv.T,
            pred_distribution=pred_distribution,
            true_distribution=batch_func,
            points=points,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), pred_score, final_true_score
    def debug_and_save(avg_loss, pred_score, true_score, model, optimizer,cond_encoder=None, epoch=None, batch_index=None, avg_train_wait_count=None):
        if not use_conditional:
            print(f"Epoch {epoch+1}/{settings.training.num_epochs} | Loss: {avg_loss:.6f}")
        else:
            print(f"Batch {batch_index} | Loss: {avg_loss:.6f}")
        print(f"[DEBUG] pred_score stats (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            pred_score.min().item(), pred_score.max().item(), pred_score.mean().item(), pred_score.std().item()))
        print(f"[DEBUG] true_score stats (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            true_score.min().item(), true_score.max().item(), true_score.mean().item(), true_score.std().item()))
        diff = true_score - pred_score
        print(f"[DEBUG] score diff (last batch): min={{:.4g}}, max={{:.4g}}, mean={{:.4g}}, std={{:.4g}}".format(
            diff.min().item(), diff.max().item(), diff.mean().item(), diff.std().item()))
        if avg_train_wait_count is not None:
            print(f"[DEBUG] Average train_wait_count per batch: {avg_train_wait_count:.2f}")
        # Save every epoch if use_conditional is True (conditional training), otherwise use interval logic
        if (not use_conditional) and ((epoch + 1) % settings.training.epochs_per_checkpoint == 0 or (epoch + 1) == settings.training.num_epochs):
            checkpoint_path = settings.model.checkpoint.save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            if batch_index is not None:
                checkpoint_path = checkpoint_path.replace('.pth', f'_batch_{batch_index}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        if use_conditional:
            checkpoint_path = settings.model.checkpoint.save_path.replace('.pth', f'_batch_{batch_index}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'cond_encoder_state_dict': cond_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Only use threaded prefetch for conditional training
    if use_conditional:
        prefetch_queue = queue.Queue(maxsize=3)
        stop_event = threading.Event()

        def prefetch_batches(prefetch_queue, stop_event):
            for batch in batch_provider(batch_size):
                if stop_event.is_set():
                    break
                prefetch_queue.put(batch)
            prefetch_queue.put(None)  # Sentinel for end
        
        num_train_examples = settings.training.num_train_examples
        total_batches = int(num_train_examples // batch_size)
        batch_log_interval = settings.training.batch_log_interval
        prefetch_thread = threading.Thread(target=prefetch_batches, args=(prefetch_queue, stop_event))
        prefetch_thread.start()
        next_batch = prefetch_queue.get()
        running_loss = 0.0
        num_batches = 0
        train_wait_counts = []
        current_batch = next_batch
        batch_losses = []
        progress_bar = tqdm(range(total_batches), desc="Training (conditional)",
                             total=total_batches, leave=False)

        for batch_idx in progress_bar:
            wait_count = 0
            saved_on_current_batch = False

            # Try to fetch a fresh batch; if not ready, reuse current_batch
            while True:
                try:
                    next_batch = prefetch_queue.get(block=False)
                    if next_batch is None:
                        break  # End of data, exit batch loop
                    break
                except queue.Empty:
                    # No new batch available, train again on current_batch
                    loss, pred_score, true_score = train_on_batch(current_batch, score_model, cond_encoder,
                                                                    device, points, optimizer)
                    if loss is None:
                        current_batch = next_batch
                        continue  # Skip if training failed
                    running_loss += loss
                    batch_losses.append(loss)
                    num_batches += 1
                    wait_count += 1
                    if batch_idx % batch_log_interval == 0 and (not batch_idx == 0) and (not saved_on_current_batch):
                        avg_interval_loss = sum(batch_losses[-batch_log_interval:]) / batch_log_interval
                        debug_and_save(avg_interval_loss, pred_score, true_score, score_model, optimizer, cond_encoder=cond_encoder, batch_index=batch_idx)
                        saved_on_current_batch = True
            train_wait_counts.append(wait_count)
            loss, pred_score, true_score = train_on_batch(current_batch, score_model, cond_encoder,
                                                            device, points, optimizer)
            if loss is None:
                current_batch = next_batch
                continue  # Skip if training failed
            running_loss += loss
            batch_losses.append(loss)
            num_batches += 1
            current_batch = next_batch
            if batch_idx % batch_log_interval == 0 and (not batch_idx == 0):
                avg_interval_loss = sum(batch_losses[-batch_log_interval:]) / batch_log_interval
                debug_and_save(avg_interval_loss, pred_score, true_score, score_model, optimizer, cond_encoder=cond_encoder, batch_index=batch_idx)
        stop_event.set()
        prefetch_thread.join()
    else:
        for epoch in range(settings.training.num_epochs):
            running_loss = 0.0
            num_batches = 0
            total_batches = int(train_size // batch_size) + int(train_size % batch_size != 0)
            progress_bar = tqdm(batch_provider(batch_size), desc=f"Epoch {epoch+1}", total=total_batches, leave=False)
            for batch in progress_bar:
                loss, pred_score, true_score = train_on_batch(batch, score_model, None, device, points, optimizer)
                running_loss += loss
                num_batches += 1
            avg_loss = running_loss / max(1, num_batches)
            debug_and_save(avg_loss, pred_score, true_score, score_model, optimizer, epoch=epoch)

if __name__ == '__main__':
    train()
