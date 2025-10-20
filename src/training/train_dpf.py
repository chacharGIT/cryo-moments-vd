import torch
from tqdm import tqdm
import os
import zarr
import threading
import queue
from config.config import settings
from src.data.emdb_vmf_subspace_moments_dataloader import create_subspace_moments_dataloader
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.networks.dpf.sample_generation import build_network_input
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.conditional_moment_encoder import CryoMomentsConditionalEncoder
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule
from src.networks.dpf.loss import dpf_score_matching_loss

def train():
    if settings.device.use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{settings.device.cuda_device}')
    else:
        device = torch.device('cpu')

    use_conditional = settings.dpf.conditional_moment_encoder.use_encoder
    # Freeze all model weights only for conditional training
    model = S2ScoreNetwork().to(device)
    if use_conditional:
        for param in model.parameters():
            param.requires_grad = True
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

    quadrature_n = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    points = fibonacci_sphere_points(quadrature_n)
    points = torch.from_numpy(points).float().to(device)  # [n_points, 3]
    batch_size = settings.training.batch_size
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
    else:
        zarr_path = "/data/shachar/zarr_files/emdb_vmf_top_eigen.zarr"
        z = zarr.open(zarr_path, mode='r')
        num_examples = z['distribution_evaluations'].shape[0]
        dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=1, shuffle=False)
        sample_batch = next(iter(dataloader))
        D = sample_batch['eigen_images'].shape[1]  # [B, D, N]
        batch_log_interval = settings.dpf.conditional_moment_encoder.batch_log_interval
        cond_encoder = CryoMomentsConditionalEncoder(
            output_dim= settings.dpf.perceiver.latent_dim,
            num_queries_eig= settings.dpf.conditional_moment_encoder.num_queries_eig,
            unet_out_channels= settings.dpf.conditional_moment_encoder.first_moment_unet_out_channels,
            D=D
        ).to(device)
        print("[INFO] Using conditional encoder for training.")

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
                                                             shuffle=True, debug=True)
            for batch in dataloader:
                # Remap keys to match unconditional convention
                mapped_batch = {
                    'distribution_evaluations': batch['distribution_evaluations'],
                    'kappa': batch['s2_distribution_kappas'],
                    'mu': batch['s2_distribution_means'],
                    'weights': batch['s2_distribution_weights'],
                    # Add compressed moment data for conditional training
                    'eigen_images': batch['eigen_images'],
                    'eigen_values': batch['eigen_values'],
                    'first_moments': batch['first_moments'],
                    'volume_id': batch['volume_id'],
                }
                yield mapped_batch
                
    # Shared batch training function for both conditional and unconditional cases
    def train_on_batch(batch, model, cond_encoder=None, device=None, points=None, optimizer=None):
        if batch is None:
            return None, None, None
        batch_func = batch['func_data']
        batch_kappa = batch['kappa']
        batch_mu = batch['mu']
        batch_weights = batch['weights']
        batch_func = batch_func / (batch_func.std(dim=1, keepdim=True))
        current_batch_size = batch_func.shape[0]
        t = torch.rand(current_batch_size, device=batch_func.device)
        x_t = q_sample(batch_func, t)
        context_encoding = build_network_input(
            points, t, x_t,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )
        query_encoding = context_encoding
        context_encoding = context_encoding.to(dtype=torch.float32)
        query_encoding = query_encoding.to(dtype=torch.float32)
        if cond_encoder is not None:
            conditional_encoding = cond_encoder(batch['eigen_images'].transpose(1, 2), batch['first_moments'], batch['eigen_values'])
            pred_score = model(context=context_encoding, queries=query_encoding, cond_feat=conditional_encoding)
        else:
            pred_score = model(context=context_encoding, queries=query_encoding)
        scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_t.dim() - 1)))
        true_score = -(x_t - torch.sqrt(scaling_t) * batch_func) /(1 - scaling_t)
        if pred_score.dim() == true_score.dim() + 1 and pred_score.shape[-1] == 1:
            pred_score = pred_score.squeeze(-1)
        loss = dpf_score_matching_loss(pred_score, true_score, scale_invariant=True, variance_matching=False,
                                       correlation_matching=False, third_cumulant_matching=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), pred_score, true_score

    def debug_and_save(epoch, avg_loss, pred_score, true_score, model, optimizer, batch_index=None, avg_train_wait_count=None):
        print(f"Epoch {epoch+1}/{settings.training.num_epochs} | Loss: {avg_loss:.6f}")
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
        if use_conditional or (epoch + 1) % settings.training.epochs_per_checkpoint == 0 or (epoch + 1) == settings.training.num_epochs:
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

    # Only use threaded prefetch for conditional training
    if use_conditional:
        dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=batch_size, shuffle=True, debug=True)
        prefetch_queue = queue.Queue(maxsize=3)
        stop_event = threading.Event()
        def remap_conditional_batch(batch):
            return {
                'func_data': batch['distribution_evaluations'],
                'kappa': batch['s2_distribution_kappas'],
                'mu': batch['s2_distribution_means'],
                'weights': batch['s2_distribution_weights'],
                'eigen_images': batch['eigen_images'],
                'eigen_values': batch['eigen_values'],
                'first_moments': batch['first_moments'],
                'volume_id': batch['volume_id'],
            }

        def prefetch_batches(dataloader, prefetch_queue, stop_event):
            for batch in dataloader:
                if stop_event.is_set():
                    break
                mapped_batch = remap_conditional_batch(batch)
                prefetch_queue.put(mapped_batch)
            prefetch_queue.put(None)  # Sentinel for end
        
        for epoch in range(settings.training.num_epochs):
            prefetch_thread = threading.Thread(target=prefetch_batches, args=(dataloader, prefetch_queue, stop_event))
            prefetch_thread.start()
            next_batch = prefetch_queue.get()
            running_loss = 0.0
            num_batches = 0
            train_wait_counts = []
            total_batches = int(num_examples // batch_size) + int(num_examples % batch_size != 0)
            progress_bar = tqdm(range(total_batches), desc=f"Epoch {epoch+1}", total=total_batches, leave=False)
            current_batch = next_batch
            batch_losses = []
            for batch_idx in progress_bar:
                wait_count = 0
                saved_on_current_batch = False
                while True:
                    try:
                        next_batch = prefetch_queue.get(block=False)
                        if next_batch is None:
                            break  # End of data, exit batch loop
                        break
                    except queue.Empty:
                        # No new batch available, train again on current_batch
                        loss, pred_score, true_score = train_on_batch(current_batch, model, cond_encoder,
                                                                       device, points, optimizer)
                        running_loss += loss
                        batch_losses.append(loss)
                        num_batches += 1
                        wait_count += 1
                        if batch_idx % batch_log_interval == 0 and (not batch_idx == 0) and (not saved_on_current_batch):
                            avg_interval_loss = sum(batch_losses[-batch_log_interval:]) / batch_log_interval
                            debug_and_save(epoch, avg_interval_loss, pred_score, true_score, model, optimizer, batch_index=batch_idx)
                            saved_on_current_batch = True
                train_wait_counts.append(wait_count)
                loss, pred_score, true_score = train_on_batch(current_batch, model, cond_encoder,
                                                               device, points, optimizer)
                if loss is None:
                    break  # train_on_batch returned None, start next epoch
                running_loss += loss
                current_batch = next_batch
            avg_loss = running_loss / max(1, num_batches)
            avg_train_wait_count = sum(train_wait_counts) / len(train_wait_counts)
            debug_and_save(epoch, avg_loss, pred_score, true_score, model, optimizer, avg_train_wait_count=avg_train_wait_count)
        stop_event.set()
        prefetch_thread.join()
    else:
        for epoch in range(settings.training.num_epochs):
            running_loss = 0.0
            num_batches = 0
            total_batches = int(train_size // batch_size) + int(train_size % batch_size != 0)
            progress_bar = tqdm(batch_provider(batch_size), desc=f"Epoch {epoch+1}", total=total_batches, leave=False)
            for batch in progress_bar:
                loss, pred_score, true_score = train_on_batch(batch, model, None, device, points, optimizer)
                running_loss += loss
                num_batches += 1
            avg_loss = running_loss / max(1, num_batches)
            debug_and_save(epoch, avg_loss, pred_score, true_score, model, optimizer)

if __name__ == '__main__':
    train()
