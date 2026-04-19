import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
import os
import sys
import zarr
import threading
import queue
import signal
import traceback

from config.config import settings
from src.data.emdb_downloader import load_aspire_volume
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.networks.dpf.sample_generation import build_network_input
from src.networks.dpf.score_network import S2ScoreNetwork
from src.networks.dpf.loss import dpf_score_matching_loss, partial_moment_loss
from src.networks.dpf.torch_utils import rotate_s2_function_interpolated

def handle_sigterm(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, handle_sigterm)

def adjust_checkpoint_keys(ckpt_state, model_state):
    """
    Adjust checkpoint state dict keys to match model state dict keys,
     handling 'module.' prefix from DDP wrapping.

    Parameters:
        ckpt_state: state dict from checkpoint (may have 'module.' prefix or not)
        model_state: state dict from model (may have 'module.' prefix or not)
    
    Returns:
        ckpt_state: adjusted checkpoint state dict with keys modified to match model_state keys
    """
    if all(k.startswith('module.') for k in model_state.keys()) and not all(k.startswith('module.') for k in ckpt_state.keys()):
        ckpt_state = {'module.' + k if not k.startswith('module.') else k: v for k, v in ckpt_state.items()}
    
    # If checkpoint keys have 'module.' but model does not, remove it
    elif not all(k.startswith('module.') for k in model_state.keys()) and all(k.startswith('module.') for k in ckpt_state.keys()):
        ckpt_state = {k.replace('module.', '', 1): v for k, v in ckpt_state.items()}
    return ckpt_state

def load_filtered_state_dict_compat(model, ckpt_state, verbose=True):
    """
    Load checkpoint weights into model:
      - fix 'module.' prefix mismatch
      - keep only keys that exist AND match shape
      - strict=False
    
    Parameters:
        model: the PyTorch model to load weights into
        ckpt_state: the state dict from the checkpoint (already adjusted for 'module.' prefix)
        verbose: if True, print skipped keys due to shape mismatch or missing keys
    """
    model_state = model.state_dict()
    ckpt_state = adjust_checkpoint_keys(ckpt_state, model_state)

    filtered = {}
    for k, v in ckpt_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            if verbose:
                print(f"[CKPT] skipping key due to shape mismatch or missing: {k} "
                        f"ckpt={tuple(v.shape)} "
                        f"model={tuple(model_state.get(k, torch.empty(0)).shape)}")

    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)

def debug_compare_items(items: dict, prefix: str = ""):
    def print_stats(name, tensor):
        print(f"[DEBUG] {prefix} {name} stats: min={tensor.min().item():.4g}, max={tensor.max().item():.4g}, mean={tensor.mean().item():.4g}, std={tensor.std().item():.4g}")
    if not isinstance(items, dict) or len(items) != 2:
        raise ValueError("debug_compare_scores expects a dict with exactly 2 items")
    (name_a, a), (name_b, b) = list(items.items())

    if a.squeeze(-1).shape == b.shape:
        a = a.squeeze(-1)
    print_stats(name_a, a)
    print_stats(name_b, b)
    print_stats(f"{name_a} - {name_b}", a-b)

def train(local_rank):
    CURRICULUM_SINGLE_EXAMPLE = False
    CURRICULUM_SINGLE_VOLUME = True
    if CURRICULUM_SINGLE_EXAMPLE or CURRICULUM_SINGLE_VOLUME:
        cached_first_moment = None
        cached_second_moment = None

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    batch_size = settings.training.batch_size
    use_conditional = settings.dpf.conditional_moment_encoder.use_encoder
    separate_fourier_modes = settings.data_generation.separate_fourier_modes
    score_model = S2ScoreNetwork().to(device)
    score_model = DDP(score_model, device_ids=[local_rank], find_unused_parameters=True)
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
        p_drop_cond = settings.dpf.p_drop_cond # Dropout probability for conditional features during training
        p_calc_partial_moment_loss = settings.dpf.p_calc_partial_moment_loss # Probability of calculating partial moment loss for a training batch
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
            #from src.networks.dpf.conditional_separated_moment_encoder import CryoMomentsConditionalEncoder
            #cond_encoder = CryoMomentsConditionalEncoder().to(device)
            from src.networks.dpf.simple_conditional_moment_encoder import DictComplexToBert
            cond_encoder = DictComplexToBert().to(device)
            zarr_path = settings.data_generation.zarr.separated_modes_data_save_path
            if not CURRICULUM_SINGLE_VOLUME:
                dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=batch_size,
                                                             shuffle=True, debug=True, mode='train', device=device)
            else:
                dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=batch_size,
                                                             shuffle=True, debug=True, mode='train', device=device, single_volume_id="emd_19777")
                
            print("[INFO] Using conditional SEPARATED moment encoder for training.")
        else:
            from src.data.emdb_vmf_subspace_moments_dataloader import create_subspace_moments_dataloader
            from src.networks.dpf.conditional_moment_encoder import CryoMomentsConditionalEncoder
            zarr_path = settings.data_generation.zarr.full_data_save_path
            z = zarr.open(zarr_path, mode='r')
            num_examples = z['distribution_evaluations'].shape[0]
            dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=1, shuffle=True, device=device)
            sample_batch = next(iter(dataloader))
            D = sample_batch['eigen_images'].shape[1]  # [B, D, N]
            cond_encoder = CryoMomentsConditionalEncoder(
                output_dim= settings.dpf.perceiver.latent_dim,
                num_queries_eig= settings.dpf.conditional_moment_encoder.num_queries_eig,
                unet_out_channels= settings.dpf.conditional_moment_encoder.first_moment_unet_out_channels,
                D=D
            ).to(device)
            print("[INFO] Using conditional FULL moment encoder for training.")
        cond_encoder = DDP(cond_encoder, device_ids=[local_rank], find_unused_parameters=True)

    # Load model parameters
    ckpt_path = settings.get("model.checkpoint.load_path", None)
    checkpoint = None
    if ckpt_path:
        ckpt_path = os.path.expanduser(ckpt_path)
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            load_filtered_state_dict_compat(
                score_model,
                checkpoint['model_state_dict'],
                verbose=False
            )
        else:
            if local_rank == 0:
                print(f"Model parameters file not found at {ckpt_path}")
    else:
        if local_rank == 0:
            print("No model checkpoint specified, starting training from scratch.")

    if use_conditional:
        learning_rate = settings.training.learning_rate
        base_model_learning_rate = learning_rate  # Lower LR for base model
        if checkpoint and 'cond_encoder_state_dict' in checkpoint:
            load_filtered_state_dict_compat(
                cond_encoder,
                checkpoint['cond_encoder_state_dict'],
                verbose=False
            )
        """
        # Reinitialize conditional modules in the score model
        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        smp = score_model.module.perceiver
        smp.cond_cross_attn_layers.apply(_reset)
        smp.t_latent_embedding.apply(_reset)

        smp.gamma.apply(_reset)
        torch.nn.init.zeros_(smp.gamma.weight)
        if smp.gamma.bias is not None:
            torch.nn.init.zeros_(smp.gamma.bias)

        smp.beta.apply(_reset)
        torch.nn.init.zeros_(smp.beta.weight)
        if smp.beta.bias is not None:
            torch.nn.init.zeros_(smp.beta.bias)
        
        with torch.no_grad():
            for p in smp.cond_scales:
                p.fill_(-5.0)
        print("[INFO] Reinitialized conditional modules in the score model after loading checkpoint.")
        """
        # Lower LR for base model except for the conditional modules
        smp = score_model.module.perceiver
        conditional_modules = [smp.cond_cross_attn_layers, smp.cond_scales,
                                smp.t_latent_embedding, smp.gamma, smp.beta]
        cond_param_ids = {id(p) for m in conditional_modules for p in m.parameters()}

        backbone_parameters = [p for p in score_model.parameters() if id(p) not in cond_param_ids]
        conditional_parameters = [p for p in score_model.parameters() if id(p) in cond_param_ids]

        optimizer = torch.optim.Adam([
            {"params": backbone_parameters, "lr": base_model_learning_rate},
            {"params": conditional_parameters, "lr": learning_rate},
            {"params": cond_encoder.parameters(), "lr": learning_rate},
        ])
    else:
        optim_params = [p for p in score_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(optim_params, lr=settings.training.learning_rate)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if local_rank == 0:
        print(f"Loaded model parameters from {ckpt_path} (strict=False)")

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
        batch_func_base = batch['func_data']
        batch_func_base = batch_func_base / (batch_func_base.std(dim=1, keepdim=True) + 1e-8)
        current_batch_size = batch_func_base.shape[0]
        t = torch.zeros(current_batch_size, device=batch_func_base.device, dtype=torch.float32)  # const t=0.0

        if use_conditional:
            volume_id = batch["volume_id"][0]
            R_vol_np = emdb_volumes_rotations[volume_id]["rotation"].astype(np.float32)
            R_inv = R_vol_np.T
            mus = batch['mu']
            kappas = batch['kappa']
            weights = batch['weights']

            batch_func_candidates = []
            for signs in sign_configs:
                S = np.diag(signs).astype(np.float32)
                # (R * S)^{-1} = S * R^{-1} for diagonal S with ±1 entries
                R_inv_variant = S @ R_inv
                bf = rotate_s2_function_interpolated(points, batch_func_base, R_inv_variant, mus, kappas, weights)
                bf = bf / (bf.std(dim=1, keepdim=True) + 1e-8)
                batch_func_candidates.append(bf)

            batch_func = batch_func_candidates[0]  # Select the first candidate as the batch function
        else:
            batch_func = batch_func_base
            batch_func_candidates = None
            R_inv = None
        
        x_in = torch.zeros_like(batch_func)
        context_encoding = build_network_input(
            points, t, x_in,
            time_enc_len=settings.dpf.time_encoding_len,
            sph_enc_len=settings.dpf.pos_encoding_max_harmonic_degree
        )

        context_encoding = context_encoding.to(dtype=torch.float32)
        query_encoding = context_encoding

        pred_func = None
        if use_conditional:
            drop_cond = torch.rand((), device=device) < p_drop_cond
            if dist.is_initialized():
                dist.broadcast(drop_cond, src=0)
            drop_cond = bool(drop_cond.item())

            if drop_cond:
                pred_func = model(context=context_encoding, queries=query_encoding, cond_feat=None)
            else:
                try:
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
                    pred_func = model(context=context_encoding, queries=query_encoding, cond_feat=conditional_encoding)
                except Exception:
                    print(f"\n[ERROR] conditional path failed:\n{traceback.format_exc()}", flush=True)
                    return None, None, None
        else:
            pred_func = model(context=context_encoding, queries=query_encoding)
            
        if pred_func.dim() == batch_func.dim() + 1 and pred_func.shape[-1] == 1:
            pred_func = pred_func.squeeze(-1)

        # Rotate predicted score on S^2 using the per-volume rotation (conditional case)
        if use_conditional and not drop_cond and not (CURRICULUM_SINGLE_VOLUME or CURRICULUM_SINGLE_EXAMPLE):
            per_item_losses = []
            final_func_list = []
            best_k_list = []
            B = pred_func.shape[0]
            for b in range(B):
                losses_b = []
                for c in range(len(batch_func_candidates)):
                    losses_b.append(dpf_score_matching_loss(pred_func[b:b+1],
                                                            batch_func_candidates[c][b:b+1]))
                losses_b = torch.stack(losses_b)  # [4]
                best_k = int(torch.argmin(losses_b).item())
                per_item_losses.append(losses_b[best_k])
                final_func_list.append(batch_func_candidates[best_k][b:b+1])
                best_k_list.append(best_k)
                
            loss = torch.stack(per_item_losses).mean()
            final_true_func = torch.cat(final_func_list, dim=0)  
            final_k_per_item = torch.tensor(best_k_list, device=device, dtype=torch.long)  # [B]   
            # print(f"[DEBUG] number of items with best_k: {(final_k_per_item == 0).sum().item()} / {B}, {(final_k_per_item == 1).sum().item()} / {B}, {(final_k_per_item == 2).sum().item()} / {B}, {(final_k_per_item == 3).sum().item()} / {B}") 
            # print(f"[DEBUG] mean t: {t.mean().item()}")
        else:
            final_true_func = batch_func
            loss = dpf_score_matching_loss(pred_func, final_true_func)
            final_k_per_item = None

        if use_conditional and not drop_cond:
            aspire_volume = load_aspire_volume(
                settings.data_generation.emdb.download_folder + "/" + batch['volume_id'][0] + ".map.gz",
                downsample_size=settings.data_generation.downsample_size,
            )
        
        """
        from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
        from src.core.volume_distribution_model import VolumeDistributionModel
        so3_rotations, so3_weights = create_in_plane_invariant_distribution(
            points.cpu().numpy(), batch_func_base[0].cpu().numpy(), 
            num_in_plane_rotations=128)
        
        V1 =  VolumeDistributionModel(
            aspire_volume, rotations=so3_rotations, distribution=so3_weights, in_plane_invariant_distribution=False)
        m11 = V1.first_analytical_moment()
        m21 = V1.second_analytical_moment(batch_size=50, show_progress=True)
        so3_rotations, so3_weights = create_in_plane_invariant_distribution(
            points.cpu().numpy(), batch_func[0].cpu().numpy(),
            num_in_plane_rotations=128)
        
        so3_rotations = np.matmul(R_inv.T[None, :, :], so3_rotations)
        v2 =  VolumeDistributionModel(
            aspire_volume, rotations=so3_rotations, distribution=so3_weights, in_plane_invariant_distribution=False)
        m12 = v2.first_analytical_moment()
        m22 = v2.second_analytical_moment(batch_size=50, show_progress=True)
        print(np.linalg.norm(m11-m12)/np.linalg.norm(m11), np.linalg.norm(m21-m22)/np.linalg.norm(m21))
        print(np.linalg.norm(m11+m12)/np.linalg.norm(m11), np.linalg.norm(m21+m22)/np.linalg.norm(m21))
        raise Exception("Debugging moment consistency.")
        """

        pred_distribution = pred_func
        if (CURRICULUM_SINGLE_EXAMPLE or CURRICULUM_SINGLE_VOLUME) and use_conditional and not drop_cond:
            nonlocal cached_first_moment, cached_second_moment
            if cached_first_moment is None or cached_second_moment is None:
                print("[INFO] Computing and caching first and second moment components for curriculum training...")
                moment_loss, cached_first_moment, cached_second_moment = partial_moment_loss(
                    volume=aspire_volume,
                    back_rotation=R_inv.T,
                    pred_distribution=pred_distribution,
                    true_distribution=batch_func,
                    points=points,
                    single_volume_training=True,
                    # batch_func_base=batch_func_base,
                    # aspire_volume=aspire_volume
                )
                loss += moment_loss
                print("[INFO] Cached first moment shape:", cached_first_moment.shape)
                print("[INFO] Cached second moment shape:", cached_second_moment.shape)
            else:
                loss += partial_moment_loss(
                    volume=aspire_volume,
                    back_rotation=R_inv.T,
                    pred_distribution=pred_distribution,
                    true_distribution=batch_func,
                    points=points,
                    single_volume_training=True,
                    cached_first_moment=cached_first_moment,
                    cached_second_moment=cached_second_moment
                )
        elif not drop_cond:
            calculate_partial_moment_loss = torch.rand((), device=device) < p_calc_partial_moment_loss
            if dist.is_initialized():
                dist.broadcast(calculate_partial_moment_loss, src=0)
            calculate_partial_moment_loss = bool(calculate_partial_moment_loss.item())
            if calculate_partial_moment_loss:
                for k in range(len(sign_configs)):
                        mask = (final_k_per_item == k)
                        if not bool(mask.any().item()):
                            continue
                        idx = mask.nonzero(as_tuple=False).squeeze(1)

                        S = np.diag(sign_configs[k]).astype(np.float32)
                        R_inv_variant = S @ R_inv
                        back_rotation_variant = R_inv_variant.T  # = R_inv.T @ S

                        dpf_partial_moment_loss = partial_moment_loss(
                            volume=aspire_volume,
                            back_rotation=back_rotation_variant,
                            pred_distribution=pred_distribution[idx],
                            true_distribution=final_true_func[idx],
                            points=points,
                        )
                        if dpf_partial_moment_loss is not None and torch.isfinite(dpf_partial_moment_loss):
                            loss += dpf_partial_moment_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), pred_func, final_true_func
    def debug_and_save(avg_loss, pred_score, true_score, model, optimizer,cond_encoder=None, epoch=None, batch_index=None, avg_train_wait_count=None):
        if local_rank != 0:
            return  # Only let rank 0 print and save
        if not use_conditional:
            print(f"Epoch {epoch+1}/{settings.training.num_epochs} | Loss: {avg_loss:.6f}")
        else:
            print(f"Batch {batch_index} | Loss: {avg_loss:.6f}")
        debug_compare_items({'pred_score': pred_score, 'true_score': true_score})
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
        current_batch = prefetch_queue.get()
        running_loss = 0.0
        num_batches = 0
        batch_losses = []
        progress_bar = tqdm(range(total_batches), desc="Training (conditional)",
                             total=total_batches, leave=False)
        if CURRICULUM_SINGLE_EXAMPLE:
            # Get a single batch and train on it repeatedly
            from itertools import count
            single_batch = current_batch
            batch_idx = 0
            progress_bar = tqdm(count(), desc="Curriculum Training (conditional, infinite)", leave=False)
            for _ in progress_bar:
                loss, pred_score, true_score = train_on_batch(single_batch, score_model, cond_encoder,
                                                            device, points, optimizer)
                if loss is None:
                    continue
                running_loss += loss
                batch_losses.append(loss)
                num_batches += 1
                batch_idx += 1
                if batch_idx % batch_log_interval == 0 and (not batch_idx == 0):
                    avg_interval_loss = sum(batch_losses[-batch_log_interval:]) / batch_log_interval
                    debug_and_save(avg_interval_loss, pred_score, true_score, score_model, optimizer, cond_encoder=cond_encoder, batch_index=batch_idx)
        else:
            try:
                for batch_idx in progress_bar:

                    current_batch = prefetch_queue.get()
                    end_local = int(current_batch is None)
                    if dist.is_initialized():
                        end_flag = torch.tensor(end_local, device=device, dtype=torch.int32)
                        dist.all_reduce(end_flag, op=dist.ReduceOp.MAX)
                        if end_flag.item() == 1:
                            break
                    if current_batch is None:
                        break  # End of data, exit batch loop

                    loss, pred_score, true_score = train_on_batch(current_batch, score_model, cond_encoder,
                                                                    device, points, optimizer)
                    if loss is None:
                        continue  # Skip if training failed
                    running_loss += loss
                    batch_losses.append(loss)
                    num_batches += 1

                    if batch_idx % batch_log_interval == 0 and (not batch_idx == 0):
                        avg_interval_loss = sum(batch_losses[-batch_log_interval:]) / batch_log_interval
                        debug_and_save(avg_interval_loss, pred_score, true_score, score_model, optimizer, cond_encoder=cond_encoder, batch_index=batch_idx)
                stop_event.set()
                prefetch_thread.join()
            except KeyboardInterrupt:
                print("Gracefully stopping training loop...")
            finally:
                if dist.is_initialized():
                    try:
                        dist.destroy_process_group()
                    except:
                        pass
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    try:
        train(local_rank)
    except KeyboardInterrupt:
        print(f"\n[RANK {local_rank}] Caught Ctrl+C. Cleaning up...", flush=True)
    except Exception:
        print(f"\n[RANK {local_rank}] Unhandled exception:\n{traceback.format_exc()}", flush=True)
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass
        torch.cuda.empty_cache()
        sys.exit(0)
