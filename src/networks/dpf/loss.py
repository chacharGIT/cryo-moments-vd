import torch
import numpy as np

from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
from src.core.volume_distribution_model import VolumeDistributionModel

def variance_weighting(base_value, true, pred):
    """
    Applies variance-based weighting to a loss function.

    Args:
        base_value (float): Base loss value to be weighted.
        true (torch.Tensor): True values.
        pred (torch.Tensor): Predicted values.
    Returns:
        torch.Tensor: Weighted loss value.
    """
    var_true = true.var()
    var_pred = pred.var()
    weighted_value = 1/2* (base_value / (var_true + 1e-8) + base_value / (var_pred + 1e-8))
    return weighted_value

def dpf_score_matching_loss(
    pred_score, true_score,
    scale_invariant=True,
    variance_matching=False, lambda_var=2.5,
    correlation_matching=False, lambda_corr=0.3,
    third_cumulant_matching=False, lambda_cum3=0.15
):
    """
    Computes the DPF score matching loss between predicted and true scores, with optional regularization terms.
    
    Args:
        pred_score (torch.Tensor): Predicted score from the network, shape [batch, ...].
        true_score (torch.Tensor): True score (precomputed), shape [batch, ...].
        scale_invariant (bool, optional): If True, normalizes the MSE loss by the variance of true_score.
        variance_matching (bool, optional): If True, adds a penalty for the difference in variance between pred_score and true_score.
        lambda_var (float, optional): Weight for the variance matching penalty.
        correlation_matching (bool, optional): If True, adds a penalty for the difference in Pearson correlation between pred_score and true_score.
        lambda_corr (float, optional): Weight for the correlation matching penalty.
        third_cumulant_matching (bool, optional): If True, adds a penalty for the difference in third cumulant (skewness) between pred_score and true_score.
        lambda_cum3 (float, optional): Weight for the third cumulant matching penalty.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    mse = torch.mean((pred_score - true_score) ** 2)
    if scale_invariant:
        mse = variance_weighting(mse, true_score, pred_score)
    if variance_matching:
        var_loss = variance_weighting((pred_score.var() - true_score.var()).abs(), true_score, pred_score)
        mse = mse + lambda_var * var_loss
    if correlation_matching:
        # Pearson correlation loss: 1 - corr(pred, true)
        pred_flat = pred_score.flatten()
        true_flat = true_score.flatten()
        pred_centered = pred_flat - pred_flat.mean()
        true_centered = true_flat - true_flat.mean()
        corr = (pred_centered * true_centered).sum() / (torch.sqrt((pred_centered ** 2).sum()) *
                                                         torch.sqrt((true_centered ** 2).sum()) + 1e-8)
        corr_loss = 1 - corr
        mse = mse + lambda_corr * corr_loss
    if third_cumulant_matching:
        # Third cumulant: E[(x - mean)^3]
        pred_flat = pred_score.flatten()
        true_flat = true_score.flatten()
        pred_cum3 = torch.mean((pred_flat - pred_flat.mean()) ** 3)
        true_cum3 = torch.mean((true_flat - true_flat.mean()) ** 3)
        cum3_loss = (pred_cum3 - true_cum3).abs()
        mse = mse + lambda_cum3 * cum3_loss
    return mse

def partial_moment_loss(
    pred_distribution, true_distribution, volume, back_rotation, points,
     num_sampled_first=64, num_sampled_second=16, num_inplane_rotations=10,
     lambda_1=0.2, lambda_2=1.2, single_volume_training=False, cached_first_moment=None,
    cached_second_moment=None, batch_func_base=None, aspire_volume=None
):
    """
    Computes L2 loss between partial second moments of predicted and true distributions,
    using a random subset of points. First moment is computed over all points.

    Args:
        pred_distribution (torch.Tensor): Predicted distribution, shape [N_points, ...]
        true_distribution (torch.Tensor): True distribution, shape [N_points, ...]
        volume (Volume): Aspire volume object
        back_rotation (torch.Tensor): Rotation matrix to rotate the volume back from 
        R_inv, shape [3, 3]
        points (torch.Tensor): S2 points, shape [N_points, 3]
        num_sampled_first (int): Number of points to subsample for first moment
        num_sampled_second (int): Number of points to subsample for second moment
        num_inplane_rotations (int): Number of in-plane rotations to per S2 point.
        lambda_1 (float): Weight for first moment loss
        lambda_2 (float): Weight for second moment loss
        single_volume_training (bool): If True, uses only a single volume for training.
        cached_first_moment (torch.Tensor or None): If provided, uses this precomputed first moment for all points.
        cached_second_moment (torch.Tensor or None): If provided, uses this precomputed second moment for all points.

    Returns:
        torch.Tensor: Scalar loss value
    """
    max_loss = 1e3
    device = points.device
    # Subsample points with probability proportional to mean true_distribution over batch
    probs = true_distribution.mean(dim=0)
    probs = probs / (probs.sum() + 1e-8)
    idx_first = torch.multinomial(probs, num_sampled_first, replacement=False)
    # Sample points for second moment from first moment points
    probs_first = probs[idx_first]
    probs_first = probs_first / (probs_first.sum() + 1e-8)
    idx_second_in_first = torch.multinomial(probs_first, num_sampled_second, replacement=False)
    idx_second = idx_first[idx_second_in_first]
    sampled_idx_set_second = set(idx_second.tolist())

    if single_volume_training and cached_first_moment is not None and cached_second_moment is not None:
        first_moment_components = cached_first_moment
        second_moment_components = cached_second_moment
    else:
        # Compute first and (if in sampled points) second moment components
        first_moment_components = []
        second_moment_components = []
        for i in idx_first:
            s2_point = points[i].unsqueeze(0).cpu().numpy()  # shape [1, 3]
            # Create in-plane invariant distribution for this point
            so3_rotations, so3_weights = create_in_plane_invariant_distribution(
                s2_points=s2_point,
                num_in_plane_rotations=num_inplane_rotations,
                is_s2_uniform=True
            )
            so3_rotations = np.matmul(back_rotation[None, :, :], so3_rotations)
            # Compute second moment component (returns numpy array)
            vdm = VolumeDistributionModel(
                volume, rotations=so3_rotations, distribution=so3_weights,
                in_plane_invariant_distribution=False, device=device)
            first_moment = vdm.first_analytical_moment(dtype=pred_distribution.dtype, return_torch=True)
            first_moment_components.append(first_moment.to(device=device, dtype=first_moment.dtype))
            if i.item() in sampled_idx_set_second:
                second_moment = vdm.second_analytical_moment(dtype=pred_distribution.dtype, return_torch=True)
                second_moment_components.append(second_moment.to(device=device, dtype=torch.float16))
        first_moment_components = torch.stack(first_moment_components, dim=0)  # [N, ...]
        second_moment_components = torch.stack(second_moment_components, dim=0)  # [num_sampled, ...]
    
    if batch_func_base is not None and aspire_volume is not None:
        batch_func_base = batch_func_base / batch_func_base.sum(dim=1, keepdim=True)
        so3_rotations, so3_weights = create_in_plane_invariant_distribution(
            points.cpu().numpy(), batch_func_base[0].cpu().numpy(), 
            num_in_plane_rotations=128)
        
        V1 =  VolumeDistributionModel(
            aspire_volume, rotations=so3_rotations, distribution=so3_weights, in_plane_invariant_distribution=False)
        m11 = V1.first_analytical_moment()
        m21 = V1.second_analytical_moment(batch_size=50, show_progress=True)

        # Compute weighted sum for first moment and second moment differences
        weights = true_distribution[:, idx_first]  # [B, N]
        m12 = torch.einsum('bn,n...->b...', weights.to(torch.float32), first_moment_components)
        m12 = m12[0].cpu().numpy()
        weights = weights[:, idx_second_in_first]  # [B, num_sampled]
        m22 = torch.einsum('bn,n...->b...', weights.to(torch.float16), second_moment_components)
        m22 = m22[0].cpu().numpy()
        print(np.linalg.norm(m11-m12)/np.linalg.norm(m11), np.linalg.norm(m21-m22)/np.linalg.norm(m21))
        print(np.linalg.norm(m11+m12)/np.linalg.norm(m11), np.linalg.norm(m21+m22)/np.linalg.norm(m21))
        raise Exception("Debugging moment consistency.")

    # Compute weighted sum for first moment and second moment differences
    weights = pred_distribution[:, idx_first] - true_distribution[:, idx_first]  # [B, N]
    first_moment_diff = torch.einsum('bn,n...->b...', weights, first_moment_components)
    first_moment_loss = first_moment_diff.flatten(1).norm(dim=1).mean()
    weights = weights[:, idx_second_in_first]  # [B, num_sampled]
    partial_second_moment_diff = torch.einsum('bn,n...->b...', weights.to(torch.float16), second_moment_components)
    second_moment_loss = partial_second_moment_diff.flatten(1).norm(dim=1).mean()
    base_loss = (lambda_1 * first_moment_loss + lambda_2 * second_moment_loss)
    weighted_loss = base_loss
    loss = torch.clamp(weighted_loss, max=max_loss)
    if single_volume_training:
        if cached_first_moment is None or cached_second_moment is None:
            # Return loss and cached tensors for first call
            return loss, first_moment_components, second_moment_components
        else:
            return loss
    else:
        return loss
