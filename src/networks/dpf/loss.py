import torch
from src.networks.dpf.forward_diffusion import q_sample, cosine_signal_scaling_schedule

def dpf_score_matching_loss(
    pred_score, true_score,
    scale_invariant=False,
    variance_matching=True, lambda_var=0.4,
    correlation_matching=False, lambda_corr=0.2,
    third_cumulant_matching=False, lambda_cum3=0.03
):
    """
    Computes the DPF score matching loss between predicted and true scores, with optional regularization terms.
    
    Args:
        pred_score (torch.Tensor): Predicted score from the network, shape [batch, ...].
        true_score (torch.Tensor): True score (precomputed), shape [batch, ...].
        scale_invariant (bool, optional): If True, normalizes the MSE loss by the variance of true_score. Default: False.
        variance_matching (bool, optional): If True, adds a penalty for the difference in variance between pred_score and true_score. Default: False.
        lambda_var (float, optional): Weight for the variance matching penalty.
        correlation_matching (bool, optional): If True, adds a penalty for the difference in Pearson correlation between pred_score and true_score. Default: False.
        lambda_corr (float, optional): Weight for the correlation matching penalty.
        third_cumulant_matching (bool, optional): If True, adds a penalty for the difference in third cumulant (skewness) between pred_score and true_score. Default: False.
        lambda_cum3 (float, optional): Weight for the third cumulant matching penalty.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    mse = torch.mean((pred_score - true_score) ** 2)
    if scale_invariant:
        mse = mse / (true_score.var() + 1e-8)
    if variance_matching:
        var_loss = (pred_score.var() - true_score.var()).abs()
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
