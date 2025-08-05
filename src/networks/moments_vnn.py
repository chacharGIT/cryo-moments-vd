import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.vnn_layer import VNNLayer
from config.config import settings



class VNN(nn.Module):
    def __init__(self, degree=2, hidden_dim=64, distribution_type=None, distribution_metadata=None,
                  activation=None, alpha=0.2):
        super(VNN, self).__init__()
        if distribution_type is None:
            raise ValueError("distribution_type must be provided.")
        if distribution_metadata is None:
            raise ValueError("distribution_metadata must be provided.")
        
        self.distribution_type = distribution_type
        weights = distribution_metadata.get('weights', None)
        if weights is None:
            raise ValueError(f"'{distribution_type}' requires 'weights' in distribution_metadata.")
        self.num_points = len(weights)
        
        if self.distribution_type == 's2_delta_mixture':
            output_dim = self.num_points * 4 # points (3D), weights
        elif self.distribution_type == 'vmf_mixture':
            output_dim = self.num_points * 5  # means (3D), kappas, weights
        else:
            raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")

        self.layer1 = VNNLayer(degree, in_channels=1, out_channels=hidden_dim)
        self.layer2 = VNNLayer(degree, in_channels=hidden_dim, out_channels=hidden_dim)
        self.layer3 = VNNLayer(degree, in_channels=hidden_dim, out_channels=hidden_dim)
        self.activation = activation if activation is not None else nn.LeakyReLU(negative_slope=alpha)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, M2, M1):
        batch_size = M2.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(M2)

        # Layer 1: (batch_size, N, 1) -> (batch_size, N, hidden_dim)
        z1 = self.layer1(eigenvalues, eigenvectors, M1)
        x1 = self.activation(z1)

        # Layer 2: (batch_size, N, hidden_dim) -> (batch_size, N, hidden_dim)
        z2 = self.layer2(eigenvalues, eigenvectors, x1)
        x2 = self.activation(z2)

        # Layer 3: (batch_size, N, hidden_dim) -> (batch_size, N, hidden_dim)
        z3 = self.layer3(eigenvalues, eigenvectors, x2)
        x3 = self.activation(z3)

        # Global mean pooling to produce fixed-size output
        # x3 shape: (batch_size, N, hidden_dim) -> (batch_size, hidden_dim)
        pooled = torch.mean(x3, dim=1)  # Average over N dimension
        
        # Final linear layer: (batch_size, hidden_dim) -> (batch_size, output_dim)
        out = self.fc_out(pooled)

        if self.distribution_type == 's2_delta_mixture':
            points_flat = out[:, :self.num_points * 3]
            weights_raw = out[:, self.num_points * 3:]
            points_3d = points_flat.view(batch_size, self.num_points, 3)
            s2_points = F.normalize(points_3d, p=2, dim=2)
            weights = F.softmax(weights_raw, dim=1)
            return {'s2_points': s2_points, 'weights': weights}
        elif self.distribution_type == 'vmf_mixture':
            means_flat = out[:, :self.num_points * 3]
            kappas_raw = out[:, self.num_points * 3:self.num_points * 4]
            weights_raw = out[:, self.num_points * 4:self.num_points * 5]
            means_3d = means_flat.view(batch_size, self.num_points, 3)
            means = F.normalize(means_3d, p=2, dim=2)
            # Ensure kappas are >= 1 by using sigmoid to map to [0,1] then scale to [1, 101]
            # This gives a reasonable range while ensuring minimum of 1
            kappas = 1.0 + 100.0 * torch.sigmoid(kappas_raw)
            weights = F.softmax(weights_raw, dim=1)
            return {'means': means, 'kappas': kappas, 'weights': weights}

def general_distribution_loss(pred, target, distribution_type, blur=None):
    """
    Compute the loss between predicted and target distribution parameters for different distribution types.

    Parameters
    ----------
    pred : dict
        Dictionary of predicted outputs. Keys depend on distribution_type.
    target : dict
        Dictionary of target outputs. Keys must match those in pred.
    distribution_type : str
        Type of distribution.
    blur : float, optional
        Sinkhorn blur parameter. If None, uses the default from settings.

    Returns
    -------
    loss : torch.Tensor
        Scalar tensor representing the total loss for the batch.
        Also returns dict with additional regularization losses.
    """
    from geomloss import SamplesLoss
    # Use provided blur parameter or fall back to settings default
    blur_param = blur if blur is not None else settings.loss.sinkhorn.blur
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=blur_param, diameter=1.0)
    kappa_weight = settings.loss.weights.vmf_kappa

    # Common weight processing
    pred_weights = pred['weights']
    target_weights = target['weights']

    # Compute Sinkhorn loss using batched tensors
    if distribution_type == 's2_delta_mixture':
        main_loss = sinkhorn(pred_weights, pred['s2_points'], target_weights, target['s2_points'])

    elif distribution_type == 'vmf_mixture':
        # Set vectors: (mean_x, mean_y, mean_z, log_kappa * kappa_weight)
        # Use log(kappa) instead of kappa directly for better numerical stability
        pred_vec = torch.cat([
            pred['means'],
            torch.log(pred['kappas'])[..., None] * kappa_weight
        ], dim=2)  # Shape: (B, N, 4)
        target_vec = torch.cat([
            target['means'],
            torch.log(target['kappas'])[..., None] * kappa_weight
        ], dim=2)  # Shape: (B, N, 4)
        
        # Use batched Sinkhorn computation: (B,N,D) and (B,N)
        main_loss = sinkhorn(pred_weights, pred_vec, target_weights, target_vec)
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")

    total_loss = settings.loss.weights.sinkhorn * main_loss  
    return total_loss


def l2_distribution_loss(pred, target, distribution_type):
    """
    Compute simple L2 loss between predicted and target distribution parameters.

    Parameters
    ----------
    pred : dict
        Dictionary of predicted outputs. Keys depend on distribution_type.
    target : dict
        Dictionary of target outputs. Keys must match those in pred.
    distribution_type : str
        Type of distribution.

    Returns
    -------
    loss : torch.Tensor
        Scalar tensor representing the total L2 loss for the batch.
    """
    # Always compute weights loss
    weights_loss = F.mse_loss(pred['weights'], target['weights'])
    
    if distribution_type == 's2_delta_mixture':
        points_loss = F.mse_loss(pred['s2_points'], target['s2_points'])
        return weights_loss + points_loss
        
    elif distribution_type == 'vmf_mixture':
        means_loss = F.mse_loss(pred['means'], target['means'])
        kappa_loss = F.mse_loss(torch.log(pred['kappas']), torch.log(target['kappas']))
        return weights_loss + means_loss + kappa_loss
        
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")
