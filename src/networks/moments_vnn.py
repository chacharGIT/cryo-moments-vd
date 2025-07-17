import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.vnn_layer import VNNLayer



class VNN(nn.Module):
    def __init__(self, degree=2, hidden_dim=64, distribution_metadata=None, activation=None, alpha=0.2):
        super(VNN, self).__init__()
        if distribution_metadata is None or 'type' not in distribution_metadata:
            raise ValueError("distribution_metadata must be provided and contain a 'type' key.")
        self.distribution_type = distribution_metadata['type']
        if self.distribution_type == 's2_delta_mixture':
            s2_weights = distribution_metadata.get('s2_weights', None)
            if s2_weights is None:
                raise ValueError("'s2_delta_mixture' requires 's2_weights' in distribution_metadata.")
            self.num_points = len(s2_weights)
            output_dim = self.num_points * 4 # points (3D), weights
        elif self.distribution_type == 'vmf_mixture':
            vmf_weights = distribution_metadata.get('von_mises_mixture_weights', None)
            if vmf_weights is None:
                raise ValueError("'vmf_mixture' requires 'von_mises_mixture_weights' in distribution_metadata.")
            self.num_points = len(vmf_weights)
            output_dim = self.num_points * 5  # means (3D), kappas, weights
        else:
            raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")

        # Channel progression: 1 -> 64 -> 64 -> 64
        self.layer1 = VNNLayer(degree, in_channels=1, out_channels=64)
        self.layer2 = VNNLayer(degree, in_channels=64, out_channels=64)
        self.layer3 = VNNLayer(degree, in_channels=64, out_channels=64)
        self.activation = activation if activation is not None else nn.LeakyReLU(negative_slope=alpha)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, M2, M1):
        batch_size = M2.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(M2)

        # Layer 1: (batch_size, N, 1) -> (batch_size, N, 64)
        z1 = self.layer1(eigenvalues, eigenvectors, M1)
        x1 = self.activation(z1)

        # Layer 2: (batch_size, N, 64) -> (batch_size, N, 64)
        z2 = self.layer2(eigenvalues, eigenvectors, x1)
        x2 = self.activation(z2)

        # Layer 3: (batch_size, N, 64) -> (batch_size, N, 64)
        z3 = self.layer3(eigenvalues, eigenvectors, x2)
        x3 = self.activation(z3)

        # Global mean pooling to produce fixed-size output
        # x3 shape: (batch_size, N, 64) -> (batch_size, 64)
        pooled = torch.mean(x3, dim=1)  # Average over N dimension
        
        # Final linear layer: (batch_size, 64) -> (batch_size, output_dim)
        out = self.fc_out(pooled)

        if self.distribution_type == 's2_delta_mixture':
            points_flat = out[:, :self.num_points * 3]
            weights_raw = out[:, self.num_points * 3:]
            points_3d = points_flat.view(batch_size, self.num_points, 3)
            s2_points = F.normalize(points_3d, p=2, dim=2)
            weights = F.softmax(weights_raw, dim=1)
            return {'s2_points': s2_points, 's2_weights': weights}
        elif self.distribution_type == 'vmf_mixture':
            means_flat = out[:, :self.num_points * 3]
            kappas_raw = out[:, self.num_points * 3:self.num_points * 4]
            weights_raw = out[:, self.num_points * 4:self.num_points * 5]
            means_3d = means_flat.view(batch_size, self.num_points, 3)
            means = F.normalize(means_3d, p=2, dim=2)
            kappas = F.softplus(kappas_raw)  # Ensure kappas are positive
            weights = F.softmax(weights_raw, dim=1)
            return {'means': means, 'kappas': kappas, 'weights': weights}

def general_distribution_loss(pred, target, distribution_type):
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

    Returns
    -------
    loss : torch.Tensor
        Scalar tensor representing the total loss for the batch.
    """
    if distribution_type == 's2_delta_mixture':
        # L2 loss for S2 points
        points_loss = torch.mean((pred['s2_points'] - target['s2_points']) ** 2)
        # L2 loss for weights
        weights_loss = torch.mean((pred['s2_weights'] - target['s2_weights']) ** 2)
        total_loss = points_loss + weights_loss
        return total_loss
    elif distribution_type == 'vmf_mixture':
        # L2 loss for means
        means_loss = torch.mean((pred['means'] - target['means']) ** 2)
        # L2 loss for kappas
        kappas_loss = torch.mean((pred['kappas'] - target['kappas']) ** 2)
        # L2 loss for weights
        weights_loss = torch.mean((pred['weights'] - target['weights']) ** 2)
        total_loss = means_loss + kappas_loss + weights_loss
        return total_loss
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")
