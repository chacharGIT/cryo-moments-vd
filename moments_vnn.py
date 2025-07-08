import torch
import torch.nn as nn
import torch.nn.functional as F
from vnn_layer import VNNLayer

class VNN(nn.Module):
    def __init__(self, degree=2, hidden_dim=64, num_s2_points=5, activation=None, alpha=0.2):
        super(VNN, self).__init__()
        self.num_s2_points = num_s2_points
        
        # Channel progression: 1 -> 64 -> 64 -> 64
        self.layer1 = VNNLayer(degree, in_channels=1, out_channels=64)
        self.layer2 = VNNLayer(degree, in_channels=64, out_channels=64)
        self.layer3 = VNNLayer(degree, in_channels=64, out_channels=64)
        self.activation = activation if activation is not None else nn.LeakyReLU(negative_slope=alpha)
        
        # Output layers for S2 points and weights
        # 3D points on sphere (will be normalized) + weights (will be softmax normalized)
        output_dim = num_s2_points * 3 + num_s2_points  # 3D coords + weights
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, M2, M1):
        """
        Args:
            M1: First moment - tensor of shape (batch_size, N, 1)
            M2: Second moment - symmetric covariance matrix of shape (batch_size, N, N) 
        Returns:
            s2_points: Tensor of shape (batch_size, num_s2_points, 3) - normalized 3D points on unit sphere
            weights: Tensor of shape (batch_size, num_s2_points) - normalized weights that sum to 1
        """
        batch_size = M2.shape[0]
        
        # eigenvalues: (batch_size, N), eigenvectors: (batch_size, N, N)
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
        
        # Split output into 3D points and weights
        points_flat = out[:, :self.num_s2_points * 3]  # (batch_size, num_s2_points * 3)
        weights_raw = out[:, self.num_s2_points * 3:]   # (batch_size, num_s2_points)
        
        # Reshape points and normalize to unit sphere
        points_3d = points_flat.view(batch_size, self.num_s2_points, 3)  # (batch_size, num_s2_points, 3)
        s2_points = F.normalize(points_3d, p=2, dim=2)  # Normalize to unit sphere
        
        # Normalize weights using softmax
        weights = F.softmax(weights_raw, dim=1)  # (batch_size, num_s2_points)
        
        return s2_points, weights


def sinkhorn_algorithm(cost_matrix, num_iters=10, reg=0.1):
    """
    Differentiable approximation to optimal transport using Sinkhorn iterations.
    
    Args:
        cost_matrix: (batch_size, n, n) - cost matrix
        num_iters: number of Sinkhorn iterations
        reg: regularization parameter
    
    Returns:
        transport_matrix: (batch_size, n, n) - soft assignment matrix
    """
    batch_size, n, _ = cost_matrix.shape
    
    # Convert cost to similarity with regularization
    K = torch.exp(-cost_matrix / reg)  # (batch_size, n, n)
    
    # Initialize uniform distributions
    eps = 1e-8
    u = torch.ones((batch_size, n), device=cost_matrix.device)  # (batch_size, n)
    v = torch.ones((batch_size, n), device=cost_matrix.device)  # (batch_size, n)
    
    # Sinkhorn iterations
    for _ in range(num_iters):
        u = 1.0 / (K @ v.unsqueeze(-1)).clamp(min=eps).squeeze(-1)
    v = 1.0 / (K.transpose(-2, -1) @ u.unsqueeze(-1)).clamp(min=eps).squeeze(-1)
    
    # Compute transport matrix
    transport = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # (batch_size, n, n)
    return transport


def l2_norm_loss(pred_points, pred_weights, target_points, target_weights):
    """
    Simple L2 norm loss between predicted and target points/weights.
    This assumes the points are already in corresponding order (no permutation handling).
    
    Args:
        pred_points: (batch_size, num_points, 3) - predicted 3D points on unit sphere
        pred_weights: (batch_size, num_points) - predicted weights
        target_points: (batch_size, num_points, 3) - target 3D points on unit sphere  
        target_weights: (batch_size, num_points) - target weights
    
    Returns:
        loss: scalar tensor - L2 norm loss
    """
    # L2 loss for points
    points_loss = torch.mean((pred_points - target_points) ** 2)
    
    # L2 loss for weights
    weights_loss = torch.mean((pred_weights - target_weights) ** 2)
    
    # Combine both losses
    total_loss = points_loss + weights_loss
    
    return total_loss


def permutation_invariant_loss(pred_points, pred_weights, target_points, target_weights):
    """
    Efficient permutation-invariant loss using Sinkhorn algorithm for optimal transport.
    
    Args:
        pred_points: (batch_size, num_points, 3) - predicted 3D points on unit sphere
        pred_weights: (batch_size, num_points) - predicted weights
        target_points: (batch_size, num_points, 3) - target 3D points on unit sphere  
        target_weights: (batch_size, num_points) - target weights
    
    Returns:
        loss: scalar tensor - permutation invariant loss
    """
    batch_size, num_points, _ = pred_points.shape
    
    # Compute pairwise distances between all predicted and target points
    pred_expanded = pred_points.unsqueeze(2)      # (batch_size, num_points, 1, 3)
    target_expanded = target_points.unsqueeze(1)  # (batch_size, 1, num_points, 3)
    
    # Compute squared distances between all pairs
    point_distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)  # (batch_size, num_points, num_points)
    
    # Compute weight differences
    pred_weights_expanded = pred_weights.unsqueeze(2)      # (batch_size, num_points, 1)
    target_weights_expanded = target_weights.unsqueeze(1)  # (batch_size, 1, num_points)
    weight_distances = (pred_weights_expanded - target_weights_expanded) ** 2  # (batch_size, num_points, num_points)
    
    # Combined distance matrix (points + weights)
    total_distances = point_distances + weight_distances  # (batch_size, num_points, num_points)
    
    # Use Sinkhorn algorithm for differentiable optimal transport
    transport_matrix = sinkhorn_algorithm(total_distances)  # (batch_size, num_points, num_points)
    loss = torch.sum(transport_matrix * total_distances, dim=(-2, -1))  # (batch_size,)
    
    return loss.mean()
