import torch
import torch.nn as nn
from src.networks.vnn.torch_utils import apply_all_C_powers

class VNNLayer(nn.Module):
    def __init__(self, degree: int, in_channels, out_channels):
        super(VNNLayer, self).__init__()
        self.degree = degree
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h = nn.Parameter(torch.randn(out_channels, in_channels, degree + 1)) # Initialize weights

    def forward(self, eigenvalues, eigenvectors, x):
        """
        Args:
            eigenvalues: Tensor of shape (batch_size, N)
            eigenvectors: Tensor of shape (batch_size, N, N)
            x: Tensor of shape (batch_size, N, F_in)
        Returns:
            Tensor of shape (batch_size, N, F_out)
        """
        batch_size, N, F_in = x.shape
        
        # Apply C^k for all degrees: (degree+1, batch_size, N, F_in)
        hx = apply_all_C_powers(eigenvalues, eigenvectors, x, self.degree)
        
        # Implement: x_out[f] = σ(∑_{g=1}^{F_in} H_{fg}(Ĉ^n) x_in[g])
        # h shape: (F_out, F_in, degree+1)
        # hx shape: (degree+1, batch_size, N, F_in)
        
        # Reshape for efficient computation
        # hx: (degree+1, batch_size, N, F_in) -> (batch_size, N, degree+1, F_in)
        hx = hx.permute(1, 2, 0, 3)  # (batch_size, N, degree+1, F_in)
        
        # Reshape for matrix multiplication
        # hx: (batch_size, N, degree+1, F_in) -> (batch_size, N, degree+1 * F_in)
        hx_flat = hx.reshape(batch_size, N, -1)  # (batch_size, N, (degree+1) * F_in)
        
        # h: (F_out, F_in, degree+1) -> (F_out, (degree+1) * F_in)
        h_flat = self.h.reshape(self.out_channels, -1)  # (F_out, (degree+1) * F_in)
        
        # Matrix multiplication: (batch_size, N, (degree+1) * F_in) @ (F_out, (degree+1) * F_in).T
        # -> (batch_size, N, F_out)
        z = torch.einsum('bni,oi->bno', hx_flat, h_flat)
        
        return z
