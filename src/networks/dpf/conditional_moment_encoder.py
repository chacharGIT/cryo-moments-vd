import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import settings
from cyclic_equivariant_attention import CyclicEquivariantAttentionBlock
from set_transformer import SetTransformerBlock

    
class FirstMoment1DUNET(nn.Module):
    """
    A compact 1D UNet-style convolutional network for encoding a real 1D vector.
    Uses strided convolutions for downsampling and includes skip connections at each level.
    """
    def __init__(self, out_channels, base_channels=16, kernel_size=5):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size, padding=kernel_size//2), nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=kernel_size//2), nn.ReLU()
        )
        self.down1 = nn.Conv1d(base_channels, base_channels*2, kernel_size, stride=2, padding=kernel_size//2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*2, kernel_size, padding=kernel_size//2), nn.ReLU(),
            nn.Conv1d(base_channels*2, base_channels*2, kernel_size, padding=kernel_size//2), nn.ReLU()
        )
        self.down2 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size, stride=2, padding=kernel_size//2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size, padding=kernel_size//2), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*2, kernel_size, padding=kernel_size//2), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose1d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels, kernel_size, padding=kernel_size//2), nn.ReLU()
        )
        self.out_conv = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x):
        # x: [B, L] -> [B, 1, L]
        orig_len = x.shape[-1]
        pad_len = (4 - orig_len % 4) % 4
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='constant', value=0)
        x = x.unsqueeze(1)
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bottleneck(d2)
        u2 = self.up2(b)
        # Crop or pad to match e2
        min_len2 = min(u2.shape[-1], e2.shape[-1])
        u2 = torch.cat([u2[..., :min_len2], e2[..., :min_len2]], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        min_len1 = min(u1.shape[-1], e1.shape[-1])
        u1 = torch.cat([u1[..., :min_len1], e1[..., :min_len1]], dim=1)
        d1 = self.dec1(u1)
        out = self.out_conv(d1)
        # Crop output back to original length if padded
        if pad_len > 0:
            out = out[..., :orig_len]
        return out
    

class CryoMomentsConditionalEncoder(nn.Module):
    def __init__(self, radial_encoder_kwargs, set_transformer_kwargs, cyclic_attention_kwargs):
        super().__init__()
        self.second_moment_set_transformer = SetTransformerBlock(**set_transformer_kwargs)
        self.cyclic_blocks = nn.ModuleList([
            CyclicEquivariantAttentionBlock(**cyclic_attention_kwargs)
            for _ in range(6)
        ])
        self.first_moment_1d_unet = FirstMoment1DUNET(**radial_encoder_kwargs)

    def forward(self, second_moment_subspace, first_moment_radial, eigen_values, return_basis_images=False):
        """
        Args:
            second_moment_subspace (Tensor): [B, N, l] second moment vectors, each with one angular fourier frequency (flattened D=l^2).
            first_moment_radial (Tensor): [B, l] tensor of the first moment (mean projection) image.
            eigen_values (Tensor): [B, N] tensor of eigenvalues for weighting the second moment subspace.

        Returns:
            cond_feat (Tensor): [B, output_dim] conditional feature vector for downstream tasks.
        """
        B, N, D = second_moment_subspace.shape
        # --- Eigenvalue attention ---
        eig_weighted = second_moment_subspace * eigen_values.sqrt().unsqueeze(-1)  # [B, N, D]
        # Second moment pipeline
        x2 = self.second_moment_set_transformer(eig_weighted)
        x1 = self.first_moment_1d_unet(first_moment_radial)
        for block in self.cyclic_blocks:
            x2 = block(x2)
        # First moment pipeline
        return x1, x2