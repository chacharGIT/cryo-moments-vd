from cvxpy import neg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config.config import settings
from src.networks.dpf.cyclic_equivariant_attention import CyclicEquivariantAttentionBlock
from src.networks.dpf.set_transformer import SetTransformerBlock
from src.networks.dpf.random_projection import random_projections

class FirstMoment1DUNET(nn.Module):
    """
    A compact 1D UNet-style convolutional network for encoding a real 1D vector.
    Uses strided convolutions for downsampling and includes skip connections at each level.
    Input:  [B, L]
    Output: [B, C, L] where C = out_channels
    """
    def __init__(self, out_channels, kernel_size=5, nonlinearity=F.relu):
        super().__init__()
        self.nonlinearity = nonlinearity

        g1 = min(8, out_channels)
        g2 = min(8, out_channels * 2)
        g4 = min(8, out_channels * 4)

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g1, out_channels),
            self.nonlinearity,
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g1, out_channels),
            self.nonlinearity,
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, kernel_size, stride=2, padding=kernel_size // 2),
            nn.GroupNorm(g2, out_channels * 2),
            self.nonlinearity
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g2, out_channels * 2),
            self.nonlinearity,
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g2, out_channels * 2),
            self.nonlinearity,
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels * 4, kernel_size, stride=2, padding=kernel_size // 2),
            nn.GroupNorm(g4, out_channels * 4),
            self.nonlinearity
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels * 4, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g4, out_channels * 4),
            self.nonlinearity,
        )
        self.up2 = nn.ConvTranspose1d(out_channels * 4, out_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels * 2, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g2, out_channels * 2),
            self.nonlinearity,
        )
        self.up1 = nn.ConvTranspose1d(out_channels * 2, out_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g1, out_channels),
            self.nonlinearity,
        )
        self.out_conv = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        # x: [B, L]
        orig_len = x.shape[-1]
        pad_len = (4 - orig_len % 4) % 4
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="constant", value=0)
        x = x.unsqueeze(1)  # [B, 1, L]

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
    def __init__(self):
        super().__init__()
        self.R = settings.dpf.conditional_separated_moment_encoder.radial_vector_length
        self.n_queries_per_m = np.load(settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.Nms_path)
        self.first_moment_1d_unet = FirstMoment1DUNET(
            out_channels=settings.dpf.conditional_separated_moment_encoder.first_moment_1d_unet.out_channels,
            kernel_size=5,
            nonlinearity=nn.SiLU(inplace=True)
        )
        self.second_moment_set_transformer = SetTransformerBlock(
            R = self.R,
            d_k = settings.dpf.conditional_separated_moment_encoder.set_transformer.d_k,
            d_v = settings.dpf.conditional_separated_moment_encoder.set_transformer.d_v,
            num_heads = settings.dpf.conditional_separated_moment_encoder.set_transformer.num_heads,
            ff_hidden_dim = settings.dpf.conditional_separated_moment_encoder.set_transformer.ff_hidden_dim,
            n_queries_per_m = self.n_queries_per_m
        )
        self.T = 2 * len(self.n_queries_per_m) - 1
        self.D = settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.D  
    
        cyclic_attention_args = ({
            'd_k': settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.d_k,
            'd_v': settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.d_v,
            'num_heads': settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.num_heads,
            'D': self.D,
            'R': self.R,
            'ff_hidden_dim': settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.ff_hidden_dim,
            'nonlinearity': nn.GELU(),
        })
        self.cyclic_blocks_pre = nn.ModuleList([
            CyclicEquivariantAttentionBlock(**cyclic_attention_args, n_queries_per_m=self.n_queries_per_m)
            for _ in range(settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.num_blocks_pre)
        ])
        n_queries_per_m_post = self.n_queries_per_m
        n_queries_per_m_post[0] += settings.dpf.conditional_separated_moment_encoder.first_moment_1d_unet.out_channels
        self.cyclic_blocks_post = nn.ModuleList([
            CyclicEquivariantAttentionBlock(**cyclic_attention_args, n_queries_per_m=n_queries_per_m_post)
            for _ in range(settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.num_blocks_post - 1)
        ])
        self.last_cyclic_block = CyclicEquivariantAttentionBlock(**cyclic_attention_args,
            n_queries_per_m=n_queries_per_m_post, project_back=False)

        self.reduce_activation = nn.GELU()
        self.latent_dim = settings.dpf.perceiver.latent_dim
        self.num_cond_queries = settings.dpf.conditional_separated_moment_encoder.num_cond_queries
        self.reduce_R = nn.Linear(self.R, self.latent_dim)
        self.reduce_TD = nn.Linear(self.T * self.D, self.num_cond_queries)
        
    def forward(self, second_moment_radial_subspace, second_moment_eigen_values, first_moment_radial, mask_dict):
        """
        Args:
            second_moment_radial_subspace (Tensor): [B, N, R] second moment vectors, each with one angular fourier frequency (flattened D=l^2).
            first_moment_radial (Tensor): [B, R] tensor of the first moment (mean projection) image.
            second_moment_eigen_values (Tensor): [B, N] tensor of eigenvalues for weighting the second moment subspace.
            mask_dict: dict m -> [B, Ñ_m] (bool)

        Returns:
            cond_feat (Tensor): [B, output_dim] conditional feature vector for downstream tasks.
        """
        device = first_moment_radial.device
        B = first_moment_radial.shape[0]
        x2 = {}
        for m in range(len(self.n_queries_per_m)):
            if m in second_moment_radial_subspace:
                # Eigenvalue weighting
                eig_weighted = second_moment_radial_subspace[m] * \
                   second_moment_eigen_values[m].sqrt().unsqueeze(-1)  # [B, Ñ_m, R]
                mask = mask_dict[m]
            else:
                eig_weighted = torch.zeros(
                    B, 0, self.R,
                    dtype=torch.cfloat,
                    device=device,
                )
                mask = None

            if not torch.isfinite(eig_weighted).all():
                return None  # Early return on non-finite values

            x2[str(m)] = self.second_moment_set_transformer(eig_weighted, m, mask=mask)
        for i in range(len(self.cyclic_blocks_pre)):
            delta = self.cyclic_blocks_pre[i](x2)
            for k in delta.keys():
                x2[k] = x2[k] + delta[k]

        # Concatenate first moment encoding to V_0
        x1 = self.first_moment_1d_unet(first_moment_radial)  # [B, out_channels, R]
        V_0 = x2["0"]  # [B, N_0, R]
        V_0_prime = torch.cat([V_0, x1], dim=1)  # [B, N_0 + out_channels, R]
        x2["0"] = V_0_prime

        # Second half of cyclic blocks
        for i in range(len(self.cyclic_blocks_post)):
            delta = self.cyclic_blocks_post[i](x2)
            for k in delta.keys():
                x2[k] = x2[k] + delta[k]    
        x2 = self.last_cyclic_block(x2) # [B, T, D, R]

        # axis-wise reductions: [B, T, D, R] -> [B, num_cond_queries, latent_dim]
        B, T, D, R = x2.shape
        x = x2.reshape(B, T * D, R)
        x = self.reduce_R(x) # [B, T*D, latent_dim]
        x = self.reduce_activation(x)
        x = x.permute(0, 2, 1)
        x = self.reduce_TD(x) # [B, latent_dim, num_cond_queries]
        cond_feat = x.permute(0, 2, 1)
        return cond_feat
    