import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FirstMomentUNet(nn.Module):
    """
    A compact UNet-style convolutional network for encoding the first moment image (mean projection image).
    Uses strided convolutions for learnable downsampling and includes skip connections at each level.
    """
    def __init__(self, out_channels, base_channels=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU()
        )
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU()
        )
        self.down2 = nn.Conv2d(base_channels*2, base_channels*2, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1), nn.ReLU()
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, base_channels, H, W]
        d1 = self.down1(e1)  # [B, base_channels, H/2, W/2]
        e2 = self.enc2(d1)   # [B, base_channels*2, H/2, W/2]
        d2 = self.down2(e2)  # [B, base_channels*2, H/4, W/4]
        # Bottleneck
        b = self.bottleneck(d2)  # [B, base_channels*4, H/4, W/4]
        # Decoder with skip connections
        u2 = self.up2(b)  # [B, base_channels*2, H/2, W/2]
        u2 = torch.cat([u2, e2], dim=1)  # [B, base_channels*4, H/2, W/2]
        d2 = self.dec2(u2)  # [B, base_channels*2, H/2, W/2]
        u1 = self.up1(d2)  # [B, base_channels, H, W]
        u1 = torch.cat([u1, e1], dim=1)  # [B, base_channels*2, H, W]
        d1 = self.dec1(u1)  # [B, base_channels, H, W]
        out = self.out_conv(d1)
        return out


class CryoMomentsConditionalEncoder(nn.Module):
    """
    Neural network for conditional encoding of cryo-EM moment features.

    This module fuses information from the second moment subspace (set of orthogonal images)
    and the first moment image (mean projection) for downstream tasks such as distribution modeling.
    - The second moment subspace is processed via attention over learned queries, producing a set of attended images.
    - The first moment image is processed in parallel by a compact UNet (FirstMomentUNet) to extract spatial features.
    - The attended second moment images and first moment features are concatenated channel-wise and passed through a CNN.
    - The output is a conditional feature vector for use in downstream models.

    Args:
        output_dim (int): Output feature dimension from the final linear layer.
        unet_out_channels (int): Number of output channels from the first moment UNet.
    """
    def __init__(self, output_dim, num_queries_eig, unet_out_channels, D):
        super().__init__()
        self.output_dim = output_dim
        self.num_queries_eig = num_queries_eig
        self.unet_out_channels = unet_out_channels
        self.D = D
        self.queries_eig = nn.Parameter(torch.randn(num_queries_eig, D))
        self.first_moment_unet = FirstMomentUNet(out_channels=unet_out_channels)
        # Conv to reduce attended images channels
        reduced_channels = num_queries_eig // 2
        self.second_moment_conv = nn.Conv2d(num_queries_eig, reduced_channels, kernel_size=3, padding=1)
        self.reduced_channels = reduced_channels
        # CNN stack: [B, C, H, W] -> [B, final_out_ch, 4, 4]
        l = int(D ** 0.5)
        in_ch = reduced_channels + unet_out_channels
        num_layers = int(math.log2(l // 4))
        chs = [256, 128, 128, 64]
        layers = []
        for i in range(num_layers):
            out_ch = chs[i] if i < len(chs) else chs[-1]
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        # Final linear layer
        final_feat_dim = out_ch * 4 * 4
        self.final_linear = nn.Linear(final_feat_dim, output_dim)

    def forward(self, second_moment_subspace, first_moment_image, eigen_values, return_basis_images=False):
        """
        Args:
            second_moment_subspace (Tensor): [B, N, D] tensor of orthogonal second moment images (flattened D=l^2).
            first_moment_image (Tensor): [B, l, l] tensor of the first moment (mean projection) image.
            eigen_values (Tensor): [B, N] tensor of eigenvalues for weighting the second moment subspace.
            return_basis_images (bool): If True, also return the learned basis images (for visualization/debug).

        Returns:
            cond_feat (Tensor): [B, output_dim] conditional feature vector for downstream tasks.
            basis_images (Tensor, optional): [N, l, l] learned basis images (if return_basis_images=True).
        """
        B, N, D = second_moment_subspace.shape
        # --- Eigenvalue attention ---
        eig_weighted = second_moment_subspace * eigen_values.sqrt().unsqueeze(-1)  # [B, N, D]
        # --- Attention weights and queries ---
        attn_logits = torch.matmul(eig_weighted, self.queries_eig.t()) * (D ** -0.5)  # [B, N, num_queries_eig]
        attn_logits = attn_logits.permute(0, 2, 1)  # [B, num_queries_eig, N]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, num_queries_eig, N]
        attended = torch.bmm(attn_weights, eig_weighted)  # [B, num_queries_eig, D]
        # Reshape to images
        l = int(D ** 0.5)
        attended_images = attended.view(B, self.num_queries_eig, l, l)
        attended_images_reduced = self.second_moment_conv(attended_images)  # [B, reduced_channels, l, l]
        # --- UNet for first moment image ---
        if first_moment_image.ndim == 3:
            first_moment_image = first_moment_image.unsqueeze(1)  # [B, 1, l, l]
        unet_out = self.first_moment_unet(first_moment_image)  # [B, unet_out_channels, l, l]
        # Concatenate UNet output to reduced attended images along channel dim
        cnn_input = torch.cat([attended_images_reduced, unet_out], dim=1)  # [B, reduced_channels+unet_out_channels, l, l]
        cond_feat = self.cnn(cnn_input)
        cond_feat = cond_feat.view(cond_feat.size(0), -1)
        cond_feat = self.final_linear(cond_feat)
        if return_basis_images:
            # Return the actual basis images (queries) as well, reshaped
            l = int(D ** 0.5)
            basis_eig = self.queries_eig.view(-1, l, l) if hasattr(self, 'queries_eig') else None
            return cond_feat, basis_eig
        return cond_feat

