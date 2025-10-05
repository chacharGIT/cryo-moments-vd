import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalMomentEncoder(nn.Module):
    """
    Encodes a variable set of orthogonal input images into a fixed set of learned basis images,
    then processes them with a CNN for conditional feature extraction.

    Args:
        input_dim (int): Flattened dimension of each input image (D = H*W).
        num_queries (int): Number of learned basis elements (M).
        image_shape (tuple): (H, W) shape for reshaping basis images.
        output_dim (int): Output feature dimension from the final linear layer.
    """
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.queries = None  # Will be initialized on first forward
        self.scale = None
        self.num_queries = None
        self.input_dim = None
        self.image_shape = None

    def forward(self, keys, first_moment_projections, eigen_values, return_basis_images=False):
        """
        Args:
            keys: [B, N, D] tensor of orthogonal input images (flattened)
            values: [B, N, D] tensor (optional, defaults to keys)
        Returns:
            cond_feat: [B, cnn_out_dim] conditional feature vector
            basis_images: [B, M, H, W] learned basis images (for visualization/debug)
        """
        B, N, D = keys.shape
        # --- First moment attention ---
        valid_fmp = (first_moment_projections != 0)  # [B, N]
        keys_fmp_list = []
        for b in range(B):
            keys_fmp_b = keys[b][valid_fmp[b]] * first_moment_projections[b][valid_fmp[b]].unsqueeze(-1)  # [N_fmp, D]
            keys_fmp_list.append(keys_fmp_b)
        max_len_fmp = max(x.shape[0] for x in keys_fmp_list)
        keys_fmp = torch.stack([F.pad(x, (0,0,0,max_len_fmp-x.shape[0])) for x in keys_fmp_list], dim=0)  # [B, max_len_fmp, D]
        # --- Eigenvalue attention ---
        keys_eig = keys * eigen_values.unsqueeze(-1)  # [B, N, D]
        max_len_eig = N
        # --- Attention weights and queries ---
        # Create separate queries for each attention
        if (not hasattr(self, 'queries_fmp')) or (self.queries_fmp is None) or (self.queries_fmp.shape[0] != max_len_fmp):
            self.queries_fmp = nn.Parameter(torch.randn(max_len_fmp, D, device=keys.device))
        if (not hasattr(self, 'queries_eig')) or (self.queries_eig is None) or (self.queries_eig.shape[0] != max_len_eig):
            self.queries_eig = nn.Parameter(torch.randn(max_len_eig, D, device=keys.device))
        scale_fmp = D ** -0.5
        scale_eig = D ** -0.5
        # Attention for first moment
        attn_logits_fmp = torch.matmul(self.queries_fmp, keys_fmp.transpose(1,2)) * scale_fmp  # [max_len_fmp, B, max_len_fmp]
        attn_logits_fmp = attn_logits_fmp.permute(1,0,2)  # [B, max_len_fmp, max_len_fmp]
        attn_weights_fmp = F.softmax(attn_logits_fmp, dim=-1)  # [B, max_len_fmp, max_len_fmp]
        O_fmp = torch.bmm(attn_weights_fmp, keys_fmp)  # [B, max_len_fmp, D]
        # Attention for eigenvalues
        attn_logits_eig = torch.matmul(self.queries_eig, keys_eig.transpose(1,2)) * scale_eig  # [max_len_eig, B, max_len_eig]
        attn_logits_eig = attn_logits_eig.permute(1,0,2)  # [B, max_len_eig, max_len_eig]
        attn_weights_eig = F.softmax(attn_logits_eig, dim=-1)  # [B, max_len_eig, max_len_eig]
        O_eig = torch.bmm(attn_weights_eig, keys_eig)  # [B, max_len_eig, D]
        # Reshape to images and concatenate as channels
        l = int(D ** 0.5)
        attended_images_fmp = O_fmp.view(B, max_len_fmp, l, l)
        attended_images_eig = O_eig.view(B, max_len_eig, l, l)
        attended_images = torch.cat([attended_images_fmp, attended_images_eig], dim=1)  # [B, max_len_fmp+max_len_eig, l, l]
        # CNN: [B, C, H, W] -> [B, final_out_ch, 4, 4]
        # Build the CNN dynamically
        in_ch = attended_images.shape[1]
        import math
        num_layers = int(math.log2(l // 4))
        chs = [256, 128, 128, 64]
        layers = []
        for i in range(num_layers):
            out_ch = chs[i] if i < len(chs) else chs[-1]
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        cnn = nn.Sequential(*layers)
        cond_feat = cnn(attended_images)
        # Flatten and apply a linear layer to get [B, output_dim]
        cond_feat = cond_feat.view(cond_feat.size(0), -1)
        cond_feat = nn.Linear(cond_feat.size(1), self.output_dim, device=cond_feat.device)(cond_feat)
        if return_basis_images:
            # Return the actual basis images (queries) as well, reshaped
            l = int(D ** 0.5)
            basis_fmp = self.queries_fmp.view(-1, l, l) if hasattr(self, 'queries_fmp') else None
            basis_eig = self.queries_eig.view(-1, l, l) if hasattr(self, 'queries_eig') else None
            return cond_feat, (basis_fmp, basis_eig)
        return cond_feat

