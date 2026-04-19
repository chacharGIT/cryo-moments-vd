import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import BertConfig, BertModel

from config.config import settings

class Moments1DUNET(nn.Module):
    """
    Compact 1D UNet-style convolutional encoder for 1D signals.

    Args:
        in_channels: Number of input channels C_in
        out_channels: Number of output channels C_out
        kernel_size: Convolution kernel size
        nonlinearity: Activation module instance

    Shapes:
        Input:
            x: [B, L] or [B, C_in, L]
        Output:
            y: [B, C_out, L]
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, nonlinearity=nn.SiLU(inplace=True)):
        super().__init__()
        self.nonlinearity = nonlinearity

        g1 = min(8, out_channels)
        g2 = min(8, out_channels * 2)
        g4 = min(8, out_channels * 4)

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
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
        if x.dim() == 2:
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
    
def fixed_fourier_k_embed(k: torch.Tensor, bands: int, max_abs_k: int):
    """
    Fixed (non-learned) Fourier feature embedding for an integer Fourier mode k.

    Args:
        k: Scalar tensor (typically integer-valued) representing Fourier mode.
        bands: Number of frequency bands.
        max_abs_k: Used to normalize k into approximately [-1, 1] via k/max_abs_k.

    Returns:
        emb: Tensor of shape [2*bands] with concatenated sin/cos features.
    """
    k = k.to(torch.float32) / float(max_abs_k)
    freqs = (2.0 ** torch.arange(bands, device=k.device, dtype=torch.float32)) * torch.pi
    x = k * freqs  # [bands]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=0)  # [2*bands]

def fixed_source_embed(src_id: torch.Tensor, bands: int):
    """
    Fixed (non-learned) embedding that indicates the source/type of a token.

    Used to distinguish tokens coming from different moment types, e.g.:
      - src_id = 0.0 : first moment tokens
      - src_id = 1.0 : second moment tokens

    Args:
        src_id: Scalar tensor (float is fine) identifying the token source.
        bands: Number of frequency bands.

    Returns:
        emb: Tensor of shape [2*bands].
    """
    src_id = src_id.to(torch.float32)
    freqs = (2.0 ** torch.arange(bands, device=src_id.device, dtype=torch.float32))
    x = src_id * freqs
    return torch.cat([torch.sin(x), torch.cos(x)], dim=0)  # [2*bands]

class TokenPooling(nn.Module):
    """
    Cross-attention pooling from a variable-length token sequence to a fixed
    number of conditional query vectors.

    The module maintains `num_queries` learnable query vectors (shape [Q, H]) and performs
    cross-attention over the input token sequence to produce a fixed-size representation.

    Args:
        hidden_size: Token embedding dimension H
        latent_dim: Output dimension after projection
        num_queries: Number of pooled query outputs Q
        heads: Number of attention heads for cross-attention
    
    Output:
        pooled: [B, Q, latent_dim]
    """
    def __init__(self, hidden_size, latent_dim, num_queries, heads=8):
        super().__init__()

        self.num_queries = num_queries

        self.query = nn.Parameter(
            torch.randn(num_queries, hidden_size) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            batch_first=True
        )

        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens, mask=None):
        # tokens: [B, T, H]

        B = tokens.shape[0]

        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B,Q,H]

        pooled, _ = self.cross_attn(
            q,        # queries
            tokens,   # keys
            tokens,   # values
            key_padding_mask=(mask == 0) if mask is not None else None
        )

        return self.proj(pooled)     # [B,Q,latent_dim]

class DictComplexToBert(nn.Module):
    """
    Builds a token sequence from first- and second-moment radial inputs, encodes it with a
    BERT-style transformer, and pools to a fixed number of conditional query vectors.

    Tokenization:
        - First moment:
            first_moment_radial: [B, R] (real)
            Moments1DUNET(1->128): [B, N1, R]
            Linear(R->H):          [B, N1, H]
            Adds k-embed (k=0) and a moment-type embed (src=0).
        - Second moment (per Fourier mode k):
            second_moment_radial_subspace[k]: [B, N_k, R] (complex)
            Weighted by sqrt(eigenvalues[k]).
            Stack real/imag -> Moments1DUNET(2->1) -> Linear(R->H): [B, N_k, H]
            Adds k-embed (k) and a moment-type embed (src=1).
            mask_dict[k] marks valid tokens (1=valid, 0=masked).

    Returns:
        cond: [B, num_cond_queries, latent_dim]
    """
    def __init__(self, hidden_size=256, bands=20, max_abs_k=67, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.bands = bands
        self.max_abs_k = max_abs_k
        self.latent_dim = settings.dpf.perceiver.latent_dim
        self.num_cond_queries = settings.dpf.conditional_separated_moment_encoder.num_cond_queries
        self.R = settings.dpf.conditional_separated_moment_encoder.radial_vector_length

        cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=1,
            type_vocab_size=2,
        )
        self.bert = BertModel(cfg, add_pooling_layer=False)

        self.first_moment_1d_unet = Moments1DUNET(in_channels=1, out_channels=128, kernel_size=5)
        self.first_moment_linear = nn.Linear(self.R, hidden_size)
        self.second_moment_1d_unet = Moments1DUNET(in_channels=2, out_channels=1, kernel_size=5)
        self.second_moment_linear = nn.Linear(self.R, hidden_size)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + 4*bands, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.pool = TokenPooling(
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_queries=self.num_cond_queries,
            heads=heads
        )

    def forward(self, second_moment_radial_subspace, second_moment_eigen_values, first_moment_radial, mask_dict):
        ks = sorted(second_moment_radial_subspace.keys())
        tok_list = []
        mask_list = []
        token_type_list = []

        x1 = self.first_moment_1d_unet(first_moment_radial)          # [B,N1,R]
        x1_tokens = self.first_moment_linear(x1.to(torch.float32))   # [B,N1,H]
        B, N1, _ = x1_tokens.shape

        if not torch.isfinite(x1_tokens).all():
            raise RuntimeError("Non-finite first-moment tokens (check first_moment_radial input).")
        
        k0_emb = fixed_fourier_k_embed(
            torch.tensor(0.0, device=x1_tokens.device),
            self.bands,
            self.max_abs_k
        ).to(torch.float32).view(1, 1, -1).expand(B, N1, -1)        # [B,N1,2*bands]


        # use a fixed "source" embed in the same slot as k-embed
        src1_emb = fixed_source_embed(
            torch.tensor(0.0, device=x1_tokens.device),  # first moment
            self.bands
        ).to(torch.float32).view(1, 1, -1).expand(B, N1, -1)        # [B,N1,2*bands]

        feat1 = torch.cat([x1_tokens, k0_emb, src1_emb], dim=-1)    # [B,N1,H+4*bands]
        mask1 = torch.ones((B, N1), device=x1_tokens.device, dtype=torch.long)  # [B,N1]
        tok_list.append(feat1)
        mask_list.append(mask1)
        token_type_list.append(torch.full((B, N1), 0, device=x1_tokens.device, dtype=torch.long))

        for k in ks:
            x = second_moment_radial_subspace[k]  # [B, N_k, R] complex
            current_mask = mask_dict[k].to(dtype=torch.long, device=x.device)  # [B, N_k]
            finite_vec = torch.isfinite(x.real).all(dim=-1) & torch.isfinite(x.imag).all(dim=-1)  # [B,N_k]
            bad_vec = (~finite_vec) & (current_mask != 0)
            if bad_vec.any():
                print(f"[WARN] Non-finite second-moment vectors for k={k}: {bad_vec.sum().item()}", flush=True)
            # Zero and mask bad vectors
            xr = torch.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0)
            xi = torch.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.complex(xr, xi)                    
            current_mask = current_mask & finite_vec.to(dtype=torch.long)    

            lambdas_k = second_moment_eigen_values[k]
            if torch.is_complex(lambdas_k) or lambdas_k.real.min() < 0:
                print(f"[WARN] Complex or negative eigenvalues for k={k}")
                lambdas_k = torch.real(lambdas_k)
                lambdas_k = torch.clamp(lambdas_k, min=0.0)

            x = x * torch.sqrt(lambdas_k).unsqueeze(-1)  # [B, N_k, R]
            if not torch.isfinite(x.real).all() or not torch.isfinite(x.imag).all():
                raise RuntimeError(f"Non-finite eig-weighted second-moment input for k={k} (check eigenvalues).")

            _, N_k, R = x.shape

            if N_k == 0:
                continue

            x_ri = torch.stack([x.real, x.imag], dim=2)          # [B, N_k,2,R]
            x_ri = x_ri.reshape(B * N_k, 2, R).to(torch.float32)   # [B*N_K,2,R]

            x_real = self.second_moment_1d_unet(x_ri)            # [B*N_k,1,R]
            x_real = x_real.squeeze(1)                           # [B*N_k,R]
            vec_emb = self.second_moment_linear(x_real)          # [B*N_k,H]
            vec_emb = vec_emb.view(B, N_k, self.hidden_size)     # [B,N_k,H]
            if not torch.isfinite(vec_emb).all():
                raise RuntimeError(f"Non-finite second-moment embeddings for k={k} (check input).")


            k_emb = fixed_fourier_k_embed(
                torch.tensor(float(k), device=x.device),
                self.bands,
                self.max_abs_k
            ).to(torch.float32)                                   # [2*bands]
            k_emb = k_emb.view(1, 1, -1).expand(B, N_k, -1)          # [B,N_k,2*bands]

            src2_emb = fixed_source_embed(
                torch.tensor(1.0, device=x.device),  # second moment
                self.bands
            ).to(torch.float32).view(1, 1, -1).expand(B, N_k, -1)      # [B,Nk,2*bands]
            
            feat = torch.cat([vec_emb, k_emb, src2_emb], dim=-1)             # [B,N_k,H+4*bands]
            # Zero out masked tokens
            feat = feat * current_mask.to(feat.dtype).unsqueeze(-1)
            tok_list.append(feat)
            mask_list.append(current_mask)
            token_type_list.append(torch.full((B, N_k), 1, device=x.device, dtype=torch.long))

        feats = torch.cat(tok_list, dim=1)                         # [B,T,H+4*bands]
        attn_mask = torch.cat(mask_list, dim=1)                    # [B,T]
        embeds = self.ln(self.fuse(feats))                         # [B,T,H]
        token_type_ids = torch.cat(token_type_list, dim=1)         # [B,T]

        B, T, _ = embeds.shape
        position_ids = torch.zeros((B, T), device=embeds.device, dtype=torch.long)   # all tokens share pos=0
        out = self.bert(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        tokens = out.last_hidden_state   # [B,T,H]
        cond = self.pool(tokens, attn_mask)
        return cond # [B, num_cond_queries, latent_dim]
    