import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import settings
from src.networks.dpf.set_transformer import ComplexLinear

class CyclicEquivariantAttentionBlock(nn.Module):
    def __init__(self, n_queries_per_m, d_k, d_v, D, R, num_heads,
                 ff_hidden_dim, nonlinearity, project_back=True):
        """Cyclic equivariant attention with learned per-mode complex projections.

        Args:
            n_queries_per_m: 1D array/list, N_m for m=0..M (len = num_modes)
            d_k: attention key/query dimension
            d_v: attention value dimension
            D: latent dimension after per-mode projection
            R: feature dimension along the radial axis
            num_heads: number of attention heads
            ff_hidden_dim: hidden dim for feedforward
            nonlinearity: nonlinearity for head mixing and feedforward
        """
        super().__init__()
        self.R = R
        self.D = D
        self.project_back = project_back
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.nonlinearity = nonlinearity

        # Mode indices as strings: "0", "1", ...
        self.ms = [str(m) for m in range(len(n_queries_per_m))]
        self.num_modes = len(self.ms)

        # Learned per-mode complex projections using ComplexLinear
        # Qms_in[m]: ComplexLinear(N_m -> D), Qms_out[m]: ComplexLinear(D -> N_m)
        self.Qms_in = nn.ModuleDict()
        self.Qms_out = nn.ModuleDict()
        for m_idx, N_m in enumerate(n_queries_per_m):
            key = str(m_idx)
            N_m_int = int(N_m)
            self.Qms_in[key] = ComplexLinear(N_m_int, D)
            self.Qms_out[key] = ComplexLinear(D, N_m_int)

        # For m = 0, start with purely real projections (imag weights = 0)
        with torch.no_grad():
            self.Qms_in["0"].imag_weight.zero_()
            self.Qms_out["0"].imag_weight.zero_()

        # Attention weights for each head
        self.W_Q = nn.Parameter(torch.randn(num_heads, self.R, d_k))
        self.W_K = nn.Parameter(torch.randn(num_heads, self.R, d_k))
        self.W_V = nn.Parameter(torch.randn(num_heads, self.R, d_v))

        # Head mixing
        self.W_H = nn.Parameter(torch.randn(num_heads, num_heads))

        # Feedforward layers (row-wise)
        in_dim = d_v * num_heads
        self.post_mha_linear = nn.Sequential(
            nn.Linear(in_dim, R),
            self.nonlinearity
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(R, ff_hidden_dim),
            self.nonlinearity,
            nn.Dropout(0.05),
            nn.Linear(ff_hidden_dim, R)
        )

        self.ln_attn_in = nn.LayerNorm(R)
        self.ln_ff_in = nn.LayerNorm(R)

    def forward(self, V):
        """
        Args:
            V: dict mapping m (str) -> V_m (B x N_m x R) torch.complex64 tensors
               (B: batch size, N_m: number of queries for mode m, R: feature dim)
        Returns:
            out: dict mapping m (str) -> (B x N_m x R) torch.complex64 tensors
        """
        ms = self.ms
        D = self.D
        H = self.num_heads
        d_k = self.d_k
        d_v = self.d_v
        num_modes = self.num_modes
        T = 2 * num_modes - 1
        R = self.R
        B = V[ms[0]].shape[0]

        # Project each mode to D x R via learned complex linears (only m >= 0, hermitian symmetry)
        A_hat = []
        for m in ms:
            V_m = V[m]
            if m == "0":
                V_m = V_m.real.to(torch.cfloat)
            # V_m: [B, N_m, R] -> [B, R, N_m] so ComplexLinear acts on last dim N_m
            V_m_perm = V_m.permute(0, 2, 1)       # [B, R, N_m]
            A_hat_m = self.Qms_in[m](V_m_perm)    # [B, R, D]
            A_hat_m = A_hat_m.permute(0, 2, 1)    # [B, D, R]
            A_hat.append(A_hat_m.unsqueeze(1))
        A_hat = torch.cat(A_hat, dim=1)           # [B, num_modes, D, R]

        # Inverse real FFT along mode index to get A_t: [B, T, D, R]
        A_t = torch.fft.irfft(A_hat, n=T, dim=1) # [B, T, D, R], real

        A_t = self.ln_attn_in(A_t)
        # Multi-head attention at each t. Projections for all heads at once:
        # A_t: [B,T,D,R], W_Q: [H,R,d_k] -> Q: [B,T,H,D,d_k]
        Q = torch.einsum('btdr,hrk->bthdk', A_t, self.W_Q)
        K = torch.einsum('btdr,hrk->bthdk', A_t, self.W_K)
        Vv = torch.einsum('btdr,hrv->bthdv', A_t, self.W_V)
        Q = Q.reshape(B * T, H, D, d_k)   # [BT, H, D, d_k]
        K = K.reshape(B * T, H, D, d_k)   # [BT, H, D, d_k]
        Vv = Vv.reshape(B * T, H, D, d_v)  # [BT, H, D, d_v]
        attn_out = F.scaled_dot_product_attention(
        Q, K, Vv, attn_mask=None, dropout_p=0.05, is_causal=False
        )
        A_t_heads = attn_out.reshape(B, T, H, D, d_v) # [B,T,H,D,d_v]
        A_t_heads = A_t_heads.permute(0, 1, 3, 4, 2)  # [B,T,D,d_v,H]

        # Head mixing: [B, T, D, d_v, H] x [H, H] -> [B, T, D, d_v, H]
        A_t_heads_mixed = torch.einsum('btdvh,hp->btdvp', A_t_heads, self.W_H)
        A_t_heads_mixed = self.nonlinearity(A_t_heads_mixed)

        # Flatten head dim, apply feedforward row-wise: [B, T, D, d_v*H] -> [B, T, D, R]
        A_t_mha = A_t_heads_mixed.reshape(B, T, D, d_v * H)
        A_t = A_t + self.post_mha_linear(A_t_mha)  # Residual connection
        A_t = self.ln_ff_in(A_t)
        A_t = A_t + self.feedforward(A_t) # [B, T, D, R]

        if self.project_back:
            # Real FFT along t to recover modes: [B, T, D, R] -> [B, num_modes, D, R]
            A_hat_out = torch.fft.rfft(A_t, dim=1)   # [B, num_modes, D, R]
            # For each mode, project back to N_m x R using Qms_out
            out = {}
            for i, m in enumerate(ms):
                A_hat_out_m = A_hat_out[:, i]                # [B, D, R]
                A_hat_out_perm = A_hat_out_m.permute(0, 2, 1)  # [B, R, D]
                out_m = self.Qms_out[m](A_hat_out_perm)        # [B, R, N_m]
                out[m] = out_m.permute(0, 2, 1)                # [B, N_m, R]
            return out
        else:
            return A_t
        