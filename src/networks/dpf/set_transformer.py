import torch
import torch.nn as nn
import torch.nn.functional as F

class AmpNonlinearity(nn.Module):
    def __init__(self, nonlinearity=torch.tanh):
        super().__init__()
        self.nonlinearity = nonlinearity
    def forward(self, z):
        amp = torch.abs(z)
        phase = torch.angle(z)
        amp_nl = self.nonlinearity(amp)
        return amp_nl * torch.exp(1j * phase)
    
class AmpLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, z):
        amp = torch.abs(z)                  # [*, R]
        amp_norm = self.ln(amp)             # normalized amplitude
        scale = amp_norm / (amp + self.eps) # real, >=0
        return z * scale                    # rescales magnitude, same phase

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, input):
        real = input.real
        imag = input.imag
        real_out = torch.matmul(real, self.real_weight.t()) - torch.matmul(imag, self.imag_weight.t())
        imag_out = torch.matmul(real, self.imag_weight.t()) + torch.matmul(imag, self.real_weight.t())
        return torch.complex(real_out, imag_out)

class MultiHeadAttention(nn.Module):
    def __init__(self, R, d_k, d_v, num_heads, use_query_weights=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.use_query_weights = use_query_weights
        if self.use_query_weights:
            self.W_Q = ComplexLinear(R, d_k * num_heads)
        self.W_K = ComplexLinear(R, d_k * num_heads)
        self.W_V = ComplexLinear(R, d_v * num_heads)
        self.W_O = ComplexLinear(d_v * num_heads, R)

    def forward(self, Q, K, V, mask=None):
        B, Nq, _ = Q.shape
        Nk = K.shape[1]
        Nh = self.num_heads

        if self.use_query_weights:
            Q_proj = self.W_Q(Q).view(B, Nq, Nh, self.d_k).transpose(1, 2)  # [B, Nh, Nq, d_k]
        else:
            Q_proj = Q.view(B, Nq, Nh, self.d_k).transpose(1, 2)  # [B, Nh, Nq, d_k]
        K_proj = self.W_K(K).view(B, Nk, Nh, self.d_k).transpose(1, 2)  # [B, Nh, Nk, d_k]
        V_proj = self.W_V(V).view(B, Nk, Nh, self.d_v).transpose(1, 2)  # [B, Nh, Nk, d_v]
        attn_scores = torch.matmul(Q_proj, K_proj.conj().transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, Nh, Nq, Nk]
        attn_scores_abs = torch.abs(attn_scores)
        attn_weights = torch.zeros_like(attn_scores_abs)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Nk]
            attn_scores_abs = attn_scores_abs.masked_fill(~mask, float('-inf'))
            # Keep rows with at least one valid key
            valid_queries = mask.any(dim=-1)  # [B, 1, 1, Nk] -> [B, 1, 1] over Nk
            valid_queries = valid_queries.expand(-1, Nh, Nq)  # [B, Nh, Nq]

            if valid_queries.any():
                # flatten (B, Nh, Nq, Nk) to (N_valid, Nk) for the valid rows
                scores_valid = attn_scores_abs[valid_queries]  # [N_valid, Nk]
                weights_valid = F.softmax(scores_valid, dim=-1)  # [N_valid, Nk]
                attn_weights[valid_queries] = weights_valid.view(-1, Nk)
        else:
            attn_weights = F.softmax(attn_scores_abs, dim=-1)
            
        attn_weights = attn_weights.to(V_proj.dtype)  # Ensure same dtype
        attn_out = torch.matmul(attn_weights, V_proj)  # [B, Nh, Nq, d_v]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, Nh * self.d_v)
        return self.W_O(attn_out)  # [B, Nq, d_model]

class SetTransformerBlock(nn.Module):
    """
    Maps a variable-size set ˜Vm (B x Ñm x R) to a fixed-size Vm (B x Nm x R)
    using a self-attention block followed by pooling by multihead attention (with learnable queries).
    """
    def __init__(self, R, d_k, d_v, num_heads, ff_hidden_dim, n_queries_per_m):
        super().__init__()
        self.mha_sab = MultiHeadAttention(R, d_k, d_v, num_heads)
        self.mha_pma = MultiHeadAttention(R, d_k, d_v, num_heads, use_query_weights=False)
        self.amp_nl = AmpNonlinearity()
        self.ff = nn.Sequential(
            ComplexLinear(R, ff_hidden_dim),
            AmpNonlinearity(),
            ComplexLinear(ff_hidden_dim, R)
        )
        self.num_heads = num_heads
        self.d_k = d_k

        self.ln_sab1 = AmpLayerNorm(R)
        self.ln_sab2 = AmpLayerNorm(R)
        self.ln_pma1 = AmpLayerNorm(R)
        self.ln_pma2 = AmpLayerNorm(R)

        self.queries_dict = nn.ParameterDict()

        for m in range(len(n_queries_per_m)):
            n_q = int(n_queries_per_m[m])
            key = str(m)
            self.queries_dict[key] = nn.Parameter(
                torch.randn(1, n_q, self.num_heads * self.d_k, dtype=torch.cfloat)
            )

    def forward(self, X, m, mask=None):
        """
        X: [B, Ñm, R] (complex)
        m: frequency index (int)
        num_queries: number of queries for this m (int)
        Returns: [B, num_queries, R] (complex)
        """
        B, N_tilde, R = X.shape

        # Handle empty input set: return zeros with fixed size [B, num_queries, R]
        if N_tilde == 0:
            n_q = self.queries_dict[str(m)].shape[1]
            out = torch.zeros(B, n_q, R, dtype=torch.cfloat, device=X.device)
            if m == 0:
                out = out.real   # keep strictly real for m=0
            return out # [B, num_queries, R]
        # Self-attention block (SAB)
        H = X + self.mha_sab(X, X, X, mask=mask)
        H = self.ln_sab1(H)
        H = H + self.amp_nl(self.ff(H))
        H = self.ln_sab2(H)
        # Pooling by multihead attention (PMA)
        Q = self.queries_dict[str(m)].expand(B, -1, -1)  # [B, num_queries, R]
        H = self.mha_pma(Q, H, H, mask=mask)
        H = self.ln_pma1(H)
        H = H + self.amp_nl(self.ff(H))
        H = self.ln_pma2(H)
        if m == 0:
            H = H.real  # For m=0, output real values
        return H  # [B, num_queries, R]
    