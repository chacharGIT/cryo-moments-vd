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

    def forward(self, Q, K, V):
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
        attn_weights = F.softmax(torch.abs(attn_scores), dim=-1)
        attn_out = torch.matmul(attn_weights, V_proj)  # [B, Nh, Nq, d_v]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, Nh * self.d_v)
        return self.W_O(attn_out)  # [B, Nq, d_model]

class SetTransformerBlock(nn.Module):
    """
    Maps a variable-size set ˜Vm (B x Ñm x R) to a fixed-size Vm (B x Nm x R)
    using a self-attention block followed by pooling by multihead attention (with learnable queries).
    """
    def __init__(self, R, d_k, d_v, num_heads, ff_hidden, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, R, dtype=torch.cfloat))
        self.mha = MultiHeadAttention(R, d_k, d_v, num_heads)
        self.amp_nl = AmpNonlinearity()
        self.ff = nn.Sequential(
            ComplexLinear(R, ff_hidden),
            AmpNonlinearity(),
            ComplexLinear(ff_hidden, R)
        )

    def forward(self, X):
        # Self-attention block (SAB)
        H = X + self.mha(X, X, X)
        H = H + self.amp_nl(self.ff(H))
        # Pooling by multihead attention (PMA)
        B = X.shape[0]
        Q = self.queries.expand(B, -1, -1)  # [B, num_queries, R]
        H = self.mha(Q, H, H)
        H = H + self.amp_nl(self.ff(H))
        return H  # [B, num_queries, R]
    