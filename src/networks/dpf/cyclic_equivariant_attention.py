import torch
import torch.nn as nn
import torch.nn.functional as F

class CyclicEquivariantAttentionLayer(nn.Module):
    def __init__(self, Pms_in, Pms_out, D, d_k, d_v, num_heads, ff_hidden=128, ff_layers=2, nonlinearity=F.relu):
        """
        Args:
            Pms_in: dict mapping m to P_m_in (D x N_m) torch.complex64 tensors (input projections)
            Pms_out: dict mapping m to P_m_out (R x N_m) torch.complex64 tensors (output projections)
            D: projection dimension
            d_k: attention key/query dimension
            d_v: attention value dimension
            num_heads: number of attention heads
            ff_hidden: hidden dim for feedforward
            ff_layers: number of feedforward layers
            nonlinearity: nonlinearity for head mixing and feedforward
        """
        super().__init__()
        self.Pms_in = Pms_in
        self.Pms_out = Pms_out
        self.R = Pms_in[self.ms[0]].shape[1]
        self.ms = sorted(Pms_in.keys())
        self.D = D
        self.num_modes = len(self.ms)
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.nonlinearity = nonlinearity

        # Attention weights for each head
        self.W_Q = nn.Parameter(torch.randn(num_heads, 1, 1, self.R, d_k)) # [H, 1, 1, R, d_k]
        self.W_K = nn.Parameter(torch.randn(num_heads, 1, 1, self.R, d_k))
        self.W_V = nn.Parameter(torch.randn(num_heads, 1, 1, self.R, d_v))

        # Head mixing
        self.W_H = nn.Parameter(torch.randn(num_heads, num_heads))

        # Feedforward layers (row-wise)
        ff_layers_list = []
        in_dim = d_v * num_heads
        for _ in range(ff_layers - 1):
            ff_layers_list.append(nn.Linear(in_dim, ff_hidden))
            ff_layers_list.append(nn.ReLU())
            in_dim = ff_hidden
        ff_layers_list.append(nn.Linear(in_dim, self.R))
        self.feedforward = nn.Sequential(*ff_layers_list)

    def forward(self, V):
        """
        Args:
            V: dict mapping m to V_m (N_m x R) torch.complex64 tensors
        Returns:
            out: dict mapping m to output (N_m x R) torch.complex64 tensors
        """
        device = self.W_Q.device
        ms = self.ms
        D = self.D
        H = self.num_heads
        d_k = self.d_k
        d_v = self.d_v
        num_modes = self.num_modes
        T = 2 * num_modes - 1
        R = self.R

        # Project each mode to D x R (only m >= 0, hermitian symmetry)
        A_hat = []
        for m in ms:
            Pm_in = self.Pms_in[m].to(device)  # [D, N_m]
            V_m = V[m].to(device)         # [N_m, R]
            A_hat_m = torch.matmul(Pm_in, V_m)  # [D, R]
            A_hat.append(A_hat_m.unsqueeze(0))
        A_hat = torch.cat(A_hat, dim=0)  # [num_modes, D, R]

        # Inverse real FFT along mode index to get A_t: [num_modes, D, R] -> [T, D, R]
        # T = 2*num_modes-1
        A_t = torch.fft.irfft(A_hat, n=T, dim=0)  # [T, D, R], real

        # Multi-head attention at each t
        A_t_heads = []
        for h in range(H):
            # Linear projections: [T, D, R] x [1, 1, R, d_k/d_v] -> [T, D, d_k/d_v]
            Q_h = torch.matmul(A_t, self.W_Q[h])  # [T, D, d_k]
            K_h = torch.matmul(A_t, self.W_K[h])  # [T, D, d_k]
            V_h = torch.matmul(A_t, self.W_V[h]) # [T, D, d_v]

            # Attention scores: [T, D, d_k] x [T, d_k, D] -> [T, D, D]
            attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / (d_k ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # softmax over D

            # Output: [T, D, D] x [T, D, d_v] -> [T, D, d_v]
            A_t_h = torch.matmul(attn_weights, V_h)
            A_t_heads.append(A_t_h.unsqueeze(-1))  # [T, D, d_v, 1]
        A_t_heads = torch.cat(A_t_heads, dim=-1)  # [T, D, d_v, H]

        # Head mixing: [T, D, d_v, H] x [H, H] -> [T, D, d_v, H]
        A_t_heads_mixed = torch.einsum('tdvh,hp->tdvp', A_t_heads, self.W_H)
        A_t_heads_mixed = self.nonlinearity(A_t_heads_mixed)  # [T, D, d_v, H]

        # Flatten head dimension and apply row-wise feedforward: [T, D, d_v*H]
        A_t_ff = A_t_heads_mixed.reshape(T, D, d_v * H)
        A_t_ff = self.feedforward(A_t_ff)  # [T, D, R]

        # Real FFT along t to recover mode structure: [T, D, R] -> [num_modes, D, R]
        A_hat_out = torch.fft.rfft(A_t_ff, dim=0)  # [num_modes, D, R]

        # For each mode, project back to N_m x R using Pms_out
        out = {}
        for i, m in enumerate(ms):
            # [D, R] x [R, N_m] -> [N_m, R]
            out_m = torch.matmul(self.Pms_out[str(m)].conj().transpose(0, 1), A_hat_out[i])  # [N_m, R]
            out[m] = out_m
        return out