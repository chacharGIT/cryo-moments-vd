import torch
import torch.nn as nn

class BilinearEquivariantLayer(nn.Module):
    def __init__(self, D, d, Pms, num_heads):
        """
        Args:
            D (int): Ambient dimension of P_m.
            d (int): Output dimension after projection.
            Pms (dict): Dictionary mapping m to P_m (D x N_m) numpy arrays or torch tensors.
        """
        super().__init__()
        self.D = D
        self.d = d
        self.Pms = {m: torch.tensor(Pms[m], dtype=torch.cfloat) if not torch.is_tensor(Pms[m]) else Pms[m] for m in Pms}
        self.ms = sorted(Pms.keys())
        self.M = max(abs(m) for m in self.ms)
        self.num_heads = num_heads

        # Learnable weight matrices for each head
        self.W1 = nn.Parameter(torch.randn(num_heads, d, D, dtype=torch.float))
        self.W2 = nn.Parameter(torch.randn(num_heads, d, D, dtype=torch.float))
        # Learnable mixing matrix (H x H)
        self.head_mixer = nn.Parameter(torch.randn(num_heads, num_heads, dtype=torch.cfloat))

    def forward(self, V):
        """
        Args:
            V: dict mapping m to V_m (N_m x R) real or complex torch tensors.
        Returns:
            U: dict mapping m to U_m (R x R) complex torch tensors.
        """
        device = self.W1.device
        ms = self.ms
        R = next(iter(V.values())).shape[1]
        d = self.d
        H = self.num_heads
        
        # Only compute for m >= 0 (hermitian symmetry)
        ms_pos = [m for m in ms if m >= 0]
        A_pos = []
        for m in ms_pos:
            P_m = self.Pms[m].to(device)  # [D, N_m]
            V_m = V[m].to(device)         # [N_m, R]
            A_m = torch.matmul(P_m, V_m)  # [D, R]
            A_pos.append(A_m.unsqueeze(0))
        A_pos = torch.cat(A_pos, dim=0)  # [M+1, D, R]
        # Inverse real FFT to get real-space signal (along m)
        A_real = torch.fft.irfft(A_pos, n=2*len(ms_pos)-1, dim=0)  # [2M+1, D, R]

        # Multi-head bilinear product in real space
        U_heads = []
        for h in range(H):
            W1h = self.W1[h]  # [d, D]
            W2h = self.W2[h]  # [d, D]
            U_h = []
            for k in range(A_real.shape[0]):
                W1A = torch.matmul(W1h, A_real[k])  # [d, R]
                W2A = torch.matmul(W2h, A_real[k])  # [d, R]
                U_k = torch.matmul(W1A.conj().transpose(0, 1), W2A)  # [R, d] x [d, R] -> [R, R]
                U_h.append(U_k.unsqueeze(0))
            U_h = torch.cat(U_h, dim=0)  # [2M+1, R, R]
            U_heads.append(U_h.unsqueeze(-1))  # [2M+1, R, R, 1]
        U_heads = torch.cat(U_heads, dim=-1)  # [2M+1, R, R, H]

        # If you want to return in frequency space:
        U_heads_freq = torch.fft.rfft(U_heads, dim=0)  # [M+1, R, R, H]

        # Mix heads with learned matrix (in frequency space)
        U_mixed = torch.einsum('mrri,ij->mrrj', U_heads_freq, self.head_mixer)  # [M+1, R, R, H]

        # Return as dict mapping m to U_m (R x R x H) for m in ms_pos
        U = {m: U_mixed[i] for i, m in enumerate(ms_pos)}
        return U