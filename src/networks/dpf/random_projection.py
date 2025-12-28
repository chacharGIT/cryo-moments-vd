import os
import numpy as np

def random_projections(D, N_m, save_path="pms.npy"):
    """
    For each m in [-M, M], construct a random subspace of R^D of dimension N_m[m+M],
    and return the projection matrices P_m for each m. If a file exists at save_path,
    load the matrices from disk instead of creating new ones.
    """
    if os.path.exists(save_path):
        print(f"Loading projection matrices from {save_path}")
        Pms = np.load(save_path, allow_pickle=True)
        return list(Pms)
    else:
        print(f"Creating new complex projection matrices and saving to {save_path}")
        M = len(N_m) - 1
        Pms = {}
        for m in range(0, M+1):
            n = N_m[m]
            A_real = np.random.randn(D, n)
            A_imag = np.random.randn(D, n)
            A = A_real + 1j * A_imag
            # Orthonormalize columns (QR decomposition for complex)
            Q, _ = np.linalg.qr(A)
            Pm = Q @ Q.conj().T  # Complex projection matrix: D x D
            # Assign for both +m and -m
            Pms[m] = Pm
            if m != 0:
                Pms[-m] = Pm  # Share for negative m
        np.save(save_path, Pms)
        return Pms