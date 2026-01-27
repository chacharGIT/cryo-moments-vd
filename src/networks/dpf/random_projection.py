import os
import numpy as np

def random_projections(D, N_m, N0_post, save_path):
    """
    For each m in [0, M], construct a random N_m[m]-dimensional subspace of C^D,
    and return an orthonormal basis matrix Q_m (D x N_m[m]) for each m.
    Also create a '0_post' basis for m=0 with dimension N0_post.
    If a file exists at save_path, load the matrices from disk instead of creating new ones.
    """
    if os.path.exists(save_path):
        print(f"Loading projection matrices from {save_path}")
        Qms = np.load(save_path, allow_pickle=True).item()
        return Qms
    else:
        print(f"Creating new complex projection matrices and saving to {save_path}")
        Qms = {}
        for m in range(0, len(N_m)):
            A_real = np.random.randn(D, N_m[m])
            if m != 0:
                A_imag = np.random.randn(D, N_m[m])
            else:
                A_imag = np.zeros((D, N_m[m]))
            A = A_real + 1j * A_imag
            # Orthonormalize columns (QR decomposition for complex)
            Qm, _ = np.linalg.qr(A)
            # Assign for both +m and -m
            Qms[str(m)] = Qm
        # Add post-concatenation Q_0 if requested
        if N0_post is not None:
            A_real = np.random.randn(D, N0_post)
            A_imag = np.zeros((D, N0_post))
            A = A_real + 1j * A_imag
            Qm_post, _ = np.linalg.qr(A)
            Qms["0_post"] = Qm_post
        np.save(save_path, Qms)
        return Qms