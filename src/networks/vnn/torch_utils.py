import torch

def apply_all_C_powers(eigenvalues, eigenvectors, x, degree):
    """
    Efficiently compute C^k x for all k in [0, degree] using spectral decomposition.
    
    Args:
        eigenvalues: Tensor of shape (batch_size, N)
        eigenvectors: Tensor of shape (batch_size, N, N)
        x: Tensor of shape (batch_size, N, F_in)
        degree: int
    
    Returns:
        Tensor of shape (degree + 1, batch_size, N, F_in)
    """
    batch_size, N, F_in = x.shape
    # Project x onto eigenvector space for each batch element
    x_proj = torch.bmm(eigenvectors.transpose(-2, -1), x)  # (batch_size, N, F_in)

    # Compute all eigenvalue powers at once: (degree+1, batch_size, N)
    powers = torch.arange(degree + 1, device=x.device).view(-1, 1, 1)
    eigenval_powers = torch.pow(eigenvalues.clamp(min=1e-12), powers)  # (degree+1, batch_size, N)

    # Scale: (degree+1, batch_size, N, 1) * (batch_size, N, F_in) -> (degree+1, batch_size, N, F_in)
    scaled_proj = eigenval_powers.unsqueeze(-1) * x_proj.unsqueeze(0)

    # Reshape for batched bmm: (degree+1)*batch_size, N, F_in
    scaled_proj_flat = scaled_proj.permute(1,0,2,3).reshape(-1, N, F_in)
    eigenvectors_flat = eigenvectors.repeat(degree+1,1,1)
    result_flat = torch.bmm(eigenvectors_flat, scaled_proj_flat)
    # Reshape back: (batch_size, degree+1, N, F_in) -> (degree+1, batch_size, N, F_in)
    result = result_flat.view(batch_size, degree+1, N, F_in).permute(1,0,2,3)
    return result
