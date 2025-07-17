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
    batch_size, N , F_in = x.shape
    
    # Project x onto eigenvector space for each batch element
    # eigenvectors: (batch_size, N, N), x: (batch_size, N, F_in)
    x_proj = torch.bmm(eigenvectors.transpose(-2, -1), x)  # Shape: (batch_size, N, F_in)
    
    outputs = []
    for k in range(degree + 1):
        # eigenvalues: (batch_size, N) -> (batch_size, N, 1) for broadcasting
        eigenval_powers = (eigenvalues ** k).unsqueeze(-1)  # Shape: (batch_size, N, 1)
        scaled_proj = eigenval_powers * x_proj  # Shape: (batch_size, N, F_in)
        
        # Apply eigenvectors: (batch_size, N, N) @ (batch_size, N, F_in) -> (batch_size, N, F_in)
        result = torch.bmm(eigenvectors, scaled_proj)
        outputs.append(result)
    
    return torch.stack(outputs)  # Shape: (degree + 1, batch_size, N, F_in)
