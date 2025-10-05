import numpy as np
import torch
from typing import Optional, Union
from config.config import settings


def generate_goe_matrix(downsample_size: int, 
                       device: Optional[str] = None,
                       dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
    """
    Generate a random Gaussian Orthogonal Ensemble (GOE) matrix.
    
    The GOE matrix has size (downsample_size^2, downsample_size^2) and represents
    a symmetric random matrix where:
    - Diagonal elements are drawn from N(0, 2)
    - Off-diagonal elements are drawn from N(0, 1) 
    - The matrix is symmetric: M[i,j] = M[j,i]
    
    This corresponds to the space where vectors have size downsample_size^2,
    and the matrix operates on flattened images of size (downsample_size, downsample_size).
    
    Parameters:
    -----------
    downsample_size : int
        The linear dimension of the image space. The resulting matrix will have
        size (downsample_size^2, downsample_size^2)
    device : str, optional
        Device to use for computation ('cuda' or 'cpu'). If None, uses GPU if available.
    dtype : torch.dtype, optional
        Data type for the matrix. If None, uses torch.float32 for efficiency.
        
    Returns:
    --------
    goe_matrix : torch.Tensor
        Symmetric random matrix of shape (downsample_size^2, downsample_size^2)
        following GOE statistics.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    # Matrix dimension
    n = downsample_size ** 2
    
    # Generate random matrix with GOE statistics
    # Start with a random matrix from standard normal distribution
    random_matrix = torch.randn(n, n, device=device, dtype=dtype)
    
    # Make it symmetric: M = (A + A^T) / sqrt(2)
    # The factor 1/sqrt(2) ensures the variance of off-diagonal elements is 1
    goe_matrix = (random_matrix + random_matrix.T) / np.sqrt(2)
    
    # Adjust diagonal elements to have variance 2 (GOE convention)
    # Current diagonal variance is 2 (since diag(A + A^T) = 2*diag(A))
    # We want variance 2, so we need to multiply diagonal by 1/sqrt(2) * sqrt(2) = 1
    # Actually, GOE diagonal elements should have variance 2, so we multiply by sqrt(2)
    diagonal_mask = torch.eye(n, device=device, dtype=torch.bool)
    goe_matrix[diagonal_mask] *= np.sqrt(2)
    
    return goe_matrix


def generate_gaussian_matrix(shape, device: str = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate a standard Gaussian random matrix of given shape.
    Entries are drawn from N(0, 1).
    Args:
        shape: tuple or list of ints
            Desired output shape (e.g., (d, s)).
        device: str or torch.device, optional
            Device for computation ('cuda' or 'cpu').
        dtype: torch.dtype, optional
            Data type for the matrix.
    Returns:
        torch.Tensor of shape `shape` with standard normal entries.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(*shape, device=device, dtype=dtype)


if __name__ == "__main__":
    print("=== GOE Matrix Generation Test ===")    
    # Generate test matrix
    goe_test = generate_goe_matrix(downsample_size=settings.data_generation.downsample_size)
    print(f"Matrix shape: {goe_test.shape}")
    print(f"Matrix dtype: {goe_test.dtype}")
    print(f"Is symmetric: {torch.allclose(goe_test, goe_test.T)}")
