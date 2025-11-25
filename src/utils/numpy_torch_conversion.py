import numpy as np
import torch

mapping = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.float16: torch.float16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

def dtype_to_torch(dtype):
    # Use dtype.type for comparison
    if dtype.type in mapping:
        return mapping[dtype.type]
    raise ValueError(f"Unsupported dtype: {dtype}")