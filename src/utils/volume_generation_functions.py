import numpy as np

def white_noise_on_unit_ball(resolution, mean=0.0, std=1.0):
    """
    Generate a scalar function on the unit ball in R^3 with white noise values.
    The function is discretized on a cubic grid of given resolution, and values outside the unit ball are set to zero.

    Parameters:
    -----------
    resolution : int
        Number of grid points along each axis (the grid is resolution x resolution x resolution).
    mean : float
        Mean of the white noise (default 0.0).
    std : float
        Standard deviation of the white noise (default 1.0).

    Returns:
    --------
    volume : np.ndarray
        3D numpy array of shape (resolution, resolution, resolution) with white noise inside the unit ball, zero outside.
    """
    # Create a grid of coordinates in [-1, 1]^3
    lin = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mask = R <= 1.0
    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    volume[mask] = np.random.normal(mean, std, size=np.count_nonzero(mask))
    return volume

if __name__ == "__main__":
    from config.config import settings
    volume = white_noise_on_unit_ball(resolution=settings.data_generation.downsample_size,
                                       mean=settings.data_generation.gaussian_white_noise.mean,
                                       std=settings.data_generation.gaussian_white_noise.std)
    print(f"shape {volume.shape}")
