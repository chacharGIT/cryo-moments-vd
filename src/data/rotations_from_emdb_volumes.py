import os
import glob
import numpy as np

from config.config import settings
from src.data.emdb_downloader import load_aspire_volume

def volume_second_moment(volume):
    """
    Compute the second-moment / covariance matrix
        C(phi) = ∫ x x^T phi(x) dx
    for a 3D volume on a regular grid.

    Parameters
    ----------
    volume : aspire.volume.Volume or 3D numpy array
        Volume data, shape (Nx, Ny, Nz).
    Returns
    -------
    C : ndarray, shape (3, 3)
        Second-moment matrix.
    """
    # Get underlying numpy data
    rho = volume._data[0]
    print(rho.shape)
    assert rho.ndim == 3 and rho.shape[0] == rho.shape[1] == rho.shape[2], \
        "Expected cubic volume (N x N x N)."
    N = rho.shape[0]
    coords = np.indices((N, N, N), dtype=np.float64)  # shape (3, N, N, N)
    coords -= (N - 1) / 2.0  # center at geometric center

    # C_ab = sum_{i,j,k} rho[i,j,k] * coords[a,i,j,k] * coords[b,i,j,k]
    C = np.einsum('aijk,ijk,bijk->ab', coords, rho, coords)
    return C

def rotation_from_second_moment(C):
    """
    Canonical SO(3) frame from a 3×3 second-moment matrix.
    
    Parameters
    ----------
    C : ndarray, shape (3, 3)
        Symmetric second-moment matrix.
    Returns
    -------
    R_p : ndarray, shape (3, 3)
        Canonical rotation matrix whose columns are the associated eigenvectors
    """
    C = np.asarray(C, dtype=np.float64)

    # Eigen-decomposition: vals ascending, columns of vecs are eigenvectors
    vals, vecs = np.linalg.eigh(C)

    # Sort eigenvalues descending and reorder eigenvectors accordingly
    idx = np.argsort(vals)[::-1]
    lambdas = vals[idx]
    R_p = vecs[:, idx].copy()  # columns v1, v2, v3

    # Fix signs of first two eigenvectors by their largest-magnitude entry
    for k in range(2):
        v = R_p[:, k]
        j = np.argmax(np.abs(v))
        if v[j] < 0:
            R_p[:, k] = -v

    # Enforce det(R_p) = +1 by possibly flipping the last eigenvector
    if np.linalg.det(R_p) < 0:
        R_p[:, 2] *= -1

    return R_p

if __name__ == "__main__":  
    volumes_folder = settings.data_generation.emdb.download_folder
    save_path = settings.data_generation.emdb.volume_rotations_path

    pattern = os.path.join(volumes_folder, "*.map.gz")
    paths = sorted(glob.glob(pattern))

    print(f"Found {len(paths)} volumes in {volumes_folder}")

    results = {}

    for map_path in paths:
        name = os.path.basename(map_path)
        volume_id = name.split(".")[0]  # e.g. 'emd_70864' from 'emd_70864.map.gz'
        try:
            print(f"\nProcessing {name} (id={volume_id})")
            volume = load_aspire_volume(map_path, downsample_size=settings.data_generation.downsample_size)
            C = volume_second_moment(volume)
            R = rotation_from_second_moment(C)
            results[volume_id] = {
                "second_moment": C,
                "rotation": R,
            }
            print("Rotation R (columns = eigenvectors):")
            print(R)
        except Exception as e:
            print(f"Failed to process {map_path}: {e}")

    print(f"\nSaving results for {len(results)} volumes to {save_path}")
    np.save(save_path, results, allow_pickle=True)
