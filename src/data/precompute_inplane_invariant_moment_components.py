from tqdm import tqdm
import zarr
import numpy as np
import os
import random
from aspire.volume import Volume
from config.config import settings
from src.core.volume_distribution_model import VolumeDistributionModel
from src.data.emdb_downloader import search_emdb_asymmetric_ids, download_emdb_map, load_aspire_volume
from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution, fibonacci_sphere_points
from src.utils.subspace_moment_utils import compute_second_moment_eigendecomposition, num_components_for_energy_threshold

def compute_moments_volume_point(volume: Volume, point: np.ndarray):
    so3_rotations, so3_weights = create_in_plane_invariant_distribution(np.array([point]), np.array([1]),
    num_in_plane_rotations=settings.data_generation.in_plane_invariant_distributions.num_in_plane_rotations)

    vdm = VolumeDistributionModel(volume, rotations=so3_rotations, distribution=so3_weights)
    first_moment = vdm.first_analytical_moment()
    second_moment = vdm.second_analytical_moment(batch_size=settings.data_generation.second_moment_batch_size)
    eigvals, eigvecs = compute_second_moment_eigendecomposition(second_moment)
    return first_moment, eigvals, eigvecs

def main():
    save_path = "/data/shachar/zarr_files/emdb_in_plane_invariant_moment_subspace_components"
    n_quadrature = 512

    quadrature_points = fibonacci_sphere_points(n_quadrature)
    emdb_folder = settings.data_generation.emdb.download_folder
    emdb_files = [os.path.join(emdb_folder, f) for f in os.listdir(emdb_folder) if f.endswith('.map.gz')]

    for emdb_file in tqdm(emdb_files, desc="EMDB Volumes"):
        emdb_id = os.path.basename(emdb_file).split('.')[0]
        zarr_file_path = os.path.join(save_path, f"{emdb_id}.zarr")
        expected_shape = (quadrature_points.shape[0],)
        
        # Check if zarr file exists and is complete
        skip = False
        try:
            zarr_file = zarr.open(zarr_file_path, mode='r')
            # Check all arrays exist and have correct shape
            for arr_name in ['eigen_images', 'eigen_values', 'first_moments']:
                if arr_name not in zarr_file:
                    skip = False
                    break
                arr = zarr_file[arr_name]
                if arr.shape[0] != expected_shape[0]:
                    skip = False
                    break
            else:
                skip = True
        except Exception as e:
            print(f"Cannot access {emdb_id}.zarr")
            skip = False
        if skip:
            print(f"Zarr file for EMDB {emdb_id} already exists and is complete. Skipping...")
            continue

        try:
            volume = load_aspire_volume(emdb_file, downsample_size=settings.data_generation.downsample_size)
        except (EOFError, OSError, IOError, Exception) as e:
            print(f"Error loading EMDB {emdb_file}: {type(e).__name__}: {e}")
            print(f"Skipping EMDB {emdb_file} due to corrupted or invalid file")
            continue
        random_indices = random.sample(range(len(quadrature_points)), 10)
        n_keep_list = []
        for idx in random_indices:
            _, eigvals, _ = compute_moments_volume_point(volume, quadrature_points[idx])
            n_keep_curr = num_components_for_energy_threshold(eigvals,
                                        settings.data_generation.second_moment_energy_truncation_threshold)
            n_keep_list.append(n_keep_curr)
        n_keep_max = max(n_keep_list)
        n_keep = int(1.1 * n_keep_max)
        print(f"Using n_keep={n_keep} for all quadrature points for EMDB {emdb_file}")

        all_eigen_images = []
        all_eigen_values = []
        all_first_moments = []
        for point in tqdm(quadrature_points, desc="Quadrature Points", leave=False):
            first_moment, eigvals, eigvecs = compute_moments_volume_point(volume, point)
            eigvals_keep = eigvals[:n_keep]
            eigvecs_keep = eigvecs[:, :n_keep]
            all_eigen_images.append(eigvecs_keep)
            all_eigen_values.append(eigvals_keep)
            all_first_moments.append(first_moment)

        # Convert lists to arrays
        all_eigen_images = np.array(all_eigen_images)
        all_eigen_values = np.array(all_eigen_values)
        all_first_moments = np.array(all_first_moments)

        # Save to zarr
        zarr_file = zarr.open(zarr_file_path, mode='w')
        def create_zarr_array(name, arr):
            chunk_size = max(1, quadrature_points.shape[0] // 8)
            chunks = (chunk_size,) + arr.shape[1:]
            chunks = (min(100, arr.shape[0]),) + arr.shape[1:]
            if name not in zarr_file:
                zarr_file.create_array(
                    name,
                    shape=(arr.shape[0],) + arr.shape[1:],
                    chunks=chunks,
                    dtype=arr.dtype,
                    overwrite=True,
                    fill_value=0
                )
            zarr_file[name][:] = arr
        create_zarr_array('eigen_images', all_eigen_images)
        create_zarr_array('eigen_values', all_eigen_values)
        create_zarr_array('first_moments', all_first_moments)
if __name__ == "__main__":
    zarr_path = "/data/shachar/zarr_files/emdb_in_plane_invariant_moment_subspace_components/emd_54247.zarr"
    zarr_file = zarr.open(zarr_path, mode='r')
    for name in zarr_file.array_keys():
        arr = zarr_file[name]
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"Chunk size: {zarr_file['first_moments'].chunks[0]}")
    
    main()