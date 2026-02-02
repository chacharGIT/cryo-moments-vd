import warnings
# Suppress Zarr warnings about Python type strings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
warnings.filterwarnings("ignore", message=".*object arrays.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*string.*", category=UserWarning)
import os
from tqdm import tqdm
import zarr
import numpy as np
import torch

from config.config import settings
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.utils.von_mises_fisher_distributions import evaluate_vmf_mixture, generate_random_vmf_parameters
from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.subspace_moment_utils import compute_second_moment_eigendecomposition, num_components_for_energy_threshold, extract_dominant_eigenvector_modes
from src.utils.polar_transform import cartesian_to_polar

def main():
    # Generate quadrature points once
    n_quadrature = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    quadrature_points = fibonacci_sphere_points(n_quadrature)
    separate_fourier_modes = settings.data_generation.separate_fourier_modes

    if separate_fourier_modes:
        save_path = settings.data_generation.zarr.separated_modes_data_save_path
    else:
        save_path = settings.data_generation.zarr.save_path
    save_interval = 100  # Save every save_interval samples
    root = zarr.open(save_path, mode='a')  # Use append mode
    # Use CUDA device from settings if enabled
    if settings.device.use_cuda:
        device = f"cuda:{settings.device.cuda_device}"
    else:
        device = "cpu"
    
    # Preallocate lists for all samples (to be periodically flushed)
    all_means = []
    all_kappas = []
    all_weights = []
    if separate_fourier_modes:
        all_eigen_radial_profiles = []
        all_eigen_m_detected = []
        all_eigen_energy_fractions = []
        all_first_moments_radial = []
        n_theta = settings.data_generation.cartesian_to_polar_n_theta
    else:
        all_eigen_images = []
        all_first_moments = []
    all_eigen_values = []
    all_volume_ids = []
    all_distribution_evaluations = []
    total_samples = 0

    emdb_folder = settings.data_generation.emdb.download_folder
    emdb_files = [os.path.join(emdb_folder, f) for f in os.listdir(emdb_folder) if f.endswith('.map.gz')]
    np.random.shuffle(emdb_files)
    precomputed_folder = settings.data_generation.zarr.precomputed_moments_path

    num_mixtures_per_volume = settings.data_generation.von_mises_fisher.num_generated_examples_per_volume
    print("number of mixtures per volume:", num_mixtures_per_volume)
    for emdb_file in tqdm(emdb_files, desc="EMDB Volumes"):
        emdb_id = os.path.basename(emdb_file).split('.')[0]
        print(f"Processing EMDB {emdb_id}")
        zarr_path = os.path.join(precomputed_folder, f"{emdb_id}.zarr")
        try:
            pre_z = zarr.open(zarr_path, mode='r')
        except Exception as e:
            print(f"Error opening precomputed data for EMDB {emdb_id}: {e}")
            continue
        # Expect arrays: first_moments (n_q, L, L), eigen_images (n_q, L^2, k), eigen_values (n_q, k)
        first_moments_arr = pre_z['first_moments']          # shape (n_q, L, L)
        eigen_images_arr = pre_z['eigen_images']             # shape (n_q, L^2, k)
        eigen_values_arr = pre_z['eigen_values']             # shape (n_q, k)
        n_pixels = eigen_images_arr.shape[1]
        data_type = eigen_images_arr.dtype
        torch_data_type = torch.float64 if data_type == np.float64 else torch.float32
        M_arr = np.zeros((n_quadrature, n_pixels, n_pixels), dtype=data_type)
        for qp in tqdm(range(n_quadrature), desc=f"Extracting second moment components for {emdb_id}", leave=False):
            U = torch.tensor(eigen_images_arr[qp], dtype=torch_data_type, device=device)  # (n_pixels, k)
            lam = torch.tensor(eigen_values_arr[qp], dtype=torch_data_type, device=device)     # (k,)
            if U.size == 0 or lam.size == 0:
                raise ValueError(f"Empty eigen images or values for quadrature point {qp} in EMDB {emdb_id}")
            # Reconstruct second moment matrix for quadrature point from truncated eigendecomposition
            M_i = (U * lam[None, :]) @ U.T  # (n_pixels, n_pixels)
            M_arr[qp] = M_i.cpu().numpy()

        if emdb_id in root:
            volume_group = root[emdb_id]
        else:
            volume_group = None
        if volume_group is None:
            # Dynamically determine n_eigen_target
            n_probe = 15
            n_keep_list = []
            for _ in range(n_probe):
                # Generate random mixtures, compute second moments, and determine n_keep
                mu, kappa, weights = generate_random_vmf_parameters(
                    settings.data_generation.von_mises_fisher.num_distributions,
                    settings.data_generation.von_mises_fisher.kappa_start,
                    settings.data_generation.von_mises_fisher.kappa_mean)
                mixture_eval = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)
                second_moment = np.tensordot(mixture_eval, M_arr, axes=(0, 0))
                L = int(np.sqrt(n_pixels))
                second_moment = second_moment.reshape(L, L, L, L) 
                eigvals, _ = compute_second_moment_eigendecomposition(second_moment)
                n_keep = num_components_for_energy_threshold(eigvals, settings.data_generation.second_moment_energy_truncation_threshold)
                n_keep_list.append(n_keep)
            n_eigen_target = int(max(n_keep_list) * 1.1) # Add 10% buffer
            print(f"Determined n_eigen_target for {emdb_id}: {n_eigen_target}")
            volume_group = root.require_group(emdb_id)
            volume_group.attrs['n_eigen_target'] = n_eigen_target  # Save for future reference
        else:
            n_eigen_target = volume_group["eigen_values"].shape[-1]
            print(f"Loaded n_eigen_target for {emdb_id}: {n_eigen_target}")

        for _ in tqdm(range(num_mixtures_per_volume), desc=f"Mixtures for {emdb_id}", leave=False):
            # Sample vMF mixture
            mu, kappa, weights = generate_random_vmf_parameters(
                settings.data_generation.von_mises_fisher.num_distributions,
                settings.data_generation.von_mises_fisher.kappa_start,
                settings.data_generation.von_mises_fisher.kappa_mean)
            mixture_eval = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)

            # Compute weighted first and second moments
            first_moment = np.tensordot(mixture_eval, first_moments_arr[0:n_quadrature], axes=(0,0))
            second_moment = np.tensordot(mixture_eval, M_arr, axes=(0, 0))
            L = int(np.sqrt(n_pixels))
            second_moment = second_moment.reshape(L, L, L, L) 

            # Spectral decomposition (full)
            eigvals, eigvecs = compute_second_moment_eigendecomposition(second_moment)
            eigvals_keep = eigvals[:n_eigen_target]
            eigvecs_keep = eigvecs[:, :n_eigen_target]

            if separate_fourier_modes:
                polar, _ , _ = cartesian_to_polar(first_moment, n_theta=n_theta)
                first_moment_radial = polar.mean(axis=-1).astype(np.float32)
                all_first_moments_radial.append(first_moment_radial)
                compressed_eigenspaces = extract_dominant_eigenvector_modes(eigvecs_keep)
                # Determine radial profile length from first eigenvector
                if compressed_eigenspaces[0]['m_detected'] == 0:
                    len_r = compressed_eigenspaces[0]['radial_profile'].shape[0]
                    data_type = compressed_eigenspaces[0]['radial_profile'].dtype
                else:
                    len_r = compressed_eigenspaces[0]['cos_component'].shape[0]
                    data_type = compressed_eigenspaces[0]['cos_component'].dtype
                all_eigen_m_detected.append(np.array([d['m_detected'] for d in compressed_eigenspaces], dtype=np.int32))
                all_eigen_energy_fractions.append(np.array([d['energy_fraction'] for d in compressed_eigenspaces]))

                # Collect radial profiles for all eigenvectors in this sample
                radial_profiles = np.zeros((2 * len_r, len(compressed_eigenspaces)), dtype=data_type)
                for j, d in enumerate(compressed_eigenspaces):
                    if d['m_detected'] == 0:
                        radial_profiles[:len_r, j] = d['radial_profile']
                        radial_profiles[len_r:, j] = 0  # pad sin component with zeros
                    else:
                        radial_profiles[:len_r, j] = d['cos_component']
                        radial_profiles[len_r:, j] = d['sin_component']
                all_eigen_radial_profiles.append(radial_profiles)
            else:
                all_eigen_images.append(eigvecs_keep)
                all_first_moments.append(first_moment)
            all_means.append(mu)
            all_kappas.append(kappa)
            all_weights.append(weights)
            all_eigen_values.append(eigvals_keep)
            all_volume_ids.append(emdb_id)
            all_distribution_evaluations.append(mixture_eval)
            total_samples += 1

            # Periodic save and append
            if total_samples % save_interval == 0:
                data_dict = {
                        "s2_distribution_means": all_means,
                        "s2_distribution_kappas": all_kappas,
                        "s2_distribution_weights": all_weights,
                        "eigen_values": all_eigen_values,
                        "volume_ids": all_volume_ids,
                        "distribution_evaluations": all_distribution_evaluations,
                }
                if separate_fourier_modes:
                    data_dict.update({
                        "eigen_radial_profiles": all_eigen_radial_profiles,
                        "eigen_m_detected": all_eigen_m_detected,
                        "eigen_energy_fractions": all_eigen_energy_fractions,
                        "first_moments_radial": all_first_moments_radial,
                    })
                else:
                    data_dict.update({
                        "eigen_images": all_eigen_images,
                        "first_moments": all_first_moments,
                    })
                _flush_to_zarr(volume_group, data_dict)

                if separate_fourier_modes:
                    all_eigen_radial_profiles.clear()
                    all_eigen_m_detected.clear()
                    all_eigen_energy_fractions.clear()
                else:
                    all_eigen_images.clear()
                    all_first_moments.clear()
                all_means.clear()
                all_kappas.clear()
                all_weights.clear()
                all_eigen_values.clear()
                all_volume_ids.clear()
                all_distribution_evaluations.clear()

def _flush_to_zarr(
    group,
    data_dict,
):
    def pad_or_cut(arr, n_eigen_target):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            out = np.zeros(n_eigen_target, dtype=arr.dtype)
            n = min(arr.shape[0], n_eigen_target)
            out[:n] = arr[:n]
            return out
        elif arr.ndim == 2:
            # n_pixels is always fixed, only pad/cut the second dimension
            out = np.zeros((arr.shape[0], n_eigen_target), dtype=arr.dtype)
            n = min(arr.shape[1], n_eigen_target)
            out[:, :n] = arr[:, :n]
            return out
        else:
            raise ValueError("pad_or_cut only supports 1D or 2D arrays")
    # Convert lists to arrays (padding eigenvectors/values if needed)
    # Pad eigen arrays to max save amount
    n_eigen_target = group.attrs.get('n_eigen_target')
    arrays_to_save = {}
    for key, arr_list in data_dict.items():
        if key.startswith("eigen_"):
            arrays_to_save[key] = np.stack([pad_or_cut(e, n_eigen_target) for e in arr_list])
        else:
            arrays_to_save[key] = np.stack(arr_list) if isinstance(arr_list[0], (np.ndarray, list)) else np.array(arr_list)

    # If datasets do not exist, create them; else, append
    def append_or_create(group, name, arr, axis=0):
        chunks = (min(100, arr.shape[0]),) + arr.shape[1:]
        if name not in group:
            group.create_array(
                name,
                shape=(0,) + arr.shape[1:],
                chunks=chunks,
                dtype=arr.dtype,
                overwrite=False,
                fill_value=0
            )
        zarr_arr = group[name]
        zarr_arr.append(arr, axis=axis)
        
    for key, arr in arrays_to_save.items():
        append_or_create(group, key, arr)

if __name__ == "__main__":
    main()
