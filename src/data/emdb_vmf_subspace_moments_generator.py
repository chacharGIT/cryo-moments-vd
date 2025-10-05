import warnings
# Suppress Zarr warnings about Python type strings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
warnings.filterwarnings("ignore", message=".*object arrays.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*string.*", category=UserWarning)

from tqdm import tqdm
import zarr
import numpy as np
from src.data.emdb_downloader import search_emdb_asymmetric_ids, download_emdb_map, load_aspire_volume
from src.utils.von_mises_fisher_distributions import generate_random_von_mises_fisher_parameters
from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.spectral_analysis import compute_second_moment_eigendecomposition
from config.config import settings

def main():
    save_path = "/data/shachar/zarr_files/emdb_vmf_top_eigen.zarr"
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
    all_eigen_images = []
    all_eigen_values = []
    all_first_moments = []
    all_volume_ids = []
    total_samples = 0

    emdb_ids = search_emdb_asymmetric_ids(
        resolution_cutoff=settings.data_generation.emdb.resolution_cutoff,
        max_results=settings.data_generation.emdb.max_results
    )
    np.random.shuffle(emdb_ids)
    num_mixtures_per_volume = settings.data_generation.von_mises_fisher.num_generated_examples // len(emdb_ids)
    print("number of mixtures per volume:", num_mixtures_per_volume)
    for emdb_id in tqdm(emdb_ids, desc="EMDB Volumes"):
        # Download map and load as aspire Volume
        print(f"Processing EMDB {emdb_id}")
        try:
            map_path = download_emdb_map(emdb_id, settings.data_generation.emdb.download_folder)
            volume = load_aspire_volume(map_path, downsample_size=settings.data_generation.downsample_size)
        except (EOFError, OSError, IOError, Exception) as e:
            print(f"Error loading EMDB {emdb_id}: {type(e).__name__}: {e}")
            print(f"Skipping EMDB {emdb_id} due to corrupted or invalid file")
            continue

        for i in tqdm(range(num_mixtures_per_volume), desc=f"Mixtures for {emdb_id}", leave=False):
            # Sample vMF mixture
            mu, kappa, weights = generate_random_von_mises_fisher_parameters(
                settings.data_generation.von_mises_fisher.num_distributions,
                settings.data_generation.von_mises_fisher.kappa_start,
                settings.data_generation.von_mises_fisher.kappa_mean)
            # Create SO(3) distribution
            rotations, distribution = create_in_plane_invariant_distribution(
                mu, weights, num_in_plane_rotations=settings.data_generation.von_mises_fisher.num_in_plane_rotations)
            # Build VDM
            vdm = VolumeDistributionModel(volume, rotations, distribution, distribution_metadata={
                "type": "vmf_mixture",
                "means": mu,
                "kappas": kappa,
                "weights": weights
            })
            # Compute moments
            first_moment = vdm.first_analytical_moment(device=device)
            second_moment = vdm.second_analytical_moment(batch_size=200, device=device)
            # Spectral decomposition (full)
            eigvals, eigvecs = compute_second_moment_eigendecomposition(second_moment, device=device)
            L = second_moment.shape[0]
            total_energy = np.trace(second_moment.reshape(L*L, L*L))
            cum_energy = np.cumsum(eigvals) / total_energy
            n_keep = np.searchsorted(cum_energy, 1 - settings.data_generation.second_moment_eigen_energy_dismiss_fraction) + 1
            if i == 0:
                print(f"EMDB {emdb_id}: n_keep for first mixture = {n_keep}")
            eigvals_keep = eigvals[:n_keep]
            eigvecs_keep = eigvecs[:, :n_keep]
            all_means.append(mu)
            all_kappas.append(kappa)
            all_weights.append(weights)
            all_eigen_images.append(eigvecs_keep)
            all_eigen_values.append(eigvals_keep)
            all_first_moments.append(first_moment)
            all_volume_ids.append(emdb_id)
            total_samples += 1

            # Periodic save and append
            if total_samples % save_interval == 0:
                print(f"Reached {total_samples} samples, saving to Zarr...")
                _flush_to_zarr(root, all_means, all_kappas, all_weights, all_eigen_images, all_eigen_values, all_first_moments, all_volume_ids)
                all_means.clear()
                all_kappas.clear()
                all_weights.clear()
                all_eigen_images.clear()
                all_eigen_values.clear()
                all_first_moments.clear()
                all_volume_ids.clear()

    # Final flush for any remaining samples
    if all_means:
        print(f"Final flush: saving remaining {len(all_means)} samples to Zarr...")
        _flush_to_zarr(root, all_means, all_kappas, all_weights, all_eigen_images, all_eigen_values, all_first_moments, all_volume_ids)
        print(f"Saved {total_samples} samples from {len(emdb_ids)} volumes to {save_path}")


def _flush_to_zarr(root, all_means, all_kappas, all_weights, all_eigen_images, all_eigen_values, all_first_moments, all_volume_ids):
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
    n_eigen_target = settings.data_generation.max_eigen_vector_save_amount
    # Pad eigen arrays to max save amount
    # Always pad/cut eigen_images to [image_dim, max_amount] and eigen_values to [max_amount]
    eigen_images_arr = np.stack([pad_or_cut(e, n_eigen_target) for e in all_eigen_images])
    eigen_values_arr = np.stack([pad_or_cut(ev, n_eigen_target) for ev in all_eigen_values])
    # Stack first moments
    first_moments_arr = np.stack(all_first_moments)
    # Stack other arrays
    means_arr = np.stack(all_means)
    kappas_arr = np.stack(all_kappas)
    weights_arr = np.stack(all_weights)
    volume_ids_arr = np.array(all_volume_ids)

    # If datasets do not exist, create them; else, append
    def append_or_create(name, arr, axis=0):
        chunks = (min(100, arr.shape[0]),) + arr.shape[1:]
        if name not in root:
            root.create_array(
                name,
                shape=(0,) + arr.shape[1:],
                chunks=chunks,
                dtype=arr.dtype,
                overwrite=False,
                fill_value=0
            )
        zarr_arr = root[name]
        zarr_arr.append(arr, axis=axis)
    append_or_create("s2_distribution_means", means_arr)
    append_or_create("s2_distribution_kappas", kappas_arr)
    append_or_create("s2_distribution_weights", weights_arr)
    append_or_create("eigen_images", eigen_images_arr)
    append_or_create("eigen_values", eigen_values_arr)
    append_or_create("first_moments", first_moments_arr)
    append_or_create("volume_ids", volume_ids_arr)


if __name__ == "__main__":
    main()
