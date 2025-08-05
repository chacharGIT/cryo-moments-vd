import os
import zarr
import random
from config.config import settings
from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.distribution_generation_functions import fibonacci_sphere_points
from src.utils.von_mises_fisher_distributions import (
    generate_random_von_mises_fisher_parameters,
    so3_distribution_from_von_mises_mixture
)
from src.data.emdb_downloader import load_aspire_volume



def save_vdm_to_zarr_group(zarr_root, emdb_id, vdm: VolumeDistributionModel, vmf_params: dict, override=False):
    """
    Save all VDM data for a single EMDB ID as a Zarr group.

    Parameters:
        zarr_root: Zarr root group or file.
        emdb_id: The EMDB ID (str or int) for this group.
        vdm: VolumeDistributionModel object containing the data.
        vmf_params: Dictionary of VMF mixture parameters (means, kappas, weights).
        override: If True, overwrite existing group. If False, skip if group exists.
    """
    group_name = str(emdb_id)
    if group_name in zarr_root and not override:
        print(f"Group {group_name} already exists, skipping (override=False).")
        return
    A_1 = vdm.first_analytical_moment(device=settings.device.cuda_device)
    A_2 = vdm.second_analytical_moment(batch_size=settings.data_generation.second_moment_batch_size,
                                        show_progress=True, device=settings.device.cuda_device)
    g = zarr_root.create_group(group_name, overwrite=True)
    g['first_moment'] = A_1
    g['second_moment'] = A_2
    g['volume'] = vdm.volume.asnumpy()
    g['rotation_matrices'] = vdm.rotations.matrices
    g['distribution_weights'] = vdm.distribution
    g['vmf_means'] = vmf_params['means']
    g['vmf_kappas'] = vmf_params['kappas']
    g['vmf_weights'] = vmf_params['weights']
    print(f"Saved data for {emdb_id} to Zarr group.")



def process_emdb_volumes(override=False, max_examples=None):
    """
    Process all EMDB map files in the download folder, generate VDMs, and store them in a single Zarr file.

    For each map file:
        - Loads the volume
        - Generates a random VMF mixture and SO(3) distribution
        - Computes analytical moments
        - Saves all data to a Zarr group (skips if group exists unless override=True)

    Parameters:
        override: If True, overwrite existing Zarr groups. If False, skip existing groups.
        max_examples: If specified, randomly select at most this many examples from available files.
                     If None, process all available files.
    """
    quadrature_points = fibonacci_sphere_points(n=settings.data_generation.von_mises_fisher.fibonacci_spiral_n)
    num_vmf = settings.data_generation.von_mises_fisher.num_distributions
    download_dir = settings.data_generation.emdb.download_folder
    files = [f for f in os.listdir(download_dir) if f.endswith('.map') or f.endswith('.map.gz')]
    
    # Randomly select a subset of files if max_examples is specified
    if max_examples is not None and max_examples < len(files):
        files = random.sample(files, max_examples)
        print(f"Randomly selected {len(files)} files out of {len(os.listdir(download_dir))} available files")
    else:
        print(f"Processing all {len(files)} available files")
    
    random.shuffle(files)
    split_ratio = settings.training.train_val_split
    split_idx = int(split_ratio * len(files))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    os.makedirs(settings.data_generation.zarr.save_dir, exist_ok=True)
    zarr_train_path = os.path.join(settings.data_generation.zarr.save_dir, "train_vdms.zarr")
    zarr_val_path = os.path.join(settings.data_generation.zarr.save_dir, "val_vdms.zarr")
    zroot_train = zarr.open(zarr_train_path, mode='w')
    zroot_val = zarr.open(zarr_val_path, mode='w')

    def process_files(file_list, zroot):
        for idx, filename in enumerate(file_list):
            import re
            emdb_id = re.sub(r'\.map(\.gz)?$', '', filename)
            map_path = os.path.join(download_dir, filename)
            zarr_file = os.path.basename(getattr(zroot.store, 'path', str(zroot.store)))
            print(f"[{idx+1}/{len(file_list)}] Processing {filename} (EMDB {emdb_id}) for {zarr_file}")
            try:
                vol = load_aspire_volume(map_path, downsample_size=settings.data_generation.downsample_size)
                mu_directions, kappa_values, mixture_weights = generate_random_von_mises_fisher_parameters(
                    num_vmf, kappa_range=tuple(settings.data_generation.von_mises_fisher.kappa_range)
                )
                rotations, distribution = so3_distribution_from_von_mises_mixture(
                    quadrature_points, mu_directions, kappa_values, mixture_weights, 
                    settings.data_generation.von_mises_fisher.num_in_plane_rotations
                )
                distribution_metadata = {
                    "type": "vmf_mixture",
                    "means": mu_directions,
                    "kappas": kappa_values,
                    "weights": mixture_weights
                }
                vdm = VolumeDistributionModel(vol, rotations, distribution, distribution_metadata=distribution_metadata)
                save_vdm_to_zarr_group(zroot, emdb_id, vdm, distribution_metadata, override=override)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    process_files(train_files, zroot_train)
    process_files(val_files, zroot_val)


if __name__ == "__main__":
    # Get max_examples from settings configuration
    max_examples = settings.data_generation.zarr.max_examples
    
    process_emdb_volumes(override=False, max_examples=max_examples)
    
    # Check and print the keys of the first group in the train and validation Zarr files for verification
    import zarr
    import os
    for zarr_name in ["train_vdms.zarr", "val_vdms.zarr"]:
        zarr_path = os.path.join(settings.data_generation.zarr.save_dir, zarr_name)
        print(f"\nChecking {zarr_name}...")
        zroot = zarr.open(zarr_path, mode='r')
        group_names = list(zroot.group_keys())
        if group_names:
            for group_name in group_names:
                group = zroot[group_name]
                print(f"Group: {group_name}")
                print("Keys:", list(group.keys()))
                if 'second_moment' in group:
                    second_moment = group['second_moment']
                    print("  second_moment shape:", second_moment.shape)
                    print("  second_moment dtype:", second_moment.dtype)
                else:
                    print("  'second_moment' not found in this group.")
        else:
            print("No groups found in the Zarr file.")
