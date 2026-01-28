import argparse
import os

import numpy as np
import zarr

from src.utils.polar_transform import cartesian_to_polar


def precompute_first_moments_radial(
    zarr_path: str,
    n_theta: int = 512,
    overwrite: bool = False,
):
    """
    For each volume group in the root Zarr, read `first_moments` [N, H, W],
    convert each to polar, average over angle to get a 1D radial profile,
    and store the result as `first_moments_radial` [N, R] in the same group.
    """

    root = zarr.open(zarr_path, mode="a")
    volume_ids = list(root.group_keys())
    print(f"Found {len(volume_ids)} volume groups in {zarr_path}")

    for vid in volume_ids:
        grp = root[vid]
        try:
            fm_arr = grp["first_moments"]
        except KeyError:
            print(f"[{vid}] 'first_moments' not found, skipping")
            continue
        n_samples, H, W = fm_arr.shape
        print(f"[{vid}] first_moments shape: {fm_arr.shape}")

        # Compute radial length R from the first sample
        fm0 = fm_arr[0]
        polar0, r_vals, _ = cartesian_to_polar(fm0, n_theta=n_theta)
        R = polar0.shape[0]
        radial_name = "first_moments_radial"
        # If radial already exists and not overwriting, skip
        if radial_name in grp.array_keys() and not overwrite:
            if grp[radial_name].shape == (n_samples, R):
                print(f"[{vid}] '{radial_name}' already exists, skipping")
                continue
            else:
                print(f"[{vid}] '{radial_name}' exists but shape mismatch, recomputing")
                del grp[radial_name]

        radial_all = np.empty((n_samples, R), dtype=np.float32)

        for i in range(n_samples):
            fm_i = fm_arr[i]  # [H, W]
            polar_i, _, _ = cartesian_to_polar(fm_i, n_theta=n_theta)
            # Average over angular coordinate -> 1D radial profile [R]
            radial_i = polar_i.mean(axis=-1)
            radial_all[i] = radial_i.astype(np.float32)

            if (i + 1) % 100 == 0 or (i + 1) == n_samples:
                print(f"[{vid}] processed {i + 1}/{n_samples} samples", end="\r")

        print()  # newline after progress

        # Use similar chunking/compression as first_moments along sample axis
        radial_chunks = (100,) + radial_all.shape[1:]
        grp.create_array(
                "first_moments_radial",
                shape=(0,) + radial_all.shape[1:],
                chunks=radial_chunks,
                dtype=radial_all.dtype,
                overwrite=overwrite,
                fill_value=0
            )
        zarr_arr = grp["first_moments_radial"]
        zarr_arr.append(radial_all, axis=0)
        print(print(grp["first_moments_radial"].shape))
        print(f"[{vid}] wrote 'first_moments_radial' with shape {radial_all.shape}")

    print("Done precomputing first_moments_radial for all volumes.")


if __name__ == "__main__":
    """
    precompute_first_moments_radial(
        zarr_path = "/data/shachar/zarr_files/emdb_vmf_subspace_moments_separated.zarr"
    )
    """
    zarr_file = "/data/shachar/zarr_files/emdb_vmf_subspace_moments_separated.zarr"
    root = zarr.open(zarr_file, mode="r")
    group = root["emd_62916"]
    print(group["eigen_m_detected"].shape, group["first_moments"].shape, group["first_moments_radial"].shape)