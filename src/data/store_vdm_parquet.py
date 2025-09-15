import dis
import numpy as np
import os
import torch
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.matrix_generation_functions import generate_goe_matrix
from config.config import settings


def save_vdm_pickle(vdm: VolumeDistributionModel, output_path):
    """
    Save the VolumeDistributionModel object as a pickle file.
    
    Parameters:
    -----------
    vdm : VolumeDistributionModel
        The volume distribution model to save
    output_path : str
        Path where the pickle file will be saved
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the VDM object as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(vdm, f)
    
    print(f"Saved VDM object to {output_path}")



def save_distribution_data(vdm: VolumeDistributionModel, output_path):
    """
    Save analytical moments, volume, distribution, and metadata to a parquet file.

    Parameters:
    -----------
    vdm : VolumeDistributionModel
        The volume distribution model to save. Its distribution_metadata field determines
        what extra information is stored (e.g., S2 points/weights or VMF mixture parameters).
    output_path : str
        Path where the parquet file will be saved
    """
    print("Calculating analytical moments")
    A_1 = vdm.first_analytical_moment(device=settings.device.cuda_device)
    #A_2 = vdm.second_analytical_moment(batch_size=settings.data_generation.second_moment_batch_size,
    #                                    show_progress=True, device=settings.device.cuda_device)
    #print(f"First moment shape: {A_1.shape}, Second moment shape: {A_2.shape}")

    # Generate random GOE matrix instead of computing analytical second moment
    print("Generating random GOE matrix for second moment...")
    goe_matrix = generate_goe_matrix(downsample_size=settings.data_generation.downsample_size)

    # Reshape GOE matrix to (L, L, L, L) format to match analytical second moment shape
    L = A_1.shape[0]  # Image dimension
    A_2 = goe_matrix.reshape(L, L, L, L).cpu().numpy()
    print(f"Random second moment shape: {A_2.shape}")

    # Flatten the moments for storage
    first_moment_flat = A_1.flatten()
    second_moment_flat = A_2.flatten()

    # Get volume data and rotation matrices
    volume_data = vdm.volume.asnumpy().flatten()
    rotation_matrices = vdm.rotations.matrices

    data_dict = {
        'first_moment': pa.array([first_moment_flat.tolist()]),
        'first_moment_shape': pa.array([list(A_1.shape)]),
        'second_moment': pa.array([second_moment_flat.tolist()]),
        'second_moment_shape': pa.array([list(A_2.shape)]),
        'volume_data': pa.array([volume_data.tolist()]),
        'volume_shape': pa.array([list(vdm.volume.asnumpy().shape)]),
        'rotation_matrices': pa.array([rotation_matrices.tolist()]),
        'num_rotations': pa.array([len(vdm.rotations)]),
        'distribution_weights': pa.array([vdm.distribution.tolist()]),
        'distribution_type': pa.array([vdm.distribution_metadata.get('type') if vdm.distribution_metadata is not None else None])
    }
    print(vdm.distribution_metadata)
    # Add distribution metadata
    if vdm.distribution_metadata is not None:
        dist_type = vdm.distribution_metadata.get('type', None)
        if dist_type == 's2_delta_mixture':
            # Expect s2_points and weights in metadata
            s2_points = vdm.distribution_metadata.get('s2_points', None)
            s2_weights = vdm.distribution_metadata.get('weights', None)
            if s2_points is None or s2_weights is None:
                raise ValueError("'s2_delta_mixture' requires both 's2_points' and 'weights' in distribution_metadata.")
            data_dict['s2_points'] = pa.array([np.asarray(s2_points).tolist()])
            data_dict['s2_weights'] = pa.array([np.asarray(s2_weights).tolist()])
        elif dist_type == 'vmf_mixture':
            # Expect means, kappas, weights in metadata
            means = vdm.distribution_metadata.get('means', None)
            kappas = vdm.distribution_metadata.get('kappas', None)
            weights = vdm.distribution_metadata.get('weights', None)
            if means is None or kappas is None or weights is None:
                raise ValueError("'vmf_mixture' requires 'means', 'kappas', and 'weights' in distribution_metadata.")
            data_dict['means'] = pa.array([np.asarray(means).tolist()])
            data_dict['kappas'] = pa.array([np.asarray(kappas).tolist()])
            data_dict['weights'] = pa.array([np.asarray(weights).tolist()])
    table = pa.Table.from_pydict(data_dict)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving distribution data to {output_path}...")
    pq.write_table(table, output_path)


if __name__ == "__main__":
    # Get a sample volume
    from src.data.vdm_generator import generate_vdm_from_volume
    from aspire.downloader import emdb_2660
    vol = emdb_2660().downsample(settings.data_generation.downsample_size)
    print(f"Volume resolution: {vol.resolution}")

    # Generate VDM using the vdm_generator
    print("Generating VDM with von Mises Fisher mixture...")
    vdm = generate_vdm_from_volume(vol, distribution_type='vmf_mixture')
    
    print(f"Created VDM with {len(vdm.rotations)} rotations")
    print(f"Distribution type: {vdm.distribution_metadata.get('type')}")
    
    # Save all distribution data, including von Mises mixture parameters
    print("Saving distribution data parquet file")
    save_distribution_data(vdm, settings.data_generation.parquet_path)

    # Also save the VDM object as pickle
    pickle_path = settings.data_generation.parquet_path.replace('.parquet', '_vdm.pkl')
    save_vdm_pickle(vdm, pickle_path)
