import numpy as np
import os
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from aspire.downloader import emdb_2660
from volume_distribution_model import VolumeDistributionModel
from distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution
from config import settings


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


def generate_and_save_analytical_moments(vdm: VolumeDistributionModel, output_path):
    """
    Generate clean analytical moments from a VolumeDistributionModel and save them to a parquet file.
    
    Parameters:
    -----------
    vdm : VolumeDistributionModel
        The volume distribution model to generate analytical moments from
    output_path : str
        Path where the parquet file will be saved
    """
    A_1 = vdm.first_analytical_moment()
    A_2 = vdm.second_analytical_moment()
    
    # Flatten the moments for storage
    first_moment_flat = A_1.flatten()
    second_moment_flat = A_2.flatten()
    
    # Get volume data and rotation matrices
    volume_data = vdm.volume.asnumpy().flatten()
    rotation_matrices = vdm.rotations.matrices  # Get rotation matrices
    
    # Create a dictionary to store the data
    data_dict = {
        'first_moment': pa.array([first_moment_flat.tolist()]),  # Store as single row
        'first_moment_shape': pa.array([list(A_1.shape)]),  # Store original shape
        'second_moment': pa.array([second_moment_flat.tolist()]),  # Store as single row
        'second_moment_shape': pa.array([list(A_2.shape)]),  # Store original shape
        'volume_data': pa.array([volume_data.tolist()]),  # Store flattened volume
        'volume_shape': pa.array([list(vdm.volume.asnumpy().shape)]),  # Store original volume shape
        'rotation_matrices': pa.array([rotation_matrices.tolist()]),  # Store rotation matrices
        'num_rotations': pa.array([len(vdm.rotations)]),
        'distribution_weights': pa.array([vdm.distribution.tolist()])
    }
    
    # Add S2 points and weights if available
    if vdm.s2_points is not None and vdm.s2_weights is not None:
        data_dict['s2_points'] = pa.array([vdm.s2_points.tolist()])  # 3D Cartesian coordinates
        data_dict['s2_weights'] = pa.array([vdm.s2_weights.tolist()])  # S2 point weights
        data_dict['num_s2_points'] = pa.array([len(vdm.s2_points)])
    else:
        # Store None values to maintain schema consistency
        data_dict['s2_points'] = pa.array([None])
        data_dict['s2_weights'] = pa.array([None])
        data_dict['num_s2_points'] = pa.array([0])
        print("No S2 data available in VDM")
    
    # Create a table from the dictionary
    table = pa.Table.from_pydict(data_dict)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the table to a parquet file
    print(f"Saving analytical moments to {output_path}...")
    pq.write_table(table, output_path)


if __name__ == "__main__":
    # Get a sample volume
    vol = emdb_2660().downsample(settings.data_generation.downsample_size)
    print(f"Volume resolution: {vol.resolution}")
    
    # Generate random S2 points with non-uniform weights using the new function
    s2_coords, s2_weights = generate_weighted_random_s2_points(
        num_points=settings.data_generation.num_s2_points
    )
    
    # Create in-plane invariant distribution with the weighted S2 points
    rotations, distribution = create_in_plane_invariant_distribution(
        s2_coords, s2_weights, num_in_plane_rotations=settings.data_generation.num_in_plane, is_s2_uniform=False
    )

    # Convert S2 spherical coordinates to 3D Cartesian coordinates
    phi, theta = s2_coords[:, 0], s2_coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    s2_points_3d = np.column_stack([x, y, z])
    
    # Create the VolumeDistributionModel with S2 points and weights
    vdm = VolumeDistributionModel(vol, rotations, distribution, s2_points=s2_points_3d, s2_weights=s2_weights)
    
    # Generate and save the analytical moments
    generate_and_save_analytical_moments(vdm, settings.data.parquet_path)
    
    # Also save the VDM object as pickle
    pickle_path = settings.data.parquet_path.replace('.parquet', '_vdm.pkl')
    save_vdm_pickle(vdm, pickle_path)
