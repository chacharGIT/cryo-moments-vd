import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.downloader import emdb_2660
from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.distribution_generation_functions import generate_weighted_random_s2_points, create_in_plane_invariant_distribution
from config.config import settings


def generate_and_save_noisy_data(vdm: VolumeDistributionModel, num_images, output_path, sigma=0.1, downsample_size=64):
    """
    Generate noisy projections from the input volume and save them to a parquet file.
    
    Parameters:
    -----------
    vdm : VolumeDistributionModel
        The volume distribution model to generate projections from
    num_images : int
        Number of noisy projections to generate
    output_path : str
        Path where the parquet file will be saved
    sigma : float, optional
        Standard deviation of the Gaussian noise to add. Default is 0.1.
    downsample_size : int, optional
        Size to downsample the volume to. Default is 64.
    """  
    
    print(f"Generating {num_images} noisy projections with sigma={sigma}...")
    # Use the new generate_noisy_projections function
    noisy_projections, sampled_rotations = vdm.generate_projections(
        num_projections=num_images, sigma=sigma
    )
    
    # Convert the data to a format suitable for parquet
    print("Preparing data for parquet file...")
    image_data = noisy_projections
    rotation_matrices = sampled_rotations
    
    # Create a dictionary to store the data
    data_dict = {
        'image_data': pa.array(image_data.reshape(num_images, -1).tolist()),  # Flatten images for storage
        'image_shape': pa.array([list(image_data.shape[1:])]),  # Store original shape
        'rotation_matrices': pa.array(rotation_matrices.tolist()),
        'sigma': pa.array([sigma]),
        'volume_resolution': pa.array([vdm.volume.resolution])
    }
    
    # Create a table from the dictionary
    table = pa.Table.from_pydict(data_dict)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the table to a parquet file
    print(f"Saving data to {output_path}...")
    pq.write_table(table, output_path)
    
    print(f"Successfully saved {num_images} noisy projections to {output_path}")
    print(f"Data shape: {image_data.shape}")
    print(f"Rotation matrices shape: {rotation_matrices.shape}")


if __name__ == "__main__":
    # Get a sample volume
    vol = emdb_2660().downsample(settings.data_generation.downsample_size)
    
    # Generate random S2 points with non-uniform weights using the new function
    s2_coords, s2_weights = generate_weighted_random_s2_points(
        num_points=settings.data_generation.noisy.num_points_s2
    )
    
    # Create in-plane invariant distribution with the weighted S2 points
    rotations, distribution = create_in_plane_invariant_distribution(
        s2_coords, s2_weights, num_in_plane_rotations=settings.data_generation.noisy.num_points_s1, is_s2_uniform=False
    )
    
    # Convert S2 spherical coordinates to 3D Cartesian coordinates
    phi, theta = s2_coords[:, 0], s2_coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    s2_points_3d = np.column_stack([x, y, z])
    
    # Create the VolumeDistributionModel
    vdm = VolumeDistributionModel(vol, rotations, distribution, s2_points=s2_points_3d, s2_weights=s2_weights)
    
    generate_and_save_noisy_data(
        vdm,
        settings.data_generation.noisy.num_images,
        settings.data_generation.noisy.output_path,
        settings.data_generation.noisy.sigma,
        settings.data_generation.downsample_size
    )
