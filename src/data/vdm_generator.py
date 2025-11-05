import numpy as np
from typing import Tuple, Optional, Union, Any
from aspire.volume import Volume
from aspire.utils.rotation import Rotation

from src.core.volume_distribution_model import VolumeDistributionModel
from src.utils.distribution_generation_functions import (
    fibonacci_sphere_points, 
    generate_weighted_random_s2_points, 
    create_in_plane_invariant_distribution
)
from src.utils.von_mises_fisher_distributions import (
    generate_random_vmf_parameters,
    so3_distribution_from_vmf
)
from config.config import settings


def generate_vdm_from_volume(
    volume: Volume,
    distribution_type: str,
    downsample_size: Optional[int] = None
) -> VolumeDistributionModel:
    """
    Generate a VDM object from a given ASPIRE volume and distribution type.
    
    This function creates distributions randomly based on the specified type and returns
    a complete VDM object ready for analysis. All parameters are taken from the configuration.
    
    Parameters:
    -----------
    volume : Volume
        ASPIRE Volume object to use for projections
    distribution_type : str
        Type of distribution to generate. Supported values:
        - 'vmf_mixture': von Mises-Fisher mixture distribution
        - 's2_delta_mixture': S2 delta mixture distribution
    downsample_size : int, optional
        Resolution to downsample the volume to. If None, uses config setting.
        
    Returns:
    --------
    vdm : VolumeDistributionModel
        Complete VDM object with volume, rotations, and distribution
    """
    
    # Downsample volume if requested
    if downsample_size is not None:
        volume = volume.downsample(downsample_size)
    
    print(f"=== VDM Generation - {distribution_type} ===")
    print(f"Volume resolution: {volume.resolution}")
    
    if distribution_type == 'vmf_mixture':
        return _generate_vmf_mixture_vdm(volume)
    elif distribution_type == 's2_delta_mixture':
        return _generate_s2_delta_mixture_vdm(volume)
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}. "
                        f"Supported types: 'vmf_mixture', 's2_delta_mixture'")


def _generate_vmf_mixture_vdm(volume: Volume) -> VolumeDistributionModel:
    """
    Generate VDM with von Mises-Fisher mixture distribution.
    All parameters are taken from the configuration.
    
    Parameters:
    -----------
    volume : Volume
        ASPIRE Volume object
        
    Returns:
    --------
    vdm : VolumeDistributionModel
        VDM with vMF mixture distribution
    """
    
    # Get parameters from config
    num_vmf_distributions = settings.data_generation.von_mises_fisher.num_distributions
    fibonacci_n = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    num_in_plane = settings.data_generation.von_mises_fisher.num_in_plane_rotations
    kappa_range = tuple(settings.data_generation.von_mises_fisher.kappa_range)
    
    print(f"vMF mixture parameters:")
    print(f"  - Number of vMF distributions: {num_vmf_distributions}")
    print(f"  - S2 quadrature points: {fibonacci_n}")
    print(f"  - In-plane rotations: {num_in_plane}")
    print(f"  - Kappa range: {kappa_range}")
    
    # Generate S2 quadrature points for rotations
    print("\n1. Generating rotation quadrature...")
    s2_quadrature_points = fibonacci_sphere_points(fibonacci_n)
    print(f"   Generated {len(s2_quadrature_points)} S2 points")
    
    # Generate random von Mises-Fisher mixture parameters
    print("\n2. Generating vMF mixture distribution...")
    mu_directions, kappa_values, mixture_weights = generate_random_vmf_parameters(
        num_vmf_distributions, kappa_range
    )
    
    print(f"   Generated {num_vmf_distributions} vMF distributions:")
    for i, (mu, kappa, weight) in enumerate(zip(mu_directions, kappa_values, mixture_weights)):
        print(f"     vMF_{i+1}: μ=({mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}), κ={kappa:.2f}, weight={weight:.3f}")
    
    # Create S2 von Mises-Fisher mixture distribution on SO(3)
    print("\n3. Creating SO(3) distribution from vMF mixture...")
    rotations, rotation_weights = so3_distribution_from_vmf(
        s2_quadrature_points, mu_directions, kappa_values, mixture_weights, num_in_plane
    )
    
    print(f"   Created SO(3) distribution with {len(rotations)} rotations")
    print(f"   Rotation weights sum: {np.sum(rotation_weights):.6f}")
    
    # Create VDM
    print("\n4. Creating Volume Distribution Model...")
    distribution_metadata = {
        'type': 'vmf_mixture',
        'means': mu_directions,
        'kappas': kappa_values,
        'weights': mixture_weights
    }
    
    vdm = VolumeDistributionModel(
        volume=volume,
        rotations=rotations,
        distribution=rotation_weights,
        distribution_metadata=distribution_metadata
    )
    print("   VDM created successfully")
    
    return vdm


def _generate_s2_delta_mixture_vdm(volume: Volume) -> VolumeDistributionModel:
    """
    Generate VDM with S2 delta mixture distribution.
    All parameters are taken from the configuration.
    
    Parameters:
    -----------
    volume : Volume
        ASPIRE Volume object
        
    Returns:
    --------
    vdm : VolumeDistributionModel
        VDM with S2 delta mixture distribution
    """
    
    # Get parameters from config
    num_s2_points = settings.data_generation.s2_delta_mixture.num_s2_points
    num_in_plane_rotations = settings.data_generation.s2_delta_mixture.num_in_plane_rotations
    uniform_weights = False  # Default to non-uniform weights
    
    print(f"S2 delta mixture parameters:")
    print(f"  - Number of S2 points: {num_s2_points}")
    print(f"  - In-plane rotations: {num_in_plane_rotations}")
    print(f"  - Uniform weights: {uniform_weights}")
    
    # Generate random S2 points with weights
    print("\n1. Generating S2 delta points...")
    if uniform_weights:
        # Generate random S2 points without weights (will be made uniform)
        from src.utils.distribution_generation_functions import generate_random_s2_points
        s2_coords = generate_random_s2_points(num_s2_points)
        s2_weights = None  # Will be made uniform in create_in_plane_invariant_distribution
        print(f"   Generated {num_s2_points} uniform-weighted S2 points")
    else:
        # Generate random S2 points with non-uniform weights
        s2_coords, s2_weights = generate_weighted_random_s2_points(num_s2_points)
        print(f"   Generated {num_s2_points} non-uniform-weighted S2 points")
        print(f"   Weight range: [{np.min(s2_weights):.4f}, {np.max(s2_weights):.4f}]")
    
    # Convert S2 spherical coordinates to 3D Cartesian coordinates for metadata
    phi, theta = s2_coords[:, 0], s2_coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    s2_points_3d = np.column_stack([x, y, z])
    
    # Create in-plane invariant distribution with the S2 points
    print("\n2. Creating SO(3) distribution...")
    rotations, distribution = create_in_plane_invariant_distribution(
        s2_coords, s2_weights, num_in_plane_rotations=num_in_plane_rotations, 
        is_s2_uniform=uniform_weights
    )
    
    print(f"   Created SO(3) distribution with {len(rotations)} rotations")
    print(f"   Total rotations: {num_s2_points} × {num_in_plane_rotations} = {len(rotations)}")
    print(f"   Rotation weights sum: {np.sum(distribution):.6f}")
    
    # Create VDM
    print("\n3. Creating Volume Distribution Model...")
    distribution_metadata = {
        'type': 's2_delta_mixture',
        's2_points': s2_points_3d,
        'weights': s2_weights if s2_weights is not None else np.ones(num_s2_points) / num_s2_points
    }
    
    vdm = VolumeDistributionModel(
        volume=volume,
        rotations=rotations,
        distribution=distribution,
        distribution_metadata=distribution_metadata
    )
    print("   VDM created successfully")
    
    return vdm
