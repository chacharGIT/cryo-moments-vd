import numpy as np

from aspire.utils.rotation import Rotation

def fibonacci_sphere_points(n: int):
    """
    Generate N approximately uniformly distributed points on the unit sphere S²
    using the Fibonacci (golden spiral) method.

    Parameters
    ----------
    N : int
        Number of points to generate.

    Returns
    -------
    points : ndarray of shape (n, 3)
        Cartesian coordinates of points on the unit sphere.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    i = np.arange(n)

    z = 1 - (2*i + 1)/n                 # z-coordinates (from north to south)
    theta = 2 * np.pi * i / phi        # golden angle increments
    r = np.sqrt(1 - z**2)              # radius at each height

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.column_stack((x, y, z))
    return points

def cartesian_to_spherical(points):
    """
    Convert normalized points in R^3 to spherical coordinates (phi, theta).
    Assumes input points are normalized (on the unit sphere).

    Parameters:
    -----------
    points : ndarray
        Array of shape (N, 3) representing N points in R^3.

    Returns:
    --------
    spherical_coords : ndarray
        Array of shape (N, 2), where each row is (phi, theta).
        phi in [0, 2*pi), theta in [0, pi].
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.column_stack([phi, theta])

def spherical_to_cartesian(spherical_coords):
    """
    Convert spherical coordinates (phi, theta) to Cartesian coordinates (x, y, z).
    Assumes phi in [0, 2*pi), theta in [0, pi].

    Parameters:
    -----------
    spherical_coords : ndarray
        Array of shape (N, 2), where each row is (phi, theta).

    Returns:
    --------
    cartesian_coords : ndarray
        Array of shape (N, 3), where each row is (x, y, z).
    """
    phi, theta = spherical_coords[:, 0], spherical_coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def generate_random_s2_points(num_points):
    """
    Generate random points on the sphere S2 in spherical coordinates.
    
    Parameters:
    -----------
    num_points : int
        Number of points to generate on S2
    
    Returns:
    --------
    spherical_coords : ndarray
        Array of shape (num_points, 2) containing (phi, theta) in spherical coordinates
        phi: azimuthal angle [0, 2π]
        theta: polar angle [0, π]
    """
    
    # Generate random points on S2 using uniform sampling on the sphere
    # Method: Generate random points in 3D and normalize to unit sphere
    cartesian_points = np.random.randn(num_points, 3)
    cartesian_points = cartesian_points / np.linalg.norm(cartesian_points, axis=1, keepdims=True)
    
    # Convert to spherical coordinates
    # cartesian_points = [x, y, z] where x^2 + y^2 + z^2 = 1
    theta = np.arccos(np.clip(cartesian_points[:, 2], -1, 1))  # Polar angle [0, π]
    phi = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuthal angle [-π, π]
    
    # Ensure phi is in [0, 2π]
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    
    spherical_coords = np.column_stack([phi, theta])
    return spherical_coords


def generate_weighted_random_s2_points(num_points):
    """
    Generate random points on the sphere S2 with random non-uniform weights that sum to 1.
    
    Parameters:
    -----------
    num_points : int
        Number of points to generate on S2
    
    Returns:
    --------
    s2_points : ndarray
        Array of shape (num_points, 2) containing (phi, theta) in spherical coordinates
        phi: azimuthal angle [0, 2π]
        theta: polar angle [0, π]
    weights : ndarray
        Array of shape (num_points,) containing random weights that sum to 1
    """
    
    # Generate random S2 points in Cartesian coordinates
    s2_points = np.random.randn(num_points, 3)
    s2_points = s2_points / np.linalg.norm(s2_points, axis=1, keepdims=True)

    # Generate random weights uniformly from [0,1] and normalize to sum to 1
    weights = np.random.uniform(0, 1, num_points)
    weights = weights / np.sum(weights)

    return s2_points, weights


def create_in_plane_invariant_distribution(s2_points, s2_weights=None, num_in_plane_rotations=8, is_s2_uniform=False):
    """
    Create a distribution from given S2 points (in Cartesian coordinates) with weights 
    and uniform in-plane rotations.
    
    Parameters:
    -----------
    s2_coords : ndarray
        Array of shape (num_s2_points, 3) containing (x, y, z) in Cartesian coordinates
    s2_weights : ndarray or None
        Array of shape (num_s2_points,) containing weights for each S2 point
        (ignored if is_s2_uniform=True)
    num_in_plane_rotations : int
        Number of uniform points for in-plane rotations
    is_s2_uniform : bool
        If True, ignore s2_weights and create uniform weights for S2 points
    
    Returns:
    --------
    rotations : Rotation
        A set of rotations in SO(3) representing the distribution
    distribution : ndarray
        Array of weights for each rotation
    """
    num_s2_points = len(s2_points)
    
    # Create uniform weights if requested
    if is_s2_uniform or s2_weights is None:
        s2_weights = np.ones(num_s2_points) / num_s2_points
    
    # Convert S2 Cartesian points to spherical coordinates (phi, theta)
    phi = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    theta = np.arccos(np.clip(s2_points[:, 2], -1, 1))

    # Generate uniform in-plane rotation angles with a random starting angle
    psi_0 = np.random.uniform(0, 2*np.pi, num_s2_points)
    # For each S2 point, generate evenly spaced psi angles starting from psi_0
    in_plane_angles = psi_0[:, None] + np.arange(num_in_plane_rotations) * 2 * np.pi / num_in_plane_rotations
    psi_in_plane = in_plane_angles % (2 * np.pi)
    # Use meshgrid to create all combinations efficiently
    phi_s2 = np.repeat(phi[:, None], num_in_plane_rotations, axis=1)
    theta_s2 = np.repeat(theta[:, None], num_in_plane_rotations, axis=1)
    
    # Flatten to get 1D arrays
    phi_array = phi_s2.flatten()
    theta_array = theta_s2.flatten()
    psi_array = psi_in_plane.flatten()
    
    distribution = np.repeat(s2_weights / num_in_plane_rotations, num_in_plane_rotations)
    
    # Stack angles for ASPIRE's from_euler
    euler_angles = np.stack([phi_array, theta_array, psi_array], axis=1)
    # Create Rotation object (ZYZ convention)
    rotations = Rotation.from_euler(euler_angles, dtype=np.float32)

    return rotations, distribution


if __name__ == "__main__":

    from aspire.downloader import emdb_2660
    
    # Get a sample volume
    vol_ds = emdb_2660().downsample(64)
    L = vol_ds.resolution
    
    # Test the new distribution generation functions
    print("Testing distribution generation functions")
    
    # Generate 3 random points on S2
    s2_coords = generate_random_s2_points(num_points=3)
    print(f"Generated {len(s2_coords)} random S2 points in spherical coordinates:")
    print(f"S2 coordinates (phi, theta):\n{s2_coords}")
    
    # Create in-plane invariant distribution with 8 in-plane rotations
    num_in_plane = 8
    rotations, rotation_weights = create_in_plane_invariant_distribution(
        s2_coords, None, num_in_plane_rotations=num_in_plane, is_s2_uniform=True
    )
    
    print(f"\nCreated in-plane invariant distribution:")
    print(f"Number of S2 points: {len(s2_coords)}")
    print(f"Number of in-plane rotations per S2 point: {num_in_plane}")
    print(f"Total number of rotations: {len(rotations)}")
    print(f"Rotation weights sum: {np.sum(rotation_weights):.6f}")
    print(f"First 5 rotation weights: {rotation_weights[:5]}")
        
    # Save the values for later usage
    saved_rotations = rotations
    saved_distribution = rotation_weights
    
    print(f"\nSaved rotations and distribution for later usage:")
    print(f"saved_rotations: Rotation object with {len(saved_rotations)} rotations")
    print(f"saved_distribution: Array of {len(saved_distribution)} weights")
