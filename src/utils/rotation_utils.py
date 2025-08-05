import numpy as np

def geodesic_distance_SO3(R1, R2):
    """
    Compute the geodesic distance (in radians) between two SO(3) rotations.

    Parameters
    ----------
    R1 : np.ndarray or object
        First rotation. Either a 3x3 numpy array (rotation matrix) or an object with a .as_matrix() method.
    R2 : np.ndarray or object
        Second rotation. Either a 3x3 numpy array (rotation matrix) or an object with a .as_matrix() method.

    Returns
    -------
    theta : float
        Geodesic distance (in radians) between R1 and R2, in [0, pi].
    """
    # Convert to numpy arrays if needed
    if hasattr(R1, 'as_matrix'):
        R1 = R1.as_matrix()
    if hasattr(R2, 'as_matrix'):
        R2 = R2.as_matrix()
    # Compute relative rotation
    R = np.dot(R1, R2.T)
    # The geodesic distance is arccos((trace(R) - 1)/2)
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta
