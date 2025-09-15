import numpy as np

def exp_map_s2(base, tangent):
    """
    Exponential map on S2 manifold.
    base: np.ndarray, shape (3,) - base point on S2 (unit vector)
    tangent: np.ndarray, shape (3,) - tangent vector at base (orthogonal to base)
    Returns: np.ndarray, shape (3,) - point on S2
    """
    norm_tan = np.linalg.norm(tangent)
    if norm_tan < 1e-8:
        return base
    return np.cos(norm_tan) * base + np.sin(norm_tan) * (tangent / norm_tan)


def geodesic_distance_s2(x, y):
    """
    Geodesic (arc) distance between two points on S2.
    x, y: np.ndarray, shape (3,)
    Returns: float
    """
    dot = np.clip(np.dot(x, y), -1.0, 1.0)
    return np.arccos(dot)
