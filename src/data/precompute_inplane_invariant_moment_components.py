import numpy as np
from aspire.volume import Volume
from aspire.image import Image
from aspire.operators import Projector
from aspire.utils.rotation import Rotation
from src.utils.empirical_moment_calculation import first_empirical_moment, second_empirical_moment

# Parameters (customize as needed)
N_proj = 100  # Number of viewing directions (quadrature points)
N_rot = 36    # Number of SO(2) in-plane rotations per direction
volume_path = 'path_to_emdb_or_mrc_file'  # TODO: set actual path

# Generate viewing directions (Fibonacci sphere)
def fibonacci_sphere_points(n):
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    z = 1 - (2*i + 1)/n
    theta = 2 * np.pi * i / phi
    r = np.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, z))
    return points

# Main computation
def compute_inplane_invariant_moment_components(volume_path, N_proj, N_rot):
    vol = Volume.load(volume_path)
    projector = Projector(vol)
    viewing_dirs = fibonacci_sphere_points(N_proj)
    moment_components = []
    for idx, vdir in enumerate(viewing_dirs):
        # Convert viewing direction to ASPIRE rotation (z-axis is reference)
        rot = Rotation.from_viewing_direction(vdir)
        # Generate SO(2) in-plane rotations (angles from 0 to 2pi)
        inplane_angles = np.linspace(0, 2*np.pi, N_rot, endpoint=False)
        projections = []
        for alpha in inplane_angles:
            # Compose rotation: first align to viewing direction, then rotate in-plane
            full_rot = rot * Rotation.from_euler('Z', alpha)
            proj_img = projector.project(full_rot)
            projections.append(proj_img._data)
        projections = np.stack(projections, axis=0)  # Shape: (N_rot, H, W)
        # Compute empirical moments
        first_mom = np.mean(projections, axis=0)
        second_mom = np.einsum('nij,nkl->ijkl', projections, projections) / N_rot
        moment_components.append({
            'viewing_direction': vdir,
            'first_moment': first_mom,
            'second_moment': second_mom
        })
        print(f"Done: {idx+1}/{N_proj} viewing directions")
    return moment_components

if __name__ == "__main__":
    # Example usage
    results = compute_inplane_invariant_moment_components(volume_path, N_proj, N_rot)
    # TODO: Save results to disk (e.g., with numpy or zarr)
    print("Finished computing in-plane invariant moment components.")
