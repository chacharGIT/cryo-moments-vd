import numpy as np
from scipy.ndimage import map_coordinates
from config.config import settings

def cartesian_to_polar(cartesian_image, n_theta=settings.data_generation.cartesian_to_polar_n_theta, order=3, boundary_mode='nearest'):
    """
    Convert a 2D Cartesian image to its polar coordinate representation using interpolation.

    Parameters
    ----------
    cartesian_image : ndarray
        2D array representing the image.
    r_vals : ndarray or None
        Array of radii to sample. If None, uses all unique pixel radii from the image center.
    n_theta : int
        Number of angular samples (columns in polar image).
    center : tuple or None
        (y, x) coordinates of the center. If None, uses image center.
    order : int
        Interpolation order for map_coordinates (default: 3).
    boundary_mode : str
        Boundary mode for map_coordinates (default: 'reflect').

    Returns
    -------
    polar_image : ndarray
        2D array of shape (len(r_vals), n_theta) representing the polar image.
    r_vals : ndarray
        Radii used for sampling.
    theta_vals : ndarray
        Angles used for sampling.
    """
    cartesian_image = np.asarray(cartesian_image)
    ny, nx = cartesian_image.shape
    cy, cx = ny / 2 , nx / 2
    Y, X = np.indices(cartesian_image.shape)
    R_pixel = np.hypot(Y - cy, X - cx)
    r_vals = np.sort(np.unique(R_pixel))
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R_grid, Theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
    X_grid = cx + R_grid * np.cos(Theta_grid)
    Y_grid = cy + R_grid * np.sin(Theta_grid)
    coords = np.vstack([Y_grid.ravel(), X_grid.ravel()])
    polar_image = map_coordinates(cartesian_image, coords, order=order, mode=boundary_mode)
    polar_image = polar_image.reshape(len(r_vals), len(theta_vals))
    return polar_image, r_vals, n_theta

def polar_to_cartesian(polar_image, r_vals, n_theta, output_shape, order=3, boundary_mode='nearest'):
    """
    Convert a polar image back to Cartesian coordinates using interpolation.

    Parameters
    ----------
    polar_image : ndarray
        2D array of shape (len(r_vals), len(theta_vals)) representing the polar image.
    r_vals : ndarray
        Radii used for sampling.
    theta_vals : ndarray
        Angles used for sampling.
    output_shape : tuple or None
        Shape of the output Cartesian image. If None, uses a square image with side 2*max(r_vals)+1.
    center : tuple or None
        (y, x) coordinates of the center. If None, uses image center.
    order : int
        Interpolation order for map_coordinates (default: 3).
    boundary_mode : str
        Boundary mode for map_coordinates (default: 'reflect').

    Returns
    -------
    image : ndarray
        2D array representing the reconstructed Cartesian image.
    """
    polar_image = np.asarray(polar_image)
    ny, nx = output_shape
    cy, cx = ny / 2, nx / 2
    Y, X = np.indices(output_shape)
    R = np.hypot(Y - cy, X - cx)
    Theta = np.arctan2(Y - cy, X - cx)
    r_idx = np.interp(R.ravel(), r_vals, np.arange(len(r_vals)))
    # Map Theta to theta index (assumes uniform theta_vals). Handle wrap-around by extending the polar image
    theta_idx = (Theta.ravel() / (2*np.pi) * n_theta)
    theta_idx = theta_idx % n_theta  # wrap-around
    coords = np.vstack([r_idx, theta_idx])
    cartesian_image = map_coordinates(polar_image, coords, order=order, mode=boundary_mode)
    cartesian_image = cartesian_image.reshape(output_shape)
    return cartesian_image

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from aspire.downloader import emdb_2660
    from src.core.volume_distribution_model import VolumeDistributionModel
    from src.utils.von_mises_fisher_distributions import generate_random_von_mises_fisher_parameters, evaluate_von_mises_fisher_mixture
    from src.utils.distribution_generation_functions import fibonacci_sphere_points

    vol_ds = emdb_2660().downsample(settings.data_generation.downsample_size)
    L = vol_ds.resolution
    # Get VMF parameters from config
    vmf_cfg = settings.data_generation.von_mises_fisher

    # Generate S2 quadrature points
    quadrature_points = fibonacci_sphere_points(n=vmf_cfg.fibonacci_spiral_n)

    # Generate VMF mixture parameters
    mu, kappa, weights = generate_random_von_mises_fisher_parameters(
        vmf_cfg.num_distributions, kappa_start=vmf_cfg.kappa_start, kappa_mean=vmf_cfg.kappa_mean
    )
    mixture_eval = evaluate_von_mises_fisher_mixture(quadrature_points, mu, kappa, weights)

    vdm = VolumeDistributionModel(vol_ds, rotations=quadrature_points, distribution=mixture_eval,
                                   distribution_metadata={
                                        'type': 'vmf_mixture',
                                        'means': mu,
                                        'kappas': kappa,
                                        'weights': weights
                                    }, in_plane_invariant_distribution=True)
    image = vdm.generate_projections(num_projections=1, sigma=0)[0]
    polar_image, r_vals, n_theta = cartesian_to_polar(image, n_theta=settings.data_generation.cartesian_to_polar_n_theta)
    reconstructed = polar_to_cartesian(polar_image, r_vals, n_theta, output_shape=image.shape)
    diff_l2_norm = np.linalg.norm(image - reconstructed)
    image_l2_norm = np.linalg.norm(image)
    print(f"Relative L2 norm of difference: {diff_l2_norm/image_l2_norm:.4e}")

    # Plot and save results
    os.makedirs("outputs/tmp_figs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(polar_image, aspect='auto', cmap='gray')
    axes[1].set_title('Polar Image')
    axes[1].axis('off')
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title('Reconstructed Image')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("outputs/tmp_figs/polar_transform_test.png", dpi=150)
    plt.close(fig)