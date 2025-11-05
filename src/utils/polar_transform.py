import numpy as np
from scipy.ndimage import map_coordinates
from config.config import settings

def cartesian_to_polar(cartesian_image, n_theta=settings.data_generation.cartesian_to_polar_n_theta,
                       order=5, boundary_mode='nearest', restrict_to_disk=True):
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
    restrict_to_disk : bool
        If True, only sample up to the inscribed disk.

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
    if restrict_to_disk:
        max_radius = min(cy, cx)
        r_vals = r_vals[r_vals <= max_radius]
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R_grid, Theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
    X_grid = cx + R_grid * np.cos(Theta_grid)
    Y_grid = cy + R_grid * np.sin(Theta_grid)
    coords = np.vstack([Y_grid.ravel(), X_grid.ravel()])
    polar_image = map_coordinates(cartesian_image, coords, order=order, mode=boundary_mode)
    polar_image = polar_image.reshape(len(r_vals), len(theta_vals))
    return polar_image, r_vals, n_theta

def polar_to_cartesian(polar_image, r_vals, n_theta=settings.data_generation.cartesian_to_polar_n_theta,
                        output_shape=None, order=5, boundary_mode='nearest'):
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
    if output_shape is None:
        L = settings.data_generation.downsample_size
        output_shape = (L, L)
    polar_image = np.asarray(polar_image)
    ny, nx = output_shape
    cy, cx = ny / 2, nx / 2
    Y, X = np.indices(output_shape)
    R = np.hypot(Y - cy, X - cx)
    Theta = np.arctan2(Y - cy, X - cx)
    r_idx = np.interp(R.ravel(), r_vals, np.arange(len(r_vals)))
    # Map Theta to theta index (assumes uniform theta_vals). Handle wrap-around by extending the polar image
    theta_idx = (Theta.ravel() / (2*np.pi) * n_theta) % n_theta
    polar_extended = np.hstack([polar_image, polar_image[:, :order]])
    coords = np.vstack([r_idx, theta_idx])
    cartesian_image = map_coordinates(polar_extended, coords, order=order, mode=boundary_mode)
    cartesian_image = cartesian_image.reshape(output_shape)
    return cartesian_image

def remove_global_phase(radial_vector: np.ndarray) -> np.ndarray:
    """
    Remove global phase from a complex radial vector so its sum is real and positive.

    Parameters
    ----------
    radial_vector : ndarray
        1D complex array.

    Returns
    -------
    radial_vector_real : ndarray
        1D real array with global phase removed.
    """
    phase = np.angle(np.sum(radial_vector))
    radial_real = np.real(radial_vector * np.exp(-1j * phase))
    if np.sum(radial_real) < 0:
        radial_real = -radial_real
    return radial_real

def extract_dominant_angular_fourier_mode(polar_image: np.ndarray):
    """
    For a real polar image, find the angular mode m for which the sum of energies of
    plus/minus m Fourier components is maximal. For both +m and -m, extract the radial
    profile, remove global phase, align their sign and return their average.

    Parameters
    ----------
    polar_image : ndarray
        2D real array of shape (num_r, num_theta).

    Returns
    -------
    m_detected : int
        Detected angular frequency m.
    radial_profile : ndarray
        1D array (num_r,) averaged from +m and -m components, global phase removed.
    energy_fraction : float
        Fraction of total energy in the detected modes.
    """
    num_r, num_theta = polar_image.shape
    fft_coeffs = np.fft.fft(polar_image, axis=1)
    total_energy = np.sum(np.abs(fft_coeffs)**2)

    # Compute energy for each mode as sum of plus/minus
    energies = np.zeros(num_theta // 2 + 1)
    for m in range(num_theta // 2 + 1):
        idx_pos = m
        idx_neg = (-m) % num_theta
        if m == 0:
            energy = np.sum(np.abs(fft_coeffs[:, idx_pos])**2)
        else:
            energy = np.sum(np.abs(fft_coeffs[:, idx_pos])**2) + np.sum(np.abs(fft_coeffs[:, idx_neg])**2)
        energies[m] = energy

    m_detected = np.argmax(energies)
    if m_detected == 0:
        radial_profile = remove_global_phase(fft_coeffs[:, 0])
        energy_fraction = energies[0] / total_energy
        return 0, radial_profile, energy_fraction
    else:
        idx_pos = m_detected
        idx_neg = (-m_detected) % num_theta
        radial_plus = fft_coeffs[:, idx_pos]
        radial_minus = fft_coeffs[:, idx_neg]
        radial_plus_real = remove_global_phase(radial_plus)
        radial_minus_real = remove_global_phase(radial_minus)
        radial_profile = (radial_plus_real + radial_minus_real)
        radial_profile /= np.sum(radial_profile)  # Normalize
        energy_fraction = energies[m_detected] / total_energy
        return m_detected, radial_profile, energy_fraction

def extract_dominant_angular_fourier_mode_deprecated(polar_image: np.ndarray):
    """
    Detect the dominant angular frequency m in a complex polar image using FFT along the angular axis.
    If the input is real, returns the radial average.

    Parameters
    ----------
    polar_image : ndarray
        2D array of shape (num_r, num_theta) representing the polar image.

    Returns
    -------
    m_detected : int
        Detected angular frequency m.
    radial_profile : ndarray
        1D array (num_r,) of the real part of the FFT component for m_detected, phase-corrected,
        or the radial average if the input is real.
    energy_fraction : float, optional
        Fraction of total energy in the detected mode. Returns 1.0 if input is real.
    """
    _ , num_theta = polar_image.shape
    if np.isrealobj(polar_image):
        radial_profile = np.mean(polar_image, axis=1)
        return 0, radial_profile, 1.0
    else:
        fft_coeffs = np.fft.fft(polar_image, axis=1)
        spectrum = np.sum(np.abs(fft_coeffs), axis=0)
        # Find the frequency with the largest magnitude (excluding DC if desired)
        m_detected = np.argmax(spectrum)
        # For negative frequencies (if needed)
        if m_detected > num_theta // 2:
            m_detected -= num_theta

        idx_detected = m_detected
        energy_m = np.sum(np.abs(fft_coeffs[:, idx_detected])**2)
        total_energy = np.sum(np.abs(fft_coeffs)**2)
        energy_fraction = energy_m / total_energy

        radial_fft_component = fft_coeffs[:, m_detected]
        global_phase = np.angle(np.sum(np.abs(radial_fft_component)**2 * radial_fft_component))
        radial_profile = np.real(radial_fft_component * np.exp(-1j * global_phase))
        return m_detected, radial_profile, energy_fraction

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from aspire.downloader import emdb_2660
    from src.core.volume_distribution_model import VolumeDistributionModel
    from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture
    from src.utils.distribution_generation_functions import fibonacci_sphere_points

    vol_ds = emdb_2660().downsample(settings.data_generation.downsample_size)
    L = vol_ds.resolution
    # Get VMF parameters from config
    vmf_cfg = settings.data_generation.von_mises_fisher

    # Generate S2 quadrature points
    quadrature_points = fibonacci_sphere_points(n=vmf_cfg.fibonacci_spiral_n)

    # Generate VMF mixture parameters
    mu, kappa, weights = generate_random_vmf_parameters(
        vmf_cfg.num_distributions, kappa_start=vmf_cfg.kappa_start, kappa_mean=vmf_cfg.kappa_mean
    )
    mixture_eval = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)

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
    ny, nx = image.shape[:2]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    max_r = (min(nx, ny) / 2.0) -1  # biggest polar radius (half image width / height)
    Y, X = np.ogrid[:ny, :nx]
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask = dist2 <= (max_r**2)
    diff_vals = image[mask] - reconstructed[mask]
    img_vals = image[mask]
    diff_l2_norm = np.linalg.norm(diff_vals)
    image_l2_norm = np.linalg.norm(img_vals)
    print(f"Relative L2 norm of difference: {diff_l2_norm/image_l2_norm:.4e}")

    # Plot and save results
    os.makedirs("outputs/tmp_figs", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(polar_image, aspect='auto', cmap='gray')
    axes[1].set_title('Polar Image')
    axes[1].axis('off')
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title('Reconstructed Image')
    axes[2].axis('off')
    diff_image = image - reconstructed
    diff_image[~mask] = 0
    axes[3].imshow(diff_image, cmap='RdBu_r')
    axes[3].set_title('Difference Image')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/tmp_figs/polar_transform_test.png", dpi=150)
    plt.close(fig)