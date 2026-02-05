from matplotlib.pylab import f
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Optional, Union

from config.config import settings
from src.utils.polar_transform import cartesian_to_polar, extract_dominant_angular_fourier_mode
from src.utils.numpy_torch_conversion import dtype_to_torch

def compute_second_moment_eigendecomposition(second_moment: Union[np.ndarray, torch.Tensor]):
    """
    Compute the eigendecomposition of the second moment tensor.

    The second moment tensor has shape (L, L, L, L) and can be viewed as a linear operator
    acting on the image space of size (L, L). This function reshapes it to (L², L²) and 
    computes its eigendecomposition.

    Parameters
    ----------
    second_moment : ndarray or torch.Tensor
        Second moment tensor of shape (L, L, L, L)
    device : str, optional
        Device to use for computation ('cuda' or 'cpu'). If None, uses GPU if available.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues in descending order, shape (L²,)
    eigenvectors : ndarray
        Eigenvectors as columns, shape (L², L²)
    """
    if settings.device.use_cuda:
        device = f"cuda:{settings.device.cuda_device}"
    else:
        device = "cpu"
    dtype = second_moment.dtype
    dtype = dtype_to_torch(dtype) if isinstance(dtype, np.dtype) else dtype
    # Convert to torch tensor if needed
    if isinstance(second_moment, np.ndarray):
        second_moment_t = torch.from_numpy(second_moment).to(device=device, dtype=dtype)
    else:
        second_moment_t = second_moment.to(device=device, dtype=dtype)
    if second_moment_t.dim() != 4:
        raise ValueError("second_moment must be a 4D tensor (L, L, L, L)")
    L = second_moment_t.shape[0]
    # Reshape from (L, L, L, L) to (L², L²) to view as a matrix operator
    M = second_moment_t.reshape(L*L, L*L)
    # Compute eigendecomposition, the matrix should be symmetric/Hermitian 
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    # Sort in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[idx].cpu().numpy(), eigenvectors[:, idx].cpu().numpy()

def num_components_for_energy_threshold(eigenvalues: np.ndarray, truncation_threshold: float) -> int:
    """
    Given sorted singular/eigenvalues and a truncation threshold,
    returns the number of components needed to retain (1 - truncation_threshold) of the energy.

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted singular/eigenvalues (largest first).
    truncation_threshold : float
        Fraction of energy to discard (e.g., 1e-3 for 99.9% energy kept).

    Returns
    -------
    num_components : int
        Number of components needed to keep desired energy.
    """
    total_energy = np.sum(eigenvalues**2)
    cumulative_energy = np.cumsum(eigenvalues**2) / total_energy
    required_energy = 1 - truncation_threshold
    idx = np.where(cumulative_energy >= required_energy)[0]
    return int(idx[0] + 1) if len(idx) > 0 else len(eigenvalues)

def extract_dominant_eigenvector_modes(eigenvectors: np.ndarray):
    """
    For each eigenvector, extract the dominant angular Fourier modes from its polar image.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of eigenvalues (sorted in descending order).
    eigenvectors : ndarray
        Array of eigenvectors as columns (shape: (L², N)).

    Returns
    -------
    compressed_eigenspaces : list of dict
        For each eigenvector, a dictionary containing:
            - 'm_detected': int, the detected angular frequency (mode)
            - 'radial_profile': ndarray (for m=0), the real radial profile
            - 'cos_component': ndarray (for m>0), the cosine quadrature radial profile
            - 'sin_component': ndarray (for m>0), the sine quadrature radial profile
            - 'energy_fraction': float, fraction of total energy in the detected mode
    """
    compressed_eigenspaces = []
    L = int(np.sqrt(eigenvectors.shape[0]))
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        polar_image, r_vals, _ = cartesian_to_polar(vec.reshape(L, L))
        m_detected, radial_profile, energy_fraction = extract_dominant_angular_fourier_mode(polar_image, r_vals)
        if int(m_detected) == 0:
            compressed_eigenspaces.append({
                'm_detected': int(m_detected),
                'radial_profile': radial_profile,
                'energy_fraction': energy_fraction
            })
        else:
            compressed_eigenspaces.append({
                'm_detected': int(m_detected),
                'cos_component': radial_profile[0],
                'sin_component': radial_profile[1],
                'energy_fraction': energy_fraction
            })
    return compressed_eigenspaces

def plot_angular_mode(eigenvector: np.ndarray, mode: int, radial_profile: np.ndarray, r_vals: np.ndarray, save_path: str):
    """
    Plot a 2D image from a radial profile and angular mode using polar_to_cartesian.
    Plot the radial profile in the same figure.

    Parameters
    ----------
    eigenvector : ndarray
        The original eigenvector, shape (L²,) or (L, L).
    mode : int
        Angular Fourier mode (m).
    radial_profile : ndarray or list of ndarray
        For m=0: 1D array of radial values.
        For m>0: list of two 1D arrays (cosine and sine quadratures).
    r_vals : ndarray
        1D array of radial coordinates.
    save_path : str
        Saves the plot to this path.
    """
    from src.utils.polar_transform import polar_to_cartesian
    import matplotlib.pyplot as plt

    L = int(np.sqrt(eigenvector.shape[0]))
    eigen_img = eigenvector.reshape(L, L)
    theta_vals = np.linspace(0, 2 * np.pi, settings.data_generation.cartesian_to_polar_n_theta, endpoint=False)
    if mode == 0:
        polar_recon = radial_profile[:, None] * np.ones_like(theta_vals)[None, :]
    else:
        polar_recon = radial_profile[0][:, None] * np.cos(mode * theta_vals)[None, :] - \
                        radial_profile[1][:, None] * np.sin(mode * theta_vals)[None, :]
    # Convert to Cartesian
    recon_img = polar_to_cartesian(polar_recon, r_vals)
    plt.figure(figsize=(18, 6))
    # Original eigenvector
    plt.subplot(1, 3, 1)
    plt.imshow(np.real(eigen_img), cmap='RdBu_r', aspect='equal')
    plt.title("Original Eigenvector")
    plt.axis('off')
    plt.colorbar()
    # Reconstruction
    plt.subplot(1, 3, 2)
    plt.imshow(np.real(recon_img), cmap='RdBu_r', aspect='equal')
    plt.title(f"Reconstruction (mode={mode})")
    plt.axis('off')
    plt.colorbar()
    # Radial profile
    plt.subplot(1, 3, 3)
    if isinstance(radial_profile, (list, tuple)) or (isinstance(radial_profile, np.ndarray) and radial_profile.ndim == 2):
        plt.plot(r_vals, np.real(radial_profile[0]), 'b-', linewidth=2, label='cos')
        plt.plot(r_vals, np.real(radial_profile[1]), 'r-', linewidth=2, label='sin')
        plt.legend()
    else:
        plt.plot(r_vals, np.real(radial_profile), 'b-', linewidth=2, label='radial')
    plt.title("Radial Profile")
    plt.xlabel("Radius")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Eigenvector comparison saved to {save_path}")
    plt.close()

def plot_m_detected_average_histogram(zarr_path, save_path=None):
    import zarr
    import numpy as np
    import matplotlib.pyplot as plt

    root = zarr.open(zarr_path, mode='r')
    all_m_detected = []
    # Collect all m_detected arrays per experiment
    for volume_id in root.group_keys():
        group = root[volume_id]
        if "eigen_m_detected" in group:
            m_detected_arr = group["eigen_m_detected"][:]  # shape: [num_experiments_in_group, ...]
            # Loop over experiments in this group
            for m_detected in m_detected_arr:
                all_m_detected.append(m_detected.flatten())
    if not all_m_detected:
        print("No m_detected data found.")
        return

    # Find global min/max m
    m_min = min(m.min() for m in all_m_detected)
    m_max = max(m.max() for m in all_m_detected)
    bins = np.arange(m_min, m_max + 2) - 0.5  # one bin per integer

    # Compute histogram for each experiment
    histograms = []
    for m in all_m_detected:
        hist, _ = np.histogram(m, bins=bins)
        histograms.append(hist)
    histograms = np.stack(histograms, axis=0)  # shape: [num_experiments, num_bins]

    avg_counts = histograms.mean(axis=0)

    avg_total = avg_counts.sum()

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(m_min, m_max + 1), avg_counts, width=1, edgecolor='black')
    plt.xlabel('Detected Angular Mode (m)')
    plt.ylabel('Average Count per Experiment')
    plt.title('Average Histogram of Detected Angular Modes Across All Experiments')
    plt.legend([f"Avg. vectors/experiment: {avg_total:.1f}"])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved average histogram to {save_path}")
    plt.close()

    return avg_counts

def plot_eigenvalues(eigenvalues: np.ndarray, 
                    log_scale: bool = True,
                    save_path: Optional[str] = None,
                    show_cumulative: bool = True,
                    num_show: Optional[int] = None) -> None:
    """
    Plot the eigenvalues of the second moment operator.
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Eigenvalues in descending order
    title : str
        Title for the plot
    log_scale : bool
        Whether to use log scale for y-axis
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    show_cumulative : bool
        Whether to show cumulative energy plot as well
    num_show : int, optional
        Number of eigenvalues to show. If None, shows all.
    """
    if num_show is not None:
        eigenvalues = eigenvalues[:num_show]
    
    fig, axes = plt.subplots(1, 2 if show_cumulative else 1, figsize=(12 if show_cumulative else 6, 5))
    if not show_cumulative:
        axes = [axes]
    
    # Plot eigenvalues
    ax1 = axes[0]
    ax1.plot(eigenvalues, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Second Moment Eigenvalues')
    ax1.grid(True, alpha=0.3)
    
    if log_scale and np.any(eigenvalues > 0):
        ax1.set_yscale('log')
    
    # Add statistics text
    total_energy = np.sum(eigenvalues)
    max_eig = np.max(eigenvalues)
    min_eig = np.min(eigenvalues)
    
    stats_text = f'Total Energy: {total_energy:.2e}\n'
    stats_text += f'Max: {max_eig:.2e}\n'
    stats_text += f'Min: {min_eig:.2e}\n'
    stats_text += f'Ratio: {max_eig/max(min_eig, 1e-16):.1e}'
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot cumulative energy if requested
    if show_cumulative:
        ax2 = axes[1]
        cumulative_energy = np.cumsum(eigenvalues) / total_energy
        ax2.plot(cumulative_energy, 'r-', linewidth=2)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Energy Fraction')
        ax2.set_title('Cumulative Energy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Add lines for common thresholds
        for threshold in [0.9, 0.95, 0.99]:
            idx = np.where(cumulative_energy >= threshold)[0]
            if len(idx) > 0:
                ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
                ax2.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.7)
                ax2.text(idx[0], threshold, f' {threshold*100:.0f}% @ {idx[0]}', 
                        fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Eigenvalue plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_eigenvectors(eigenvectors: np.ndarray, 
                          eigenvalues: np.ndarray,
                          save_dir: str,
                          num_show: int,
                          title: str = "Principal Eigenvectors") -> None:
    """
    Visualize the principal eigenvectors as images, creating multiple plots if needed.
    
    Parameters:
    -----------
    eigenvectors : ndarray
        Eigenvectors as columns, shape (L², L²)
    eigenvalues : ndarray
        Corresponding eigenvalues
    num_show : int
        Number of eigenvectors to display
    title : str
        Base title for the plots
    save_dir : str
        Directory to save the plots. Creates an 'eigenvector_visualization' subfolder within this directory.
    """
    num_show = min(num_show, eigenvectors.shape[1])
    
    # Infer L from eigenvector shape: L² = eigenvectors.shape[0]
    L = int(np.sqrt(eigenvectors.shape[0]))

    is_complex = np.iscomplexobj(eigenvectors)
    # Determine subplot arrangement
    if is_complex:
        vectors_per_plot = 4
        rows, cols = 2, 4
    else:
        vectors_per_plot = 9
        rows, cols = 3, 3
    
    # Create eigenvector_visualization subfolder
    eigenvec_dir = os.path.join(save_dir, "eigenvector_visualization")
    os.makedirs(eigenvec_dir, exist_ok=True)
    
    # Calculate number of plots needed
    num_plots = (num_show + vectors_per_plot - 1) // vectors_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * vectors_per_plot
        end_idx = min(start_idx + vectors_per_plot, num_show)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()
        
        for i,idx in enumerate(range(start_idx, end_idx)):
            # Reshape eigenvector to image
            eigenvec_img = eigenvectors[:, idx].reshape(L, L)

            if is_complex:
                # Real part (top row)
                ax_real = axes[i]
                im_real = ax_real.imshow(np.real(eigenvec_img), cmap='RdBu_r', aspect='equal')
                ax_real.set_title(f'Real λ_{idx+1}={eigenvalues[idx]:.2e}', fontsize=10)
                plt.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)
                ax_real.axis('off')
                # Imaginary part (bottom row)
                ax_imag = axes[i + cols]
                im_imag = ax_imag.imshow(np.imag(eigenvec_img), cmap='RdBu_r', aspect='equal')
                ax_imag.set_title(f'Imag λ_{idx+1}={eigenvalues[idx]:.2e}', fontsize=10)
                plt.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04)
                ax_imag.axis('off')
            else:
                ax = axes[i]
                im = ax.imshow(eigenvec_img, cmap='RdBu_r', aspect='equal')
                ax.set_title(f'λ_{idx+1} = {eigenvalues[idx]:.2e}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')
        
        # Hide unused subplots
        for j in range((end_idx-start_idx)*2 if is_complex else end_idx-start_idx, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        
        if num_plots > 1:
            filename = f"eigenvectors_part_{plot_idx + 1:02d}.png"
        else:
            filename = "eigenvectors.png"
        save_path = os.path.join(eigenvec_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Eigenvector visualization saved to {save_path}")
        
        plt.close()
    
    if num_plots > 1:
        print(f"Created {num_plots} eigenvector plots in {eigenvec_dir}")

def compare_images_with_optimal_normalization(image1: np.ndarray, image2: np.ndarray, save_diff_image_path: Optional[str] = None):
    """
    Compare two L×L images using L2 normalization and measuring L2 difference.
    Handles potential sign differences by testing both orientations.
    
    Parameters:
    -----------
    image1 : ndarray
        First image of shape (L, L)
    image2 : ndarray  
        Second image of shape (L, L)
    save_diff_image_path : str, optional
        If provided, saves a plot of the difference image to this path
        
    Returns:
    --------
    l2_distance : float
        Minimum L2 distance after L2 normalization (considering sign flip)
    """
    # L2 normalize both images
    img1_normalized = image1 / (np.linalg.norm(image1) + 1e-16)
    img2_normalized = image2 / (np.linalg.norm(image2) + 1e-16)
    
    # Compute L2 distance for both orientations
    distance_normal = np.linalg.norm(img1_normalized - img2_normalized)
    distance_flipped = np.linalg.norm(img1_normalized - (-img2_normalized))
    
    # Determine optimal orientation and return accordingly
    if distance_normal <= distance_flipped:
        min_distance = distance_normal
        img2_normalized_optimal = img2_normalized
    else:
        min_distance = distance_flipped
        img2_normalized_optimal = -img2_normalized
    
    # Save difference image if path is provided
    if save_diff_image_path is not None:
        difference_image = img1_normalized - img2_normalized_optimal
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(difference_image, cmap='RdBu_r', aspect='equal')
        plt.title(f'Difference Image (L2 distance: {min_distance:.4f})')
        plt.colorbar(im)
        plt.axis('off')
        plt.savefig(save_diff_image_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Difference image saved to {save_diff_image_path}")
    
    return min_distance

def main():
    import os
    from config.config import settings
    from src.data.emdb_downloader import load_aspire_volume
    from src.utils.distribution_generation_functions import fibonacci_sphere_points, create_in_plane_invariant_distribution
    from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture
    
    # Configuration from settings
    downsample_size = settings.data_generation.downsample_size
    second_moment_batch_size = settings.data_generation.second_moment_batch_size
    device_str = f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda else "cpu"
    
    print("=== Spectral Analysis Workflow ===")
    emdb_id = "emd_52988"
    emdb_path = f"/data/shachar/emdb_downloads/{emdb_id}.map.gz"
    save_dir = f"outputs/spectral_analysis/{emdb_id}"

    volume = load_aspire_volume(emdb_path, downsample_size=settings.data_generation.downsample_size)
    volume = volume.downsample(downsample_size)

    from src.utils.distribution_generation_functions import generate_weighted_random_s2_points
    from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
    from src.core.volume_distribution_model import VolumeDistributionModel
    n_quadrature = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    quadrature_points = fibonacci_sphere_points(n_quadrature)
    mu, kappa, weights = generate_random_vmf_parameters(
                settings.data_generation.von_mises_fisher.num_distributions,
                settings.data_generation.von_mises_fisher.kappa_start,
                settings.data_generation.von_mises_fisher.kappa_mean)
    s2_distribution = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)
    # quadrature_points, s2_distribution = generate_weighted_random_s2_points(1)
    so3_rotations, so3_weights = create_in_plane_invariant_distribution(quadrature_points, s2_distribution, 
                                                                            num_in_plane_rotations=320)
    vdm = VolumeDistributionModel(volume, rotations=so3_rotations, distribution=so3_weights, fourier_domain=False)

    # vdm = generate_vdm_from_volume(volume, 'vmf_mixture', downsample_size=downsample_size)
    # Compute moments
    print("\n Computing analytical moments...")
    first_moment = vdm.first_analytical_moment()
    second_moment = vdm.second_analytical_moment(
        batch_size=second_moment_batch_size, 
        show_progress=True
    )
    
    # Perform spectral analysis
    print("\n Performing decomposition...")
    eigenvalues, eigenvectors = compute_second_moment_eigendecomposition(second_moment)
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot eigenvalues
    print("\n Plotting eigen images...")
    plot_eigenvalues(
        eigenvalues, 
        save_path=os.path.join(save_dir, "eigenvalue_spectrum.png")
    )
    
    # Visualize principal eigenvectors
    visualize_eigenvectors(
        eigenvectors, eigenvalues,
        save_dir,
        num_show=9,
        title=f"Principal Eigenvectors"
    )
    n_keep = num_components_for_energy_threshold(eigenvalues, 1e-8)
    print(f"Number of components to keep 1-1e-8 energy: {n_keep}")
    compressed_eigenspaces = extract_dominant_eigenvector_modes(eigenvectors[:, :n_keep])
    
    # Save first moment as image
    plt.figure(figsize=(8, 6))
    plt.imshow(first_moment, cmap='RdBu_r', aspect='equal')
    plt.title('First Analytical Moment')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "first_moment.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot histogram of m_detected
    m_vals = [eig['m_detected'] for eig in compressed_eigenspaces]
    bins = np.arange(min(m_vals), max(m_vals)+2) - 0.5
    plt.figure(figsize=(8, 5))
    plt.hist(m_vals, bins=bins, edgecolor='black')
    plt.xlabel('Detected Angular Mode (m)')
    plt.ylabel('Count')
    plt.title('Histogram of Detected Angular Modes')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "m_detected_histogram.png"), dpi=150)
    plt.close()
    print("Saved histogram of m_detected to", os.path.join(save_dir, "m_detected_histogram.png"))

    print(f"SO(3) quadrature: {len(vdm.rotations):,} rotations")
    print(f"Spectral properties:")
    print(f"  1 - 1e-4 energy in top: {num_components_for_energy_threshold(eigenvalues, 1e-4)} components")
    print(f"  1 - 1e-8 of energy in top: {num_components_for_energy_threshold(eigenvalues, 1e-8)} components")
    idx = np.random.randint(len(compressed_eigenspaces))
    idx = 0
    eig = compressed_eigenspaces[idx]
    eigenvector = eigenvectors[:, idx]
    L = int(np.sqrt(eigenvectors.shape[0]))
    _, r_vals, _ = cartesian_to_polar(eigenvector.reshape(L, L))
    save_path = os.path.join(save_dir, f"angular_mode_example_{idx+1}_m.png")
    if eig['m_detected'] == 0:
        plot_angular_mode(eigenvector, eig['m_detected'], eig['radial_profile'], r_vals, save_path=save_path)
    else:
        plot_angular_mode(eigenvector, eig['m_detected'], [eig['cos_component'], eig['sin_component']], r_vals, save_path=save_path)
    print(f"Saved angular mode plot for eigenspace {idx} to {save_path}")

    for i, eig in enumerate(compressed_eigenspaces[:180]):
        print(f"--- Compressed Eigenspace {i+1} ---")
        for k, v in eig.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"{k}: {v}")

if __name__ == "__main__":
    avg_counts = plot_m_detected_average_histogram(zarr_path="/data/shachar/zarr_files/emdb_vmf_subspace_moments_separated.zarr",
                                       save_path="outputs/spectral_analysis/m_detected_average_histogram.png")
    rounded_counts = np.round(avg_counts * 1.1).astype(int)
    # Remove trailing zeros
    last_nonzero = np.max(np.nonzero(rounded_counts)) + 1 if np.any(rounded_counts) else 0
    rounded_counts_trimmed = rounded_counts[:last_nonzero]
    print(f"Average m_detected counts (length {len(rounded_counts_trimmed)}):", rounded_counts_trimmed)
    # Save trimmed array
    print(f"Sum of rounded counts: {rounded_counts_trimmed.sum()}")