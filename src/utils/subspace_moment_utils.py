from matplotlib.pylab import f
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Optional, Union

from zarr import save
from config.config import settings
from src.utils.polar_transform import cartesian_to_polar, extract_dominant_angular_fourier_mode

def compute_second_moment_eigendecomposition(second_moment: Union[np.ndarray, torch.Tensor]):
    """
    Compute the eigendecomposition of the second moment tensor.

    The second moment tensor has shape (L, L, L, L) and can be viewed as a linear operator
    acting on the image space of size L×L. This function reshapes it to (L², L²) and 
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
    # Convert to torch tensor if needed
    if isinstance(second_moment, np.ndarray):
        second_moment_t = torch.from_numpy(second_moment).to(device=device, dtype=torch.float64)
    else:
        second_moment_t = second_moment.to(device=device, dtype=torch.float64)
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
    For each eigenvector, extract the dominant angular Fourier mode from its polar image.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of eigenvalues (sorted in descending order).
    eigenvectors : ndarray
        Array of eigenvectors as columns (shape: (L², N)).

    Returns
    -------
    compressed_eigenspaces : list of dict
        Each dict contains:
            'm_detected': int,
            'radial_profile': ndarray,
            'energy_fraction': float
    """
    compressed_eigenspaces = []
    L = int(np.sqrt(eigenvectors.shape[0]))
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        polar_image, r_vals, _ = cartesian_to_polar(vec.reshape(L, L))
        m_detected, radial_profile, energy_fraction = extract_dominant_angular_fourier_mode(polar_image)
        compressed_eigenspaces.append({
            'm_detected': int(np.abs(m_detected)),
            'radial_profile': radial_profile,
            'energy_fraction': energy_fraction
        })
    return compressed_eigenspaces

def extract_dominant_eigenvector_modes_deprecated(eigenvalues: np.ndarray, eigenvectors: np.ndarray, rel_tol: float = 5e-2):
    """
    Groups eigenvectors by unique eigenvalues (up to relative tolerance and geometric multiplicity 2).
    For each eigen space, extracts angular Fourier mode information.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of eigenvalues (sorted in descending order).
    eigenvectors : ndarray
        Array of eigenvectors as columns (shape: (L², L²)).
    rel_tol : float
        Relative tolerance for considering eigenvalues equal.

    Returns
    -------
    compressed_eigenspaces : list of dict
        Each dict contains 'eigenvalue', 'm_detected', 'radial_profile', and 'energy_fraction'.
        - 'm_detected': the detected angular frequency (natural number),
        - 'radial_profile': the radial profile for the dominant mode,
        - 'energy_fraction': the fraction of energy in the detected mode.
    """
    eigenspaces = []
    used = np.zeros(len(eigenvalues), dtype=bool)
    for i, val in enumerate(eigenvalues):
            if used[i]:
                continue
            # Find all eigenvalues close to val
            idx = np.where(np.abs(eigenvalues - val) / (np.abs(val) + 1e-16) < rel_tol)[0]
            # Only mark as used the ones you actually select
            if len(idx) > 2:
                eigvecs = []
                for j in idx:
                    if not used[j]:
                        eigvecs.append(eigenvectors[:, j])
                        used[j] = True
                        if len(eigvecs) == 2:
                            break
            else:
                eigvecs = [eigenvectors[:, j] for j in idx]
                used[idx] = True
            eigenspaces.append((val, eigvecs))

    compressed_eigenspaces = []
    for eigenvalue, vectors in eigenspaces:
        multiplicity = len(vectors)

        if multiplicity == 2:
            polar_image_0, r_vals, _ = cartesian_to_polar(vectors[0].reshape(int(np.sqrt(eigenvectors.shape[0])), -1))
            polar_image_1 = cartesian_to_polar(vectors[1].reshape(int(np.sqrt(eigenvectors.shape[0])), -1))[0]
            m_detected, radial_profile, energy_fraction = extract_dominant_angular_fourier_mode(polar_image_0 + 1j*polar_image_1)
        else:
            polar_image = cartesian_to_polar(vectors[0].reshape(int(np.sqrt(eigenvectors.shape[0])), -1))[0]
            m_detected, radial_profile, energy_fraction = extract_dominant_angular_fourier_mode(polar_image)
        compressed_eigenspaces.append({
            'eigenvalue': eigenvalue,
            'm_detected': int(np.abs(m_detected)),
            'radial_profile': radial_profile,
            'energy_fraction': energy_fraction
        })
    return compressed_eigenspaces

def plot_angular_mode(eigenvector: np.ndarray, mode: int, radial_profile: np.ndarray, r_vals: np.ndarray, save_path: str = None):
    """
    Plot a 2D image from a radial profile and angular mode using polar_to_cartesian,
    and also plot the radial profile in the same figure.

    Parameters
    ----------
    mode : int
        Angular Fourier mode (m).
    radial_profile : ndarray
        1D array of radial values.
    r_vals : ndarray
        1D array of radial coordinates.
    save_path : str, optional
        If provided, saves the plot to this path.
    """
    from src.utils.polar_transform import polar_to_cartesian
    import matplotlib.pyplot as plt

    L = int(np.sqrt(eigenvector.shape[0]))
    eigen_img = eigenvector.reshape(L, L)
    theta_vals = np.linspace(0, 2 * np.pi, settings.data_generation.cartesian_to_polar_n_theta, endpoint=False)
    polar_recon = radial_profile[:, None] * np.exp(1j * mode * theta_vals)[None, :]
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
    plt.plot(r_vals, np.real(radial_profile), 'b-', linewidth=2)
    plt.title("Radial Profile")
    plt.xlabel("Radius")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Eigenvector comparison saved to {save_path}")
    plt.close()

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
    
    # Fixed 3x3 grid (9 vectors per plot)
    vectors_per_plot = 9
    
    # Create eigenvector_visualization subfolder
    eigenvec_dir = os.path.join(save_dir, "eigenvector_visualization")
    os.makedirs(eigenvec_dir, exist_ok=True)
    
    # Calculate number of plots needed
    num_plots = (num_show + vectors_per_plot - 1) // vectors_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * vectors_per_plot
        end_idx = min(start_idx + vectors_per_plot, num_show)
        current_num_show = end_idx - start_idx
        
        # Determine subplot arrangement (3x3 grid for 9 vectors)
        cols = 3
        rows = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(current_num_show):
            global_idx = start_idx + i
            # Reshape eigenvector to image
            eigenvec_img = eigenvectors[:, global_idx].reshape(L, L)
            
            ax = axes[i]
            im = ax.imshow(eigenvec_img, cmap='RdBu_r', aspect='equal')
            ax.set_title(f'λ_{global_idx+1} = {eigenvalues[global_idx]:.2e}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(current_num_show, len(axes)):
            axes[i].set_visible(False)
        
        # Create plot-specific title
        if num_plots > 1:
            plot_title = f"{title} - Part {plot_idx + 1}/{num_plots} (λ_{start_idx+1}-λ_{end_idx})"
        else:
            plot_title = title
        
        plt.suptitle(plot_title, fontsize=16)
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


if __name__ == "__main__":
    import os
    from config.config import settings
    from src.data.emdb_downloader import load_aspire_volume
    from src.utils.distribution_generation_functions import fibonacci_sphere_points, create_in_plane_invariant_distribution
    from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture
    
    # Configuration from settings
    downsample_size = settings.data_generation.downsample_size
    second_moment_batch_size = settings.data_generation.second_moment_batch_size
    device_str = f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda else "cpu"
    
    print("=== Spectral Analysis Workflow - vMF Mixture ===")
    print(f"Downsample size: {downsample_size}")
    print(f"Device: {device_str}")
    emdb_id = "emd_47365"
    emdb_path = f"/data/shachar/emdb_downloads/{emdb_id}.map.gz"
    save_dir = f"outputs/spectral_analysis/{emdb_id}"
    
    # Generate VDM using the generator
    print("\n1. Loading volume and generating VDM...")
    volume = load_aspire_volume(emdb_path, downsample_size=settings.data_generation.downsample_size)
    volume = volume.downsample(downsample_size)

    from src.utils.distribution_generation_functions import generate_weighted_random_s2_points
    from src.utils.distribution_generation_functions import create_in_plane_invariant_distribution
    from src.core.volume_distribution_model import VolumeDistributionModel
    # quadrature_points, s2_distribution = generate_weighted_random_s2_points(1)
    n_quadrature = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    quadrature_points = fibonacci_sphere_points(n_quadrature)
    mu, kappa, weights = generate_random_vmf_parameters(
                settings.data_generation.von_mises_fisher.num_distributions,
                settings.data_generation.von_mises_fisher.kappa_start,
                settings.data_generation.von_mises_fisher.kappa_mean)
    s2_distribution = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)
    so3_rotations, so3_weights = create_in_plane_invariant_distribution(quadrature_points, s2_distribution, 
                                                                            num_in_plane_rotations=32)
    vdm = VolumeDistributionModel(volume, rotations=so3_rotations, distribution=so3_weights)

    # vdm = generate_vdm_from_volume(volume, 'vmf_mixture', downsample_size=downsample_size)
    # Compute moments
    print("\n2. Computing analytical moments...")
    first_moment = vdm.first_analytical_moment()
    print(f"   First moment computed: shape {first_moment.shape}")
    
    second_moment = vdm.second_analytical_moment(
        batch_size=second_moment_batch_size, 
        show_progress=True, 
        dtype=torch.float64
    )
    print(f"   Second moment computed: shape {second_moment.shape}")
    
    # Perform spectral analysis
    print("\n3. Performing spectral analysis...")
    eigenvalues, eigenvectors = compute_second_moment_eigendecomposition(second_moment)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot eigenvalues
    print("\n4. Creating visualizations...")
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
    compressed_eigenspaces = extract_dominant_eigenvector_modes(eigenvectors[:, :n_keep])
    
    # Save first moment as image
    plt.figure(figsize=(8, 6))
    plt.imshow(first_moment, cmap='RdBu_r', aspect='equal')
    plt.title('First Analytical Moment')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "first_moment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    

    print(f"\n6. Analysis complete!")
    print(f"   All outputs saved to: {save_dir}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"SO(3) quadrature: {len(vdm.rotations):,} rotations")
    print(f"Spectral properties:")
    print(f"  1 - 1e-4 energy in top: {num_components_for_energy_threshold(eigenvalues, 1e-4)} components")
    print(f"  1 - 1e-8 of energy in top: {num_components_for_energy_threshold(eigenvalues, 1e-8)} components")
    idx = np.random.randint(len(compressed_eigenspaces))
    eig = compressed_eigenspaces[idx]
    eigenvector = eigenvectors[:, idx]
    L = int(np.sqrt(eigenvectors.shape[0]))
    _, r_vals, _ = cartesian_to_polar(eigenvector.reshape(L, L))
    save_path = os.path.join(save_dir, f"angular_mode_example_{idx}_m.png")
    plot_angular_mode(eigenvector, eig['m_detected'], eig['radial_profile'], r_vals, save_path=save_path)
    for i, eig in enumerate(compressed_eigenspaces[:9]):
        print(f"--- Compressed Eigenspace {i+1} ---")
        for k, v in eig.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"{k}: {v}")
