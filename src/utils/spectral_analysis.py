import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, Union


def compute_second_moment_eigendecomposition(second_moment: Union[np.ndarray, torch.Tensor], 
                                           device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigendecomposition of the second moment tensor.
    
    The second moment tensor has shape (L, L, L, L) and can be viewed as a linear operator
    acting on the image space of size L×L. This function reshapes it to (L², L²) and 
    computes its eigendecomposition.
    
    Parameters:
    -----------
    second_moment : ndarray or torch.Tensor
        Second moment tensor of shape (L, L, L, L)
    device : str, optional
        Device to use for computation ('cuda' or 'cpu'). If None, uses GPU if available.
        
    Returns:
    --------
    eigenvalues : ndarray
        Eigenvalues in descending order, shape (L²,)
    eigenvectors : ndarray
        Eigenvectors as columns, shape (L², L²)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to torch tensor if needed
    if isinstance(second_moment, np.ndarray):
        second_moment_t = torch.from_numpy(second_moment).to(device=device, dtype=torch.float64)
    else:
        second_moment_t = second_moment.to(device=device, dtype=torch.float64)
    
    L = second_moment_t.shape[0]
    
    # Reshape from (L, L, L, L) to (L², L²) to view as a matrix operator
    M = second_moment_t.reshape(L*L, L*L)
    
    # Compute eigendecomposition, the matrix should be symmetric/Hermitian 
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    # Sort in descending order
    idx = torch.argsort(eigenvalues, descending=True)

    return eigenvalues[idx].cpu().numpy(), eigenvectors[:, idx].cpu().numpy()


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
    from src.data.vdm_generator import generate_vdm_from_volume
    from config.config import settings
    from src.networks.vnn.torch_utils import apply_all_C_powers
    
    # Configuration from settings
    downsample_size = settings.data_generation.downsample_size
    second_moment_batch_size = settings.data_generation.second_moment_batch_size
    device_str = f"cuda:{settings.device.cuda_device}" if settings.device.use_cuda else "cpu"
    
    print("=== Spectral Analysis Workflow - vMF Mixture ===")
    print(f"Downsample size: {downsample_size}")
    print(f"Device: {device_str}")
    save_dir = "outputs/spectral_analysis/vmf_mixture_emdb_2984/"
    
    # Generate VDM using the generator
    print("\n1. Loading volume and generating VDM...")
    from aspire.downloader import emdb_2984
    volume = emdb_2984()
    vdm = generate_vdm_from_volume(volume, 'vmf_mixture', downsample_size=downsample_size)
    
    # Compute moments
    print("\n2. Computing analytical moments...")
    first_moment = vdm.first_analytical_moment(device=device_str)
    print(f"   First moment computed: shape {first_moment.shape}")
    
    second_moment = vdm.second_analytical_moment(
        batch_size=second_moment_batch_size, 
        show_progress=True, 
        device=device_str,
        dtype=torch.float64
    )
    print(f"   Second moment computed: shape {second_moment.shape}")
    
    # Perform spectral analysis
    print("\n3. Performing spectral analysis...")
    eigenvalues, eigenvectors = compute_second_moment_eigendecomposition(
        second_moment, device=device_str
    )
    
    print(f"   Eigendecomposition complete:")
    print(f"   - {len(eigenvalues)} eigenvalues found")
    print(f"   - Largest eigenvalue: {eigenvalues[0]:.2e}")
    print(f"   - Smallest eigenvalue: {eigenvalues[-1]:.2e}")
    print(f"   - Condition number: {eigenvalues[0]/max(eigenvalues[-1], 1e-16):.1e}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot eigenvalues
    print("\n4. Creating visualizations...")
    plot_eigenvalues(
        eigenvalues, 
        save_path=os.path.join(save_dir, "eigenvalue_spectrum.png")
    )
    
    # Get distribution metadata for titles and logging
    metadata = vdm.distribution_metadata
    num_components = len(metadata['means']) if 'means' in metadata else 0
    
    # Visualize principal eigenvectors
    visualize_eigenvectors(
        eigenvectors, eigenvalues,
        save_dir,
        num_show=18,
        title=f"Principal Eigenvectors - {num_components} vMF Components"
    )
    
    # Save first moment as image
    plt.figure(figsize=(8, 6))
    plt.imshow(first_moment, cmap='RdBu_r', aspect='equal')
    plt.title('First Analytical Moment')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "first_moment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compare first moment with leading eigenvector
    print("\n5. Comparing first moment with leading eigenvector...")
    
    # Get leading eigenvector and reshape to image
    L = first_moment.shape[0]
    leading_eigenvector = eigenvectors[:, 0].reshape(L, L)
    
    l2_distance = compare_images_with_optimal_normalization(
        first_moment, leading_eigenvector,
        save_diff_image_path=os.path.join(save_dir, "first_moment_vs_leading_eigenvector_difference.png")
    )
    
    # Apply second moment operator iteratively to leading eigenvector using torch_utils
    k_iterations = 50  # Number of iterations to apply
    
    print(f"   Applying second moment operator iteratively to first moment ({k_iterations} times)...")
    
    # Prepare inputs for apply_all_C_powers
    # eigenvalues: (1, L²) - add batch dimension
    eigenvalues_batch = torch.from_numpy(eigenvalues).unsqueeze(0).to(device=device_str, dtype=torch.float32)
    
    # eigenvectors: (1, L², L²) - add batch dimension  
    eigenvectors_batch = torch.from_numpy(eigenvectors).unsqueeze(0).to(device=device_str, dtype=torch.float32)
    
    # first_moment as input: (1, L², 1) - flatten image, add batch and feature dimensions
    x_input = torch.from_numpy(first_moment.flatten()).unsqueeze(0).unsqueeze(-1).to(device=device_str, dtype=torch.float32)
    
    # Compute all powers at once: C^0 x, C^1 x, ..., C^k x
    all_powers = apply_all_C_powers(eigenvalues_batch, eigenvectors_batch, x_input, k_iterations)
    # Shape: (k_iterations+1, 1, L², 1)
    
    # Compare each iteration with both the original first moment and leading eigenvector
    leading_eigenvalue = eigenvalues[0]  # Get the leading eigenvalue for normalization
    iteration_distances_first_moment = []
    iteration_distances_leading_eigenvector = []
    
    for k in range(1, k_iterations + 1):
        # Get the k-th power result and reshape to image
        new_image = all_powers[k, 0, :, 0].cpu().numpy().reshape(L, L)
        
        # Divide by the leading eigenvalue raised to the k-th power
        normalized_image = new_image / (leading_eigenvalue ** k)
        
        # Compare with original first moment
        distance_first_moment = compare_images_with_optimal_normalization(
            first_moment, normalized_image,
            save_diff_image_path=None
        )
        iteration_distances_first_moment.append(distance_first_moment)
        
        # Compare with leading eigenvector
        distance_leading_eigenvector = compare_images_with_optimal_normalization(
            leading_eigenvector, normalized_image,
            save_diff_image_path=None
        )
        iteration_distances_leading_eigenvector.append(distance_leading_eigenvector)
    
    print(f"   L2 distance (first moment vs leading eigenvector): {l2_distance:.4e}")
    print(f"   First moment iteration distances: {iteration_distances_first_moment}")
    print(f"   Leading eigenvector iteration distances: {iteration_distances_leading_eigenvector}")
    print(f"   Distance growth ratio vs first moment (iter {k_iterations}/iter 1): {iteration_distances_first_moment[-1] / iteration_distances_first_moment[0]:.2f}")
    print(f"   Distance growth ratio vs leading eigenvector (iter {k_iterations}/iter 1): {iteration_distances_leading_eigenvector[-1] / iteration_distances_leading_eigenvector[0]:.2f}")
    
    print(f"\n6. Analysis complete!")
    print(f"   All outputs saved to: {save_dir}")
    print(f"   Files created:")
    print(f"   - eigenvalue_spectrum.png: Eigenvalue decay and cumulative energy")
    print(f"   - eigenvector_visualization/: Subfolder with eigenvector visualizations (multiple files)")
    print(f"   - first_moment.png: First analytical moment")
    print(f"   - first_moment_vs_leading_eigenvector_difference.png: Difference image after optimal alignment")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Volume resolution: {downsample_size}³")
    print(f"Total matrix size: {downsample_size**2}² = {(downsample_size**2)**2:,}")
    print(f"vMF mixture: {num_components} components")
    print(f"SO(3) quadrature: {len(vdm.rotations):,} rotations")
    print(f"Spectral properties:")
    print(f"  - Rank (eff. dim.): {np.sum(eigenvalues > eigenvalues[0] * 1e-12)}")
    print(f"  - 90% energy in top: {np.where(np.cumsum(eigenvalues)/np.sum(eigenvalues) >= 0.9)[0][0] + 1} components")
    print(f"  - 99% energy in top: {np.where(np.cumsum(eigenvalues)/np.sum(eigenvalues) >= 0.99)[0][0] + 1} components")
