import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from config.config import settings

from src.networks.dpf.sample_generation import generate_vmf_mixture_on_s2
from src.networks.dpf.forward_diffusion import q_sample

if __name__ == "__main__":
    # Generate mixture and quadrature points
    print("Generating vMF mixture on S²...")
    quadrature_points, mixture_pdf, mixture_params = generate_vmf_mixture_on_s2()

    # Convert mixture_pdf to torch tensor for noise application
    mixture_pdf_tensor = torch.from_numpy(mixture_pdf).float()
    n_points = mixture_pdf_tensor.shape[0]
    timesteps = settings.dpf.timesteps

    # Split points into front (z > 0) and back (z < 0) hemispheres
    front_mask = quadrature_points[:, 2] > 0
    back_mask = quadrature_points[:, 2] <= 0

    num_subfigs = 14  # Number of diffusion steps (columns)
    fig, axes = plt.subplots(2, num_subfigs, figsize=(6 * num_subfigs, 10))
    for row, (mask, hemi_name) in enumerate([(front_mask, "Upper Hemisphere (z > 0)"), (back_mask, "Lower Hemisphere (z < 0)")]):
        for col in range(num_subfigs):
            if col == 0:
                vals = mixture_pdf[mask]
                title = f'Clean ({hemi_name})'
            else:
                # Use continuous t in [0, 1]
                t_val = col / (num_subfigs - 1)
                t_tensor = torch.full((n_points,), t_val, dtype=torch.float32)
                vals = q_sample(mixture_pdf_tensor, t_tensor).numpy()[mask]
                title = f'Diffused t={t_val:.2f} ({hemi_name})'
            # Project to 2D for scatter plot (x, y)
            sc = axes[row, col].scatter(quadrature_points[mask, 0], quadrature_points[mask, 1], c=vals, cmap='viridis', alpha=0.8)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            axes[row, col].set_aspect('equal')
            plt.colorbar(sc, ax=axes[row, col])
    print("Plotting complete.")
    plt.tight_layout()
    plt.savefig('outputs/tmp_figs/mixture_vmf_noised_hemispheres.png')
    plt.show()
