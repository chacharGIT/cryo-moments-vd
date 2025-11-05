import os
import zarr
import numpy as np
from config.config import settings
from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture
from src.utils.distribution_generation_functions import fibonacci_sphere_points

def main():
    num_examples = settings.data_generation.von_mises_fisher.num_generated_examples
    n_quadrature = settings.data_generation.von_mises_fisher.fibonacci_spiral_n
    num_components = settings.data_generation.von_mises_fisher.num_distributions
    kappa_start = settings.data_generation.von_mises_fisher.kappa_start
    kappa_mean = settings.data_generation.von_mises_fisher.kappa_mean

    # Output Zarr path
    zarr_path = os.path.join(settings.data_generation.zarr.save_dir, "vmf_mixtures_evaluations.zarr")
    os.makedirs(settings.data_generation.zarr.save_dir, exist_ok=True)

    # Create Zarr group and datasets
    z = zarr.open(zarr_path, mode='w')
    z.create_array('func_data', shape=(num_examples, n_quadrature), chunks=(1000, n_quadrature), dtype='f8')
    z.create_array('kappa', shape=(num_examples, num_components), chunks=(1000, num_components), dtype='f8')
    z.create_array('mu_directions', shape=(num_examples, num_components, 3), chunks=(1000, num_components, 3), dtype='f8')
    z.create_array('mixture_weights', shape=(num_examples, num_components), chunks=(1000, num_components), dtype='f8')

    # Precompute quadrature points (same for all mixtures)
    quadrature_points = fibonacci_sphere_points(n_quadrature)

    batch_size = 1000
    for i in range(0, num_examples, batch_size):
        batch_end = min(i + batch_size, num_examples)
        bsize = batch_end - i
        batch_func_data = np.zeros((bsize, n_quadrature), dtype=np.float64)
        batch_kappa = np.zeros((bsize, num_components), dtype=np.float64)
        batch_mu = np.zeros((bsize, num_components, 3), dtype=np.float64)
        batch_weights = np.zeros((bsize, num_components), dtype=np.float64)
        for j in range(bsize):
            mu, kappa, weights = generate_random_vmf_parameters(num_components, kappa_start, kappa_mean)
            pdf = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)
            batch_func_data[j] = pdf
            batch_kappa[j] = kappa
            batch_mu[j] = mu
            batch_weights[j] = weights
        z['func_data'][i:batch_end] = batch_func_data
        z['kappa'][i:batch_end] = batch_kappa
        z['mu_directions'][i:batch_end] = batch_mu
        z['mixture_weights'][i:batch_end] = batch_weights
        print(f"Stored batch {i//batch_size+1} / {num_examples//batch_size}")
    print(f"All vMF mixtures stored in {zarr_path}")

if __name__ == "__main__":
    main()
