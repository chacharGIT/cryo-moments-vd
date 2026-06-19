from tqdm import tqdm
import numpy as np
from aspire.abinitio import CLSyncVoting
from aspire.source import OrientedSource, Simulation
from aspire.utils.rotation import Rotation
from aspire.utils import mean_aligned_angular_distance
from aspire.noise.noise import WhiteNoiseAdder

from config.config import settings
from src.data.emdb_downloader import load_aspire_volume
from src.utils.von_mises_fisher_distributions import generate_random_vmf_parameters, evaluate_vmf_mixture
from src.utils.distribution_generation_functions import s2_points_to_in_plane_euler_angles
from src.core.volume_distribution_model import VolumeDistributionModel

num_imgs = 128
sigma = 0.0
downsample_size = settings.data_generation.downsample_size
dtype = np.float32
emdb_id = "emd_19777"
emdb_path = f"/data/shachar/emdb_downloads/{emdb_id}.map.gz"
save_dir = f"outputs/spectral_analysis/{emdb_id}"

# Generate VDM using the generator
print("Loading volume and generating VDM...")
volume = load_aspire_volume(emdb_path, downsample_size=downsample_size)
vol_ds = volume.downsample(downsample_size)
L = vol_ds.resolution

# Get VMF parameters from config
vmf_cfg = settings.data_generation.von_mises_fisher

# Generate S2 quadrature points
from src.utils.distribution_generation_functions import fibonacci_sphere_points
quadrature_points = fibonacci_sphere_points(n=vmf_cfg.fibonacci_spiral_n)

# Generate VMF mixture parameters
mu, kappa, weights = generate_random_vmf_parameters(
    vmf_cfg.num_distributions, kappa_start=vmf_cfg.kappa_start, kappa_mean=vmf_cfg.kappa_mean
)
s2_distribution = evaluate_vmf_mixture(quadrature_points, mu, kappa, weights)

vdm = VolumeDistributionModel(vol_ds, rotations=quadrature_points, distribution=s2_distribution,
                                   distribution_metadata={
                                        'type': 'vmf_mixture',
                                        'means': mu,
                                        'kappas': kappa,
                                        'weights': weights
                                    }, in_plane_invariant_distribution=True)

sampled_rotations = vdm.sample_rotations(num_imgs)
rots_true = sampled_rotations.matrices
print("Use downsampled map to create simulation object.")

noise_adder = WhiteNoiseAdder(var=sigma**2)
sim = Simulation(
    L=L, n=num_imgs, vols=vol_ds, angles=sampled_rotations.angles, offsets=0, dtype=dtype, noise_adder=noise_adder
)

print(
    "Estimate rotation angles and offsets using synchronization matrix and voting method."
)
orient_est = CLSyncVoting(sim, n_theta=72, mask=False)
oriented_src = OrientedSource(sim, orient_est)
rots_est = oriented_src.rotations

mean_ang_dist = mean_aligned_angular_distance(rots_est, rots_true)
print(
    f"Mean angular distance between estimates and ground truth: {mean_ang_dist:.4f} (degrees)"
)

print("Determine best rotation by evaluating the loss for each candidate rotation.")
euler_angles = s2_points_to_in_plane_euler_angles(
        fibonacci_sphere_points(512),
        num_in_plane_rotations=64,
        random_start=True,
)
rotations = Rotation.from_euler(euler_angles, dtype=np.float32)

best_loss = None
best_idx = None
best_est_rot = None
J = np.diag([1, 1, -1]).astype(np.float32)
I = np.eye(3, dtype=np.float32)
for conj_mat in [J, I]:
    for i, R in enumerate(tqdm(rotations.matrices, desc="Determining best rotation")):
        aligned_rots_est = R @ conj_mat @ rots_est @ conj_mat
        loss = np.mean((aligned_rots_est - sampled_rotations.matrices) ** 2)

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_est_rot = R
            best_aligned_rots_est = aligned_rots_est


print(rots_est.shape, sampled_rotations.matrices.shape)
print(np.linalg.norm(best_aligned_rots_est - sampled_rotations.matrices),
       np.linalg.norm(rots_est - sampled_rotations.matrices))
