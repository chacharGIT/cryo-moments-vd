import warnings
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future.",
    category=UserWarning,
    module=r"zarr.codecs.vlen_utf8"
)

import math
import torch
import zarr
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader, IterableDataset

from config.config import settings
from src.utils.polar_transform import cartesian_to_polar

class EMDBvMFSubspaceMomentsDataset(Dataset):
    """
    PyTorch Dataset for loading subspace moments data and metadata from EMDB volumes 
    with von Mises-Fisher mixture distributions.
    
    Each sample contains:
    - s2_distribution_means: vMF means on S²
    - s2_distribution_kappas: vMF concentration parameters
    - s2_distribution_weights: vMF mixture weights
    - eigen_radial_profiles: Radial profile of dominant angular Fourier components of top eigenvectors
    - eigen_values: Corresponding eigenvalues of top eigenvectors
    - eigen_m_detected: Dominant angular modes of detected eigenvectors
    - first_moments: First analytical moments
    - volume_ids: EMDB volume identifiers
    """

    def __init__(self, zarr_path, transform=None, debug=False, mode="train"):
        """
        Initialize the dataset.
        
        Args:
            zarr_path: Path to the Zarr file containing compressed moments data
            transform: Optional transform to apply to samples
            debug: If True, print debug information
        """
        self.root = zarr.open(zarr_path, mode='r')
        self.volume_ids = list(self.root.group_keys())
        self.transform = transform
        self.mode = mode
        split_path = settings.data_generation.emdb.volume_split_path

        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split = np.load(split_path, allow_pickle=True).item()
        else:
            all_volume_ids = list(self.root.group_keys())
            rng = np.random.RandomState()
            perm = rng.permutation(len(all_volume_ids))
            n_train = int(len(all_volume_ids) * settings.training.train_val_split)
            train_ids = [all_volume_ids[i] for i in perm[:n_train]]
            test_ids  = [all_volume_ids[i] for i in perm[n_train:]]
            split = {"train": train_ids, "test": test_ids}
            np.save(split_path, split)
        if mode == "train":
            self.volume_ids = split["train"]
        elif mode == "test":
            self.volume_ids = split["test"]

        # Get sample counts for each volume
        self.volume_sample_counts = {}
        for vid in self.volume_ids:
            try:
                n = self.root[vid]['eigen_m_detected'].shape[0]
            except Exception as e:
                print(f"Error accessing volume {vid}: {type(e).__name__}: {e}")
            self.volume_sample_counts[vid] = n
        # Build flat index mapping: (volume_id, sample_idx)
        self.flat_indices = []
        for vid in self.volume_ids:
            for i in range(self.volume_sample_counts[vid]):
                self.flat_indices.append((vid, i))
        self.length = len(self.flat_indices)
        self.volume_to_indices = {}
        for vid in self.volume_ids:
            self.volume_to_indices[vid] = [i for i, (v, _) in enumerate(self.flat_indices) if v == vid]
        # Load allowed m values
        n_vectors_per_m_path = settings.dpf.conditional_separated_moment_encoder.cyclic_equivariant_attention.Nms_path
        n_vectors_per_m = np.load(n_vectors_per_m_path, allow_pickle=True)
        self.allowed_m = set(int(m) for m in np.arange(len(n_vectors_per_m)))

        if debug:
            print(f"Mode '{mode}': {len(self.volume_ids)} volumes loaded (split file: {split_path})")
            print(f"Angular frequency band limit: m={n_vectors_per_m.shape[0]}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, indices):
        """
        Load a batch of samples.
        Args:
            indices: List or array of sample indices
        Returns:
            Dictionary with batched tensors, ready for neural network input
        """
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        elif isinstance(indices, (list, np.ndarray)) and all(isinstance(i, (int, np.integer)) for i in indices):
            indices = [int(i) for i in indices]
        else:
            raise TypeError(f"__getitem__ expects a single int or a list/array of integers, got: {indices}")
    
        tensor_fields = [
            's2_distribution_means', 's2_distribution_kappas', 's2_distribution_weights',
            'eigen_radial_profiles', 'eigen_values', 'eigen_m_detected',
            'first_moments_radial', 'distribution_evaluations'
        ]
        resolved = [self.flat_indices[i] for i in indices]
        local_indices = np.asarray([sidx for _, sidx in resolved], dtype=int)
    
        # Assume all samples come from the same volume
        vid = resolved[0][0]
        batch = {}
        grp = self.root[vid]
        for key in tensor_fields:
            arr = grp[key][local_indices]
            batch[key] = torch.from_numpy(np.ascontiguousarray(arr)).float()

        batch['volume_id'] = [vid] * len(indices)

        # Normalize first and second moments
        first_moments_radial = batch['first_moments_radial']
        r_vals = np.load(os.path.join("outputs", "model_static_parameters", "r_vals.npy"))
        norms = []
        for i in range(first_moments_radial.shape[0]):
            radial_i = first_moments_radial[i].cpu().numpy()
            norms.append(np.sqrt(np.sum((radial_i ** 2) * r_vals)))
        norms = torch.tensor(norms, dtype=batch['eigen_values'].dtype).view(-1, 1)

        # Divide first_moments by norm and eigen_values (second moment) by norm squared
        batch['first_moments_radial'] = first_moments_radial / norms  # [B, 2*R]
        batch['eigen_values'] = batch['eigen_values'] / (norms ** 2)

        eigen_vals = batch['eigen_values']      
        eigen_rads = batch['eigen_radial_profiles']
        eigen_rads = eigen_rads.permute(0, 2, 1).contiguous()
        eigen_m = batch['eigen_m_detected'].long()

        per_sample_vals = []
        per_sample_rads = []
        batch_all_ms = set()
        B, _, R = eigen_rads.shape
        assert R % 2 == 0, "Radial profile length R must be even to split into real/imag."
        R = R // 2
        real_part = eigen_rads[:, :, :R]          # [B, Neig, R]
        imag_part = eigen_rads[:, :, R:]          # [B, Neig, R]
        eigen_rads_complex = real_part + 1j * imag_part

        for b in range(B):
            vals_b = eigen_vals[b]          # [Neig]
            Neig = vals_b.shape[0]
            rads_b = eigen_rads_complex[b]  # [Neig, R]
            m_b    = eigen_m[b]             # [Neig]

            # Dicts for this sample: m -> list of indices
            m_to_idx = {}
            for j in range(Neig):
                m_val = int(m_b[j].item())
                if m_val not in self.allowed_m:
                    continue
                if m_val not in m_to_idx:
                    m_to_idx[m_val] = []
                m_to_idx[m_val].append(j)

            vals_dict_b = {}
            rads_dict_b = {}
            for m_val, idx_list in m_to_idx.items():
                idx_tensor = torch.tensor(idx_list, dtype=torch.long)
                vals_dict_b[m_val] = vals_b[idx_tensor]     # [N_{b,m}]
                rads_dict_b[m_val] = rads_b[idx_tensor]     # [N_{b,m}, R] (complex)
                batch_all_ms.add(m_val)

            per_sample_vals.append(vals_dict_b)
            per_sample_rads.append(rads_dict_b)

        # Build padded tensors per m
        eigen_values_by_m = {}
        eigen_radial_by_m = {}
        mask_by_m = {}

        for m_val in batch_all_ms:
            max_len = 0
            for b in range(B):
                if m_val in per_sample_vals[b]:
                    n = per_sample_vals[b][m_val].shape[0]
                    if n > max_len:
                        max_len = n
            if max_len == 0:
                continue

            vals_pad = eigen_vals.new_zeros((B, max_len))  # [B, max_len]
            rads_pad = torch.zeros(
                (B, max_len, R),
                dtype=eigen_rads_complex.dtype,
                device=eigen_rads_complex.device
            )                                              # [B, max_len, R] complex
            mask = torch.zeros(
                (B, max_len),
                dtype=torch.bool,
                device=eigen_rads_complex.device
            )                                              # [B, max_len]

            for b in range(B):
                if m_val not in per_sample_vals[b]:
                    continue
                v = per_sample_vals[b][m_val]              # [N_{b,m}]
                r = per_sample_rads[b][m_val]              # [N_{b,m}, R]
                n = v.shape[0]
                vals_pad[b, :n] = v
                rads_pad[b, :n, :] = r
                mask[b, :n] = True

            eigen_values_by_m[m_val] = vals_pad
            eigen_radial_by_m[m_val] = rads_pad
            mask_by_m[m_val] = mask

        # Attach dictionaries to batch
        batch['eigen_values_by_m'] = eigen_values_by_m      # dict[m] -> [B, N_max(m)]
        batch['eigen_radial_by_m'] = eigen_radial_by_m      # dict[m] -> [B, N_max(m), R] complex
        batch['mask_by_m'] = mask_by_m                      # dict[m] -> [B, N_max(m)]
        if self.transform:
            batch = self.transform(batch)
        return batch
    
class ChunkShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, batch_size=None, shuffle=True, device=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.device = device

        # Infer chunk size from first array of first volume
        zarr_root = dataset.root
        first_volume_id = dataset.volume_ids[0]
        first_group = zarr_root[first_volume_id]
        first_key = next(iter(first_group.array_keys()))
        self.chunk_size = first_group[first_key].chunks[0]
        
        # Enforce batch_size as multiple of chunk_size
        if batch_size is None:
            batch_size = self.chunk_size
        self.batch_size = batch_size

    def __iter__(self):
        volume_ids = list(self.dataset.volume_to_indices.keys())
        n_samples_per_vol = {
            vid: len(self.dataset.volume_to_indices[vid]) for vid in volume_ids
        }
        n_chunks_per_vol = {
            vid: math.ceil(n_samples_per_vol[vid] / self.chunk_size) for vid in volume_ids
        }
        chunk_ids_per_vol = {
            vid: list(range(n_chunks_per_vol[vid])) for vid in volume_ids
        }

        if self.shuffle:
            for vid in volume_ids:
                np.random.shuffle(chunk_ids_per_vol[vid])
        # Per-volume state: current chunk cursor and pointer inside current chunk
        chunk_ptr_per_vol = {vid: 0 for vid in volume_ids}
        offset_in_chunk_per_vol = {vid: 0 for vid in volume_ids}

        while True:
            # Pick a random volume for this batch
            vid = random.choice(volume_ids)
            vol_global_indices = self.dataset.volume_to_indices[vid]
            chunk_ids = chunk_ids_per_vol[vid]
            ptr = chunk_ptr_per_vol[vid]
            offset = offset_in_chunk_per_vol[vid]
            n_chunks = len(chunk_ids)
            batch_size = self.batch_size

            batch_indices = []
            while len(batch_indices) < batch_size:
                chunk_id = chunk_ids[ptr]
                start_local = chunk_id * self.chunk_size
                end_local = start_local + self.chunk_size

                # Start inside this chunk from current offset
                cur_start = start_local + offset
                remaining = batch_size - len(batch_indices)
                cur_end = min(end_local, cur_start + remaining)

                for local_idx in range(cur_start, cur_end):
                    batch_indices.append(vol_global_indices[local_idx])

                consumed = cur_end - cur_start
                offset += consumed

                # If we finished this chunk, move to next
                if offset >= (end_local - start_local):
                    ptr += 1
                    offset = 0

                # If we exhausted all chunks for this volume, restart and reshuffle
                if ptr >= n_chunks:
                    ptr = 0
                    offset = 0
                    if self.shuffle:
                        np.random.shuffle(chunk_ids)

            # Save updated pointers for this volume
            chunk_ptr_per_vol[vid] = ptr
            offset_in_chunk_per_vol[vid] = offset

            try:
                # Fetch and yield batch
                batch = self.dataset[batch_indices]
                if self.device is not None:
                    batch = to_device(batch, self.device)
            except Exception as e:
                print(f"Error fetching batch for volume {vid}: {type(e).__name__}: {e}")
                continue
            yield batch

def create_subspace_moments_dataloader(zarr_path: str, batch_size: int = None, 
                                      shuffle: bool = True, transform=None, num_workers: int = 0,
                                      debug: bool = False, mode="train") -> DataLoader:
    """
    Create a DataLoader for subspace moments EMDB data with vMF mixtures.
    Uses batch-level Zarr reading for efficiency with large samples.
    
    Args:
        zarr_path: Path to the Zarr file
        batch_size: Batch size (defaults to config value)
        shuffle: Whether to shuffle data
        transform: Optional transform function (applied at batch level)
        num_workers: Number of worker processes for data loading
        debug: Print debug info
    Returns:
        PyTorch DataLoader instance optimized for batch loading
    """
    if batch_size is None:
        batch_size = settings.training.batch_size
    dataset = EMDBvMFSubspaceMomentsDataset(zarr_path, transform=transform, debug=debug, mode=mode)
    if settings.device.use_cuda:
        device = f"cuda:{settings.device.cuda_device}"
    else:
        device = "cpu"
    iterable_dataset = ChunkShuffleIterableDataset(dataset, batch_size=batch_size, shuffle=shuffle, device=device)
    dataloader = DataLoader(
        iterable_dataset,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x[0]
    )
    return dataloader

def to_device(batch, device):
    """Recursively move all tensors in a batch dict to the specified device."""
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    else:
        return batch

# Example usage and testing
if __name__ == "__main__":
    import time
    zarr_path = "/data/shachar/zarr_files/emdb_vmf_subspace_moments_separated.zarr"
    
    dataloader = create_subspace_moments_dataloader(
        zarr_path, batch_size=300,
        shuffle=True,
        num_workers=0,
        debug=False
    )
    print(f"DataLoader created.")
    n_batches = 10

    start_time = time.time()
    r_vals = np.load(os.path.join("outputs", "model_static_parameters", "r_vals.npy"))
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        first_moments_radial = batch['first_moments_radial']  # [B, R]
        eigen_values = batch['eigen_values']    # [B, N]
        # Compute L1 norm of first moments per sample
        radial_0 = first_moments_radial[0].cpu().numpy()
        print(np.sum((radial_0 ** 2) * r_vals))
        # Compute sum of eigenvalues per sample
        eigen_sums = eigen_values.sum(dim=1)
        n_kept = (batch['eigen_values'][0] != 0).sum().item()
        # Print only the first item in the batch
        print(f"sum(eigen_values) = {eigen_sums[0].item():.3e}, number of kept eigenvalues = {n_kept}")
        i = np.RankWarning
        try:
            print(batch['eigen_values_by_m'][43].shape, batch['eigen_radial_by_m'][43].shape)
            print(sum(v.shape[1] for v in batch['eigen_values_by_m'].values()))
        except Exception as e:
            continue
        # Print sum of distribution_evaluations for first item if present
        dist_eval_sum = batch['distribution_evaluations'][0].sum().item()
        if batch_idx + 1 >= n_batches:
            break
    elapsed = time.time() - start_time
    print(f"Loaded {n_batches} batches in {elapsed:.2f} seconds (batch_size=300)")
