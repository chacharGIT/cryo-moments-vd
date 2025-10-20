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
from torch.utils.data import Dataset, DataLoader
from config.config import settings
from torch.utils.data import IterableDataset

class EMDBvMFSubspaceMomentsDataset(Dataset):
    """
    PyTorch Dataset for loading subspace moments data and metadata from EMDB volumes 
    with von Mises-Fisher mixture distributions.
    
    Each sample contains:
    - s2_distribution_means: vMF means on S²
    - s2_distribution_kappas: vMF concentration parameters
    - s2_distribution_weights: vMF mixture weights
    - eigen_images: Top eigenvectors from subspace moment decomposition
    - eigen_values: Corresponding eigenvalues from second moment decomposition
    - first_moments: First analytical moments
    - volume_ids: EMDB volume identifiers
    """

    def __init__(self, zarr_path: str, transform=None, debug=False):
        """
        Initialize the dataset.
        
        Args:
            zarr_path: Path to the Zarr file containing compressed moments data
            transform: Optional transform to apply to samples
        """
        self.zarr_path = zarr_path
        self.transform = transform
        self.root = zarr.open(zarr_path, mode='r')
        # Get dataset length from any of the arrays
        self.length = self.root['s2_distribution_means'].shape[0]
        self.shapes = {
            's2_distribution_means': self.root['s2_distribution_means'].shape[1:],
            's2_distribution_kappas': self.root['s2_distribution_kappas'].shape[1:],
            's2_distribution_weights': self.root['s2_distribution_weights'].shape[1:],
            'eigen_images': self.root['eigen_images'].shape[1:],
            'eigen_values': self.root['eigen_values'].shape[1:],
            'first_moments': self.root['first_moments'].shape[1:],
            'distribution_evaluations': self.root['distribution_evaluations'].shape[1:]
        }
        if debug:
            print(f"Loaded EMDB vMF subspace moments dataset with {self.length} samples")
            print(f"Sample shapes: {self.shapes}")
            print(f"Chunk size: {self.root['s2_distribution_means'].chunks[0]}")
            print("Zarr array shapes for debug:")
            for key in self.root.array_keys():
                print(f"  {key}: {self.root[key].shape}")
    
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
            'eigen_images', 'eigen_values', 'first_moments', 'distribution_evaluations'
        ]
        # Ensure indices are a plain list of ints
        indices = [int(i) for i in indices]
        batch = {}
        for key in tensor_fields:
            batch_data = self.root[key][indices]
            batch[key] = torch.from_numpy(np.ascontiguousarray(batch_data)).float()
        batch['volume_id'] = [self.root['volume_ids'][i] for i in indices]

        # --- Normalize first moments by L1 norm per sample ---
        epsilon = 1e-8
        first_moments = batch['first_moments']  # shape: [B, ...]
        # Compute L1 norm for each sample in batch
        norms = first_moments.view(first_moments.shape[0], -1).abs().sum(dim=1, keepdim=True)  # [B, 1]
        # Avoid division by zero
        norms = norms + epsilon
        # Reshape for broadcasting
        norm_shape = [first_moments.shape[0]] + [1] * (first_moments.dim() - 1)
        batch['first_moments'] = first_moments / norms.view(*norm_shape)
        
        # Normalize eigen_values (second moment) by norm squared
        batch['eigen_values'] = batch['eigen_values'] / (norms ** 2)

        if self.transform:
            batch = self.transform(batch)
        return batch
    
class ChunkShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, batch_size=None, shuffle=True, device=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.device = device
        zarr_root = dataset.root
        first_key = next(iter(zarr_root.array_keys()))
        self.chunk_size = zarr_root[first_key].chunks[0]
        self.batch_size = batch_size if batch_size is not None else self.chunk_size

    def __iter__(self):
        n_samples = len(self.dataset)
        n_chunks = math.ceil(n_samples / self.chunk_size)
        chunk_indices = list(range(n_chunks))
        if self.shuffle:
            np.random.shuffle(chunk_indices)
        current_chunk_cursor = 0
        chunk_ptr = 0  # pointer within current chunk
        batch_indices = []
        while current_chunk_cursor < len(chunk_indices):
            current_chunk = chunk_indices[current_chunk_cursor]
            start = current_chunk * self.chunk_size
            end = min(start + self.chunk_size, n_samples)
            remaining_in_chunk = end - (start + chunk_ptr)
            need = self.batch_size - len(batch_indices)
            take = min(need, remaining_in_chunk)
            batch_indices.extend(list(range(start + chunk_ptr, start + chunk_ptr + take)))
            chunk_ptr += take
            if start + chunk_ptr == end:
                current_chunk_cursor += 1
                chunk_ptr = 0
            if len(batch_indices) == self.batch_size:
                if self.device is not None:
                    batch = to_device(self.dataset[batch_indices], self.device)
                else:
                    batch = self.dataset[batch_indices]
                yield batch
                batch_indices = []
        # Yield any remaining samples as a final (smaller) batch
        if batch_indices:
            if self.device is not None:
                batch = to_device(self.dataset[batch_indices], self.device)
            else:
                batch = self.dataset[batch_indices]
            yield batch

def create_subspace_moments_dataloader(zarr_path: str, batch_size: int = None, 
                                      shuffle: bool = True, transform=None, num_workers: int = 0,
                                      debug: bool = False) -> DataLoader:
    """
    Create an optimized DataLoader for subspace moments EMDB data with vMF mixtures.
    Uses batch-level Zarr reading for maximum efficiency with large samples.
    
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
    dataset = EMDBvMFSubspaceMomentsDataset(zarr_path, transform=transform, debug=debug)
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
    zarr_path = "/data/shachar/zarr_files/emdb_vmf_top_eigen.zarr"

    try:
        dataloader = create_subspace_moments_dataloader(zarr_path, batch_size=200, shuffle=True, num_workers=0)
        print(f"DataLoader created.")
        n_batches = 25
        start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            first_moments = batch['first_moments']  # [B, ...]
            eigen_values = batch['eigen_values']    # [B, N]
            # Compute L1 norm of first moments per sample
            norms = first_moments.view(first_moments.shape[0], -1).abs().sum(dim=1)
            # Compute sum of eigenvalues per sample
            eigen_sums = eigen_values.sum(dim=1)
            n_kept = (batch['eigen_values'][0] != 0).sum().item()
            # Print only the first item in the batch
            print(f"  Item 0: L1 norm(first_moment) = {norms[0].item():.6f}, sum(eigen_values) = {eigen_sums[0].item():.3e}")
            # Print sum of distribution_evaluations for first item if present
            dist_eval_sum = batch['distribution_evaluations'][0].sum().item()
            print(f"  Item 0: sum(distribution_evaluations) = {dist_eval_sum:.6f}")
            print(f"  Item 0: number of kept eigenvalues = {n_kept}")
            if batch_idx + 1 >= n_batches:
                break
        elapsed = time.time() - start_time
        print(f"Loaded {n_batches} batches in {elapsed:.2f} seconds (batch_size=100)")
    except FileNotFoundError:
        print(f"Zarr file not found at {zarr_path}")
        print("Please run the data generation script first.")
    except Exception as e:
        print(f"Error: {e}")
