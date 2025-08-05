import torch
from torch.utils.data import Dataset
import zarr

def load_data_from_zarr(zarr_path):
    """
    Load all VDM data from a single Zarr file.

    Args:
        zarr_path (str): Path to the Zarr file containing VDM data.

    Returns:
        M1_list (list[Tensor]): List of first moment tensors.
        M2_list (list[Tensor]): List of second moment tensors.
        distribution_meta_list (list[dict]): List of distribution metadata dicts (without type).
        distribution_type (str): Distribution type string.
    """
    z = zarr.open(zarr_path, mode='r')
    M1_list, M2_list, distribution_meta_list = [], [], []
    distribution_type = 'vmf_mixture'  # Assuming all data is VMF mixture
    
    for emdb_id in z.group_keys():
        g = z[emdb_id]
        # First and second moments
        M1 = torch.tensor(g['first_moment'][...], dtype=torch.float32)
        M2 = torch.tensor(g['second_moment'][...], dtype=torch.float32)
        # VMF mixture parameters
        means = torch.tensor(g['vmf_means'][...], dtype=torch.float32)
        kappas = torch.tensor(g['vmf_kappas'][...], dtype=torch.float32)
        weights = torch.tensor(g['vmf_weights'][...], dtype=torch.float32)
        distribution_meta = {
            'means': means,
            'kappas': kappas,
            'weights': weights
        }
        M1_list.append(M1)
        M2_list.append(M2)
        distribution_meta_list.append(distribution_meta)
    return M1_list, M2_list, distribution_meta_list, distribution_type

class MomentsDataset(Dataset):
    def __init__(self, M1_list, M2_list, target_list):
        self.M1_list = M1_list
        self.M2_list = M2_list
        self.target_list = target_list

    def __len__(self):
        return len(self.M1_list)

    def __getitem__(self, idx):
        return self.M1_list[idx], self.M2_list[idx], self.target_list[idx]
