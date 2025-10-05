import torch
from torch.utils.data import Dataset
import zarr

class ZarrVMFMixtureDataset(Dataset):
    """
    PyTorch Dataset for vMF mixture data stored in a Zarr file.
    Loads the entire dataset; splitting should be handled externally.
    """
    def __init__(self, zarr_path):
        self.z = zarr.open(zarr_path, mode='r')
        self.func_data = self.z['func_data']
        self.kappa = self.z['kappa']
        self.mu_directions = self.z['mu_directions']
        self.mixture_weights = self.z['mixture_weights']
        self.num_examples = self.func_data.shape[0]
        self.quadrature_n = self.func_data.shape[1]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        func = torch.from_numpy(self.func_data[idx]).float()
        kappa = torch.from_numpy(self.kappa[idx]).float()
        mu = torch.from_numpy(self.mu_directions[idx]).float()
        weights = torch.from_numpy(self.mixture_weights[idx]).float()
        return func, kappa, mu, weights
