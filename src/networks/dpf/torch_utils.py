import torch

def linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    Linear beta schedule as in Nochil & Dhariwal 2021 (DDPM).
    Returns a tensor of betas for each timestep.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s):
    """
    Cosine beta schedule as in Improved DDPM (Nochil & Dhariwal 2021).
    Returns a tensor of betas for each timestep.
    """
    import numpy as np
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(np.clip(betas, 0, 0.999)).float()
