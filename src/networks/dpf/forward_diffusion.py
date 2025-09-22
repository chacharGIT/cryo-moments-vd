import torch
import numpy as np


def cosine_signal_scaling_schedule(t, s=0.008):
    """
    Continuous cosine signal scaling schedule.
    t: tensor of shape [...] with values in [0, 1]
    Returns: tensor of shape [...], signal scaling at each t
    """
    scaling = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
    return scaling

def beta_schedule(t, s=0.008):
    """
    Returns beta(t) for the continuous cosine noise schedule (Song et al., 2021).
    Uses the closed-form expression for the derivative of the signal scaling schedule.
    Supports batch input for t.
    Args:
        t: tensor of shape [batch] or [...], values in [0, 1]
        s: float, schedule offset (default 0.008)
    Returns:
        beta_t: tensor of same shape as t
    """
    angle = (t + s) / (1 + s) * torch.pi * 0.5
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    beta_t = (torch.pi * 0.5) * sin_angle / (cos_angle ** 2)
    return beta_t

def q_sample(x_0, t):
    """
    Domain-agnostic forward diffusion (DDPM-style) for any signal x_0 at continuous timestep t in [0, 1].
    Args:
        x_0: [batch, ...] original signal
        t: [batch] timestep (float tensor in [0, 1])
    Returns:
        x_t: [batch, ...] noised signal at timestep t
    """
    noise = torch.randn_like(x_0)
    scaling_t = cosine_signal_scaling_schedule(t).reshape(-1, *([1] * (x_0.dim() - 1)))
    x_t = torch.sqrt(scaling_t) * x_0 + torch.sqrt(1 - scaling_t) * noise
    return x_t

if __name__ == "__main__":
    # Debug: Analyze signal/noise stats for different timesteps
    import numpy as np
    from src.networks.dpf.sample_generation import generate_vmf_mixture_on_s2
    batch_size = 1
    points_np, func_data_np, _ = generate_vmf_mixture_on_s2(batch_size=batch_size)
    func_data = torch.from_numpy(func_data_np).float()
    # Normalize func_data to unit variance per function
    func_data = func_data / (func_data.std(dim=1, keepdim=True))
    t_values = torch.linspace(0, 1, 10)
    print(t_values)
    print("t\tmin\tmax\tmean\tstd")
    for t in t_values:
        t_batch = torch.full((batch_size,), t)
        x_t = q_sample(func_data, t_batch)
        print(f"{t.item():.2f}\t{x_t.min().item():.4g}\t{x_t.max().item():.4g}\t{x_t.mean().item():.4g}\t{x_t.std().item():.4g}")
