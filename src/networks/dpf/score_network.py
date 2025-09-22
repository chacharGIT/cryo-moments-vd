import torch
import torch.nn as nn
from src.networks.dpf.perceiver import PerceiverIO
from config.config import settings

class S2ScoreNetwork(nn.Module):
    """
    Score network for functions on S², using PerceiverIO as the backbone.
    Input: points on S² and (optionally) noisy function values, timestep embedding, etc.
    Output: estimated score (gradient of log-probability) at each point.
    """
    def __init__(self, perceiver_config=None):
        super().__init__()
        # Calculate PerceiverIO input and output dimensions based on encoding settings
        time_enc_len = settings.dpf.time_encoding_len  # Fourier encoding length
        sph_enc_len = settings.dpf.pos_encoding_max_harmonic_degree  # Spherical harmonics max degree
        d = 1  # Number of function data channels (change if needed)

        # Fourier encoding: 2 values (sin, cos) per frequency
        fourier_dim = time_enc_len * 2
        # Spherical harmonics: (L+1)^2-1 real harmonics - Exclude l=0 (constant) term
        sph_dim = (sph_enc_len + 1) ** 2 - 1

        context_dim = fourier_dim + sph_dim + d  # Full context encoding
        query_dim = context_dim
        output_dim = d  # Output dimension (score per point)

        # Prepare PerceiverIO config, override dims to match input construction
        if perceiver_config is None:
            perceiver_config = dict(settings.dpf.perceiver)
        perceiver_config['dim'] = context_dim
        perceiver_config['queries_dim'] = query_dim
        perceiver_config['logits_dim'] = output_dim  # output a scalar per point
        self.perceiver = PerceiverIO(**perceiver_config)

    def forward(self, context, queries, mask=None):
        # context: [batch, n_points, context_dim] (full encoding)
        # queries: [batch, n_points, query_dim] (positional encoding only)
        # mask: optional
        return self.perceiver(context, queries=queries, mask=mask)

