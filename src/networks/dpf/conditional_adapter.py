import torch
import torch.nn as nn

class ConditionalEncoder(nn.Module):
    """
    Encodes conditional information (e.g., class, auxiliary data) into a fixed-size embedding.
    Args:
        cond_dim (int): Dimension of the conditional input.
        emb_dim (int): Dimension of the output embedding.
    """
    def __init__(self, cond_dim, emb_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, cond_info):
        return self.encoder(cond_info)

class ConditionalAdapter(nn.Module):
    """
    Injects the conditional embedding into the Perceiver latent space at each block.
    Args:
        cond_emb_dim (int): Dimension of the conditional embedding.
        latent_dim (int): Dimension of the Perceiver latent vectors.
    """
    def __init__(self, cond_emb_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_emb_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, cond_emb, batch_size, seq_len):
        # cond_emb: [batch, cond_emb_dim]
        # Output: [batch, seq_len, latent_dim]
        out = self.mlp(cond_emb)  # [batch, latent_dim]
        return out.unsqueeze(1).expand(-1, seq_len, -1)

# Usage example (in your Perceiver):
# self.cond_adapters = nn.ModuleList([
#     ConditionalAdapter(cond_emb_dim, latent_dim) for _ in range(depth)
# ])
# ...
# for i, (self_attn, self_ff) in enumerate(self.layers):
#     x = self_attn(x) + x
#     if cond_emb is not None:
#         x = x + self.cond_adapters[i](cond_emb, x.shape[0], x.shape[1])
#     x = self_ff(x) + x
#     if cond_emb is not None:
#         x = x + self.cond_adapters[i](cond_emb, x.shape[0], x.shape[1])
