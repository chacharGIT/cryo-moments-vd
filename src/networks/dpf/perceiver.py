import torch
import torch.nn as nn
from einops import rearrange, repeat
from functools import wraps

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def xavier_init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)
    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    keep_prob = 1. - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices
    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    seq = seq[batch_indices, keep_indices]
    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, 'b -> b 1')
        mask = mask[batch_indices, keep_indices] & keep_mask
    return seq, mask


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Multi-head attention module for Perceiver-style architectures.
    Args:
        query_dim (int): Dimension of query input.
        context_dim (int, optional): Dimension of context input. Defaults to query_dim.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per head.
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Rearrange for multi-head attention
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        # Scaled dot-product attention
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and 'context' in kwargs and kwargs['context'] is not None:
            normed_context = self.norm_context(kwargs['context'])
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)

class PerceiverIO(nn.Module):
    """
    PerceiverIO core model.
    Args:
        depth (int): Number of Perceiver blocks.
        dim (int): Input embedding dimension.
        queries_dim (int): Query embedding dimension (output tokens).
        num_latents (int): Number of latent vectors.
        latent_dim (int): Latent vector dimension.
        cross_heads (int): Number of cross-attention heads.
        latent_heads (int): Number of latent self-attention heads.
        cross_dim_head (int): Dim per cross-attention head.
        latent_dim_head (int): Dim per latent-attn head.
        seq_dropout_prob (float): Dropout on input sequence.
        logits_dim (int, optional): Output logits dimension.
        weight_tie_layers (bool, optional): Weight tying for layers.
        decoder_ff (bool, optional): Use decoder feedforward.
    """
    def __init__(self,
                 depth,
                 dim,
                 queries_dim,
                 num_latents,
                 latent_dim,
                 cross_heads,
                 latent_heads,
                 cross_dim_head,
                 latent_dim_head,
                 seq_dropout_prob,
                 logits_dim=None,
                 ff_mult=4,
                 weight_tie_layers=False,
                 ):
        super().__init__()
        # Xavier initialization for all Linear layers
        self.apply(xavier_init_linear)
        self.seq_dropout_prob = seq_dropout_prob
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim, mult=ff_mult))
        ])
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, mult=ff_mult))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim, mult=ff_mult))
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        # Learnable scaling for function value channels (assumed last d channels)
        self.func_scale = nn.Parameter(torch.ones(1) * 1e3)
    def forward(self, data, mask=None, queries=None, cond_feat=None):
        b = data.shape[0]
        # Apply learnable scaling to function value channels (last d channels)
        # Get d from input shape
        d = 1  # Default to 1, override if known
        if hasattr(self, 'logits_dim'):
            d = self.logits_dim
        else:
            pass
        if d > 0:
            func_data = data[..., -d:] * self.func_scale
            data = torch.cat([data[..., :-d], func_data], dim=-1)
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # Additive conditioning: add cond_feat to latents if provided
        if cond_feat is not None:
            cond_feat_exp = cond_feat.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + cond_feat_exp
        cross_attn, cross_ff = self.cross_attend_blocks
        if self.training and self.seq_dropout_prob > 0.:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)
        x = cross_attn(x, context=data, mask=mask) + x
        if cond_feat is not None:
            x = x + cond_feat_exp
        x = cross_ff(x) + x
        if cond_feat is not None:
            x = x + cond_feat_exp
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            if cond_feat is not None:
                x = x + cond_feat_exp
            x = self_ff(x) + x
            if cond_feat is not None:
                x = x + cond_feat_exp
        if not exists(queries):
            return x
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)
        # Optionally inject cond_feat before decoder cross-attn
        if cond_feat is not None:
            x = x + cond_feat_exp
        latents = self.decoder_cross_attn(queries, context=x)
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)
