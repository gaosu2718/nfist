"""NFIST: Normalizing Flow Invertible Solution Transformer.

A task-conditioned, time-dependent affine coupling flow F(x, t | prompts) with
F(., 0) = id, an exact inverse, and an analytic log-determinant. Particles move as
x_t = F(x_0, t), and the score of p_t follows in closed form from the inverse.
Shapes: particles (bsz, n_pt, dim), times (bsz,).
"""
import torch
import torch.nn as nn
from torch import Tensor


def mlp(in_dim, out_dim, h_dim, n_hidden=1):
    """Token-wise MLP acting on the last axis."""
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h_dim), nn.GELU()]
        d = h_dim
    return nn.Sequential(*layers, nn.Linear(d, out_dim))


class Attention(nn.Module):
    """Multi-head attention with residual connection and LayerNorm."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        return self.norm(q + self.attn(q, kv, kv)[0])


class SetTransformer(nn.Module):
    """Self-attention block over the samples of a particle-cloud prompt."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.mlp = mlp(dim, dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.attn(x, x)
        return self.norm(h + self.mlp(h))


class CouplingLayer(nn.Module):
    """y = x_A + x_B * exp(alpha) + beta, with (alpha, beta) computed from (x_A, t, context).

    mask = 1 on the conditioning coordinates A. Both alpha and beta are scaled by t,
    so the layer is the identity at t = 0. log|det dy/dx| = sum(alpha).
    """
    def __init__(self, dim, mask, fea_dim, h_dim, n_heads):
        super().__init__()
        self.register_buffer('mask', mask)
        self.embed = mlp(dim + 1, fea_dim, h_dim, n_hidden=2)
        self.self_attn = Attention(fea_dim, n_heads)
        self.cross_attn = Attention(fea_dim, n_heads)
        self.head = mlp(fea_dim, 2 * dim, h_dim)

    def coeffs(self, x, t, ctx):
        t = t[:, None, None]
        h = self.embed(torch.cat([x * self.mask, t.expand(-1, x.shape[1], 1)], -1))
        h = self.self_attn(h, h)        # interaction among query particles
        h = self.cross_attn(h, ctx)     # conditioning on the task prompt
        alpha, beta = self.head(h).chunk(2, -1)
        scale = (1 - self.mask) * t
        return torch.tanh(alpha) * scale, beta * scale

    def forward(self, x, t, ctx):
        alpha, beta = self.coeffs(x, t, ctx)
        return x * alpha.exp() + beta, alpha.sum(-1)

    def inverse(self, y, t, ctx):
        alpha, beta = self.coeffs(y, t, ctx)   # y_A = x_A, so the same coefficients
        return (y - beta) * (-alpha).exp(), -alpha.sum(-1)


class NFIST(nn.Module):
    """Maps task prompts to a SolutionOperator.

    prompt_dims:   {name: token dim}, e.g. {'p0': 2, 'p1': 2}.
    cloud_prompts: names passed as particle clouds (bsz, n_samples, d) instead of
                   parametric tokens (bsz, n_tokens, d); encoded with set-transformer blocks.
    """
    def __init__(self, dim, prompt_dims, cloud_prompts=(), n_coupling=8,
                 fea_dim=256, h_dim=64, n_heads=4, n_set_tf=2):
        super().__init__()
        self.encoders = nn.ModuleDict({
            k: nn.Sequential(mlp(d, fea_dim, h_dim),
                             *[SetTransformer(fea_dim, n_heads) for _ in range(n_set_tf if k in cloud_prompts else 0)])
            for k, d in prompt_dims.items()})
        half, idx = max(1, dim // 2), torch.arange(dim)
        masks = [(idx < half if i % 2 == 0 else idx >= half).float() for i in range(n_coupling)]
        self.layers = nn.ModuleList(CouplingLayer(dim, m, fea_dim, h_dim, n_heads) for m in masks)

    def forward(self, **prompts):
        ctx = torch.cat([enc(prompts[k]) for k, enc in self.encoders.items()], dim=1)
        return SolutionOperator(self.layers, ctx)


class SolutionOperator:
    """Flow for one batch of encoded tasks: G(x, t0, t1) = F(F^{-1}(x, t0), t1)."""
    def __init__(self, layers, ctx):
        self.layers, self.ctx = layers, ctx

    def _ctx(self, n):
        # time steps may be stacked (time-major) along the batch axis
        return self.ctx.repeat(n // self.ctx.shape[0], 1, 1)

    def forward(self, x, t):
        """F(x, t) and log|det dF/dx|."""
        ctx, logdet = self._ctx(x.shape[0]), 0
        for layer in self.layers:
            x, ld = layer(x, t, ctx)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, y, t):
        """F^{-1}(y, t) and log|det dF^{-1}/dy|."""
        ctx, logdet = self._ctx(y.shape[0]), 0
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y, t, ctx)
            logdet = logdet + ld
        return y, logdet

    def flowmap(self, x, t0, t1):
        return self.forward(self.inverse(x, t0)[0], t1)[0]

    def traj(self, x0, t):
        """Particle paths (bsz, n_time, n_pt, dim) on the time grid t: (n_time,)."""
        bsz, n = x0.shape[0], len(t)
        xt, _ = self.forward(x0.repeat(n, 1, 1), t.repeat_interleave(bsz))
        return xt.view(n, *x0.shape).transpose(0, 1)

    def score(self, y, t, log_p0):
        """grad_y log p_t(y) = grad_y [log p0(F^{-1}(y, t)) + log|det dF^{-1}/dy|]."""
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if not y.requires_grad:
                y = y.detach().requires_grad_()
            x, logdet = self.inverse(y, t)
            return torch.autograd.grad((log_p0(x) + logdet).sum(), y, create_graph=create_graph)[0]


def fp_energy(op, x0, t, gamma, log_p0):
    """Fokker-Planck energy int_0^T E|dx/dt + gamma * score|^2 dt (midpoint rule).

    The control of the probability-flow ODE is u = dx/dt + gamma * grad log p_t, with
    gamma = sigma^2 / 2. log_p0 maps x (n_time, bsz, n_pt, dim) to log p0(x) up to a constant.
    Returns paths (bsz, n_time, n_pt, dim) and energy (bsz,).
    """
    bsz, n = x0.shape[0], len(t)
    tt = t.repeat_interleave(bsz)
    xt, _ = op.forward(x0.repeat(n, 1, 1), tt)
    st = op.score(xt, tt, lambda x: log_p0(x.view(n, *x0.shape)).flatten(0, 1))
    xt, st = (z.view(n, *x0.shape).transpose(0, 1) for z in (xt, st))
    dt = t.diff()
    u = xt.diff(dim=1) / dt[:, None, None] + gamma * (st[:, 1:] + st[:, :-1]) / 2
    return xt, (u.pow(2).sum(-1).mean(-1) * dt).sum(-1)
