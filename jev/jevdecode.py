"""Incremental decoding with a KV cache, so the autoregressive baseline is timed honestly.

Without a cache every step re-reads the whole prefix and AR looks artificially bad.
The pad mask has to be carried through the cache too, or the decode path silently
differs from the training path.
"""
import torch, torch.nn as nn
from .jevmodel import attend


class CachedAttn(nn.Module):
    def __init__(s, src, h):
        super().__init__(); s.src, s.h = src, h; s.k = s.v = s.pad = None

    def reset(s):
        s.k = s.v = s.pad = None

    def forward(s, x, kpad):
        a = s.src; B, T, D = x.shape
        sh = lambda t: t.view(B, T, s.h, D // s.h).transpose(1, 2)
        k, v = sh(a.k(x)), sh(a.v(x))
        s.k   = k if s.k is None else torch.cat([s.k, k], 2)
        s.v   = v if s.v is None else torch.cat([s.v, v], 2)
        s.pad = kpad if s.pad is None else torch.cat([s.pad, kpad], 1)
        Tk = s.k.shape[2]
        causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
        m = causal[None, None] & s.pad[:, None, None, :]      # same mask as training
        return a.o(attend(sh(a.q(x)), s.k, s.v, m).transpose(1, 2).reshape(B, T, D))


class ARDecoder:
    """prefill(ids) -> logits for the next token; then step(id) repeatedly."""
    def __init__(s, model, h):
        s.m = model; s.h = h
        s.caches = [CachedAttn(b.a1, h) for b in model.enc.blocks]; s.t = 0

    def reset(s):
        for c in s.caches:
            c.reset()
        s.t = 0

    def _run(s, ids):
        e = s.m.enc; B, T = ids.shape
        kpad = ids != 0
        x = e.tok(ids) + e.pos(torch.arange(s.t, s.t + T, device=ids.device))[None]
        for b, c in zip(e.blocks, s.caches):
            x = x + c(b.n1(x), kpad); x = x + b.m(b.n3(x))
        s.t += T
        return s.m.head(e.ln(x))[:, -1]

    prefill = _run
    step    = _run
