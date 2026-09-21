"""A System One model, written out.

No nn.MultiheadAttention, no nn.TransformerEncoder, no AutoModelForAnything.
Attention is softmax(QK'/sqrt(d))V and the output head is a dot product against
option text.  The three public primitives -- Choice, Score, Noul -- are the same
operation at m<=255, ordered m in 2..10, and m=2.
"""
import math, torch, torch.nn as nn, torch.nn.functional as Fn


def attend(q, k, v, mask=None):
    """q:(B,H,Tq,dh) k,v:(B,H,Tk,dh)  mask:(B,1,Tq,Tk) bool, True = keep."""
    a = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        a = a.masked_fill(~mask, torch.finfo(a.dtype).min)
    return torch.softmax(a.float(), -1).to(v.dtype) @ v


class Attn(nn.Module):
    """One module, two jobs: self-attention when kv is None, cross-attention otherwise."""
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.q = nn.Linear(d, d, bias=False); s.k = nn.Linear(d, d, bias=False)
        s.v = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False)

    def forward(s, x, kv=None, mask=None):
        kv = x if kv is None else kv
        B, T, D = x.shape; S = kv.shape[1]
        sh = lambda t, n: t.view(B, n, s.h, D // s.h).transpose(1, 2)
        y = attend(sh(s.q(x), T), sh(s.k(kv), S), sh(s.v(kv), S), mask)
        return s.o(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(s, d, m=4):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, m * d), nn.GELU(), nn.Linear(m * d, d))

    def forward(s, x):
        return s.f(x)


class Block(nn.Module):
    """Pre-LN. `cross=True` adds a second attention that reads an external memory."""
    def __init__(s, d, h, cross=False, self_attn=True):
        super().__init__()
        s.self_attn = self_attn
        if self_attn:
            s.n1, s.a1 = nn.LayerNorm(d), Attn(d, h)
        s.cross = cross
        if cross:
            s.n2, s.a2 = nn.LayerNorm(d), Attn(d, h)
        s.n3, s.m = nn.LayerNorm(d), MLP(d)

    def forward(s, x, mem=None, smask=None, xmask=None):
        if s.self_attn:
            x = x + s.a1(s.n1(x), mask=smask)
        if s.cross:
            x = x + s.a2(s.n2(x), kv=mem, mask=xmask)
        return x + s.m(s.n3(x))


class TextEncoder(nn.Module):
    """Shared trunk. causal=False -> every token sees every token."""
    def __init__(s, V, d, h, L, maxlen, causal=False):
        super().__init__()
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(maxlen, d); s.causal = causal
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); s.ln = nn.LayerNorm(d)

    def forward(s, ids):
        B, T = ids.shape
        pad = ids != 0
        x = s.tok(ids) + s.pos(torch.arange(T, device=ids.device))[None]
        m = pad[:, None, None, :].expand(B, 1, T, T)
        if s.causal:
            m = m & torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))[None, None]
        for b in s.blocks:
            x = b(x, smask=m)
        return s.ln(x), pad


def masked_mean(h, pad):
    w = pad.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1)


class JEV(nn.Module):
    """state + typed questions -> one forward pass -> a distribution per question.

    query_self_attn=False is the non-autoregressive factorisation: questions never see
    each other.  Turning it on does NOT buy back the joint -- the queries remain a
    deterministic function of the state, so the output is still a product of marginals.
    See R7.

    Every question's options are encoded in ONE batched pass and scored with ONE einsum,
    so wall-clock stays flat in K.  Doing it in a python loop is O(K) kernel launches,
    which at batch 1 on a T4 is the entire cost -- see R3.
    """
    def __init__(s, V, d=256, h=4, L_state=4, L_opt=2, L_dec=2, maxlen=128,
                 causal=False, query_self_attn=False):
        super().__init__()
        s.enc = TextEncoder(V, d, h, L_state, maxlen, causal=causal)
        s.opt = TextEncoder(V, d, h, L_opt, 32)          # encodes option / instruction TEXT
        s.dec = nn.ModuleList([Block(d, h, cross=True, self_attn=query_self_attn)
                               for _ in range(L_dec)])
        s.q_proj = nn.Linear(d, d); s.o_proj = nn.Linear(d, d)
        s.ln_q = nn.LayerNorm(d); s.d = d

    def embed_text(s, ids):
        h, pad = s.opt(ids)
        return masked_mean(h, pad)

    def forward(s, state_ids, instr_ids, option_ids):
        """state_ids (B,T) | instr_ids (K,Ti) | option_ids: list of K tensors (m_k, To)."""
        mem, pad = s.enc(state_ids)
        B, K = state_ids.shape[0], instr_ids.shape[0]
        q = s.q_proj(s.embed_text(instr_ids))[None].expand(B, K, -1)
        xmask = pad[:, None, None, :].expand(B, 1, K, pad.shape[1])
        for b in s.dec:
            q = b(q, mem=mem, xmask=xmask)
        q = s.ln_q(q)

        sizes = [o.shape[0] for o in option_ids]; mmax = max(sizes)
        L = max(o.shape[1] for o in option_ids)
        flat = torch.cat([Fn.pad(o, (0, L - o.shape[1])) for o in option_ids], 0)  # (sum m_k, L)
        O = s.o_proj(s.embed_text(flat))                                          # (sum m_k, d)
        Op = O.new_zeros(K, mmax, O.shape[-1]); i = 0
        for k, m in enumerate(sizes):
            Op[k, :m] = O[i:i + m]; i += m
        logits = torch.einsum('bkd,kmd->bkm', q, Op) / math.sqrt(s.d)              # (B,K,mmax)
        return [logits[:, k, :m] for k, m in enumerate(sizes)]   # views: no extra kernels


def confidence(logits):
    """1 - H(p)/log m.  For m=2 this is a function of p alone -- which is exactly why the
    public API gives Choice and Score a confidence field and says Noul's confidence is
    'built into the probability itself'."""
    p = torch.softmax(logits.float(), -1)
    H = -(p * (p + 1e-12).log()).sum(-1)
    return 1 - H / math.log(p.shape[-1])


def score_readout(logits):
    """Expected level. This is why a Score 'may fall between levels'."""
    p = torch.softmax(logits.float(), -1)
    return (p * torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)).sum(-1)


class ARBaseline(nn.Module):
    """Same trunk, causal, with an LM head: answers are emitted as TOKENS, one at a time.
    This is the thing a System One model deletes.  It can emit a string that is not a
    valid option -- and it has a search problem that the factorised head does not."""
    def __init__(s, V, d=256, h=4, L=6, maxlen=160):
        super().__init__()
        s.enc = TextEncoder(V, d, h, L, maxlen, causal=True)
        s.head = nn.Linear(d, V, bias=False); s.head.weight = s.enc.tok.weight

    def forward(s, ids):
        h, _ = s.enc(ids)
        return s.head(h)
