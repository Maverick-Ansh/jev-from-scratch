"""The whole ladder, end to end.  ~7 minutes on a free Colab T4.

    python run_ladder.py            # everything
    python run_ladder.py r1 r3      # selected rungs (earlier rungs they depend on still run)

R1  how much does the independence factorisation actually throw away
R2  the causal mask ablation
R3  latency vs number of questions
R4  structured output: theorem vs metric
R5  four training objectives, and what RLCD has to be
R6  throughput, and where "output tokens: free" comes from
R7  self-contradictory answer tuples
"""
import sys, math, json, time, random, pathlib
import numpy as np, torch, torch.nn.functional as Fn

from jev import jevbench as JB
from jev import jevmodel as JM
from jev.jevdecode import ARDecoder

ROOT = pathlib.Path(__file__).parent
(ROOT / "ckpt").mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.float16 if (DEV == "cuda" and torch.cuda.get_device_capability(0)[0] < 8) else torch.bfloat16
WANT = set(a.lower() for a in sys.argv[1:]) or {f"r{i}" for i in range(1, 8)}


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(s)


# ─── data ────────────────────────────────────────────────────────────────────
seed_all(0)
TRAIN, TEST = JB.sample(40000, seed=1), JB.sample(4000, seed=3)

OPTION_TEXTS = {
    "category": ["a login or authentication failure", "an incorrect charge or invoice problem",
                 "records or objects were destroyed", "the system is slow but working",
                 "a fraudulent message trying to steal credentials", "a machine or disk fault",
                 "someone is asking for permission to a resource", "the service is completely down"],
    "severity": ["no impact, informational only", "minor annoyance for a few users",
                 "a real problem with a workaround", "serious impact, needs attention today",
                 "emergency, everything stops for this"],
    "contains_pii": ["no personal data is present", "personal data is present"],
    "page_oncall": ["do not wake anyone", "page the oncall engineer now"],
    "route_team": ["identity and access team", "finance operations", "core platform",
                   "security operations", "front line helpdesk"],
}
INSTR = {"category": "what kind of incident is this",
         "severity": "how bad is this incident for the business",
         "contains_pii": "does the ticket contain personal data about a person",
         "page_oncall": "should we wake the oncall engineer right now",
         "route_team": "which team should own this ticket"}
ANS_WORDS = [JB.CAT, JB.SEV, ["pii_no", "pii_yes"], ["page_no", "page_yes"], JB.TEAM]
AR_TOKENS = ["<sep>"] + [f"<q:{n}>" for n in JB.QNAMES] + [w for g in ANS_WORDS for w in g]


def words(s):
    return s.replace(".", " . ").replace(":", " : ").split()


vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in TRAIN:
    for w in words(ex["text"]):
        vocab.setdefault(w, len(vocab))
for t in [o for n in OPTION_TEXTS for o in OPTION_TEXTS[n]] + list(INSTR.values()):
    for w in words(t):
        vocab.setdefault(w, len(vocab))
for w in AR_TOKENS:
    vocab.setdefault(w, len(vocab))
V = len(vocab)


def encode(s, L):
    ids = [2] + [vocab.get(w, 1) for w in words(s)][:L - 1]
    return ids + [0] * (L - len(ids))


LMAX = max(len(words(e["text"])) for e in TRAIN) + 1
LOPT = max(len(words(o)) for n in OPTION_TEXTS for o in OPTION_TEXTS[n]) + 1
LINS = max(len(words(t)) for t in INSTR.values()) + 1


def tensorize(d):
    return (torch.tensor([encode(e["text"], LMAX) for e in d]),
            torch.tensor([e["y"] for e in d]),
            torch.tensor(np.stack([e["fired"] for e in d]), dtype=torch.bool))


Xtr, Ytr, _ = tensorize(TRAIN)
Xte, Yte, Fte = tensorize(TEST)
INSTR_IDS = torch.tensor([encode(INSTR[n], LINS) for n in JB.QNAMES]).to(DEV)
OPT_LIST = [torch.tensor([encode(o, LOPT) for o in OPTION_TEXTS[n]]).to(DEV) for n in JB.QNAMES]
NOUT = [k for _, _, k in JB.QUESTIONS]
POST = [torch.tensor(np.stack([e["post"][k] for e in TEST]), dtype=torch.float32) for k in range(5)]
print(f"vocab={V}  state<={LMAX}  train={len(TRAIN)}  test={len(TEST)}  dev={DEV}/{AMP}")

# ─── exact posterior machinery (this is what makes the ladder falsifiable) ───
LL = np.where(Fte.numpy()[:, None, :], JB.LIK[None], 1 - JB.LIK[None])
W = JB.PZ[None] * LL.prod(2); W /= W.sum(1, keepdims=True)          # (N,80) exact p(z|s)
A = np.array([JB.answers_of(*z) for z in JB.Z])                     # (80,5)
M = [np.eye(k)[A[:, i]] for i, k in enumerate(NOUT)]
PM = [W @ m for m in M]
EPS = 1e-12

# ─── harness ─────────────────────────────────────────────────────────────────
def ece(p, y, bins=15):
    conf, pred = p.max(1); acc = (pred == y).float(); e = 0.
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return e.item()


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval(); P = [[] for _ in NOUT]
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
            outs = model(X[i:i + bs].to(DEV), INSTR_IDS, OPT_LIST)
        for k, o in enumerate(outs):
            P[k].append(torch.softmax(o.float(), -1).cpu())
    return [torch.cat(p) for p in P]


def evaluate(model):
    P = predict(model, Xte); rows = []
    for k, (n, kind, m) in enumerate(JB.QUESTIONS):
        y, pk, ps = Yte[:, k], P[k], POST[k]
        rows.append(dict(q=n, acc=(pk.argmax(1) == y).float().mean().item(),
                         bayes=ps.max(1).values.mean().item(),
                         nll=-(pk[torch.arange(len(y)), y] + EPS).log().mean().item(),
                         kl=(ps * ((ps + EPS).log() - (pk + EPS).log())).sum(1).mean().item(),
                         ece=ece(pk, y),
                         brier=((pk - torch.eye(m)[y]) ** 2).sum(1).mean().item()))
    return rows, P


def show(rows, title):
    print(f"\n-- {title} --")
    print(f"{'question':<14}{'acc':>7}{'bayes':>7}{'NLL':>8}{'KL->bayes':>11}{'ECE':>8}")
    for r in rows:
        print(f"{r['q']:<14}{r['acc']:>7.3f}{r['bayes']:>7.3f}{r['nll']:>8.4f}{r['kl']:>11.4f}{r['ece']:>8.4f}")
    g = lambda f: np.mean([f(r) for r in rows])
    print(f"{'MEAN':<14}{g(lambda r: r['acc']):>7.3f}{g(lambda r: r['bayes']):>7.3f}"
          f"{g(lambda r: r['nll']):>8.4f}{g(lambda r: r['kl']):>11.4f}{g(lambda r: r['ece']):>8.4f}")


def make_jev(**kw):
    return JM.JEV(V, d=256, h=4, L_state=4, L_opt=2, L_dec=2, maxlen=max(LMAX, 128), **kw)


def loss_ce(outs, yb):
    return sum(Fn.cross_entropy(o.float(), yb[:, k]) for k, o in enumerate(outs))


def loss_brier(outs, yb):
    tot = 0.
    for k, o in enumerate(outs):
        p = torch.softmax(o.float(), -1)
        tot = tot + ((p - Fn.one_hot(yb[:, k], p.shape[-1]).float()) ** 2).sum(1).mean()
    return tot


def loss_rlvr(outs, yb):
    """REINFORCE on r = 1[sampled answer correct].  Expected reward is LINEAR in p, so its
    maximum is a vertex of the simplex: the optimum of this objective is a point mass."""
    tot = 0.
    for k, o in enumerate(outs):
        logp = torch.log_softmax(o.float(), -1)
        a = torch.multinomial(logp.exp(), 1).squeeze(1)
        r = (a == yb[:, k]).float()
        tot = tot - ((r - r.mean()) * logp.gather(1, a[:, None]).squeeze(1)).mean()
    return tot


def train_jev(model, epochs=3, bs=256, lr=3e-4, loss_fn=loss_ce, log=True, seed=0):
    seed_all(seed); model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda"); t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr)); run = 0.
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
                loss = loss_fn(model(xb, INSTR_IDS, OPT_LIST), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); run += loss.item()
        if log:
            print(f"  ep{ep + 1} loss={run / math.ceil(len(Xtr) / bs):.4f}  {time.time() - t0:.0f}s")
    return model


OUT = {}

# ─── R1 ──────────────────────────────────────────────────────────────────────
logPk = sum(np.log(PM[i][np.arange(len(W))[:, None], A[None, :, i]] + EPS) for i in range(5))
TC = (W * (np.log(W + EPS) - logPk)).sum(1)
Hk = np.array([-(p * np.log(p + EPS)).sum(1).mean() for p in PM])
if "r1" in WANT:
    print("\n=== R1: what the independence factorisation throws away ===")
    print(f"{'question':<14}{'H(a|s)':>10}{'Bayes acc':>11}{'majority':>10}")
    for i, n in enumerate(JB.QNAMES):
        maj = np.bincount(Ytr[:, i].numpy(), minlength=NOUT[i]).max() / len(Ytr)
        print(f"{n:<14}{Hk[i]:>10.3f}{PM[i].max(1).mean():>11.3f}{maj:>10.3f}")
    CMI = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            J = np.einsum('nz,za,zb->nab', W, M[i], M[j])
            Pi, Pj = J.sum(2)[:, :, None], J.sum(1)[:, None, :]
            CMI[i, j] = CMI[j, i] = (J * (np.log(J + EPS) - np.log(Pi * Pj + EPS))).sum((1, 2)).mean()
    np.set_printoptions(precision=3, suppress=True)
    print(f"\nsum_k H(a_k|s) = {Hk.sum():.4f} | TC = {TC.mean():.4f} nats "
          f"({TC.mean() / Hk.sum():.1%} of it)\nI(a_i;a_j|s):\n{CMI}")
    OUT["r1"] = dict(TC=float(TC.mean()), H=Hk.tolist(), CMI=CMI.tolist())

# ─── R2 ──────────────────────────────────────────────────────────────────────
print("\n=== R2: the causal mask ===")
JEV = None
for causal in (False, True):
    tag = "causal" if causal else "bidirectional"
    ck = ROOT / "ckpt" / f"jev_{tag}.pt"
    seed_all(0); m = make_jev(causal=causal).to(DEV)
    if ck.exists():
        m.load_state_dict(torch.load(ck, map_location=DEV)); print(f"  [{tag}] loaded")
    else:
        train_jev(m); torch.save(m.state_dict(), ck)
    rows, P = evaluate(m)
    show(rows, f"JEV / {tag}")
    OUT.setdefault("r2", {})[tag] = rows
    if not causal:
        JEV, JEV_P = m, P
    if "r2" not in WANT:
        break

# ─── AR baseline (needed by R3, R4, R6, R7) ─────────────────────────────────
SEP = vocab["<sep>"]
QTOK = [vocab[f"<q:{n}>"] for n in JB.QNAMES]
ATOK = [[vocab[w] for w in g] for g in ANS_WORDS]
BLK, KMAX = 2 * len(QTOK), 50


def ar_seq(X, Y):
    tail = torch.zeros(len(X), 1 + BLK, dtype=torch.long); tail[:, 0] = SEP
    for k in range(len(QTOK)):
        tail[:, 1 + 2 * k] = QTOK[k]
        tail[:, 2 + 2 * k] = torch.tensor(ATOK[k])[Y[:, k]]
    return torch.cat([X, tail], 1)


Str, Ste = ar_seq(Xtr, Ytr), ar_seq(Xte, Yte)
seed_all(0)
AR = JM.ARBaseline(V, d=256, h=4, L=8, maxlen=LMAX + 1 + 2 * KMAX).to(DEV)
ck = ROOT / "ckpt" / "ar.pt"
if ck.exists():
    AR.load_state_dict(torch.load(ck, map_location=DEV)); print("\n  [ar] loaded")
else:
    print("\n  [ar] training")
    opt = torch.optim.AdamW(AR.parameters(), lr=3e-4, weight_decay=0.01)
    EP, BS = 3, 256
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP * math.ceil(len(Str) / BS),
                                              pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    for ep in range(EP):
        AR.train(); perm = torch.randperm(len(Str)); run = 0.
        for i in range(0, len(Str), BS):
            sb = Str[perm[i:i + BS]].to(DEV)
            with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
                lg = AR(sb)[:, LMAX:LMAX + BLK]
                loss = Fn.cross_entropy(lg.float().reshape(-1, V), sb[:, LMAX + 1:LMAX + BLK + 1].reshape(-1))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(AR.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); run += loss.item()
        print(f"  ep{ep + 1} loss={run / math.ceil(len(Str) / BS):.4f}")
    torch.save(AR.state_dict(), ck)

with torch.no_grad():
    sb = Ste[:2048].to(DEV)
    with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
        lg = AR(sb)[:, LMAX:LMAX + BLK]
    ar_joint = Fn.cross_entropy(lg.float().reshape(-1, V),
                                sb[:, LMAX + 1:LMAX + BLK + 1].reshape(-1)).item() * BLK
jev_sum = sum(r["nll"] for r in OUT["r2"]["bidirectional"])
print(f"\nAR teacher-forced NLL over answers : {ar_joint:.3f}  (Bayes joint H(a|s) = {Hk.sum() - TC.mean():.3f})")
print(f"JEV sum of per-question NLL        : {jev_sum:.3f}  (Bayes sum_k H(a_k|s) = {Hk.sum():.3f})")
print(f"difference                         : {jev_sum - ar_joint:.3f}  (total correlation = {TC.mean():.3f})")
OUT["r6_nll"] = dict(ar_joint=ar_joint, jev_sum=jev_sum)


@torch.no_grad()
def ar_decode(S, bs=512):
    AR.eval(); dec = ARDecoder(AR, h=4); T, P = [], []
    for i in range(0, len(S), bs):
        dec.reset()
        with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
            cur = dec.prefill(S[i:i + bs, :LMAX + 1].to(DEV))
        tk, pr = [], []
        for t in range(BLK):
            nx = cur.argmax(-1); tk.append(nx.cpu()); pr.append(torch.softmax(cur.float(), -1).cpu())
            if t < BLK - 1:
                with torch.autocast("cuda", dtype=AMP, enabled=DEV == "cuda"):
                    cur = dec.step(nx[:, None])
        T.append(torch.stack(tk, 1)); P.append(torch.stack(pr, 1))
    return torch.cat(T), torch.cat(P)


ARTOK, ARP = ar_decode(Ste)
ar_pred = np.stack([np.array([ATOK[k].index(t) if t in ATOK[k] else -1
                              for t in ARTOK[:, 2 * k + 1].tolist()]) for k in range(5)], 1)

# ─── R4 ──────────────────────────────────────────────────────────────────────
if "r4" in WANT:
    print("\n=== R4: structured output -- theorem vs metric ===")
    print(f"{'question':<14}{'AR acc':>8}{'JEV acc':>9}{'bayes':>8}{'invalid':>9}{'off-schema mass':>17}")
    for k in range(5):
        allowed = torch.tensor(ATOK[k])
        inv = (~torch.isin(ARTOK[:, 2 * k + 1], allowed)).float().mean().item()
        acc = (ar_pred[:, k] == Yte[:, k].numpy()).mean()
        off = (1 - ARP[:, 2 * k + 1][:, allowed].sum(1)).mean().item()
        print(f"{JB.QNAMES[k]:<14}{acc:>8.3f}{OUT['r2']['bidirectional'][k]['acc']:>9.3f}"
              f"{PM[k].max(1).mean():>8.3f}{inv:>9.4f}{off:>17.2e}")
    zstar = W.argmax(1); jointA = A[zstar]
    print(f"\nmarginal argmax {np.mean([PM[k].max(1).mean() for k in range(5)]):.3f} | "
          f"exact joint MAP {(jointA == Yte.numpy()).mean():.3f} | "
          f"AR greedy {(ar_pred == Yte.numpy()).mean():.3f}")
    print("AR is below the ceiling because greedy decoding is a SEARCH; the factorised head has none.")

# ─── R3 / R6 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def bench(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


if "r3" in WANT or "r6" in WANT:
    print("\n=== R3: latency vs number of questions (batch 1) ===")
    JEV.eval(); AR.eval()
    x1, s1 = Xte[:1].to(DEV), Ste[:1, :LMAX + 1].to(DEV)
    dec = ARDecoder(AR, h=4)
    print(f"{'K':>5}{'JEV ms':>9}{'AR ms':>9}{'speedup':>9}{'JEV depth':>11}{'AR depth':>10}")
    LAT = []
    for K in (1, 2, 5, 10, 20, 50):
        ii = INSTR_IDS[torch.arange(K) % 5]; oo = [OPT_LIST[j % 5] for j in range(K)]

        def jf():
            JEV(x1, ii, oo)

        def af():
            dec.reset(); cur = dec.prefill(s1)
            for _ in range(2 * K - 1):
                cur = dec.step(cur.argmax(-1)[:, None])

        j, a = bench(jf), bench(af)
        LAT.append((K, j, a)); print(f"{K:>5}{j:>9.2f}{a:>9.2f}{a / j:>8.1f}x{1:>10}{2 * K:>10}")
    OUT["r3"] = LAT

if "r6" in WANT:
    print("\n=== R6: throughput, and where 'output tokens: free' comes from ===")
    BS = 256
    xb, sb2 = Xte[:BS].to(DEV), Ste[:BS, :LMAX + 1].to(DEV)
    dec = ARDecoder(AR, h=4)

    def jb():
        JEV(xb, INSTR_IDS, OPT_LIST)

    def ab():
        dec.reset(); cur = dec.prefill(sb2)
        for _ in range(BLK - 1):
            cur = dec.step(cur.argmax(-1)[:, None])

    jt, at = bench(jb, iters=20), bench(ab, iters=20)
    print(f"batch {BS}, K=5: JEV {jt:.1f} ms ({5 * BS / jt * 1000:.0f} decisions/s) | "
          f"AR {at:.1f} ms ({5 * BS / at * 1000:.0f} decisions/s) | ratio {at / jt:.1f}x")
    print(f"generated tokens per request: JEV 0, AR {BLK}.  The response is {sum(NOUT)} floats.")
    OUT["r6"] = dict(jev_ms=jt, ar_ms=at, batch=BS)

# ─── R5 ──────────────────────────────────────────────────────────────────────
if "r5" in WANT:
    print("\n=== R5: four objectives ===")
    ARMS = {}
    for name, lf, ep, lr, init in [("CE", loss_ce, 3, 3e-4, None),
                                   ("Brier", loss_brier, 3, 3e-4, None),
                                   ("RLVR", loss_rlvr, 3, 3e-4, None),
                                   ("CE->RLVR", loss_rlvr, 1, 5e-5, "CE")]:
        ck = ROOT / "ckpt" / f"arm_{name.replace('->', '_')}.pt"
        seed_all(0); m = make_jev().to(DEV)
        if init:
            m.load_state_dict(ARMS[init]["sd"])
        if ck.exists():
            m.load_state_dict(torch.load(ck, map_location=DEV))
        else:
            train_jev(m, epochs=ep, lr=lr, loss_fn=lf, log=False); torch.save(m.state_dict(), ck)
        rows, P = evaluate(m)
        conf = np.mean([p.max(1).values.mean().item() for p in P])
        acc = np.mean([r["acc"] for r in rows])
        ARMS[name] = dict(rows=rows, P=P, conf=conf, acc=acc,
                          sd={k: v.clone() for k, v in m.state_dict().items()})
        print(f"{name:<10} acc={acc:.3f} conf={conf:.3f} overconf={conf - acc:+.3f} "
              f"ECE={np.mean([r['ece'] for r in rows]):.4f} KL={np.mean([r['kl'] for r in rows]):.4f}")

    def pooled(P):
        c = torch.cat([P[k].max(1).values for k in range(5)]).numpy()
        a = torch.cat([(P[k].argmax(1) == Yte[:, k]).float() for k in range(5)]).numpy()
        return c, a

    print(f"\n{'arm':<10}{'cov@90%acc':>12}{'cov@95%acc':>12}   <- fraction safely auto-handled")
    for nm in ARMS:
        c, a = pooled(ARMS[nm]["P"]); o = np.argsort(-c)
        ra = np.cumsum(a[o]) / np.arange(1, len(a) + 1)
        cov = np.arange(1, len(a) + 1) / len(a)
        f = lambda th: cov[ra >= th].max() if (ra >= th).any() else 0.
        print(f"{nm:<10}{f(.90):>12.3f}{f(.95):>12.3f}")
        ARMS[nm].pop("sd"); ARMS[nm].pop("P")
    OUT["r5"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in ARMS.items()}

# ─── R7 ──────────────────────────────────────────────────────────────────────
if "r7" in WANT:
    print("\n=== R7: self-contradictory tuples ===")
    TEAM_OF = np.array(JB.TEAM_OF)

    def inconsistency(pred):
        cat, sev, _, page, team = [pred[:, k] for k in range(5)]
        return ((page != (sev >= 3).astype(int)).mean(), (team != TEAM_OF[cat]).mean())

    ck = ROOT / "ckpt" / "jev_qattn.pt"
    seed_all(0); QSA = make_jev(query_self_attn=True).to(DEV)
    if ck.exists():
        QSA.load_state_dict(torch.load(ck, map_location=DEV))
    else:
        train_jev(QSA, log=False); torch.save(QSA.state_dict(), ck)
    qrows, qP = evaluate(QSA)
    preds = [("exact posterior (floor)", np.stack([POST[k].argmax(1).numpy() for k in range(5)], 1)),
             ("JEV (factorised)", np.stack([JEV_P[k].argmax(1).numpy() for k in range(5)], 1)),
             ("JEV + query self-attn", np.stack([qP[k].argmax(1).numpy() for k in range(5)], 1)),
             ("autoregressive", ar_pred)]
    print(f"{'model':<26}{'acc':>7}{'all-5 exact':>13}{'page!=f(sev)':>14}{'team!=g(cat)':>14}")
    for nm, pr in preds:
        bp, bt = inconsistency(pr)
        print(f"{nm:<26}{(pr == Yte.numpy()).mean():>7.3f}{(pr == Yte.numpy()).all(1).mean():>13.3f}"
              f"{bp:>14.4f}{bt:>14.4f}")
    print("\nquery self-attention does not fix it: shared computation is not shared randomness.")
    OUT["r7"] = {nm: list(map(float, inconsistency(pr))) for nm, pr in preds}

json.dump(OUT, open(ROOT / "results.json", "w"), indent=1, default=float)
print(f"\nwrote {ROOT / 'results.json'}")
