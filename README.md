# JEV from scratch

A first-principles reconstruction of **Jev**, the non-autoregressive "System One" decision model
[TypeSafe shipped on 15 September 2026](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
Every module is written out — attention is `softmax(QK'/√d)V`, there is no `nn.TransformerEncoder`
and no `AutoModelForAnything` — and every architectural claim is turned into a measurement that was
allowed to come out against the reconstruction. Three of the seven did.

Trains end to end on a free Colab T4 in about seven minutes.

```
python run_ladder.py          # all seven rungs
python run_ladder.py r5 r7    # just the calibration and consistency rungs
```

---

## What Jev actually is

Jev is a transformer that is not a language model. It does not generate text. You pass a `state` and
a dict of **typed questions**, and in a single parallel forward pass you get back a calibrated
probability distribution per question. Nothing about the weights, scale or training corpus is public.
But the *shape* is fully determined by four published statements, and from those the architecture can
be rebuilt:

| published statement | what it forces |
|---|---|
| "generates all outputs in a single query rather than autoregressively" | no causal mask, no KV-cache loop |
| "questions evaluate **independently** against shared state — the answer to A does not become context for B" | $p(a_1..a_K \mid s) = \prod_k p(a_k \mid s)$ |
| "Choice: up to 255 options · Score: 2–10 ordered levels · Noul: a probability" | the output head has no token vocabulary; it scores the options you passed in |
| "output tokens: FREE" · "0% structured output error" | the response is $O(K)$ floats; type safety is a theorem, not a metric |

### The reconstruction

**All three primitives are the same operation.** Options arrive as text, get encoded into vectors
$o_1..o_m$, and

$$p(a=j \mid s) \;=\; \operatorname{softmax}_j\!\big(\langle q, o_j\rangle / \sqrt{d}\big)$$

- **Choice** is that, for $m \le 255$ (which is exactly one byte of option index).
- **Noul** is that, for $m = 2$.
- **Score** is that over ordered level descriptions, plus a readout $\hat y = \sum_j j\,p_j$ —
  which is why the docs say a score "may fall between levels".

There is no token vocabulary anywhere in the output path, so an invalid answer is not *representable*.

**Confidence falls out of the same place.** Take $c = 1 - H(p)/\log m$. For $m=2$ that is a
deterministic function of $p$ alone — which is precisely why the public API gives Choice and Score a
separate `confidence` field, and says Noul's confidence is *"built into the probability itself"*. The
reconstruction predicts an asymmetry in their API that was not put in by hand.

---

## The benchmark: INCIDENT-80

Measuring calibration honestly requires a task whose **Bayes-optimal posterior is known**. Almost no
benchmark gives you that, so this one is generated. Latents $z = (\text{sev}, \text{cat}, \text{pii})$,
80 combinations; twelve binary signals fire with known likelihoods; emitted signals are rendered as
English paraphrases, shuffled, and padded with information-free filler. The model sees text only. We
enumerate $p(z \mid e) \propto p(z)\prod_j p(e_j \mid z)$ and get the exact posterior for every
question on every example.

Five typed questions, chosen so that two of them are deliberately coupled:

| question | type | ground truth |
|---|---|---|
| `category` | Choice (8) | `cat` |
| `severity` | Score (5 ordered levels) | `sev` |
| `contains_pii` | Noul | `pii` |
| `page_oncall` | Noul | $\mathbb{1}[\text{sev} \ge 3]$ |
| `route_team` | Choice (5) | $g(\text{cat})$ |

---

## Results

6.63 M parameters, 40 k training examples, 3 epochs, Tesla T4 fp16. The AR baseline is the same trunk
made causal with an LM head, 6.44 M parameters, same budget.

### R1 — how much does the independence factorisation throw away?

The non-autoregressive factorisation commits a known modelling error before a single weight is
trained: the total correlation $\mathrm{TC}(s) = \mathrm{KL}\big(p(a\mid s)\,\Vert\,\prod_k p(a_k\mid s)\big)$.
Because we own the generator, we can compute it exactly.

```
sum_k H(a_k|s) = 3.5075 nats
total correlation TC = 1.1096 nats   (31.6% of it)

I(a_i ; a_j | s), nats
                 category  severity  pii   page   team
category            .        .037   .036   .013   .815
severity           .037       .     .001   .221   .024
contains_pii       .036      .001    .     .000   .030
page_oncall        .013      .221   .000    .     .008
route_team         .815      .024   .030   .008    .
```

**All 1.11 nats sit in the two pairs that were planted.** Between semantically distinct questions the
factorisation costs ≤ 0.04 nats — essentially free.

Two things follow, and they carry the whole architecture:

1. **The factorisation never costs marginal accuracy or marginal calibration.** The product of
   marginals has the *correct* marginals by construction. It is only wrong about the joint — which is
   invisible to a consumer who reads per-question answers, and fatal to one who recombines them.
2. TypeSafe's own design rule — *"ask narrow, atomic questions; compose answers in code"* — is not
   style advice. It is the condition under which their architecture is lossless. Asking `severity`
   **and** `page_oncall` in one call spends 0.221 nats re-deriving `sev >= 3`, which an `if` statement
   does for free and exactly.

The same 1.11 nats then show up a second time, from two trained networks instead of from the
generator:

| | measured | Bayes |
|---|---|---|
| AR, teacher-forced NLL over the answer block | 2.481 | 2.398 (joint $H(a\mid s)$) |
| JEV, sum of per-question NLL | 3.550 | 3.507 ($\sum_k H(a_k\mid s)$) |
| **difference** | **1.069** | **1.110** (= TC) |

The entire likelihood advantage of autoregression over this architecture is the total correlation,
and nothing else.

### R2 — deleting the causal mask · **refuted**

| | acc | Bayes | KL → posterior | ECE |
|---|---|---|---|---|
| bidirectional | 0.723 | 0.722 | 0.0112 | 0.0145 |
| causal | 0.721 | 0.722 | 0.0123 | 0.0131 |

Identical parameter count, one boolean apart, and the difference is noise. The reason is worth
keeping: the *output* path never read the encoder autoregressively. The question queries cross-attend
over all state positions in both models, so the causal mask only restricts token-to-token mixing
inside the encoder, which this task does not need. **The cost of autoregression is in how answers come
out, not in how state goes in.**

Note also the headline number: the model lands *on* the Bayes ceiling — 0.723 against 0.722 — with a
KL of 0.011 nats from the exact posterior. The architecture recovers the true posterior almost
perfectly.

### R3 — one pass for K questions · **holds**

| K | JEV | AR (with KV cache) | speedup | JEV depth | AR depth |
|---|---|---|---|---|---|
| 1 | 7.47 ms | 12.21 ms | 1.6× | 1 | 2 |
| 2 | 7.48 | 25.55 | 3.4× | 1 | 4 |
| 5 | 7.53 | 64.31 | 8.5× | 1 | 10 |
| 10 | 8.31 | 126.35 | 15.2× | 1 | 20 |
| 20 | 8.65 | 244.87 | 28.3× | 1 | 40 |
| 50 | 10.11 | 611.47 | **60.5×** | 1 | 100 |

Batch-1, T4, fp16, median of 30. Sequential depth is the architecture; the wall clock follows it.

> A trap worth recording: the first implementation encoded each question's options in its own forward
> pass, and JEV's latency grew linearly in K too — 7 → 90 ms. At batch 1 on a T4 nothing is
> FLOP-bound, it is all kernel launches. Batching the option encode into one pass and the head into
> one `einsum` is the difference between an $O(K)$ and an $O(1)$ curve, with identical mathematics.

### R4 — 0% structured output error · **refuted as an advantage, upheld as a guarantee**

| question | AR acc | JEV acc | Bayes | AR invalid | AR off-schema mass |
|---|---|---|---|---|---|
| category | 0.582 | 0.592 | 0.592 | 0.0000 | 4.0e-4 |
| severity | 0.488 | 0.504 | 0.505 | 0.0000 | 1.6e-4 |
| contains_pii | 0.907 | 0.909 | 0.906 | 0.0000 | 8.9e-5 |
| page_oncall | 0.905 | 0.908 | 0.909 | 0.0000 | 3.0e-5 |
| route_team | 0.693 | 0.701 | 0.700 | 0.0000 | 1.2e-5 |

The AR baseline also scored exactly 0.0000 malformed outputs. On a closed vocabulary it was trained
on, generation does not go off-schema — so this benchmark cannot reproduce the 5.73% figure TypeSafe
reports for frontier chat models, which comes from open-ended generation against messy real schemas,
and the comparison should not be claimed.

What the AR model *cannot* do is make it a theorem. It still placed 4×10⁻⁴ of its probability mass on
tokens that are not valid categories. At a million calls a day that is hundreds of malformed
responses. JEV's is identically zero, because those outcomes are not in the sample space.

The accuracy column matters more. **AR lands below the Bayes ceiling that JEV sits on**, and it is not
undertraining:

| | marginal argmax | exact joint MAP | AR greedy |
|---|---|---|---|
| mean accuracy | 0.722 | 0.726 | **0.715** |

The exact joint MAP — computed from the true posterior, no model involved — scores 0.726. Greedy
decoding gets 0.715. The gap is **search error**: autoregression turns answering into a sequence
search, and greedy is a bad search. The factorised head has no search at all; `argmax` of an explicit
distribution is exact. More capacity on the answer path did not help, because the problem was never
capacity.

### R5 — what RLCD has to be · **refuted as stated, and something better found**

A **proper scoring rule** $S(p,y)$ is maximised in expectation by reporting the true posterior. The
log score is strictly proper — so cross-entropy training *is already* a calibration objective. That
sets the bar: RLCD has to beat plain CE, or it is theatre.

Now look at what an accuracy reward optimises. $\mathbb E[r] = \sum_a p_a \mathbb 1[a = y]$ is
**linear in $p$**, so it is maximised at a vertex of the simplex. The optimum of an accuracy reward is
a point mass. Any model trained to convergence on it must report probability 1 on its best guess,
whatever it actually knows.

| arm | objective | proper? | acc | mean conf | overconf | ECE | KL → posterior |
|---|---|---|---|---|---|---|---|
| **CE** | $\log p_y$ | strictly | 0.723 | 0.725 | **+0.003** | 0.0145 | 0.0112 |
| **Brier** | $-\lVert p - e_y\rVert^2$ | strictly | 0.723 | 0.722 | −0.001 | 0.0154 | 0.0161 |
| **RLVR** | REINFORCE on $\mathbb 1[\hat a = y]$ | **no** | 0.462 | **1.000** | +0.538 | 0.538 | 6.40 |
| **CE → RLVR** | CE, then RLVR — the RLHF pipeline | — | 0.720 | 0.918 | **+0.199** | 0.199 | 0.529 |
| *(Bayes)* | — | — | 0.722 | 0.722 | 0.000 | 0 | 0 |

Plain cross-entropy already lands 0.003 from perfect calibration. Brier adds nothing. **RLCD does not
beat MLE on calibration, because there is nothing there to beat.**

The control is the real result. **One epoch of accuracy-reward RL at lr 5e-5, on a model that started
perfectly calibrated, moved mean confidence from 0.725 to 0.918 while accuracy stayed flat at 0.720.**
That is the overconfidence of RLHF'd chat models, reproduced from scratch, with the mechanism named:
the reward is not a proper scoring rule, so its fixed point is a vertex. Trained from scratch on that
reward, the model collapses to confidence 1.000 on everything and loses a quarter of its accuracy.

So the honest reconstruction of RLCD is not "the thing that makes Jev calibrated". It is: **if you
want to do RL at all — to learn from outcomes, preferences or deployment feedback rather than labels —
your reward has to be a proper scoring rule, or you will destroy the one property you are selling.**
That is why a decision-model company had to invent a third RL variant instead of reaching for RLHF.

And here is what that is worth, in the only units that matter — confidence-gated routing, where
software auto-handles what the model is sure about and escalates the rest:

| arm | accuracy | coverage @ 90% accuracy | coverage @ 95% accuracy |
|---|---|---|---|
| CE | 0.723 | 0.551 | **0.374** |
| CE → RLVR | 0.720 | 0.519 | **0.301** |
| RLVR | 0.462 | 0.000 | 0.000 |
| *Bayes* | 0.728 | 0.564 | 0.376 |

Two models whose accuracy differs by 0.3 points. The calibrated one safely automates **37.4%** of
cases at a 95% bar; the RL-tuned one **30.1%** — a fifth less volume. The fully collapsed model can
gate nothing at all, because with confidence pinned at 1.000 there is no ordering left to threshold.
**The confidence field is the product.**

### R6 — is the speed architecture or just a small model? · **holds at batch 1, weakens under batching**

```
batch 256, K=5:  JEV  80.6 ms -> 15878 decisions/s
                 AR  221.0 ms ->  5791 decisions/s     ratio 2.7x
```

8.5× at batch 1 (K=5) but only 2.7× at batch 256. Sequential depth amortises once you are FLOP-bound
instead of launch-bound. The 40–200× figures in the launch material are interactive-latency figures
and should be read as such.

Token accounting for one 5-question request: 38 input tokens either way; **0 generated tokens for JEV
against 10 for AR**; the response is 22 floats. "Output tokens: free" is not a subsidy, it is a
statement about the architecture — and it is the same fact as "the model cannot explain itself".

### R7 — where the factorisation actually breaks · **refuted**

R1 said the cost is 1.11 nats. Here is what that looks like to a caller: the model returns
`severity = low` **and** `page_oncall = yes` in one response. Each answer individually well
calibrated; the tuple nonsense.

| model | acc | all-5 exact | `page ≠ f(sev)` | `team ≠ g(cat)` |
|---|---|---|---|---|
| exact posterior (floor) | 0.728 | 0.289 | 0.0067 | 0.0198 |
| JEV (factorised) | 0.723 | 0.274 | 0.0073 | 0.0348 |
| JEV + query self-attention | 0.725 | 0.279 | 0.0057 | 0.0318 |
| autoregressive | 0.715 | 0.279 | **0.0000** | **0.0000** |

Three things here.

**The floor is not zero.** Even a model holding the *exact* posterior contradicts itself on 2.0% of
responses, because per-question argmaxes of correct marginals need not be mutually consistent. No
factorised model of any size beats that.

**The obvious fix is not a fix.** Turning on self-attention between the question queries — letting A
see B — changes nothing meaningful (3.18% vs 3.48%). It cannot, and the reason is worth stating
precisely: with query self-attention the queries are still a *deterministic* function of $s$, so the
output is still $\prod_k p(a_k\mid s)$. **Shared computation is not shared randomness.** Representing
a joint needs autoregression over answers, or a latent variable
$p(a\mid s) = \sum_m \pi_m(s)\prod_k p(a_k\mid s,m)$ — which is how non-autoregressive translation was
eventually fixed — or iterative refinement, a second pass conditioned on the first pass's answers.
That last one is, in API terms, *a second Jev call whose state contains the first call's answers*:
exactly the "serial calls should represent genuine information dependencies" rule in TypeSafe's docs.

**The trade is explicit.** The AR decoder never self-contradicts — it learned the deterministic
constraints perfectly, because it can condition — and pays 0.8 points of marginal accuracy for it.

---

## Verdict

| # | claim | outcome |
|---|---|---|
| R1 | the factorisation is cheap for decisions | **holds, conditionally** — free between distinct questions, 0.8 nats between coupled ones |
| R2 | deleting the causal mask is what makes it work | **refuted** — noise |
| R3 | one pass for K questions ⇒ latency flat in K | **holds** — 60× at K=50 |
| R4 | 0% structured-output error is an advantage | **refuted as an advantage, upheld as a guarantee** |
| R5 | RLCD is what makes Jev calibrated | **refuted** — CE already is; RLCD's job is not destroying it |
| R6 | the speed claim is architecture, not model size | **holds at batch 1, weakens under batching** |
| R7 | the factorisation's cost is invisible in practice | **refuted** — 3.5% self-contradictory tuples |

### Three things worth carrying away

1. **Non-autoregression is a change of estimator, not an approximation.** Autoregression models
   $p(a\mid s)$; the factorised head models $\{p(a_k\mid s)\}$. For an API that returns per-question
   answers, the second is the quantity the consumer uses — and obtaining it requires no search, so
   there is no beam, no greedy error, no degeneration.
2. **Type safety is a change of sample space, not a metric.** Every "0%" number in the launch material
   is of this kind. It cannot be beaten by a better generative model; it can only be matched in
   expectation, never guaranteed.
3. **The confidence field is the product.** Calibration was worth 7.3 points of auto-handled volume at
   a 95% bar between two models 0.3 points apart in accuracy.

### What this cannot tell you

Jev's weights, scale, pretraining corpus and actual RLCD objective are not public. Everything here is
a rebuild from the published interface and four architectural statements, trained on a synthetic world
chosen because its Bayes posterior is computable. The mechanisms are real and the measurements are
honest; transfer to their model at their scale is an assumption, not a finding.

---

## Layout

```
jev/jevbench.py    INCIDENT-80: the generator, and the exact posterior
jev/jevmodel.py    attention, the encoder, the parallel question decoder, the typed head
jev/jevdecode.py   KV-cache incremental decoding, so the AR baseline is timed fairly
run_ladder.py      all seven rungs end to end
results.json       every number in this README
```

## Sources

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe
- [Quickstart / API reference](https://docs.typesafe.ai/introduction/quickstart) — TypeSafe
- [Jev: TypeSafe's System One Model That Never Hallucinates](https://www.datacamp.com/blog/system-one-models-jev) — DataCamp
- Gu et al., [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281), 2018 — the multimodality problem this architecture is betting against
- Ghazvininejad et al., [Mask-Predict](https://arxiv.org/abs/1904.09324), 2019 — iterative refinement, the fix in R7
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 2007 — why R5 works out the way it does
