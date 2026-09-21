"""INCIDENT-80: a decision world with a closed-form Bayes posterior.

z = (sev, cat, pii) -> 5*8*2 = 80 latent states, small enough to enumerate exactly.
Each of 12 binary signals fires with a known likelihood p(e_j | z); emitted signals are
rendered as one of several English paraphrases, shuffled, padded with information-free
filler.  The model sees text only.  We see p(z | e), so we know the optimal answer to
every question on every example -- which is what makes calibration falsifiable here.
"""
import numpy as np, itertools, random

SEV  = ["informational", "low", "medium", "high", "critical"]
CAT  = ["auth_failure", "billing_error", "data_loss", "latency",
        "phishing", "hardware_fault", "access_request", "outage"]
TEAM = ["identity", "finance", "platform", "secops", "helpdesk"]
TEAM_OF = [0, 1, 2, 2, 3, 2, 4, 2]          # cat -> team  (deterministic, many-to-one)

# -- priors -------------------------------------------------------------------
P_CAT = np.array([.16, .18, .08, .15, .10, .12, .13, .08])
P_SEV_GIVEN_CAT = np.array([
    [.20, .35, .30, .12, .03],   # auth_failure
    [.30, .40, .22, .07, .01],   # billing_error
    [.02, .08, .25, .40, .25],   # data_loss
    [.18, .32, .30, .16, .04],   # latency
    [.08, .20, .32, .28, .12],   # phishing
    [.22, .34, .28, .13, .03],   # hardware_fault
    [.45, .38, .13, .03, .01],   # access_request
    [.03, .10, .27, .38, .22],   # outage
])
P_PII_GIVEN_CAT = np.array([.35, .70, .55, .10, .45, .05, .60, .12])

# -- signals: likelihood p(fire | sev, cat, pii) -------------------------------
def _lik(sev, cat, pii):
    c = CAT[cat]
    p = [
        [.02, .08, .25, .75, .95][sev],                                        # 0 pager
        min(.95, [.05, .15, .35, .60, .80][sev] + (.15 if c in
            ("billing_error", "outage", "latency") else 0)),                   # 1 customer_facing
        .70 if c in ("auth_failure", "phishing") else .05,                     # 2 credentials
        .85 if pii else .06,                                                   # 3 pii_marker
        .65 if c in ("data_loss", "latency") else .04,                         # 4 replication_lag
        .80 if c == "billing_error" else .05,                                  # 5 invoice_id
        [.03, .10, .30, .60, .85][sev],                                        # 6 multiple_reports
        .35,                                                                   # 7 repro_staging (noise)
        (.60 if c in ("phishing", "auth_failure", "data_loss") else .05)
            * (.40 + .15 * sev),                                               # 8 security_looped
        [.70, .60, .40, .20, .05][sev],                                        # 9 single_user
        [.01, .02, .05, .25, .60][sev],                                        # 10 exec_escalation
        .70 if c == "data_loss" else .03,                                      # 11 data_deleted
    ]
    return np.clip(np.array(p), .01, .99)

PHRASES = [
    ["the pager fired at 02:14", "oncall was paged automatically", "an alert page went out to the rotation"],
    ["customers are reporting this in the app", "this is visible to end users", "the public status page is affected"],
    ["the logs show repeated credential rejections", "several password attempts failed", "sso tokens were refused"],
    ["the thread quotes a full email address and phone number", "a card number appears in the attachment",
     "the ticket body contains a home address"],
    ["replication lag is climbing past two minutes", "the follower replica is far behind primary",
     "write acknowledgements are stalling"],
    ["invoice inv-48120 is attached", "the billing reference is quoted in the subject",
     "the charge id appears twice in the thread"],
    ["four separate teams filed the same report", "duplicate tickets keep arriving",
     "reports are coming in from multiple regions"],
    ["it reproduces cleanly in staging", "we reproduced it on the test cluster", "staging shows the same behaviour"],
    ["the security team is already on the call", "secops opened a parallel investigation",
     "an incident channel was created by security"],
    ["only one account appears affected", "a single user is impacted so far", "the blast radius looks like one tenant"],
    ["a vp asked for an update directly", "leadership is asking for hourly updates",
     "an executive escalation was attached"],
    ["rows are missing from the primary table", "a delete ran without a where clause",
     "objects were removed from the bucket"],
]
FILLER = [
    "the ticket was filed through the web form", "timezone is utc",
    "the reporter is on the emea rotation", "this arrived outside business hours",
    "the original message was forwarded twice", "an internal runbook link was pasted",
    "the queue was already backed up this morning", "no screenshots were attached",
    "the thread has six replies", "a follow up is scheduled",
]
HEADERS = ["incident intake", "ticket", "new report", "triage queue entry", "alert summary"]

N_SIG = len(PHRASES)
Z = list(itertools.product(range(5), range(8), range(2)))          # 80 latent states
LIK = np.stack([_lik(*z) for z in Z])                              # (80, 12)
PZ  = np.array([P_CAT[c] * P_SEV_GIVEN_CAT[c, s] * (P_PII_GIVEN_CAT[c] if p else 1 - P_PII_GIVEN_CAT[c])
                for (s, c, p) in Z])
PZ /= PZ.sum()

# questions: (name, kind, n_out) -- kind in {choice, score, noul}
QUESTIONS = [("category", "choice", 8), ("severity", "score", 5), ("contains_pii", "noul", 2),
             ("page_oncall", "noul", 2), ("route_team", "choice", 5)]
QNAMES = [q[0] for q in QUESTIONS]

def answers_of(sev, cat, pii):
    return [cat, sev, pii, int(sev >= 3), TEAM_OF[cat]]

def posterior(fired):
    """fired: bool vector (12,) -> list of exact posteriors, one per question."""
    ll = np.where(fired, LIK, 1 - LIK)                 # (80,12)
    w  = PZ * ll.prod(1)
    w /= w.sum()
    pc = np.zeros(8); ps = np.zeros(5); pp = np.zeros(2); pg = np.zeros(2); pt = np.zeros(5)
    for wi, (s, c, p) in zip(w, Z):
        pc[c] += wi; ps[s] += wi; pp[p] += wi
        pg[int(s >= 3)] += wi; pt[TEAM_OF[c]] += wi
    return [pc, ps, pp, pg, pt]

def render(fired, rng):
    parts = [PHRASES[j][rng.randrange(len(PHRASES[j]))] for j in range(N_SIG) if fired[j]]
    parts += [FILLER[rng.randrange(len(FILLER))] for _ in range(rng.randint(1, 3))]
    rng.shuffle(parts)
    return HEADERS[rng.randrange(len(HEADERS))] + " : " + " . ".join(parts) + " ."

def sample(n, seed=0):
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    idx = nrng.choice(len(Z), size=n, p=PZ)
    out = []
    for i in idx:
        sev, cat, pii = Z[i]
        fired = nrng.rand(N_SIG) < LIK[i]
        out.append(dict(text=render(fired, rng), fired=fired.copy(),
                        z=(sev, cat, pii), y=answers_of(sev, cat, pii),
                        post=posterior(fired)))
    return out
