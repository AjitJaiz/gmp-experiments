"""Fuzzy inference machinery: rule base of Rineh & Cheu (2025) Table 2, reference MFs (Table 1 ranges),
Mamdani max-min inference, D1/D2 decisions, and decision metrics."""
import numpy as np

# input order follows Table 2: G_FA, G_PA, D, G_PB ; term index 0=close 1=medium 2=far
INPUTS = ["G_FA", "G_PA", "D", "G_PB"]
_T = {"close": 0, "medium": 1, "far": 2}
_RULES = """
close close close close no|close close close medium no|close close close far no
close close medium close no|close close medium medium no|close close medium far no
close close far close no|close close far medium no|close close far far no
close medium close close no|close medium close medium no|close medium close far no
close medium medium close no|close medium medium medium no|close medium medium far no
close medium far close no|close medium far medium no|close medium far far no
close far medium close no|close far medium medium no|close far medium far no
close far far close no|close far far medium no|close far far far no
medium close close close no|medium close close medium no|medium close close far no
medium close medium close no|medium close medium medium no|medium close medium far no
medium close far close no|medium close far medium no|medium close far far no
medium medium medium close no|medium medium medium medium yes|medium medium medium far yes
medium medium far close no|medium medium far medium yes|medium medium far far yes
medium far far close no|medium far far medium yes|medium far far far yes
far close medium close no|far close medium medium no|far close medium far no
far close far close no|far close far medium no|far close far far no
far medium far close no|far medium far medium yes|far medium far far yes
far far far close no|far far far medium yes|far far far far yes
"""
_rows = [r.split() for r in _RULES.replace("\n", "|").split("|") if r.strip()]
RULES = np.array([[_T[w] for w in r[:4]] for r in _rows])          # (54,4)
RULE_YES = np.array([r[4] == "yes" for r in _rows])                # (54,)
assert RULES.shape == (54, 4) and RULE_YES.sum() == 10

# reference MF breakpoints (metres) reconstructed from the crisp ranges in Table 1:
#   G_*: close<20, medium 10..30, far>20 ; D: close<25, medium 15..35, far>25
REF = {"G": (10., 20., 30.), "D": (15., 25., 35.)}


def ref_mu(x, kind, term):
    a, m, b = REF["D" if kind == "D" else "G"]
    if term == 0: return np.clip((m - x)/(m - a), 0, 1)                              # close (shoulder)
    if term == 2: return np.clip((x - m)/(b - m), 0, 1)                              # far (shoulder)
    return np.clip(np.minimum((x - a)/(m - a), (b - x)/(b - m)), 0, 1)               # medium (triangle)


def ref_mu_matrix(X):
    """X: (n,4) physical inputs in INPUTS order -> (n,4,3)"""
    out = np.empty((len(X), 4, 3))
    for i, nm in enumerate(INPUTS):
        for t in range(3):
            out[:, i, t] = ref_mu(X[:, i], nm, t)
    return out


def infer(mu):
    """mu (n,4,3) -> (mu_yes, mu_no) by Mamdani min (antecedent) and max (group composition)."""
    fire = mu[:, np.arange(4)[None, :], RULES].min(axis=2)       # (n,54)
    return fire[:, RULE_YES].max(1), fire[:, ~RULE_YES].max(1), fire


# ---- defuzzification --------------------------------------------------------
_G = np.linspace(0, 1, 401)
_NO = np.clip((0.75 - _G)/0.5, 0, 1)
_YES = np.clip((_G - 0.25)/0.5, 0, 1)


def centroid(my, mn):
    """D2: centroid of the union of the two truncated output areas (Fig.3 of Rineh & Cheu)."""
    my = np.asarray(my)[:, None]; mn = np.asarray(mn)[:, None]
    u = np.maximum(np.minimum(_YES[None, :], my), np.minimum(_NO[None, :], mn))
    s = u.sum(1)
    return np.where(s > 0, (u*_G[None, :]).sum(1)/np.maximum(s, 1e-12), 0.5)


def decide_D1(my, mn, tau):
    return ((my >= mn) & (my > tau)).astype(int)


def decide_D2(my, mn, tau):
    return (centroid(my, mn) >= tau).astype(int)


def decide_FIS1(my, mn, fire, tau):
    """FIS1 behaviour as characterised by Rineh & Cheu: every rule contributes to one 'yes' output,
    decision = (largest rule output >= tau).  PROXY of the 51-rule model of Balal et al. (2016)."""
    return (fire.max(1) >= tau).astype(int)


def score_ratio(my, mn):
    return my/(my + mn + 1e-6)


# ---- metrics ----------------------------------------------------------------
def metrics(y, yh):
    y = np.asarray(y).astype(int); yh = np.asarray(yh).astype(int)
    TP = int(((y == 1) & (yh == 1)).sum()); TN = int(((y == 0) & (yh == 0)).sum())
    FP = int(((y == 0) & (yh == 1)).sum()); FN = int(((y == 1) & (yh == 0)).sum())
    PY = 100*TP/max(TP + FN, 1) if TP + FN else np.nan
    PN = 100*TN/max(TN + FP, 1) if TN + FP else np.nan
    PO = 100*(TP + TN)/max(len(y), 1)
    BA = np.nanmean([PY, PN]) if not (np.isnan(PY) and np.isnan(PN)) else np.nan
    prec = 100*TP/(TP + FP) if TP + FP else 0.0
    f1 = 100*2*TP/(2*TP + FP + FN) if 2*TP + FP + FN else 0.0
    return dict(PY=PY, PN=PN, PO=PO, BAcc=BA, Prec=prec, F1=f1)
