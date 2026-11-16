"""Experiment V (analytical part) - parameter sensitivity and numerical checks of GMPMF properties."""
import numpy as np, pandas as pd
from .mfs import gmp_mu, count_peaks

x = np.linspace(-6, 6, 4001)


def base():
    return dict(a=[1.0, 0.8], c=[-1.5, 1.5], b=[0.8, 0.8], p=[2.0, 2.0], gamma=1.0)


def sweep(param, values, **kw):
    out = []
    for v in values:
        d = base(); d.update(kw)
        if param == "K":
            K = int(v); cs = np.linspace(-2, 2, K) if K > 1 else [0.0]
            d.update(a=[1.0]*K, c=list(cs), b=[0.8]*K, p=[2.0]*K)
        elif param in ("c2",): d["c"] = [d["c"][0], v]
        elif param in ("b2",): d["b"] = [d["b"][0], v]
        elif param in ("a2",): d["a"] = [d["a"][0], v]
        elif param in ("p",): d["p"] = [v, v]
        elif param in ("gamma",): d["gamma"] = v
        mu, z = gmp_mu(x, **d)
        out.append((v, mu, z))
    return out


def peak_vs_separation(ratios=(1.0, 0.5, 0.25), ps=(2.0, 4.0), deltas=None):
    """N_peak of a 2-kernel GMPMF as function of delta_c=|c2-c1|/(b1+b2)  (Eq.72)."""
    deltas = np.linspace(0.05, 3.0, 60) if deltas is None else deltas
    rows = []
    for p in ps:
        for r in ratios:
            for dc in deltas:
                b = 0.8; sep = dc*(2*b)
                mu, _ = gmp_mu(x, [1.0, r], [-sep/2, sep/2], [b, b], [p, p], 1.0)
                rows.append(dict(p=p, a2_over_a1=r, delta_c=dc, Npeak=count_peaks(mu, 0.01)[0]))
    return pd.DataFrame(rows)


def numerical_checks(seed=0, n=200):
    """Randomised numerical checks of identities stated in the draft. Reports worst-case violations."""
    rng = np.random.default_rng(seed); xs = np.linspace(-8, 8, 20001)
    worst_shift, worst_dmu, max_mu, npar_ok = 0.0, 0.0, 0.0, True
    for _ in range(n):
        K = int(rng.integers(1, 5))
        a = rng.uniform(.1, 3, K); c = rng.uniform(-3, 3, K); b = rng.uniform(.2, 2, K); p = rng.uniform(.8, 5, K)
        g1, g2 = rng.uniform(.2, 3, 2)
        m1, z = gmp_mu(xs, a, c, b, p, g1); m2, _ = gmp_mu(xs, a, c, b, p, g2)
        # (i) gamma does not move the maximiser (Theorem 3 / Corollary 1)
        worst_shift = max(worst_shift, abs(xs[np.argmax(m1)] - xs[np.argmax(z)]), abs(xs[np.argmax(m2)] - xs[np.argmax(z)]))
        # (ii) d mu / d gamma = mu (1-mu) ln z   (Eq.75) vs central finite difference
        h = 1e-5; mp, _ = gmp_mu(xs, a, c, b, p, g1 + h); mm, _ = gmp_mu(xs, a, c, b, p, g1 - h)
        fd = (mp - mm)/(2*h); an = m1*(1 - m1)*np.log(np.maximum(z, 1e-300))
        sel = z > 1e-100
        worst_dmu = max(worst_dmu, np.max(np.abs(fd[sel] - an[sel])))
        # (iii) subnormality: mu < 1 for finite parameters
        max_mu = max(max_mu, m1.max())
    return pd.DataFrame([
        dict(check="argmax(mu) == argmax(z) for random gamma (max |shift| on grid, dx=8e-4)", worst=worst_shift),
        dict(check="d mu/d gamma = mu(1-mu)ln z  (max abs error vs finite difference)", worst=worst_dmu),
        dict(check="max over all trials of max_x mu(x)  (must be < 1)", worst=max_mu),
        dict(check="parameter count N_GMP = 4K+1 for K=1..4", worst=float(all(4*K+1 == n for K, n in zip(range(1, 5), (5, 9, 13, 17))))),
    ])
