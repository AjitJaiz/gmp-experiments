"""Membership-function families used in the comparison.

Every family exposes:  n (params per MF), lb/ub (bounds of the raw parameter vector),
mu(x, theta) -> membership grades in [0,1], random_init(rng) -> theta.
Positive quantities (widths, strengths, shapes, gamma) are stored as logs, so that
CMA-ES / least squares can work in an unconstrained-ish, bounded space.
"""
import numpy as np
from scipy.interpolate import BSpline


class Family:
    name = "base"
    n = 0
    def __init__(self, lo=0.0, hi=1.0):
        self.lo, self.hi = float(lo), float(hi)
        self.R = self.hi - self.lo
    def random_init(self, rng):
        return rng.uniform(self.lb, self.ub)


class Tri(Family):
    """Eq.(47): max{1-|x-c|/b, 0}.  theta=[c, log b]"""
    name = "Triangular"; n = 2
    def __init__(self, lo=0, hi=1):
        super().__init__(lo, hi)
        self.lb = np.array([lo - .25*self.R, np.log(.02*self.R)])
        self.ub = np.array([hi + .25*self.R, np.log(2*self.R)])
    def mu(self, x, th):
        return np.maximum(1 - np.abs(x - th[0]) / np.exp(th[1]), 0.0)


class Gau(Family):
    """Eq.(48): exp(-(x-c)^2/(2 s^2)).  theta=[c, log s]"""
    name = "Gaussian"; n = 2
    def __init__(self, lo=0, hi=1):
        super().__init__(lo, hi)
        self.lb = np.array([lo - .25*self.R, np.log(.01*self.R)])
        self.ub = np.array([hi + .25*self.R, np.log(1.5*self.R)])
    def mu(self, x, th):
        return np.exp(-(x - th[0])**2 / (2*np.exp(th[1])**2))


class GaussMix(Family):
    """Eq.(50): min{1, sum_j w_j exp(-(x-c_j)^2/(2 s_j^2))}.  theta=[w,c,log s]*M"""
    def __init__(self, M=2, lo=0, hi=1):
        super().__init__(lo, hi)
        self.M = M; self.n = 3*M; self.name = f"Gaussian mixture (M={M})"
        self.lb = np.tile([0.0, lo - .25*self.R, np.log(.01*self.R)], M)
        self.ub = np.tile([1.0, hi + .25*self.R, np.log(1.5*self.R)], M)
    def mu(self, x, th):
        t = th.reshape(self.M, 3); s = 0.0
        for w, c, ls in t:
            s = s + w*np.exp(-(x - c)**2 / (2*np.exp(ls)**2))
        return np.minimum(1.0, s)


class BSp(Family):
    """Eq.(49): sum_j w_j B_{j,d}(x), w_j in [0,1], uniform clamped knots on [lo,hi]."""
    def __init__(self, M=5, d=3, lo=0, hi=1):
        super().__init__(lo, hi)
        assert M >= d + 1
        self.M, self.d = M, d; self.n = M; self.name = f"B-spline (M={M})"
        interior = np.linspace(lo, hi, M - d + 1)[1:-1]
        self.t = np.r_[[lo]*(d+1), interior, [hi]*(d+1)]
        self.lb = np.zeros(M); self.ub = np.ones(M)
    def basis(self, x):
        xc = np.clip(x, self.lo, self.hi - 1e-9)
        return BSpline.design_matrix(xc, self.t, self.d).toarray()
    def mu(self, x, th):
        return np.clip(self.basis(np.asarray(x, float)) @ np.clip(th, 0, 1), 0, 1)


class PaperTwoSet(Family):
    """Fixed (untrained) two-set relative-speed fuzzy sets of Rineh & Cheu (2026), Fig.6a:
    'closing' (1 below -5.95, 0 above -3.35) and 'moving away' (0 below 3.35, 1 above 5.95).
    A(dV) = max(closing, moving away): high only for a large |dV|, low in the middle band."""
    name = "Paper 2-set (fixed)"; n = 0
    def __init__(self, R=15.0):
        super().__init__(0, 1); self.Rv = R
        self.lb = np.array([]); self.ub = np.array([])
    def mu(self, x, th):
        dv = np.asarray(x)*2*self.Rv - self.Rv
        cl = np.clip((-3.35 - dv)/(5.95 - 3.35), 0, 1); aw = np.clip((dv - 3.35)/(5.95 - 3.35), 0, 1)
        return np.maximum(cl, aw)


class GMP(Family):
    """Eqs.(43)-(44).  theta=[log a, c, log b, log p]*K + [log gamma]  ->  N=4K+1."""
    def __init__(self, K=2, lo=0, hi=1):
        super().__init__(lo, hi)
        self.K = K; self.n = 4*K + 1; self.name = f"GMP (K={K})"
        kb = [np.log(1e-2), lo - .25*self.R, np.log(.01*self.R), np.log(.5)]
        ku = [np.log(1e2),  hi + .25*self.R, np.log(2*self.R),   np.log(8.)]
        self.lb = np.r_[np.tile(kb, K), np.log(.1)]
        self.ub = np.r_[np.tile(ku, K), np.log(10.)]
    @staticmethod
    def z_of(x, th, K):
        z = 0.0
        for k in range(K):
            la, c, lb, lp = th[4*k:4*k+4]
            z = z + np.exp(la) * np.exp(-np.abs((x - c)/np.exp(lb))**np.exp(lp))
        return z
    def mu(self, x, th):
        z = self.z_of(x, th, self.K)
        zg = np.maximum(z, 1e-300)**np.exp(th[-1])
        return zg / (1.0 + zg)


def gmp_mu(x, a, c, b, p, gamma):
    """Plain-parameter GMPMF (used by the analytical experiments)."""
    a, c, b, p = map(np.atleast_1d, (a, c, b, p))
    z = sum(ai*np.exp(-np.abs((x - ci)/bi)**pi) for ai, ci, bi, pi in zip(a, c, b, p))
    zg = np.maximum(z, 1e-300)**gamma
    return zg/(1+zg), z


def count_peaks(y, prom=0.01):
    from scipy.signal import find_peaks
    yy = np.r_[y[0]-1e-9, y, y[-1]-1e-9] if False else y
    pk, _ = find_peaks(yy, prominence=prom*max(np.max(yy), 1e-12))
    return len(pk), pk
