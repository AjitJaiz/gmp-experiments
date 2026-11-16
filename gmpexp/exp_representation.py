"""Experiment I - membership-function representation (no traffic data needed)."""
import numpy as np, pandas as pd
from .mfs import Tri, Gau, GaussMix, BSp, GMP, count_peaks
from .fit import fit_profile

X = np.linspace(-3, 3, 601)


def targets():
    """Analytical targets that are NOT members of any compared family."""
    sym = 0.95*np.where(np.abs(X) < 2, 0.5*(1 + np.cos(np.pi*X/2)), 0.0)          # raised cosine
    u = np.clip(X + 2.5, 0, None); g = u**2*np.exp(-1.6*u); asym = 0.9*g/g.max()  # skewed gamma-like
    lor = lambda a, c, w: a/(1 + ((X - c)/w)**2)
    multi = np.clip(lor(.9, -1.2, .5) + lor(.65, 1.5, .6), 0, 1)                   # two Lorentzians
    return {"symmetric": sym, "asymmetric": asym, "multimodal": multi}


def families(budget="default"):
    lo, hi = X.min(), X.max()
    if budget == "default":
        return [Tri(lo, hi), Gau(lo, hi), BSp(5, 3, lo, hi), GaussMix(2, lo, hi),
                GMP(1, lo, hi), GMP(2, lo, hi), GMP(3, lo, hi)]
    # matched parameter budget (9 parameters)
    return [GaussMix(3, lo, hi), BSp(9, 3, lo, hi), GMP(2, lo, hi)]


def run(restarts=40, seed=0):
    rows, fits = [], {}
    tg = targets()
    for setting in ("default", "matched9"):
        for fam in families("default" if setting == "default" else "matched"):
            for tname, y in tg.items():
                rng = np.random.default_rng(seed)
                th, rmse, emax = fit_profile(fam, X, y, restarts=restarts, rng=rng)
                yh = fam.mu(X, th); npk, _ = count_peaks(yh)
                rows.append(dict(setting=setting, family=fam.name, params=fam.n, target=tname,
                                 RMSE=rmse, Emax=emax, Npeak_fit=npk, Npeak_target=count_peaks(y)[0]))
                fits[(setting, fam.name, tname)] = yh
    return pd.DataFrame(rows), fits, tg
