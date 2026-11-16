import numpy as np
from scipy.optimize import least_squares


def fit_profile(fam, x, y, restarts=30, rng=None, init=None, max_nfev=300):
    """Least-squares fit of one MF family to a target profile; multi-start. Returns (theta, rmse, emax)."""
    rng = rng or np.random.default_rng(0)
    best = None
    starts = [] if init is None else [np.clip(init, fam.lb, fam.ub)]
    starts += [fam.random_init(rng) for _ in range(restarts)]
    for th0 in starts:
        try:
            r = least_squares(lambda th: fam.mu(x, th) - y, th0, bounds=(fam.lb, fam.ub), max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    th = best.x; e = fam.mu(x, th) - y
    return th, float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))
