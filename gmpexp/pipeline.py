"""Experiments II, III, IV, VI, VII, VIII, IX: controlled FIS comparison.

Fixed across all controlled models: 4 gap inputs, 3 terms each, the 54-rule base of Rineh & Cheu (2025),
Mamdani min/max inference, score = mu_yes/(mu_yes+mu_no), decision = score >= tau (tau picked on validation).
Only the membership-function family (and its trained parameters) changes.
Calibration: CMA-ES, identical evaluation budget for every model, objective Eq.(51) = BCE + lambda*Omega.
"""
import time, warnings
import numpy as np, cma
from . import fis, data as D
from .mfs import Tri, Gau, GaussMix, BSp, GMP, PaperTwoSet
from .fit import fit_profile

warnings.filterwarnings("ignore")
TAU_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
REF_TAU_GRID = np.round(np.arange(0.20, 0.5001, 0.01), 2)
ETAS = [0.0, 0.01, 0.03, 0.05, 0.10]
DV_RANGE = 15.0     # bounded operating domain of the speed-adjustment variable, m/s

# name -> (gap family factory, speed family factory or None)
SPECS = {
    "Tri-FIS":        (lambda: Tri(0, 1),        None),
    "Gau-FIS":        (lambda: Gau(0, 1),        None),
    "Bsp-FIS":        (lambda: BSp(5, 3, 0, 1),  None),
    "GM-FIS(9)":      (lambda: GaussMix(3, 0, 1), None),     # matched 9 params / MF
    "Bsp-FIS(9)":     (lambda: BSp(9, 3, 0, 1),  None),      # matched 9 params / MF
    "GMP-1-FIS":      (lambda: GMP(1, 0, 1),     None),
    "GMP-2-FIS":      (lambda: GMP(2, 0, 1),     None),      # = GMP-K-FIS (K=2)
    "GMP-3-FIS":      (lambda: GMP(3, 0, 1),     None),
    "GMP-4-FIS":      (lambda: GMP(4, 0, 1),     None),
    "GMP-K-FIS-S":    (lambda: GMP(2, 0, 1),     lambda: GMP(2, 0, 1)),
    "Gap+speed, Gaussian mixture": (lambda: GMP(2, 0, 1), lambda: GaussMix(2, 0, 1)),
    "Gap+speed, single Gaussian":  (lambda: GMP(2, 0, 1), lambda: Gau(0, 1)),
    "Gap+speed, paper 2-set (fixed)": (lambda: GMP(2, 0, 1), lambda: PaperTwoSet(DV_RANGE)),
    # all four relative speeds dV_PO, dV_PT, dV_FT, dV_FPT, each with its own concept A_j (same family for all four)
    "Gap+4 speeds, GMPFM":             (lambda: GMP(2, 0, 1), lambda: GMP(2, 0, 1), 4),
    "Gap+4 speeds, Gaussian mixture":  (lambda: GMP(2, 0, 1), lambda: GaussMix(2, 0, 1), 4),
    "Gap+4 speeds, single Gaussian":   (lambda: GMP(2, 0, 1), lambda: Gau(0, 1), 4),
    "Gap+4 speeds, paper 2-set (fixed)": (lambda: GMP(2, 0, 1), lambda: PaperTwoSet(DV_RANGE), 4),
}
REF_MODELS = ["FIS1", "FIS2-D1", "FIS2-D2"]


# ------------------------------------------------------------------ normalisation / forward
class Norm:
    def __init__(self, X):                       # X: (n,4) physical, train only
        self.cap = np.maximum(np.quantile(X, 0.99, axis=0), 1.0)
        self.sd = X.std(0)
    def gap(self, X):
        return np.clip(X/self.cap, 0, 1)
    @staticmethod
    def dv(aux, nsv=1):
        """(n, nsv) normalised speed variables: var 0 = aux1 - aux0 ; var j>=1 = aux[:, 2+j] (extra relative speeds)."""
        out = np.empty((len(aux), nsv))
        out[:, 0] = (aux[:, 1] - aux[:, 0] + DV_RANGE)/(2*DV_RANGE)
        for j in range(1, nsv):
            out[:, j] = (aux[:, 2 + j] + DV_RANGE)/(2*DV_RANGE)
        return np.clip(out, 0, 1)


def _prep(fam, x):
    return fam.basis(x) if isinstance(fam, BSp) else x


def _ev(fam, pre, th):
    if isinstance(fam, BSp):
        return np.clip(pre @ np.clip(th, 0, 1), 0, 1)
    return fam.mu(pre, th)


class Model:
    def __init__(self, name, gap_fam, sp_fam, norm, nsv=1):
        self.name, self.gf, self.sf, self.norm, self.nsv = name, gap_fam, sp_fam, norm, nsv
        self.ng = 12*gap_fam.n; self.ns = nsv*sp_fam.n if sp_fam else 0
        self.n_params = self.ng + self.ns
    def pre(self, Xn, dvn):
        p = [_prep(self.gf, np.ascontiguousarray(Xn[:, i])) for i in range(4)]
        return p, ([_prep(self.sf, np.ascontiguousarray(dvn[:, j])) for j in range(self.nsv)] if self.sf else None)
    def forward(self, th, pre):
        pg, ps = pre; n = len(pg[0] if not isinstance(self.gf, BSp) else pg[0]); k = self.gf.n
        n = pg[0].shape[0]
        mu = np.empty((n, 4, 3))
        for i in range(4):
            for t in range(3):
                o = (3*i + t)*k
                mu[:, i, t] = _ev(self.gf, pg[i], th[o:o + k])
        fire = mu[:, 0, fis.RULES[:, 0]]
        for i in range(1, 4):
            fire = np.minimum(fire, mu[:, i, fis.RULES[:, i]])
        my, mn = fire[:, fis.RULE_YES].max(1), fire[:, ~fis.RULE_YES].max(1)
        if self.sf is not None:
            k2 = self.sf.n
            for j in range(self.nsv):     # one learned concept A_j per relative speed; all must hold for 'yes'
                A = _ev(self.sf, ps[j], th[self.ng + j*k2: self.ng + (j + 1)*k2])
                my, mn = np.minimum(my, A), np.maximum(mn, 1 - A)
        return fis.score_ratio(my, mn), fire
    def score(self, th, X, aux):
        Xn = self.norm.gap(X); p = self.pre(Xn, Norm.dv(aux, self.nsv)); return self.forward(th, p)


def _init_theta(model, rng, restarts=6):
    """Fit every MF to the reference (Table-1) profile so all families start from the same shape."""
    th = []; g = np.linspace(0, 1, 101); cap = model.norm.cap
    for i, nm in enumerate(fis.INPUTS):
        for t in range(3):
            y = fis.ref_mu(g*cap[i], nm, t)
            th.append(fit_profile(model.gf, g, y, restarts=restarts, rng=rng, max_nfev=150)[0])
    if model.sf is not None and model.sf.n > 0:
        dv = np.linspace(-DV_RANGE, DV_RANGE, 121); gx = (dv + DV_RANGE)/(2*DV_RANGE)
        R_ = DV_RANGE; y = 0.8*np.exp(-((dv + .4*R_)/(.2*R_))**2) + 0.8*np.exp(-((dv - .4*R_)/(.2*R_))**2)   # same start for every family
        for _ in range(model.nsv):
            th.append(fit_profile(model.sf, gx, np.minimum(y, 1), restarts=restarts, rng=rng, max_nfev=150)[0])
    return np.concatenate(th)


def _bce(s, y):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return -np.mean(y*np.log(s) + (1 - y)*np.log(1 - s))


def train(model, Xtr, auxtr, ytr, seed, evals, lam=0.01, max_train=3000):
    rng = np.random.default_rng(seed)
    if len(ytr) > max_train:
        idx = rng.choice(len(ytr), max_train, replace=False); Xtr, auxtr, ytr = Xtr[idx], auxtr[idx], ytr[idx]
    t0 = time.time()
    th0 = _init_theta(model, rng)
    lb = np.concatenate([np.tile(model.gf.lb, 12)] + ([np.tile(model.sf.lb, model.nsv)] if model.sf else []))
    ub = np.concatenate([np.tile(model.gf.ub, 12)] + ([np.tile(model.sf.ub, model.nsv)] if model.sf else []))
    th0 = np.clip(th0, lb, ub); sc = ub - lb
    pre = model.pre(model.norm.gap(Xtr), Norm.dv(auxtr, model.nsv))
    def f(u):
        th = lb + np.asarray(u)*sc
        s, _ = model.forward(th, pre)
        return _bce(s, ytr) + lam*np.mean(((th - th0)/sc)**2)                 # Eq.(51): L_cls + lambda*Omega
    u0 = (th0 - lb)/sc
    es = cma.CMAEvolutionStrategy(u0, 0.08, dict(bounds=[0, 1], maxfevals=evals, seed=seed + 1, verbose=-9))
    es.optimize(f)
    th = lb + np.clip(es.result.xbest, 0, 1)*sc
    return th, time.time() - t0


def best_tau(score, y, grid):
    best, bt = -1, 0.5
    for t in grid:
        b = fis.metrics(y, (score >= t).astype(int))["BAcc"]
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and abs(t - .5) < abs(bt - .5)):
            best, bt = b, t
    return bt


# ------------------------------------------------------------------ reference models (no training)
def ref_score(name, X):
    mu = fis.ref_mu_matrix(X); my, mn, fire = fis.infer(mu)
    if name == "FIS1": return fire.max(1), fire
    if name == "FIS2-D1": return np.where(my >= mn, my, 0.0), fire
    if name == "FIS2-D2": return fis.centroid(my, mn), fire
    raise KeyError(name)


# ------------------------------------------------------------------ one seed, all models
def run_seed(args):
    df, ext, seed, cfg = args
    global DV_RANGE
    DV_RANGE = cfg.get('dv_range', DV_RANGE)        # workers are separate processes on Windows
    t_all = time.time()
    tr, va, te = D.split_by_vehicle(df, seed)
    edges = D.regime_edges(tr)
    sv = cfg.get('speed_var', 'v_des')
    Xtr, atr = D.raw_matrix(tr, sv); Xva, ava = D.raw_matrix(va, sv); Xte, ate = D.raw_matrix(te, sv)
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values
    reg_te = D.regime_of(te, edges)
    if ext is not None:
        Xex, aex = D.raw_matrix(ext, sv); yex = ext.label.values
    norm = Norm(Xtr)
    out = {}

    def finish(name, sc_fn, tau, grid_sweep=True, extra=None):
        s_te, fire = sc_fn(Xte, ate)
        yh = (s_te >= tau).astype(int); m = fis.metrics(yte, yh)
        r = dict(seed=seed, model=name, tau=tau, **m)
        r["regime_BAcc"] = [fis.metrics(yte[reg_te == k], yh[reg_te == k])["BAcc"] if (reg_te == k).any() else np.nan for k in range(3)]
        r["sweep"] = {float(t): fis.metrics(yte, (s_te >= t).astype(int)) for t in TAU_GRID}
        # per-vehicle confusion counts for the vehicle-level bootstrap
        vc = te.assign(yh=yh).groupby("vehicle_id").apply(lambda g: np.array([
            ((g.label == 1) & (g.yh == 1)).sum(), ((g.label == 1) & (g.yh == 0)).sum(),
            ((g.label == 0) & (g.yh == 0)).sum(), ((g.label == 0) & (g.yh == 1)).sum()]))
        r["veh_counts"] = np.stack(vc.values)
        # robustness (Exp VI)
        rng = np.random.default_rng(5000 + seed); rob = {}
        s0 = (s_te >= tau).astype(int)
        for eta in ETAS:
            reps = 1 if eta == 0 else cfg["noise_reps"]; bs, fl = [], []
            for _ in range(reps):
                Xp = Xte.copy(); ap = ate.copy()
                if eta > 0:
                    for j in (0, 1, 3):   # G_FA, G_PA, G_PB perturbed; D re-derived by Eq.(37)
                        Xp[:, j] = np.clip(Xp[:, j] + rng.normal(0, eta*norm.sd[j], len(Xp)), 0, D.GAP_CAP)
                    Xp[:, 2] = Xp[:, 1] + Xp[:, 0] + ate[:, 2]
                    ap[:, 0] = ap[:, 0] + rng.normal(0, eta*atr[:, 0].std(), len(ap))
                    ap[:, 1] = ap[:, 1] + rng.normal(0, eta*atr[:, 1].std(), len(ap))
                    for jj in range(3, ap.shape[1]):
                        ap[:, jj] = ap[:, jj] + rng.normal(0, eta*atr[:, jj].std(), len(ap))
                sp, _ = sc_fn(Xp, ap); yp = (sp >= tau).astype(int)
                bs.append(fis.metrics(yte, yp)["BAcc"]); fl.append(np.mean(yp != s0))
            rob[eta] = (float(np.mean(bs)), float(np.mean(fl)))
        r["robust"] = rob
        # external generalisation (Exp IX): no re-calibration, tau from internal validation
        if ext is not None:
            se, _ = sc_fn(Xex, aex); r["ext"] = fis.metrics(yex, (se >= tau).astype(int))
        # complexity (Exp VIII)
        t0 = time.time(); sc_fn(Xte, ate); tb = time.time() - t0
        lat = []
        for k in range(min(150, len(Xte))):
            t0 = time.perf_counter(); sc_fn(Xte[k:k + 1], ate[k:k + 1]); lat.append((time.perf_counter() - t0)*1e3)
        r["t_avg_ms"] = tb/len(Xte)*1e3; r["t_p95_ms"] = float(np.percentile(lat, 95))
        r["active_rules"] = int(((fire > 0.05).mean(0) >= 0.01).sum())
        if extra: r.update(extra)
        out[name] = r

    # reference models
    for nm in REF_MODELS:
        s_va, _ = ref_score(nm, Xva)
        tau = best_tau(s_va, yva, REF_TAU_GRID)
        finish(nm, lambda X, a, nm=nm: ref_score(nm, X), tau, extra=dict(n_params=0, train_s=0.0))
    # trained models
    for nm in cfg["models"]:
        spec = SPECS[nm]; gfac, sfac = spec[0], spec[1]; nsv = spec[2] if len(spec) > 2 else 1
        mdl = Model(nm, gfac(), sfac() if sfac else None, norm, nsv)
        th, ts = train(mdl, Xtr, atr, ytr, seed, cfg["evals"])
        s_va, _ = mdl.score(th, Xva, ava)
        tau = best_tau(s_va, yva, TAU_GRID)
        finish(nm, lambda X, a, mdl=mdl, th=th: mdl.score(th, X, a), tau,
               extra=dict(n_params=mdl.n_params, train_s=ts, theta=th))
    out["_n"] = dict(train_vec=len(tr), val_vec=len(va), test_vec=len(te), train_veh=tr.vehicle_id.nunique(),
                     val_veh=va.vehicle_id.nunique(), test_veh=te.vehicle_id.nunique(),
                     test_pos=int(yte.sum()), test_neg=int((1 - yte).sum()), edges=edges, secs=time.time() - t_all)
    return out
