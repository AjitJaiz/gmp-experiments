"""Build Tables II, IV-XI (LaTeX + CSV) and Figures 5-8 (+extras) from the raw results."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from . import pipeline as P, exp_representation as E1, exp_params as E5, data as D

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
NICE = {"GMP-2-FIS": "GMP-K-FIS"}
MAIN = ["Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-1-FIS", "GMP-2-FIS"]


def nm(m): return NICE.get(m, m)
def pm(v, d=1):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return "n/a" if len(v) == 0 else (f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}")
def mean(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; return float(v.mean()) if len(v) else np.nan


def tab(path, header, rows, colfmt=None):
    colfmt = colfmt or "l" + "r"*(len(header) - 1)
    s = "\\begin{tabular}{%s}\n\\hline\n%s \\\\\n\\hline\n" % (colfmt, " & ".join(header))
    s += "\n".join(" & ".join(str(c) for c in r) + " \\\\" for r in rows) + "\n\\hline\n\\end{tabular}\n"
    open(path, "w").write(s)
    pd.DataFrame(rows, columns=header).astype(str).apply(lambda c: c.str.replace("$\\pm$", "+-", regex=False)).to_csv(path.replace(".tex", ".csv").replace("/tex/", "/csv/"), index=False)


def stamp(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA - PIPELINE TEST ONLY", ha="center", va="center", rotation=25,
                 fontsize=22, color="red", alpha=0.18, weight="bold")


def build(raw, outdir, synthetic, dsinfo, rep_I=None):
    for sub in ("tex", "csv", "figures"): os.makedirs(f"{outdir}/{sub}", exist_ok=True)
    seeds = sorted(raw); R = len(seeds)
    get = lambda m, k: [raw[s][m][k] for s in seeds if m in raw[s]]
    models = [m for m in raw[seeds[0]] if m != "_n"]
    log = []

    # ---- Table II -------------------------------------------------------------
    r = [["NGSIM Dataset 2A (reference)", 106, 4334, 3739],
         ["Processed development set (this run)", dsinfo["dev_veh"], dsinfo["dev_pos"], dsinfo["dev_neg"]],
         ["External test (this run)", dsinfo.get("ext_veh", "n/a"), dsinfo.get("ext_pos", "n/a"), dsinfo.get("ext_neg", "n/a")]]
    n0 = raw[seeds[0]]["_n"]
    r.append(["Split of seed %d: train / val / test vehicles" % seeds[0], f"{n0['train_veh']}/{n0['val_veh']}/{n0['test_veh']}",
              f"test pos {n0['test_pos']}", f"test neg {n0['test_neg']}"])
    tab(f"{outdir}/tex/tab_II.tex", ["Dataset", "Vehicles", "Positive", "Negative"], r, "lrrr")

    # ---- Table IV (Experiment I) ---------------------------------------------
    if rep_I is not None:
        df, fits, tg = rep_I
        d = df[df.setting == "default"].groupby("family", sort=False).agg(params=("params", "first"), RMSE=("RMSE", "mean"), Emax=("Emax", "mean")).reset_index()
        order = ["Triangular", "Gaussian", "B-spline (M=5)", "Gaussian mixture (M=2)", "GMP (K=1)", "GMP (K=2)", "GMP (K=3)"]
        labels = ["Triangular", "Gaussian", "B-spline ($M{=}5$)", "Gaussian mixture ($M{=}2$)", "GMP, $K=1$", "GMP, $K=2$", "GMP, $K=3$"]
        rows = [[lab, int(d.set_index("family").loc[o, "params"]), f"{d.set_index('family').loc[o, 'RMSE']:.4f}", f"{d.set_index('family').loc[o, 'Emax']:.4f}"] for o, lab in zip(order, labels)]
        tab(f"{outdir}/tex/tab_IV.tex", ["MF family", "Parameters", "RMSE", "$E_{\\max}$"], rows)
        df.to_csv(f"{outdir}/csv/tab_IV_per_target_and_matched9.csv", index=False)
        # Fig 5
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
        for a, (tn, y) in zip(ax, tg.items()):
            a.plot(E1.X, y, "k", lw=2.2, label="target")
            for c, (fam, ls) in zip(OI, zip(order, ["-", "-", "-", "--", "-", "-", "-"])):
                if "K=3" in fam: continue
                a.plot(E1.X, fits[("default", fam, tn)], ls, color=c, lw=1.2, label=fam)
            a.set_title(tn); a.set_xlabel("x")
        ax[0].set_ylabel("membership grade"); ax[2].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle("Fig. 5  Fits of MF families to symmetric / asymmetric / multimodal targets (default budgets)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig5_representation.png", dpi=200); plt.close(fig)

    # ---- Table V ----------------------------------------------------------------
    def row5(m, cols=("PY", "PN", "PO", "BAcc", "Prec", "F1")):
        return [nm(m)] + [pm(get(m, c)) for c in cols]
    rowsV = [row5(m) for m in ["FIS1", "FIS2-D1", "FIS2-D2", "Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-1-FIS", "GMP-2-FIS"] if m in models]
    H = ["Model", "$P_Y$ (\\%)", "$P_N$ (\\%)", "$P_O$ (\\%)", "BAcc (\\%)", "Precision (\\%)", "F1-score (\\%)"]
    tab(f"{outdir}/tex/tab_V.tex", H, rowsV)
    extra = [row5(m) for m in ["GM-FIS(9)", "Bsp-FIS(9)", "GMP-3-FIS", "GMP-4-FIS"] if m in models]
    if extra: tab(f"{outdir}/tex/tab_V_extra_matched_budget_and_K.tex", H, extra)
    # statistics: Holm-corrected paired t-tests over seeds (paired by seed), vehicle-level bootstrap CI
    if R >= 3 and "GMP-2-FIS" in models:
        res = []
        for metric in ("BAcc", "F1"):
            a = np.array(get("GMP-2-FIS", metric), float)
            ps = []
            for o in [m for m in ["Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-1-FIS", "GM-FIS(9)", "Bsp-FIS(9)"] if m in models]:
                b = np.array(get(o, metric), float); t, p = stats.ttest_rel(a, b)
                ps.append((o, float(np.mean(a - b)), p))
            order_ = np.argsort([p for _, _, p in ps]); m_ = len(ps); adj = [None]*m_; run = 0
            for rank, i in enumerate(order_):
                run = max(run, min(1, (m_ - rank)*ps[i][2])); adj[i] = run
            for (o, dlt, p), pa in zip(ps, adj):
                res.append(dict(metric=metric, comparison=f"GMP-K-FIS minus {o}", mean_diff=dlt, p_raw=p, p_holm=pa, n_pairs=R))
        pd.DataFrame(res).to_csv(f"{outdir}/csv/stats_paired_ttests_holm.csv", index=False)
    # Table VI comparisons: does adding A help, and does GMPFM beat the other representations of A?
    if R >= 3:
        pairs = [("GMP-K-FIS-S", "GMP-2-FIS"), ("Gap+speed, Gaussian mixture", "GMP-2-FIS"), ("Gap+speed, single Gaussian", "GMP-2-FIS"),
                 ("GMP-K-FIS-S", "Gap+speed, Gaussian mixture"), ("GMP-K-FIS-S", "Gap+speed, single Gaussian"),
                 ("Gap+4 speeds, GMPFM", "GMP-2-FIS"), ("Gap+4 speeds, GMPFM", "GMP-K-FIS-S"), ("Gap+4 speeds, GMPFM", "Gap+4 speeds, Gaussian mixture"),
                 ("Gap+4 speeds, GMPFM", "Gap+4 speeds, single Gaussian"), ("Gap+4 speeds, Gaussian mixture", "GMP-2-FIS"), ("Gap+4 speeds, single Gaussian", "GMP-2-FIS")]
        res6 = []
        for metric in ("BAcc", "F1"):
            ps = []
            for a_, b_ in [q for q in pairs if q[0] in models and q[1] in models]:
                x, y = np.array(get(a_, metric), float), np.array(get(b_, metric), float); t, p = stats.ttest_rel(x, y)
                ps.append((f"{nm(a_)} minus {nm(b_)}", float(np.mean(x - y)), p))
            order_ = np.argsort([q[2] for q in ps]); m_ = len(ps); adj = [None]*m_; run = 0
            for rank, i in enumerate(order_):
                run = max(run, min(1, (m_ - rank)*ps[i][2])); adj[i] = run
            for (c_, d_, p_), pa in zip(ps, adj):
                res6.append(dict(metric=metric, comparison=c_, mean_diff=d_, p_raw=p_, p_holm=pa, n_pairs=R))
        pd.DataFrame(res6).to_csv(f"{outdir}/csv/stats_tableVI_paired.csv", index=False)
    rng = np.random.default_rng(0); ci = []
    for m in models:
        vc = [raw[s][m]["veh_counts"] for s in seeds]; bs = []
        for _ in range(1000):
            vals = []
            for c in vc:
                c = c[rng.integers(0, len(c), len(c))].sum(0); TP, FN, TN, FP = c
                vals.append(.5*(TP/max(TP + FN, 1) + TN/max(TN + FP, 1))*100)
            bs.append(np.mean(vals))
        ci.append(dict(model=nm(m), BAcc_mean=mean(get(m, "BAcc")), CI95_lo=np.percentile(bs, 2.5), CI95_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(ci).to_csv(f"{outdir}/csv/bootstrap_CI_BAcc_vehicle_level.csv", index=False)
    # full per-model summary (all measures)
    pd.DataFrame([dict(model=nm(m), tau=mean(get(m, "tau")), **{c: mean(get(m, c)) for c in ("PY", "PN", "PO", "BAcc", "Prec", "F1")}) for m in models]).to_csv(f"{outdir}/csv/all_models_summary.csv", index=False)

    # ---- Fig 6: P_N vs P_Y ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for i, m in enumerate([m for m in models if m in ["FIS1", "FIS2-D1", "FIS2-D2"] + MAIN + ["GMP-K-FIS-S"]]):
        py, pn = get(m, "PY"), get(m, "PN")
        ax.errorbar(mean(py), mean(pn), xerr=np.nanstd(py), yerr=np.nanstd(pn), fmt="o", ms=6, color=OI[i % 8], capsize=2, label=nm(m))
    ax.plot(100, 100, "k*", ms=11, label="ideal"); ax.plot([0, 100], [100, 0], ":", color="#999")
    ax.set_xlabel("$P_Y$ (%)"); ax.set_ylabel("$P_N$ (%)"); ax.set_xlim(0, 105); ax.set_ylim(0, 105); ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Fig. 6  Positive vs negative match rate", fontsize=9); stamp(fig, synthetic)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig6_PN_vs_PY.png", dpi=200); plt.close(fig)

    # ---- Table VI ---------------------------------------------------------------
    rowsVI = []
    for lab, m in [("Gap-only FIS (GMP-K MFs)", "GMP-2-FIS"), ("Gap + speed, paper two-set MFs (fixed, Rineh \\& Cheu Fig.6)", "Gap+speed, paper 2-set (fixed)"),
                   ("Gap + speed, conventional MF (Gaussian mixture)", "Gap+speed, Gaussian mixture"),
                   ("Gap + speed, GMPFM", "GMP-K-FIS-S"), ("(extra) Gap + speed, single Gaussian", "Gap+speed, single Gaussian")]:
        if m in models: rowsVI.append([lab] + [pm(get(m, c)) for c in ("PO", "BAcc", "F1")])
    tab(f"{outdir}/tex/tab_VI.tex", ["Configuration", "$P_O$", "BAcc", "F1"], rowsVI, "lrrr")

    rows6b = []
    for lab, m in [("Gap-only FIS (GMP-K MFs)", "GMP-2-FIS"), ("Gap + $\\Delta V_{PO}$ only, GMPFM", "GMP-K-FIS-S"),
                   ("Gap + four relative speeds, paper two-set MFs (fixed)", "Gap+4 speeds, paper 2-set (fixed)"),
                   ("Gap + four relative speeds, single Gaussian", "Gap+4 speeds, single Gaussian"),
                   ("Gap + four relative speeds, Gaussian mixture", "Gap+4 speeds, Gaussian mixture"),
                   ("Gap + four relative speeds, GMPFM", "Gap+4 speeds, GMPFM")]:
        if m in models: rows6b.append([lab] + [pm(get(m, c)) for c in ("PY", "PN", "PO", "BAcc", "F1")])
    if len(rows6b) > 2: tab(f"{outdir}/tex/tab_VI_four_relative_speeds.tex", ["Configuration", "$P_Y$", "$P_N$", "$P_O$", "BAcc", "F1"], rows6b, "lrrrrr")

    # ---- Table VII (regimes) + V_R ---------------------------------------------
    rowsVII = []
    for m in [m for m in MAIN if m in models]:
        rb = np.array(get(m, "regime_BAcc"), float)      # (R,3) cong, trans, free
        cell = [pm(rb[:, k]) for k in (2, 1, 0)]          # Free, Transition, Congested
        vr = np.nanmean(np.sqrt(np.nanmean((rb - np.nanmean(rb, axis=1, keepdims=True))**2, axis=1)))
        rowsVII.append([nm(m)] + cell + [f"{vr:.2f}"])
    tab(f"{outdir}/tex/tab_VII.tex", ["Model", "Free", "Transition", "Congested", "$V_R$"], rowsVII)

    # ---- Table VIII (K sweep) ---------------------------------------------------
    rowsVIII = []
    for K in (1, 2, 3, 4):
        m = f"GMP-{K}-FIS"
        if m in models: rowsVIII.append([K, 4*K + 1, pm(get(m, "BAcc")), pm(get(m, "F1")), pm(get(m, "train_s"), 1)])
    tab(f"{outdir}/tex/tab_VIII.tex", ["$K$", "Params.", "BAcc", "F1", "Train time (s)"], rowsVIII)

    # ---- Table IX / extended (robustness) ---------------------------------------
    rowsIX, rowsIXx = [], []
    for m in [m for m in MAIN if m in models]:
        rb = [raw[s][m]["robust"] for s in seeds]
        g = lambda eta, j: [x[eta][j] for x in rb]
        rowsIX.append([nm(m), pm(g(0.0, 0)), pm(g(0.05, 0)), pm(g(0.10, 0))])
        rowsIXx.append([nm(m)] + [pm(g(e, 0)) for e in P.ETAS] + [pm(100*np.array(g(e, 1)), 1) for e in P.ETAS[1:]])
    tab(f"{outdir}/tex/tab_IX.tex", ["Model", "Clean", "$\\eta=0.05$", "$\\eta=0.10$"], rowsIX, "lrrr")
    tab(f"{outdir}/tex/tab_IX_extended.tex", ["Model"] + [f"BAcc $\\eta$={e}" for e in P.ETAS] + [f"flip\\% $\\eta$={e}" for e in P.ETAS[1:]], rowsIXx)
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for i, m in enumerate([m for m in MAIN if m in models]):
        rb = [raw[s][m]["robust"] for s in seeds]
        ax[0].plot(P.ETAS, [mean([x[e][0] for x in rb]) for e in P.ETAS], "o-", color=OI[i], label=nm(m))
        ax[1].plot(P.ETAS[1:], [100*mean([x[e][1] for x in rb]) for e in P.ETAS[1:]], "o-", color=OI[i])
    ax[0].set_xlabel("noise level $\\eta$"); ax[0].set_ylabel("BAcc (%)"); ax[1].set_xlabel("noise level $\\eta$"); ax[1].set_ylabel("decision flip rate (%)")
    ax[0].legend(fontsize=7, frameon=False); fig.suptitle("Extra figure  Robustness to input noise (Exp. VI)", fontsize=9); stamp(fig, synthetic)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/figX_robustness.png", dpi=200); plt.close(fig)

    # ---- Table X (complexity) ---------------------------------------------------
    rowsX = []
    for m in [m for m in ["Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-1-FIS", "GMP-2-FIS", "GMP-K-FIS-S"] if m in models]:
        rowsX.append([nm(m), int(mean(get(m, "n_params"))), f"81 / 54 / {mean(get(m, 'active_rules')):.0f}",
                      f"{mean(get(m, 't_avg_ms')):.4f}", f"{mean(get(m, 't_p95_ms')):.3f}", f"{mean(get(m, 'train_s')):.1f}"])
    tab(f"{outdir}/tex/tab_X.tex", ["Model", "Params.", "Rules (potential / feasible / active)", "Time (ms, batch avg)", "p95 latency (ms, single)", "Train (s)"], rowsX, "lrrrrr")

    # ---- Table XI (external) ----------------------------------------------------
    rowsXI = []
    if all(("ext" in raw[seeds[0]][m]) for m in MAIN if m in models):
        for m in [m for m in ["Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-2-FIS"] if m in models]:
            i_, e_ = np.array(get(m, "BAcc"), float), np.array([x["BAcc"] for x in get(m, "ext")], float)
            rowsXI.append([nm(m), pm(i_), pm(e_), pm(i_ - e_)])
        tab(f"{outdir}/tex/tab_XI.tex", ["Model", "NGSIM (BAcc)", "External (BAcc)", "Gap $G_M$"], rowsXI, "lrrr")

    # ---- Fig 8 (threshold sweep) -----------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for i, m in enumerate([m for m in MAIN if m in models]):
        for a, key in zip(ax, ("PY", "PN", "PO")):
            a.plot(P.TAU_GRID, [mean([raw[s][m]["sweep"][float(t)][key] for s in seeds]) for t in P.TAU_GRID], color=OI[i], label=nm(m), lw=1.4)
    for a, key in zip(ax, ("$P_Y$", "$P_N$", "$P_O$")): a.set_xlabel("threshold $\\tau$"); a.set_title(key + " (%)")
    ax[0].legend(fontsize=7, frameon=False); fig.suptitle("Fig. 8  Decision match rates vs threshold (test set)", fontsize=9); stamp(fig, synthetic)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig8_threshold.png", dpi=200); plt.close(fig)
    rowsVII_tau = [[nm(m), f"{mean(get(m, 'tau')):.2f}"] for m in models]
    tab(f"{outdir}/tex/tab_VII_selected_tau_on_validation.tex", ["Model", "Selected $\\tau$ (validation)"], rowsVII_tau, "lr")

    # ---- Fig 7 + checks (Exp V analytical part) ---------------------------------
    chk = E5.numerical_checks(); chk.to_csv(f"{outdir}/csv/numerical_checks.csv", index=False)
    x = E5.x; fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
    panels = [("K", [1, 2, 3, 4], "number of kernels $K$"), ("c2", [0.5, 1.0, 1.5, 2.5, 3.5], "second centre $c_2$"),
              ("b2", [0.3, 0.5, 0.8, 1.2], "second width $b_2$"), ("a2", [0.2, 0.5, 1.0, 2.0], "second strength $a_2$"),
              ("p", [1.0, 2.0, 4.0, 8.0], "shape $p_i$"), ("gamma", [0.3, 0.6, 1.0, 2.0, 4.0], "global $\\gamma$")]
    for a, (par, vals, ttl) in zip(ax.ravel(), panels):
        for c, (v, mu, z) in zip(OI, E5.sweep(par, vals)):
            a.plot(x, mu, color=c, lw=1.3, label=f"{v}")
        a.set_title(ttl, fontsize=9); a.legend(fontsize=7, frameon=False, ncol=2); a.set_ylim(0, 1)
    ax[1, 0].set_xlabel("x"); fig.suptitle("Fig. 7  Parameter sensitivity of the GMPMF (analytical, base: K=2, c=±1.5, b=0.8, a=(1,0.8), p=2, γ=1)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig7_sensitivity.png", dpi=200); plt.close(fig)
    pk = E5.peak_vs_separation(); pk.to_csv(f"{outdir}/csv/Npeak_vs_delta_c.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); k = 0
    for (p, r), g in pk.groupby(["p", "a2_over_a1"]):
        ax.step(g.delta_c, g.Npeak, where="post", color=OI[k % 8], ls="-" if p == 2 else "--", label=f"p={p:g}, a2/a1={r:g}"); k += 1
    ax.set_xlabel("$\\delta_c=|c_2-c_1|/(b_1+b_2)$"); ax.set_ylabel("$N_{peak}$"); ax.set_yticks([1, 2, 3]); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Kernels vs peaks (check of Fig. 2 claim: $K \\neq N_{peak}$)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/figX_Npeak_vs_deltac.png", dpi=200); plt.close(fig)

    # ---- extra figure: the speed-adjustment concept A learned by each representation ------------
    sv = dsinfo.get("speed_var", "v_des"); xl = {"dV_PO": "$\\Delta V_{PO}$ (m/s), negative = closing, positive = moving away"}.get(sv, "$v_{des}-v_{ego}$ (m/s)")
    cand = [m for m in ["Gap+speed, paper 2-set (fixed)", "Gap+speed, single Gaussian", "Gap+speed, Gaussian mixture", "GMP-K-FIS-S"] if m in models]
    if cand:
        R_ = P.DV_RANGE; dv = np.linspace(-R_, R_, 601); fig, ax = plt.subplots(1, len(cand), figsize=(3.1*len(cand), 3.0), sharey=True)
        ax = np.atleast_1d(ax)
        for a, m in zip(ax, cand):
            fam = P.SPECS[m][1]()
            for s_ in seeds:
                th = raw[s_][m].get("theta")
                if th is None: continue
                a.plot(dv, fam.mu((dv + R_)/(2*R_), th[12*P.SPECS[m][0]().n:]), color=OI[0], alpha=.35, lw=1)
            if fam.n == 0: a.plot(dv, fam.mu((dv + R_)/(2*R_), None), color=OI[1], lw=2)
            a.set_title(nm(m).replace("Gap+speed, ", ""), fontsize=8); a.set_xlabel(xl if a is ax[len(ax)//2] else "", fontsize=7); a.set_ylim(0, 1.02)
        ax[0].set_ylabel("A: significant need for speed adjustment")
        fig.suptitle("Extra figure  Shape of A learned by each MF family (one line per seed)", fontsize=9); stamp(fig, synthetic)
        fig.tight_layout(); fig.savefig(f"{outdir}/figures/figX_A_profiles.png", dpi=200); plt.close(fig)

    m4 = "Gap+4 speeds, GMPFM"
    if m4 in models:
        R_ = P.DV_RANGE; dv = np.linspace(-R_, R_, 601); fam = P.SPECS[m4][1](); k0 = 12*P.SPECS[m4][0]().n
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
        for j, (a, t_) in enumerate(zip(ax, ["$\\Delta V_{PO}$", "$\\Delta V_{PT}$", "$\\Delta V_{FT}$", "$\\Delta V_{FPT}$"])):
            for s_ in seeds:
                th = raw[s_][m4]["theta"]; a.plot(dv, fam.mu((dv + R_)/(2*R_), th[k0 + j*fam.n: k0 + (j + 1)*fam.n]), color=OI[0], alpha=.3, lw=1)
            a.set_title(t_ + " (m/s)", fontsize=9)
        ax[0].set_ylabel("A_j (GMPFM)"); fig.suptitle("Extra figure  Learned concept A_j for each relative speed (negative = gap closing)", fontsize=9); stamp(fig, synthetic)
        fig.tight_layout(); fig.savefig(f"{outdir}/figures/figX_A_profiles_four_speeds.png", dpi=200); plt.close(fig)

    # ---- master include + markdown summary --------------------------------------
    names = ["II", "IV", "V", "V_extra_matched_budget_and_K", "VI", "VI_four_relative_speeds", "VII", "VIII", "IX", "IX_extended", "X", "XI"]
    with open(f"{outdir}/tex/all_tables.tex", "w") as f:
        f.write("% Auto-generated. Needs \\usepackage{booktabs} not required; plain \\hline tables.\n")
        for n in names:
            if os.path.exists(f"{outdir}/tex/tab_{n}.tex"):
                f.write(f"\n% ---- Table {n}\n\\begin{{table}}[t]\\centering\\caption{{Table {n}}}\\small\n\\input{{tex/tab_{n}.tex}}\n\\end{{table}}\n")
    return chk
