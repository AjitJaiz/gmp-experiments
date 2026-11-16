#!/usr/bin/env python3
"""One command: runs Experiments I-IX and writes tables (LaTeX+CSV) and figures.

  python run_all.py --data ngsim_processed.csv --external highd_processed.csv     # real run
  python run_all.py --data synthetic --external synthetic --quick                 # pipeline self-test
"""
import argparse, os, pickle, sys, time, multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpexp import data as D, pipeline as P, report, exp_representation as E1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (see README) or 'synthetic'")
    ap.add_argument("--external", default="none", help="processed external CSV, 'synthetic' or 'none'")
    ap.add_argument("--seeds", type=int, default=10, help="R repeated runs (different vehicle splits + CMA seeds)")
    ap.add_argument("--evals", type=int, default=2500, help="CMA-ES evaluation budget per model (identical for all)")
    ap.add_argument("--noise_reps", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--restarts_I", type=int, default=40)
    ap.add_argument("--models", default=None, help="comma-separated subset of trained models, e.g. Gau-FIS,Bsp-FIS,GMP-2-FIS")
    ap.add_argument("--speed_var", default=None, help="dV_PO (default if present in the CSV) or v_des")
    ap.add_argument("--dv_range", default="15", help="half-width (m/s) of the relative-speed domain, a number or 'auto' (= 3 x std of dV_PO, between 3 and 15);")
    ap.add_argument("--_unused", default=None, help="half-width (m/s) of the speed-adjustment variable domain; relative speeds in NGSIM are small, try 6")
    ap.add_argument("--report_only", action="store_true", help="rebuild tables/figures/stats from an existing <out>/raw_results.pkl (no retraining)")
    ap.add_argument("--quick", action="store_true", help="tiny settings, only to test the code")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick: a.seeds, a.evals, a.noise_reps, a.restarts_I = 3, 400, 3, 8
    synthetic = (a.data == "synthetic")
    out = a.out or ("results_SYNTHETIC" if synthetic else "results")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    print("== Experiment I (representation) =="); rep = E1.run(restarts=a.restarts_I)
    df = D.synthetic(400 if not a.quick else 150, 0) if synthetic else D.load_csv(a.data)
    ext = None
    if a.external == "synthetic": ext = D.synthetic(150, 77, shift=1.3)
    elif a.external != "none": ext = D.load_csv(a.external)
    dsinfo = dict(dev_veh=df.vehicle_id.nunique(), dev_pos=int(df.label.sum()), dev_neg=int((1 - df.label).sum()))
    if ext is not None: dsinfo.update(ext_veh=ext.vehicle_id.nunique(), ext_pos=int(ext.label.sum()), ext_neg=int((1 - ext.label).sum()))
    print("dev set:", dsinfo)
    models = list(P.SPECS) if not a.quick else ["Tri-FIS", "Gau-FIS", "Bsp-FIS", "GMP-1-FIS", "GMP-2-FIS", "GMP-K-FIS-S", "Gap+speed, Gaussian mixture", "Gap+speed, paper 2-set (fixed)", "Gap+4 speeds, GMPFM", "Gap+4 speeds, Gaussian mixture"]
    if a.models: models = [m.strip() for m in a.models.split(";" if ";" in a.models else ",")]
    sv = a.speed_var or ("dV_PO" if "dV_PO" in df.columns else "v_des")
    print("speed-adjustment variable:", sv); dsinfo["speed_var"] = sv
    if str(a.dv_range).lower() == "auto" and sv in df.columns:
        a.dv_range = float(np.clip(np.ceil(3*df[sv].std()), 3, 15))
    a.dv_range = float(a.dv_range); print("relative-speed domain +-", a.dv_range, "m/s")
    P.DV_RANGE = a.dv_range
    cfg = dict(models=models, evals=a.evals, noise_reps=a.noise_reps, speed_var=sv, dv_range=a.dv_range)
    if a.report_only:
        raw = pickle.load(open(f"{out}/raw_results.pkl", "rb"))
        chk = report.build(raw, out, synthetic, dsinfo, rep_I=rep); print("report rebuilt in", out); return
    jobs = [(df, ext, s, cfg) for s in range(a.seeds)]
    print(f"== Experiments II-IV, VI-IX: {a.seeds} seeds x {len(models)+3} models, {a.evals} CMA evals each, {a.jobs} workers ==")
    if a.jobs > 1:
        with mp.Pool(a.jobs) as pool:
            res = []
            for i, r in enumerate(pool.imap(P.run_seed, jobs)):
                res.append(r); print(f"  seed {i} done ({r['_n']['secs']:.0f}s)", flush=True)
    else:
        res = []
        for i, j in enumerate(jobs):
            r = P.run_seed(j); res.append(r); print(f"  seed {i} done ({r['_n']['secs']:.0f}s)", flush=True)
    raw = {i: r for i, r in enumerate(res)}
    with open(f"{out}/raw_results.pkl", "wb") as f: pickle.dump(raw, f)
    chk = report.build(raw, out, synthetic, dsinfo, rep_I=rep)
    print("== numerical checks ==\n", chk.to_string())
    open(f"{out}/RUN_INFO.txt", "w").write(
        f"data={a.data}\nexternal={a.external}\nseeds={a.seeds}\nevals={a.evals}\nnoise_reps={a.noise_reps}\nsynthetic={synthetic}\n"
        f"models={models}\nelapsed_s={time.time()-t0:.0f}\n")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {out}/ (tex/, csv/, figures/)")


if __name__ == "__main__":
    main()
