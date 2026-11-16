#!/usr/bin/env python3
"""Builds ONE self-contained HTML report (all tables, all figures, statistics, assumptions) from one or more result folders.
   python make_report.py results_v7 results_highd --out GMPMF_experiment_report.html
Open it in a browser; File > Print > Save as PDF gives a PDF.  Nothing else is needed (images are embedded)."""
import sys, os, glob, base64, argparse, html, datetime
import pandas as pd

TITLES = [
 ("tab_II", "Table II - Datasets and split"), ("tab_IV", "Table IV (Exp. I) - Representation of fixed target profiles (mean over symmetric / asymmetric / multimodal)"),
 ("tab_IV_per_target_and_matched9", "Table IV, detail - per target, and matched 9-parameter budget"),
 ("tab_V", "Table V (Exp. II) - Decision performance, gap inputs only"), ("tab_V_extra_matched_budget_and_K", "Table V, extra - matched budgets and K = 3, 4"),
 ("all_models_summary", "All models - summary (means over seeds)"), ("bootstrap_CI_BAcc_vehicle_level", "Vehicle-level bootstrap 95% CI of BAcc"),
 ("stats_paired_ttests_holm", "Paired t-tests over seeds (Holm-corrected): GMP-K-FIS against other families"),
 ("tab_VI", "Table VI (Exp. III) - Adding the speed-adjustment concept A over dV_PO"), ("tab_VI_four_relative_speeds", "Table VI-b - Concepts over all four relative speeds"),
 ("stats_tableVI_paired", "Paired t-tests over seeds (Holm-corrected) for Table VI and VI-b"),
 ("tab_VII", "Table VII (Exp. IV) - BAcc by traffic regime, V_R = spread across regimes"), ("tab_VII_selected_tau_on_validation", "Thresholds selected on validation data"),
 ("tab_VIII", "Table VIII (Exp. V) - Number of kernels K"), ("tab_IX", "Table IX (Exp. VI) - Robustness to input noise"), ("tab_IX_extended", "Table IX, extended - all noise levels and decision flip rates"),
 ("tab_X", "Table X (Exp. VIII) - Complexity"), ("tab_XI", "Table XI (Exp. IX) - External generalisation (no re-calibration)"),
 ("numerical_checks", "Numerical checks of statements in the draft"),
]
FIGS = [
 ("fig5_representation", "Fig. 5 - MF families fitted to symmetric / asymmetric / multimodal targets"), ("fig6_PN_vs_PY", "Fig. 6 - P_N versus P_Y"),
 ("fig7_sensitivity", "Fig. 7 - Parameter sensitivity of the GMPFM"), ("fig8_threshold", "Fig. 8 - Match rates versus threshold"),
 ("figX_A_profiles", "Extra - learned shape of A (dV_PO) per representation"), ("figX_A_profiles_four_speeds", "Extra - learned A_j for the four relative speeds (GMPFM)"),
 ("figX_robustness", "Extra - robustness to noise"), ("figX_Npeak_vs_deltac", "Extra - number of peaks versus kernel separation"),
]
METHODS = """
<ul>
<li><b>Data.</b> NGSIM I-80 / highD trajectories converted to decision vectors by <code>build_trajectories.py</code>: yes = last 3 s before the lane-crossing frame, no = the preceding 3 s
(same vehicle); inputs G_FA, G_PA, D = G_PA+G_FA+L_s, G_PB (gaps, m) and relative speeds dV_PO, dV_PT, dV_FT, dV_FPT (m/s, negative = gap closing). This is a <b>reconstruction</b>
of the labelling of Rineh &amp; Cheu (t1..t4 are approximated by the crossing frame), not their Dataset 2A.</li>
<li><b>Fuzzy system.</b> Rule base of Rineh &amp; Cheu (2025) Table 2 (54 feasible rules, 10 yes / 44 no), Mamdani min/max inference; score = mu_yes/(mu_yes+mu_no); decision = score &ge; tau,
tau chosen on validation data only. FIS1 is a <b>proxy</b> (all rules mapped to yes). Reference MFs are rebuilt from the crisp ranges of Rineh &amp; Cheu (2025) Table 1.</li>
<li><b>Calibration.</b> CMA-ES, identical evaluation budget for every model, objective = binary cross-entropy + lambda*distance from the reference start (lambda = 0.01). Splits are by vehicle (60/20/20), repeated with different seeds.</li>
<li><b>Speed-adjustment concept A.</b> A membership function over a relative speed enters as mu_yes &larr; min(mu_yes, A), mu_no &larr; max(mu_no, 1-A); with four relative speeds there is one A_j per variable.
This replaces the unpublished 48,600-rule FIS3 rule base of Rineh &amp; Cheu (2026); it is <b>not</b> the same model.</li>
<li><b>Statistics.</b> Paired t-tests over seeds (splits), Holm-corrected within each family of comparisons. Several analysis choices (e.g. the relative-speed domain) were adjusted after looking at earlier results, so borderline p-values should be read cautiously.</li>
<li><b>Exp. I targets</b> (raised cosine, skewed gamma-like, two Lorentzians) are not members of any compared family.</li>
</ul>"""
CSS = "body{font-family:Segoe UI,Arial,sans-serif;max-width:1050px;margin:30px auto;padding:0 16px;color:#222}h1{font-size:24px}h2{margin-top:36px;border-bottom:2px solid #ddd;padding-bottom:4px}h3{margin-top:26px;font-size:15px}table{border-collapse:collapse;font-size:12.5px;margin:8px 0}th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#f1f1f1}td:first-child,th:first-child{text-align:left}img{max-width:100%;border:1px solid #ddd;margin:6px 0}.warn{background:#fdecea;border:1px solid #e57373;padding:8px 12px;margin:12px 0}.note{color:#555;font-size:12px}"


def clean(x):
    return str(x).replace("$", "").replace("\\%", "%").replace("\\&", "&").replace("\\Delta", "d").replace("\\eta", "eta").replace("\\tau", "tau").replace("\\pm", "+-").replace("{", "").replace("}", "").replace("_", "_")


def table_html(path):
    df = pd.read_csv(path)
    df.columns = [clean(c) for c in df.columns]
    df = df.apply(lambda c: c.map(lambda v: clean(v) if isinstance(v, str) else (f"{v:.4g}" if isinstance(v, float) else v)))
    return df.to_html(index=False, border=0, escape=True)


def img(path):
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(open(path, "rb").read()).decode()


def section(d):
    info = open(os.path.join(d, "RUN_INFO.txt")).read() if os.path.exists(os.path.join(d, "RUN_INFO.txt")) else ""
    synth = "synthetic=True" in info
    h = [f"<h2>Results folder: {html.escape(os.path.basename(os.path.abspath(d)))}</h2>"]
    if synth: h.append('<p class="warn"><b>SYNTHETIC DATA</b> - pipeline test only, these numbers are not traffic results.</p>')
    h.append("<pre class='note'>" + html.escape(info) + "</pre>")
    csvd, figd = os.path.join(d, "csv"), os.path.join(d, "figures"); used = set()
    for key, title in TITLES:
        f = os.path.join(csvd, key + ".csv")
        if os.path.exists(f): h.append(f"<h3>{html.escape(title)}</h3>" + table_html(f)); used.add(f)
    for f in sorted(glob.glob(os.path.join(csvd, "*.csv"))):
        if f not in used and "Npeak_vs_delta_c" not in f: h.append(f"<h3>{html.escape(os.path.basename(f))}</h3>" + table_html(f))
    for key, title in FIGS:
        f = os.path.join(figd, key + ".png")
        if os.path.exists(f): h.append(f"<h3>{html.escape(title)}</h3>" + img(f))
    return "\n".join(h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default="GMPMF_experiment_report.html"); a = ap.parse_args()
    body = [f"<h1>GMPMF experiments (GMP_FM_18, Section V) - results report</h1><p class='note'>Generated {datetime.date.today()}. Prepared by A. Jaiswal.</p>",
            "<h2>Method, assumptions and limitations</h2>" + METHODS]
    body += [section(d) for d in a.dirs]
    open(a.out, "w", encoding="utf-8").write(f"<!doctype html><html><head><meta charset='utf-8'><title>GMPMF experiment report</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB)")
