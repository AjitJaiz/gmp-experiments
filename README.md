# GMP_FM_18 — Section V experiments (I–IX): code, tables, figures

Everything needed to fill Tables II, IV–XI and Figures 5–8 of Bahram's draft (`GMP_FM_18.pdf`).

## 1. Install (once)
```
pip install -r requirements.txt        # numpy scipy pandas matplotlib cma
```

## 2. What is real and what is not — read this first
| Part | Needs data? | Status |
|---|---|---|
| Exp I (Table IV, Fig 5) | no | **Real result** – runs as is |
| Exp V analytical part (Fig 7, N_peak vs δc, numerical checks) | no | **Real result** |
| Exp II, III, IV, VI, VII, VIII, IX (Tables V–XI, Figs 6, 8) | **yes** | Code is complete, but I could **not** run it on NGSIM/highD (no internet to those sites). It was run only on a *synthetic* generator to test the code. Synthetic numbers must NOT go in the paper. |

Folder `example_output_SYNTHETIC/` shows what the outputs look like (every figure is stamped
"SYNTHETIC DATA – PIPELINE TEST ONLY"). Its Table IV / Fig 5 / Fig 7 are valid, the rest are not.

## 3. Real run (what you do)
1. Download NGSIM I-80 (US DOT "Next Generation Simulation" trajectory CSV) and, for Exp IX, highD.
2. Build processed vectors:
```
python -m gmpexp.build_trajectories ngsim  path/to/NGSIM_I80.csv   data_ngsim.csv
python -m gmpexp.build_trajectories highd  path/to/highD/data/      data_highd.csv
```
   (For the Rineh & Cheu April 13, 2005, 4:00–4:15 pm period use the separate 0400pm-0415pm file if you have it.)
   If Bahram can share the real **Dataset 2A**, convert it to the CSV schema below and skip this step.
3. Run everything:
```
python run_all.py --data data_ngsim.csv --external data_highd.csv
```
   Defaults: 10 seeds, 2500 CMA-ES evaluations per model. ~10–40 min. Use `--external none` if no highD yet
   (Table XI is then skipped). Output goes to `results/` : `tex/` (LaTeX tabulars + `all_tables.tex`), `csv/`, `figures/`.
4. Self-test of the code (1 min): `python run_all.py --data synthetic --external synthetic --quick`

### Processed CSV schema (if you build it yourself)
`vehicle_id, t, G_PB, G_PA, G_FA, Ls, v_ego, v_des, avg_speed, label` — metres, m/s, label 1 = change lane.
Splitting is by `vehicle_id` (60/20/20, different per seed), so there is no leakage.

## 3b. Update 16 Nov (Bahram's second email): four relative speeds, highD, report
```
# NGSIM (development) - all four relative speeds, ~20 min.  Use --external none if highD is not ready yet
python run_all.py --data data_ngsim2.csv --external data_highd.csv --seeds 30 --dv_range auto --out results_ngsim
# highD (register at highd-dataset.com first), build and run
python -m gmpexp.build_trajectories highd "C:\path\to\highD\data" data_highd.csv
python run_all.py --data data_highd.csv --external data_ngsim2.csv --seeds 30 --dv_range auto --out results_highd
# one HTML report with every table and figure (File > Print > Save as PDF)
python make_report.py results_ngsim results_highd --out GMPMF_experiment_report.html
```
`--report_only` rebuilds tables/figures/statistics from an existing `<out>/raw_results.pkl` without retraining.
New models: `Gap+4 speeds, ...` give each of dV_PO, dV_PT, dV_FT, dV_FPT its own concept A_j (same MF family for all four). This 54-rule + 4-concept
construction is the alternative to the unpublished 48,600-rule FIS3 rule base (it is NOT the same model). The highD builder was tested only on a tiny fake highD-format folder.

## 4. Design choices Bahram must confirm (all are stated in the code docstrings)
1. **FIS1 is a proxy.** The 51-rule table of Balal et al. (2016) was not re-keyed. `FIS1` = the 54 antecedents with every rule
   mapped to "yes", decision = largest rule output ≥ τ — which is the behaviour Rineh & Cheu (2025) describe (P_Y=100 %, P_N=0 %).
2. **Reference MFs** (FIS1/FIS2 and the starting point of every trained MF) are reconstructed from the crisp ranges in Rineh & Cheu Table 1,
   not from their Fig. 2.
3. **Rule base** = their 54-rule Table 2 (10 yes / 44 no), Mamdani min/max; D1/D2 as in their Section 6 (centroid example 0.432 is reproduced).
4. **Controlled models**: same rules/inference; score = μ_yes/(μ_yes+μ_no); τ ∈ [0,1] picked on validation only; MF parameters fitted with
   CMA-ES minimising BCE + λΩ (λ = 0.01, Ω = distance from the reference start), identical evaluation budget for all models.
   Parameters per MF: Tri 2, Gau 2, Bsp 5, GMP-1 5, GMP-K (K=2) 9 — a *fixed-representation* comparison. A matched-9-parameter check
   (Gaussian mixture M=3, B-spline M=9) is in `tab_V_extra_matched_budget_and_K.tex`.
5. **Speed model (Exp III) - updated 16 Nov after Bahram's email.** The concept A = "Significant Need for Speed Adjustment" is now a
   membership function over the **relative speed dV_PO** (speed of the preceding vehicle in the original lane minus the subject's; negative =
   closing, positive = moving away), read directly from NGSIM - no v_des proxy any more (`--speed_var v_des` restores the old one).
   A enters as: mu_yes <- min(mu_yes, A), mu_no <- max(mu_no, 1-A). dV is clipped to +-15 m/s. Compared representations of A:
   Rineh & Cheu's own two sets (closing / moving away, Fig.6, fixed, A = their max), single Gaussian, Gaussian mixture, GMPFM K=2.
   The other three relative speeds (dV_PT, dV_FT, dV_FPT) are in the CSV but are NOT used yet; the 48,600-rule FIS3 rule base of Rineh & Cheu (2026)
   is not published in the paper, so it is not replicated. Bahram should confirm whether A should use dV_PO only.
6. **Labelling** (no exact t1–t4): yes = last 3 s before the lane-crossing frame, no = the 3 s before that (same vehicle).
   This is a reconstruction of Dataset 2A, not Dataset 2A.
7. **Regimes (Exp IV)**: terciles of the local average speed (label-independent) → congested / transition / free.
8. **Exp I targets** are not members of any compared family (raised cosine, skewed gamma-like, two Lorentzians). A 3-term Gaussian mixture
   also fits the multimodal target well — as Bahram's text anticipates, multimodality alone is not exclusive to GMPFM.
9. Title/theory say speed adjustment (ACC); Section V tests lane-change decisions. Not resolved here.

## 5. Extra outputs
`csv/stats_paired_ttests_holm.csv` (paired by seed, Holm-corrected), `csv/bootstrap_CI_BAcc_vehicle_level.csv`,
`csv/numerical_checks.csv` (γ does not move the maximiser; ∂μ/∂γ = μ(1−μ)ln z; μ<1; N=4K+1 — only these draft statements are checked),
`figures/figX_*` (robustness, N_peak vs δc, **A profiles learned per MF family**), `raw_results.pkl` (everything).

## 6. Code map
`gmpexp/mfs.py` MF families · `fit.py` profile fitting · `exp_representation.py` Exp I · `exp_params.py` Exp V ·
`fis.py` rules, inference, D1/D2, metrics · `pipeline.py` Exps II–IV, VI–IX · `build_trajectories.py` raw → vectors ·
`report.py` tables and figures · `run_all.py` entry point. The builder was tested only on a tiny fake NGSIM-format file
(`tests/make_fake_ngsim.py`), not on real NGSIM — check its printed counts against Dataset 2A (106 vehicles, 4334 / 3739).
