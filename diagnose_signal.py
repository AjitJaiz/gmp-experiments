"""How much label information does each input carry on its own?  usage: python diagnose_signal.py data_ngsim2.csv
AUC = P(score of a random 'yes' vector > score of a random 'no' vector); 0.5 = no information, 1 or 0 = perfect.
Vehicle-grouped: also prints the AUC after removing each vehicle's own mean (within-vehicle signal)."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
df = pd.read_csv(sys.argv[1])
df["D"] = df.G_PA + df.G_FA + df.Ls
df["abs_dV_PO"] = df.dV_PO.abs()
cols = ["G_PB", "G_PA", "G_FA", "D", "dV_PO", "abs_dV_PO", "dV_PT", "dV_FT", "dV_FPT", "v_ego", "avg_speed"]
def auc(x, y):
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1*(n1 + 1)/2)/(n1*n0)
rows = []
for c in cols:
    if c not in df: continue
    x = df[c].values.astype(float); y = df.label.values
    xw = x - df.groupby("vehicle_id")[c].transform("mean").values
    rows.append((c, auc(x, y), auc(xw, y), df.loc[df.label == 1, c].mean(), df.loc[df.label == 0, c].mean(), df[c].std()))
out = pd.DataFrame(rows, columns=["variable", "AUC_all", "AUC_within_vehicle", "mean_yes", "mean_no", "std"]).round(3)
print(out.to_string(index=False)); out.to_csv("signal_diagnosis.csv", index=False)
print("\nAUC far from 0.5 = informative. 'within_vehicle' removes between-driver differences.")
