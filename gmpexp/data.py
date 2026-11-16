"""Data layer.  Processed schema (one row = one decision vector), metres / m/s:
vehicle_id, t, G_PB, G_PA, G_FA, Ls, v_ego, v_des, avg_speed, label(1=change lane, 0=keep)"""
import numpy as np, pandas as pd

COLS = ["vehicle_id", "t", "G_PB", "G_PA", "G_FA", "Ls", "v_ego", "v_des", "avg_speed", "label"]
# optional relative speeds, sign convention: POSITIVE = the gap to that vehicle is OPENING, NEGATIVE = CLOSING
#   dV_PO = V_PO - V_S, dV_PT = V_PT - V_S, dV_FT = V_S - V_FT, dV_FPT = V_PT - V_FT   (m/s)
OPT = ["dV_PO", "dV_PT", "dV_FT", "dV_FPT"]
GAP_CAP = 100.0


def load_csv(path):
    df = pd.read_csv(path)
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}; expected {COLS}")
    df = df[COLS + [c for c in OPT if c in df.columns]].dropna().copy()
    for c in ("G_PB", "G_PA", "G_FA"):
        df[c] = df[c].clip(0, GAP_CAP)
    return df.reset_index(drop=True)


def synthetic(n_veh=400, seed=0, shift=1.0, name="SYNTHETIC"):
    """Pipeline-test data ONLY. Not traffic data. Labels follow a noisy gap-acceptance rule plus a
    bimodal speed-need signal that is built into the generator."""
    rng = np.random.default_rng(seed); rows = []
    for v in range(n_veh):
        n = int(rng.integers(16, 40)); half = n//2
        avg = rng.uniform(4, 30); ls = rng.uniform(4, 6); vdes_off = rng.normal(0, 1)
        for lab in [1]*half + [0]*(n - half):
            if lab: gfa, gpa, gpb = rng.normal(20, 8), rng.normal(28, 10), rng.normal(16, 8)
            else:   gfa, gpa, gpb = rng.normal(12, 9), rng.normal(18, 11), rng.normal(20, 10)
            gfa, gpa, gpb = [max(1.0, g*shift) for g in (gfa, gpa, gpb)]
            ve = max(2.0, rng.normal(avg, 3))
            if lab:
                m = rng.choice(3, p=[.4, .35, .25]); dv = [rng.normal(4, 1.5), rng.normal(-4, 1.5), rng.normal(0.5, 1)][m]
            else:
                dv = rng.normal(0, 1.8)
            rows.append((v, len(rows), gpb, gpa, gfa, ls, ve, ve + dv + vdes_off*0, avg + rng.normal(0, 1), lab))
    df = pd.DataFrame(rows, columns=COLS)
    df["dV_PO"] = -(df.v_des - df.v_ego) + rng.normal(0, .5, len(df))     # synthetic only
    df["dV_PT"] = rng.normal(0, 3, len(df)); df["dV_FT"] = rng.normal(0, 3, len(df)); df["dV_FPT"] = df.dV_PT - df.dV_FT
    for c in ("G_PB", "G_PA", "G_FA"):
        df[c] = df[c].clip(0, GAP_CAP)
    return df


def split_by_vehicle(df, seed, frac=(.6, .2, .2)):
    rng = np.random.default_rng(1000 + seed)
    ids = df.vehicle_id.unique(); rng.shuffle(ids)
    n = len(ids); a, b = int(frac[0]*n), int((frac[0] + frac[1])*n)
    grp = {**{i: 0 for i in ids[:a]}, **{i: 1 for i in ids[a:b]}, **{i: 2 for i in ids[b:]}}
    g = df.vehicle_id.map(grp).values
    return df[g == 0], df[g == 1], df[g == 2]


def regime_edges(df):
    """Terciles of the label-independent local average speed -> congested / transition / free."""
    return np.quantile(df.avg_speed, [1/3, 2/3])


def regime_of(df, edges):
    s = df.avg_speed.values
    return np.where(s <= edges[0], 0, np.where(s <= edges[1], 1, 2))     # 0=cong 1=trans 2=free


REGIMES = ["Congested", "Transition", "Free"]


def raw_matrix(df, speed_var="v_des"):
    """Physical inputs in fis.INPUTS order [G_FA, G_PA, D, G_PB] and aux [a0, a1, Ls].
    The speed-adjustment variable is dv = a1 - a0:  speed_var='v_des' -> v_des - v_ego (old proxy),
    otherwise a relative-speed column such as 'dV_PO' (a0 = 0, a1 = that column)."""
    D = df.G_PA.values + df.G_FA.values + df.Ls.values
    X = np.c_[df.G_FA.values, df.G_PA.values, D, df.G_PB.values]
    if speed_var == "v_des":
        aux = np.c_[df.v_ego.values, df.v_des.values, df.Ls.values]
    else:
        aux = np.c_[np.zeros(len(df)), df[speed_var].values, df.Ls.values]
        if speed_var == "dV_PO" and all(c in df.columns for c in OPT[1:]):
            aux = np.c_[aux, df.dV_PT.values, df.dV_FT.values, df.dV_FPT.values]     # extra relative speeds (aux cols 3,4,5)
    return X, aux
