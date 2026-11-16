"""Raw trajectory data  ->  processed decision vectors (schema in data.COLS).

Labelling (documented, Eq.(38) of the draft; follows Rineh & Cheu (2025) in spirit):
  * a lane-change event = the frame where the lane id of a vehicle changes (crossing frame f_c);
  * vehicles with two lane changes closer than 10 s are dropped (as in Dataset 2A);
  * YES vectors  : frames in [f_c - Y, f_c - 1]  (Y = --yes_window seconds, default 3.0 s) -> label 1;
  * NO  vectors  : the same number of frames immediately before, [f_c - 2Y, f_c - Y - 1] -> label 0
                   (same vehicle, same candidate target lane)  -> equal class balance per vehicle;
  * decision variables at each frame: G_PB, G_PA, G_FA, D = G_PA+G_FA+L_s  (Eq.37), computed from
    front-bumper positions; absent neighbours -> gap capped at 100 m;
  * avg_speed = mean speed of vehicles within +-100 m in the two lanes (label-independent regime variable);
  * relative speeds dV_PO = V_PO-V_S, dV_PT = V_PT-V_S, dV_FT = V_S-V_FT, dV_FPT = V_PT-V_FT (m/s): POSITIVE = gap opening,
    NEGATIVE = closing (matches the 'closing' side of Rineh & Cheu 2026, Fig. 2); a missing neighbour gives 0;
  * v_des = 90th percentile of the subject's own speed over its whole trajectory (proxy for a desired speed;
    NGSIM/highD do not record one - tell Bahram, he must confirm the definition).
THIS IS A RECONSTRUCTION, not the authors' Dataset 2A (t1..t4 timing is approximated by the crossing frame).
"""
import argparse, glob, os, numpy as np, pandas as pd
from .data import COLS, OPT, GAP_CAP

FT = 0.3048


def process(tr, fps, stride=1, yes_window=3.0, lanes_ok=None, min_gap_events_s=10.0, verbose=True):
    """tr columns: veh, frame, ys (front-bumper position along travel, m), lane, spd (m/s), length (m)."""
    tr = tr.sort_values(["veh", "frame"]).reset_index(drop=True)
    Y = int(round(yes_window*fps)); events = []
    for veh, g in tr.groupby("veh", sort=False):
        lane = g.lane.values; fr = g.frame.values
        ch = np.where(lane[1:] != lane[:-1])[0] + 1
        ch = [i for i in ch if (lanes_ok is None or (lane[i] in lanes_ok and lane[i - 1] in lanes_ok)) and abs(lane[i] - lane[i - 1]) == 1]
        if len(ch) == 0: continue
        if len(ch) > 1 and np.min(np.diff(fr[ch])) < min_gap_events_s*fps: continue
        i = ch[0]
        if len(ch) > 1: continue                                  # keep single-event vehicles only
        f_c = fr[i]
        if fr[0] > f_c - 2*Y or lane[0] != lane[i - 1]: continue    # need 2Y frames of history in the original lane
        events.append((veh, f_c, lane[i - 1], lane[i]))
    if verbose: print(f"[build] {len(events)} single lane-change vehicles found")
    need = {}
    for veh, f_c, l0, l1 in events:
        for f in range(f_c - 2*Y, f_c, stride): need.setdefault(f, []).append((veh, l0, l1, f_c))
    sub = tr[tr.frame.isin(need.keys())]
    by_frame = {f: g for f, g in sub.groupby("frame")}
    vdes = tr.groupby("veh").spd.quantile(0.9).to_dict()
    me = tr.set_index(["veh", "frame"])
    rows = []
    for f, lst in need.items():
        g = by_frame.get(f)
        if g is None: continue
        V, YS, LN, LEN, SP = g.veh.values, g.ys.values, g.lane.values, g.length.values, g.spd.values
        for veh, l0, l1, f_c in lst:
            k = np.where(V == veh)[0]
            if len(k) == 0: continue
            k = k[0]; ys, ls, ve = YS[k], LEN[k], SP[k]
            lab = 1 if f >= f_c - Y else 0
            def nb_ahead(lane):
                m = (LN == lane) & (V != veh) & (YS > ys)
                if not m.any(): return GAP_CAP, ve
                g = YS[m] - LEN[m] - ys; j = int(np.argmin(g)); return float(g[j]), float(SP[m][j])
            def nb_behind(lane):
                m = (LN == lane) & (V != veh) & (YS <= ys)
                if not m.any(): return GAP_CAP, ve
                g = ys - ls - YS[m]; j = int(np.argmin(g)); return float(g[j]), float(SP[m][j])
            (gpb, vpo), (gpa, vpt), (gfa, vft) = nb_ahead(l0), nb_ahead(l1), nb_behind(l1)
            near = ((LN == l0) | (LN == l1)) & (np.abs(YS - ys) <= 100)
            rows.append((veh, f/fps, max(0, gpb), max(0, gpa), max(0, gfa), ls, ve, vdes[veh], SP[near].mean(), lab,
                         vpo - ve, vpt - ve, ve - vft, vpt - vft))
    out = pd.DataFrame(rows, columns=COLS + OPT)
    if verbose:
        print(f"[build] vectors {len(out)}  vehicles {out.vehicle_id.nunique()}  yes {int(out.label.sum())}  no {int((1-out.label).sum())}")
    return out


def ngsim(path, yes_window=3.0, stride=1, location="i-80", t0=None, t1=None):
    """Reads the (multi-GB) NGSIM CSV in chunks, keeping only the needed columns and rows of `location`."""
    hdr = pd.read_csv(path, nrows=0).columns
    low = {c.strip().lower(): c for c in hdr}
    need = ["vehicle_id", "frame_id", "local_y", "lane_id", "v_vel", "v_length", "location", "global_time"]
    miss = [n for n in need[:6] if n not in low]
    if miss: raise ValueError(f"columns not found: {miss}; file has {list(hdr)}")
    use = [low[n] for n in need if n in low]
    parts = []
    dt = {low[n]: ("string" if n == "location" else "float64") for n in need if n in low}
    for ch in pd.read_csv(path, usecols=use, dtype=dt, thousands=",", low_memory=False, chunksize=1_000_000):
        ch.columns = [c.strip().lower() for c in ch.columns]
        if "location" in ch.columns and location:
            ch = ch[ch["location"].astype(str).str.strip().str.lower() == location.lower()]
        if t0 is not None and "global_time" in ch.columns:
            ch = ch[(ch["global_time"] >= t0) & (ch["global_time"] <= t1)]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    print(f"[build] kept {len(df)} rows for location={location}")
    if len(df) == 0: raise ValueError("no rows kept - check the Location column values (try --location with the exact text)")
    # I-80 holds several 15-min recordings in which Vehicle_ID / Frame_ID restart (even back-to-back ones).
    # Global_Time - Frame_ID*100 ms equals the recording's start time, so it identifies the recording.
    per = np.zeros(len(df), dtype="int64")
    if "global_time" in df.columns:
        start = df["global_time"].values - df["frame_id"].values*100.0
        per = pd.factorize(np.round(start/60000.0))[0].astype("int64")
        print(f"[build] recordings detected: {int(per.max()) + 1}  (rows per recording: {np.bincount(per).tolist()})")
    df["vehicle_id"] = df["vehicle_id"].astype("int64") + per*10_000_000
    df["frame_id"] = df["frame_id"].astype("int64") + per*10_000_000
    tr = pd.DataFrame({"veh": df["vehicle_id"].astype("int64"), "frame": df["frame_id"].astype("int64"), "ys": df["local_y"]*FT,
                       "lane": df["lane_id"].astype("int64"), "spd": df["v_vel"]*FT, "length": df["v_length"]*FT})
    return process(tr, 10, stride, yes_window, lanes_ok={1, 2, 3, 4, 5})


def highd(folder, yes_window=3.0, stride=3):
    outs = []
    for tf in sorted(glob.glob(os.path.join(folder, "*_tracks.csv"))):
        meta = pd.read_csv(tf.replace("_tracks.csv", "_tracksMeta.csv")); dirn = meta.set_index("id").drivingDirection
        t = pd.read_csv(tf)
        d = t.id.map(dirn)
        front = np.where(d == 2, t.x + t.width, -t.x)                  # travel-direction coordinate of the front bumper
        tr = pd.DataFrame({"veh": t.id, "frame": t.frame, "ys": front, "lane": t.laneId,
                           "spd": np.abs(t.xVelocity), "length": t.width})
        o = process(tr, 25, stride, yes_window, verbose=True)
        o["vehicle_id"] = o.vehicle_id.astype(str) + "_" + os.path.basename(tf)[:2]
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ngsim", "highd"]); ap.add_argument("path"); ap.add_argument("out")
    ap.add_argument("--yes_window", type=float, default=3.0); ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--location", default="i-80")
    a = ap.parse_args()
    if a.kind == "ngsim": d = ngsim(a.path, a.yes_window, a.stride or 1, a.location)
    else: d = highd(a.path, a.yes_window, a.stride or 3)
    d.to_csv(a.out, index=False); print("wrote", a.out)
