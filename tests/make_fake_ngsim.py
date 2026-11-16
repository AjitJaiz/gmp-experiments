"""Tiny fake file in NGSIM column layout (feet) used ONLY to test the builder code path."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(0); rows = []; FT = 3.28084
nveh = 120; frames = 900
pos = rng.uniform(0, 600, (nveh, 1)) ; lane = rng.integers(1, 4, nveh); spd = rng.uniform(5, 15, nveh)
lane_t = np.tile(lane[:, None], (1, frames)).astype(int)
for v in range(nveh):
    if rng.random() < .5:
        fc = int(rng.integers(300, 800)); lane_t[v, fc:] = lane[v] + (1 if lane[v] < 3 else -1)
for v in range(nveh):
    y = pos[v, 0] + np.cumsum(np.full(frames, spd[v]*0.1)) + 0
    for f in range(frames):
        rows.append((v + 1, f + 1, y[f]*FT, spd[v]*FT, 14.5, lane_t[v, f], "i-80"))
pd.DataFrame(rows, columns=["Vehicle_ID", "Frame_ID", "Local_Y", "v_Vel", "v_length", "Lane_ID", "Location"]).to_csv(sys.argv[1], index=False)
