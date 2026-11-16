"""Tiny fake highD-format folder (01_tracks.csv + 01_tracksMeta.csv) used ONLY to test the builder code path."""
import numpy as np, pandas as pd, sys, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(1)
fps = 25; frames = 700; rows = []; meta = []
vid = 0
for direction, lanes, sign in ((2, [5, 6, 7], +1), (1, [2, 3, 4], -1)):
    for k in range(40):
        vid += 1; lane0 = int(rng.choice(lanes)); v = rng.uniform(20, 32); x0 = rng.uniform(0, 300) if sign > 0 else rng.uniform(300, 420)
        fc = int(rng.integers(350, 650)) if rng.random() < .6 else None
        lane_t = np.full(frames, lane0)
        if fc is not None:
            nl = lane0 + 1 if lane0 < max(lanes) else lane0 - 1; lane_t[fc:] = nl
        for f in range(frames):
            x = x0 + sign*v*f/fps
            rows.append((f + 1, vid, x - (0 if sign > 0 else 4.5) if False else x, 10, 4.5, 1.8, sign*v, 0.0, int(lane_t[f])))
        meta.append((vid, direction))
pd.DataFrame(rows, columns=["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "laneId"]).to_csv(os.path.join(out, "01_tracks.csv"), index=False)
pd.DataFrame(meta, columns=["id", "drivingDirection"]).to_csv(os.path.join(out, "01_tracksMeta.csv"), index=False)
