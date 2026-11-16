"""
evaluate_stabilizers.py - flicker "after" numbers, using the SAME metric as measure_flicker.py.

Run from the project folder:  python src/evaluate_stabilizers.py
The "baseline" row should reproduce results/flicker_baseline.txt (sanity check).
Writes results/flicker_stabilized.csv and results/flicker_stabilized.txt
"""
import csv
import os
import time

import cv2
import numpy as np

from stabilize import Stabilizer

CLIP = "data/still.mov"
WIDTH = 640
MAX_FRAMES = 150
THRESHOLDS = (4, 8, 16)

CONFIGS = [
    ("baseline",        dict(method="none")),
    ("hold 1.0",        dict(method="hold", hold=1.0)),
    ("hold 2.0",        dict(method="hold", hold=2.0)),
    ("hold 3.0",        dict(method="hold", hold=3.0)),
    ("blend a=0.6",     dict(method="blend", alpha=0.6)),
    ("blend a=0.4",     dict(method="blend", alpha=0.4)),
    ("blend a=0.25",    dict(method="blend", alpha=0.25)),
]


# --- metric code copied from measure_flicker.py so numbers are comparable ---
def load(path, width=WIDTH, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.resize(f, (width, int(h * width / w))))
    cap.release()
    return frames


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pct_changed(seq, thr):
    return np.mean([(cv2.absdiff(a, b) > thr).mean() * 100 for a, b in zip(seq[:-1], seq[1:])])


def pct_blink(masks):
    vals = []
    for a, b in zip(masks[:-1], masks[1:]):
        union = max(int((a | b).sum()), 1)
        vals.append(int((a ^ b).sum()) / union * 100)
    return np.mean(vals)
# ---------------------------------------------------------------------------


def run(frames, **kw):
    st = Stabilizer(**kw)
    outs, edges, ms = [], [], []
    for f in frames:
        t = time.perf_counter()
        o, e = st(f, return_edges=True)
        ms.append((time.perf_counter() - t) * 1000)
        outs.append(gray(o))
        edges.append(e)
    return outs, edges, float(np.mean(ms))


def main():
    frames = load(CLIP)
    print(f"{CLIP}: {len(frames)} frames, {WIDTH}px wide\n")
    rows = []
    base8 = None
    for name, kw in CONFIGS:
        outs, edges, ms = run(frames, **kw)
        ch = [pct_changed(outs, t) for t in THRESHOLDS]
        blink = pct_blink(edges)
        if base8 is None:
            base8 = ch[1]
        red8 = (1 - ch[1] / base8) * 100
        rows.append([name, *[round(c, 2) for c in ch], round(blink, 1),
                     round(ms, 1), round(1000 / ms, 1), round(red8, 1)])

    header = ["config", *[f"chg@{t}%" for t in THRESHOLDS], "blink%", "ms/frame", "fps", "reduction@8%"]
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(v).rjust(w) for v, w in zip(r, widths)) for r in [header] + rows]
    print("\n".join(lines))

    os.makedirs("results", exist_ok=True)
    with open("results/flicker_stabilized.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open("results/flicker_stabilized.txt", "w") as fh:
        fh.write(f"{CLIP}: {len(frames)} frames, {WIDTH}px wide\n" + "\n".join(lines) + "\n")
    print("\nSaved results/flicker_stabilized.csv and .txt")


if __name__ == "__main__":
    main()