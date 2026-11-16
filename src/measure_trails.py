"""
measure_trails.py - moving-clip check (deliverable 6).

Two numbers per stabilizer config, on data/moving.mov:

1. Warping error (Lai et al. 2018, Eq. 5-6): warp output frame t-1 onto frame t
   using optical flow computed on the ORIGINAL frames, then measure the squared
   difference over non-occluded pixels. Lower = steadier.
   Here that includes real motion, so it is the fair flicker metric for a moving clip.

2. Ghost-trail score (my own measure): find pixels the object occupied in the last
   TRAIL_FRAMES frames but does NOT occupy now. The true scene there is background,
   so the stabilized output should match the plain per-frame cartoon. Any extra
   difference there (compared with clean background far from the object) is a ghost.
     trail diff  = extra mean |stabilized - plain| in the trail region (gray levels)
     ghost %     = extra % of trail pixels differing by more than GHOST_THR

Run from the project folder:  python src/measure_trails.py
Writes results/trails_stabilized.csv, results/trails_stabilized.txt,
and results/trail_example.png (plain vs. most-smoothed config, worst trail frame).
"""
import csv
import os

import cv2
import numpy as np

from stabilize import Stabilizer

# Paths are resolved from the project root (the folder containing src/, data/, results/),
# so the script works no matter where you run it from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(ROOT, "data", "moving.mov")
RESULTS = os.path.join(ROOT, "results")
WIDTH = 640
MAX_FRAMES = 150
TRAIL_FRAMES = 5      # how many past frames count as "old trail"
OBJ_THR = 30          # gray-level difference from background that counts as "object"
GHOST_THR = 16        # |stabilized - plain| above this counts as a visible ghost
MIN_REGION_PX = 50    # skip frames where the trail/clean region is tiny

# Keep this list the same as in evaluate_stabilizers.py so the tables line up.
CONFIGS = [
    ("baseline",     dict(method="none")),
    ("hold 1.0",     dict(method="hold", hold=1.0)),
    ("hold 2.0",     dict(method="hold", hold=2.0)),
    ("hold 3.0",     dict(method="hold", hold=3.0)),
    ("blend a=0.6",  dict(method="blend", alpha=0.6)),
    ("blend a=0.4",  dict(method="blend", alpha=0.4)),
    ("blend a=0.25", dict(method="blend", alpha=0.25)),
]


def load(path, width=WIDTH, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
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


# ---------------------------------------------------------------- warping error
def farneback(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def flows_and_masks(frames):
    """For each t >= 1: sampling maps that pull frame t-1 onto frame t, and a
    non-occlusion mask (forward-backward flow consistency). Flow is computed on
    the ORIGINAL frames, never on the cartoon."""
    g = [gray(f) for f in frames]
    h, w = g[0].shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    maps, masks = [], []
    for t in range(1, len(g)):
        fb = farneback(g[t], g[t - 1])      # where each pixel of frame t came from in t-1
        ff = farneback(g[t - 1], g[t])      # forward flow, used only for the consistency check
        mx, my = gx + fb[..., 0], gy + fb[..., 1]
        ff_w = cv2.remap(ff, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        diff = (fb + ff_w) ** 2
        occluded = diff.sum(2) > 0.01 * ((fb ** 2).sum(2) + (ff_w ** 2).sum(2)) + 0.5
        in_bounds = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
        maps.append((mx, my))
        masks.append(~occluded & in_bounds)
    return maps, masks


def warp_error(seq, maps, masks):
    """Average over t of: mean over non-occluded pixels of squared difference
    between V_t and V_{t-1} warped to t (pixel range [0,1], mean over BGR)."""
    vals = []
    for t in range(1, len(seq)):
        m = masks[t - 1]
        if m.sum() == 0:
            continue
        mx, my = maps[t - 1]
        prev_w = cv2.remap(seq[t - 1], mx, my, cv2.INTER_LINEAR)
        d = (seq[t].astype(np.float32) / 255 - prev_w.astype(np.float32) / 255) ** 2
        vals.append(d.mean(axis=2)[m].mean())
    return float(np.mean(vals))


# ----------------------------------------------------------------- ghost trails
def object_masks(frames):
    """Moving-object masks from the ORIGINAL frames. Assumes a tripod (still
    camera) and that the object does not sit in one spot for most of the clip,
    so the per-pixel median over time is a clean background."""
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    masks = []
    for f in frames:
        d = cv2.absdiff(f, bg).max(axis=2)
        m = (d > OBJ_THR).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.dilate(m, k, iterations=2)      # safety margin around the object
        masks.append(m.astype(bool))
    return masks


def regions(obj, t):
    past = np.zeros_like(obj[0])
    for j in range(max(0, t - TRAIL_FRAMES), t):
        past |= obj[j]
    trail = past & ~obj[t]
    clean = ~(past | obj[t])
    return trail, clean


def trail_scores(outs, plain, obj):
    diffs, ghosts = [], []
    for t in range(1, len(outs)):
        trail, clean = regions(obj, t)
        if trail.sum() < MIN_REGION_PX or clean.sum() < MIN_REGION_PX:
            continue
        d = cv2.absdiff(gray(outs[t]), gray(plain[t])).astype(np.float32)
        diffs.append(d[trail].mean() - d[clean].mean())
        ghosts.append(((d[trail] > GHOST_THR).mean() - (d[clean] > GHOST_THR).mean()) * 100)
    if not diffs:
        return float("nan"), float("nan")
    return float(np.mean(diffs)), float(np.mean(ghosts))


def worst_trail_frame(obj):
    sizes = [regions(obj, t)[0].sum() for t in range(len(obj))]
    return int(np.argmax(sizes))


# ------------------------------------------------------------------------ main
def run(frames, **kw):
    st = Stabilizer(**kw)
    outs = []
    for f in frames:
        o, _ = st(f, return_edges=True)
        outs.append(o)
    return outs


def main():
    frames = load(CLIP)
    if len(frames) < 2:
        raise RuntimeError(f"Read {len(frames)} frames from {CLIP}; need at least 2.")
    print(f"{CLIP}: {len(frames)} frames, {WIDTH}px wide")
    print("computing optical flow and object masks ...")
    maps, masks = flows_and_masks(frames)
    obj = object_masks(frames)
    print(f"mean non-occluded area: {np.mean([m.mean() for m in masks]) * 100:.1f}% of pixels")

    results = {}
    for name, kw in CONFIGS:
        results[name] = run(frames, **kw)
    plain = results["baseline"]

    rows = [["input video (reference)", round(warp_error(frames, maps, masks), 5), "-", "-"]]
    for name, _ in CONFIGS:
        outs = results[name]
        td, gh = trail_scores(outs, plain, obj)
        rows.append([name, round(warp_error(outs, maps, masks), 5), round(td, 2), round(gh, 1)])

    header = ["config", "warp_err", "trail_diff", "ghost%"]
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(v).rjust(w) for v, w in zip(r, widths)) for r in [header] + rows]
    print("\n" + "\n".join(lines))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "trails_stabilized.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(RESULTS, "trails_stabilized.txt"), "w") as fh:
        fh.write(f"{CLIP}: {len(frames)} frames, {WIDTH}px wide\n" + "\n".join(lines) + "\n")

    # picture for the report / video: plain vs. most-smoothed config, worst trail frame
    t = worst_trail_frame(obj)
    last = CONFIGS[-1][0]
    trail, _ = regions(obj, t)
    shown = []
    for label, img in (("plain", plain[t]), (last, results[last][t])):
        vis = img.copy()
        cnts, _ = cv2.findContours(trail.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
        cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        shown.append(vis)
    cv2.imwrite(os.path.join(RESULTS, "trail_example.png"), np.hstack(shown))
    print(f"\nSaved results/trails_stabilized.csv/.txt and results/trail_example.png (frame {t})")


if __name__ == "__main__":
    main()
