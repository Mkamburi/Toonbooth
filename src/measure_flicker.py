import cv2, numpy as np
from cartoon import cartoon_frame

CLIP = "data/still.mov"
WIDTH = 640
MAX_FRAMES = 150
THRESHOLDS = (4, 8, 16)

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

frames = load(CLIP)
raw = [gray(f) for f in frames]
outs, edges = [], []
for f in frames:
    o, e = cartoon_frame(f, return_edges=True)
    outs.append(gray(o))
    edges.append(e)

lines = [f"{CLIP}: {len(frames)} frames, {WIDTH}px wide"]
for thr in THRESHOLDS:
    lines.append(f"threshold {thr:>2}: raw input {pct_changed(raw, thr):6.2f}% | "
                 f"cartoon {pct_changed(outs, thr):6.2f}% of pixels change")
lines.append(f"outline pixels that blink on/off between frames: {pct_blink(edges):.1f}%")
print("\n".join(lines))
with open("results/flicker_baseline.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")