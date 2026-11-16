import cv2, time
import numpy as np
from cartoon import cartoon_frame

# ---- load one frame from the still clip ----
cap = cv2.VideoCapture("data/still.mov")
ok, src = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read data/still.mov (run from the project root)")

# ---- part 1: save the original and the cartoon so you can check the look ----
h, w = src.shape[:2]
frame = cv2.resize(src, (640, int(h * 640 / w)))
cv2.imwrite("results/still_original.png", frame)
cv2.imwrite("results/still_cartoon.png", cartoon_frame(frame))
print("saved results/still_original.png and results/still_cartoon.png")

# ---- part 2: speed test ----
def timeit(fn, n=20):
    for _ in range(3):                      # warm-up runs, not counted
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return np.median(t) * 1000              # milliseconds

lines = [f"{'width':>6} {'passes':>6} {'ms/frame':>9} {'fps':>6} | {'bilateral x1':>12} {'blur x2':>8} {'colour x2':>9}"]
for width in (320, 480, 640, 960):
    f = cv2.resize(src, (width, int(h * width / w)))
    f32 = f.astype(np.float32) / 255
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)
    bil = timeit(lambda: cv2.bilateralFilter(lab, -1, 4.25, 3))
    blur = 2 * timeit(lambda: cv2.GaussianBlur(lab[..., 0], (0, 0), 2.0))
    col = 2 * timeit(lambda: cv2.cvtColor(f32, cv2.COLOR_BGR2Lab))
    for n_b in (2, 4):
        ms = timeit(lambda: cartoon_frame(f, n_b=n_b), n=10)
        lines.append(f"{width:>6} {n_b:>6} {ms:>9.1f} {1000 / ms:>6.1f} | {bil:>12.1f} {blur:>8.1f} {col:>9.1f}")

print("\n".join(lines))
with open("results/speed_baseline.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")