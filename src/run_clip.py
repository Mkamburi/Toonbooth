import cv2, sys, time
from cartoon import cartoon_frame

src = sys.argv[1] if len(sys.argv) > 1 else "data/moving.mov"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/moving_baseline.mp4"
WIDTH = 640

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
writer = None
n = 0
t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    f = cv2.resize(f, (WIDTH, int(h * WIDTH / w)))
    side = cv2.hconcat([f, cartoon_frame(f)])          # original on the left, cartoon on the right
    if writer is None:
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (side.shape[1], side.shape[0]))
    writer.write(side)
    n += 1
cap.release()
writer.release()
print(f"saved {dst}: {n} frames in {time.time() - t0:.0f} s")