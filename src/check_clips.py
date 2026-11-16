import cv2

for name in ("still.mov", "moving.mov"):
    cap = cv2.VideoCapture(f"data/{name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{name}: {w}x{h}, {fps:.1f} fps, {n} frames, {n / fps:.1f} s")
    cap.release()