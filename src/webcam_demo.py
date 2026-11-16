"""
webcam_demo.py - live webcam cartoonizer with fps + timing breakdown.



Keys
  q / ESC  quit
  space    toggle original <-> cartoon
  b        toggle side-by-side view
  + / -    more / fewer bilateral passes
  w        cycle image width (320, 480, 640, 960)
  m        run a 5 s measurement (stops by itself); appends a row to results/webcam_measurements.csv
  s        save a screenshot to results/

Timing is split so camera limits are not blamed on the filter:
  capture = time waiting in cap.read() (camera-limited), filter = cartoon_frame, display = imshow/waitKey
"""
import csv
import os
import time
from collections import deque

import cv2

# ---------------------------------------------------------------------------
# 1) Connect to cartoon.py (cartoon_frame, n_b = number of bilateral passes)
# ---------------------------------------------------------------------------
from cartoon import cartoon_frame


def apply_cartoon(frame, passes):
    # n_b = number of bilateral passes; edges are taken after pass n_e (default 2),
    # which must not exceed n_b or edge_src stays None.
    return cartoon_frame(frame, n_b=passes, n_e=min(2, passes))


# ---------------------------------------------------------------------------
# 2) SETTINGS
# ---------------------------------------------------------------------------
WIDTHS = [320, 480, 640, 960]
START_WIDTH_IDX = 2          # 640 px
START_PASSES = 4
MEASURE_SECONDS = 5
REQUEST_CAMERA_FPS = 30
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "webcam_measurements.csv")
CSV_HEADER = ["width_px", "passes", "frames", "fps",
              "capture_ms_mean", "filter_ms_mean", "filter_ms_p95",
              "display_ms_mean", "processing_latency_ms_mean", "camera_fps_reported"]


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def put_text(img, text, y):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def append_csv(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. On macOS, allow camera access for your terminal/VS Code.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # keep the buffer small so latency stays low
    cap.set(cv2.CAP_PROP_FPS, REQUEST_CAMERA_FPS)  # ask for 30 fps (camera may ignore this)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera reports {cam_fps:.0f} fps (it may still drop in dim light).")

    width_idx = START_WIDTH_IDX
    passes = START_PASSES
    show_cartoon = True
    side_by_side = False

    cap_ms = deque(maxlen=60)      # waiting for the camera
    filter_ms = deque(maxlen=60)   # time spent inside the cartoon filter only
    disp_ms = deque(maxlen=60)     # drawing text + imshow + waitKey
    fps_hist = deque(maxlen=30)
    t_prev = time.perf_counter()

    measuring = False
    m_start = 0.0
    m_cap, m_filt, m_disp = [], [], []

    print(__doc__)

    while True:
        # ---- capture (waits for the camera) ----
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            print("Failed to read from webcam.")
            break
        frame = cv2.flip(frame, 1)                       # mirror view feels natural
        frame = resize_to_width(frame, WIDTHS[width_idx])

        # ---- filter (skipped when only the original is shown) ----
        t2 = time.perf_counter()
        out = apply_cartoon(frame, passes) if (show_cartoon or side_by_side) else frame
        t3 = time.perf_counter()

        # ---- display ----
        if side_by_side:
            display = cv2.hconcat([frame, out])
        else:
            display = out if show_cartoon else frame
        display = display.copy()

        now = time.perf_counter()
        fps_hist.append(1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        put_text(display, f"{WIDTHS[width_idx]} px | passes {passes} | {'cartoon' if show_cartoon else 'original'}", 22)
        put_text(display, f"end-to-end {mean(fps_hist):.0f} fps", 44)
        put_text(display, f"capture {mean(cap_ms):.1f} | filter {mean(filter_ms):.1f} | display {mean(disp_ms):.1f} ms", 66)
        put_text(display, f"processing latency {mean(filter_ms) + mean(disp_ms):.1f} ms", 88)
        if measuring:
            put_text(display, f"MEASURING... {max(0, MEASURE_SECONDS - (now - m_start)):.1f}s", 110)

        cv2.imshow("Toonbooth live", display)
        key = cv2.waitKey(1) & 0xFF            # waitKey also flushes the window draw
        t4 = time.perf_counter()

        c_ms = (t1 - t0) * 1000
        f_ms = (t3 - t2) * 1000
        d_ms = (t4 - t3) * 1000
        cap_ms.append(c_ms); filter_ms.append(f_ms); disp_ms.append(d_ms)

        if measuring:
            m_cap.append(c_ms); m_filt.append(f_ms); m_disp.append(d_ms)
            elapsed = t4 - m_start
            if elapsed >= MEASURE_SECONDS:
                measuring = False
                n = len(m_filt)
                fs = sorted(m_filt)
                row = [WIDTHS[width_idx], passes, n, round(n / elapsed, 1),
                       round(mean(m_cap), 2), round(mean(m_filt), 2),
                       round(fs[int(0.95 * (n - 1))], 2),
                       round(mean(m_disp), 2), round(mean(m_filt) + mean(m_disp), 2),
                       round(cam_fps, 1)]
                append_csv(row)
                print("Saved measurement:", dict(zip(CSV_HEADER, row)))

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            show_cartoon = not show_cartoon
        elif key == ord("b"):
            side_by_side = not side_by_side
        elif key in (ord("+"), ord("=")):
            passes = min(passes + 1, 8)
        elif key in (ord("-"), ord("_")):
            passes = max(passes - 1, 1)
        elif key == ord("w"):
            width_idx = (width_idx + 1) % len(WIDTHS)
            cap_ms.clear(); filter_ms.clear(); disp_ms.clear(); fps_hist.clear()
        elif key == ord("m") and not measuring:
            measuring = True
            m_start = time.perf_counter()
            m_cap, m_filt, m_disp = [], [], []
        elif key == ord("s"):
            os.makedirs(RESULTS_DIR, exist_ok=True)
            path = os.path.join(RESULTS_DIR, f"webcam_{WIDTHS[width_idx]}px_{int(time.time())}.png")
            cv2.imwrite(path, display)
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()