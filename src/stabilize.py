"""
stabilize.py - temporally stabilized version of cartoon_frame (Winnemoller pipeline).

Same pipeline as cartoon.cartoon_frame, but it remembers the previous frame.
Two stabilizers, both applied at two points (the edge-source luminance after
pass n_e, and the full Lab image after the last bilateral pass):

  method="hold"   dead-band hysteresis. A pixel keeps its previous (stabilized)
                  value unless it moves by more than `hold` (Lab units) -> small
                  sensor noise cannot flip a shade or an outline, but a real
                  change updates immediately (no ghost trail).
  method="blend"  temporal blending (exponential moving average):
                  out = prev + alpha * (new - prev). Smaller alpha = steadier but
                  more ghosting. Optional motion_thr: pixels that change by more
                  than this use alpha=1 (no smoothing where things really move).
  method="none"   identical to cartoon_frame (the baseline).

Usage:
    st = Stabilizer(method="hold", hold=2.0)
    for frame in frames:
        out = st(frame)              # or: out, edges = st(frame, return_edges=True)
Call st.reset() when starting a new clip.
"""
import cv2
import numpy as np

from cartoon import soft_quantize


class Stabilizer:
    def __init__(self, method="none", n_b=4, n_e=2, sigma_d=3.0, sigma_r=4.25,
                 sigma_e=2.0, tau=0.98, phi_e=2.0, n_bins=8,
                 hold=2.0, alpha=0.4, motion_thr=None):
        assert method in ("none", "hold", "blend")
        self.method = method
        self.n_b, self.n_e = n_b, min(n_e, n_b)       # n_e must not exceed n_b
        self.sigma_d, self.sigma_r = sigma_d, sigma_r
        self.sigma_e, self.tau, self.phi_e, self.n_bins = sigma_e, tau, phi_e, n_bins
        self.hold, self.alpha, self.motion_thr = hold, alpha, motion_thr
        self.state = {}

    def reset(self):
        self.state = {}

    def _apply(self, key, x):
        """Stabilize array x (H,W) or (H,W,3) against the previous stabilized one."""
        prev = self.state.get(key)
        if self.method == "none" or prev is None or prev.shape != x.shape:
            self.state[key] = x
            return x
        d = x - prev
        delta = np.abs(d) if x.ndim == 2 else np.sqrt((d * d).sum(axis=-1))
        if self.method == "hold":
            move = delta > self.hold
            if x.ndim == 3:
                move = move[..., None]
            out = np.where(move, x, prev).astype(np.float32)
        else:  # blend
            a = np.full(delta.shape, self.alpha, np.float32)
            if self.motion_thr is not None:
                a = np.where(delta > self.motion_thr, np.float32(1.0), a).astype(np.float32)
            if x.ndim == 3:
                a = a[..., None]
            out = (prev + a * d).astype(np.float32)
        self.state[key] = out
        return out

    def __call__(self, bgr, return_edges=False):
        f = bgr.astype(np.float32) / 255.0
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
        edge_src = None
        for i in range(self.n_b):
            lab = cv2.bilateralFilter(lab, -1, self.sigma_r, self.sigma_d)
            if i + 1 == self.n_e:
                edge_src = self._apply("edge", lab[..., 0].copy())
        lab = self._apply("lab", lab).copy()     # copy: lab is edited below, state must stay clean

        s1 = cv2.GaussianBlur(edge_src, (0, 0), self.sigma_e)
        s2 = cv2.GaussianBlur(edge_src, (0, 0), 1.6 * self.sigma_e)
        diff = s1 - self.tau * s2
        D = np.where(diff > 0, 1.0, 1.0 + np.tanh(self.phi_e * diff)).astype(np.float32)
        if self.n_bins:
            lab[..., 0] = soft_quantize(lab[..., 0], self.n_bins)
        lab[..., 0] *= D
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return (out, D < 0.5) if return_edges else out