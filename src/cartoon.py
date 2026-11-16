import cv2
import numpy as np

def soft_quantize(L, n_bins=8, lam_phi=3.0, om_phi=14.0, lam_d=0.0, om_d=2.0):
    """Soft luminance quantization (paper, Eq. 6). L is in [0, 100]."""
    dq = 100.0 / n_bins
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0      # gradient per pixel
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.clip(np.sqrt(gx ** 2 + gy ** 2), lam_d, om_d)
    phi = lam_phi + (g - lam_d) / (om_d - lam_d) * (om_phi - lam_phi)   # sharper where the gradient is high
    q_near = np.round(L / dq) * dq
    return q_near + (dq / 2.0) * np.tanh(phi * (L - q_near))

def cartoon_frame(bgr, n_b=4, n_e=2, sigma_d=3.0, sigma_r=4.25,
                  sigma_e=2.0, tau=0.98, phi_e=2.0, n_bins=8,
                  return_edges=False):
    """Winnemoller et al. 2006: bilateral filter + DoG edges + soft quantization."""
    f = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)               # L in [0,100], a,b about [-127,127]
    edge_src = None
    for i in range(n_b):
        lab = cv2.bilateralFilter(lab, -1, sigma_r, sigma_d)
        if i + 1 == n_e:                                    # edges come from an early pass
            edge_src = lab[..., 0].copy()
    s1 = cv2.GaussianBlur(edge_src, (0, 0), sigma_e)
    s2 = cv2.GaussianBlur(edge_src, (0, 0), 1.6 * sigma_e)
    diff = s1 - tau * s2
    D = np.where(diff > 0, 1.0, 1.0 + np.tanh(phi_e * diff)).astype(np.float32)
    if n_bins:
        lab[..., 0] = soft_quantize(lab[..., 0], n_bins)
    lab[..., 0] *= D                                        # darken luminance where there are edges
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return (out, D < 0.5) if return_edges else out