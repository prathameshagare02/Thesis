from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from ultralytics import SAM

# ----------------------------- Optional deps -----------------------------
try:
    from skimage.feature import local_binary_pattern as _lbp_import  # noqa: F401
    LBP_AVAILABLE = True
except Exception:
    LBP_AVAILABLE = False

# Optional (better covariance shrinkage for anomaly model)
try:
    from sklearn.covariance import LedoitWolf  # type: ignore
    SKLEARN_COV_AVAILABLE = True
except Exception:
    SKLEARN_COV_AVAILABLE = False


# ----------------------------- File Picker -----------------------------
def _get_tk_root():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        return root
    except ImportError as e:
        raise RuntimeError("tkinter not available.") from e


def pick_image_file(
    title: str = "Select an image",
    filetypes=(
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
        ("All files", "*.*"),
    ),
) -> str:
    from tkinter import filedialog

    root = _get_tk_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise ValueError("No file selected.")
    return path


# ----------------------------- Anomaly Model -----------------------------
class RustAnomalyModel:
    """
    Lightweight anomaly model over per-segment feature vectors.

    - Fit a Gaussian model (mean + covariance) on "clean metal" feature vectors.
    - Inference: Mahalanobis distance -> anomaly score in [0,1].

    You can fit:
      A) per-image (no dataset): robustly choose "likely clean" segments, fit on them
      B) dataset: pass a directory of clean images to build a global model (recommended)

    Saved model is a .npz with 'mean' and 'icov'.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = float(eps)
        self.mean: Optional[np.ndarray] = None
        self.icov: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return self.mean is not None and self.icov is not None

    def save(self, path: str):
        if not self.is_ready():
            raise RuntimeError("Model not fitted.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, mean=self.mean, icov=self.icov)

    def load(self, path: str):
        d = np.load(path)
        self.mean = d["mean"].astype(np.float32)
        self.icov = d["icov"].astype(np.float32)

    @staticmethod
    def _nan_to_num(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, X: np.ndarray):
        """
        Fit mean + inverse covariance using shrinkage if available.
        X: (N, D)
        """
        X = self._nan_to_num(X)
        if X.ndim != 2 or X.shape[0] < max(10, X.shape[1] + 2):
            raise ValueError(f"Not enough samples to fit anomaly model. Got X={X.shape}")

        if SKLEARN_COV_AVAILABLE:
            lw = LedoitWolf().fit(X)
            cov = lw.covariance_.astype(np.float32)
            mean = lw.location_.astype(np.float32)
        else:
            # Simple shrinkage covariance
            mean = X.mean(axis=0).astype(np.float32)
            Xc = X - mean
            cov = np.cov(Xc, rowvar=False).astype(np.float32)

            # Shrinkage towards diagonal for stability
            diag = np.diag(np.diag(cov))
            alpha = 0.10
            cov = (1 - alpha) * cov + alpha * diag

        # Regularize
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * self.eps
        icov = np.linalg.pinv(cov).astype(np.float32)

        self.mean = mean
        self.icov = icov

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance -> mapped to [0,1] with a robust logistic mapping.
        """
        if not self.is_ready():
            raise RuntimeError("Model not fitted/loaded.")
        X = self._nan_to_num(X)

        mu = self.mean.reshape(1, -1)
        ic = self.icov

        Xc = X - mu
        # d^2 = x^T icov x
        d2 = np.einsum("bi,ij,bj->b", Xc, ic, Xc).astype(np.float32)
        d2 = np.maximum(d2, 0.0)

        # Robust scaling to [0,1] using percentiles
        p50 = float(np.percentile(d2, 50))
        p95 = float(np.percentile(d2, 95))
        scale = max(p95 - p50, 1e-6)
        z = (d2 - p50) / scale  # roughly: 0 around median, ~1 around 95th

        # Logistic mapping
        scores = 1.0 / (1.0 + np.exp(-2.0 * (z - 0.5)))
        return scores.astype(np.float32)


# ----------------------------- Core Detector -----------------------------
class FastRustDetector:
    """
    Rust detection on metal parts.

    Two modes:
      1) mode="heuristic" : your existing per-segment rust score (color+texture)
      2) mode="anomaly"   : anomaly detection over per-segment feature vectors
                            (Mahalanobis distance) + rust color prior

    Key idea for anomaly mode:
      - If you provide a directory of CLEAN metal images, it will build a global anomaly model
        and then score anomalies on the target image.
      - If you do NOT provide a clean directory/model, it will fit a *per-image* anomaly model
        using segments that look "likely clean" (robust subset selection).

    Notes:
      - "Anatoly detection" is typically meant as "Anomaly detection".
      - This code keeps your SAM-based interactive metal selection.
    """

    def __init__(
        self,
        n_segments: int = 10000,
        fast_mode: bool = False,
        verbose: bool = True,
        sam_checkpoint: str = "sam2.1_b.pt",
        # Scoring threshold behavior:
        rust_threshold_fallback: float = 0.60,
        ensure_one_positive: bool = True,
        dynamic_feature_gates: bool = True,
        dynamic_score_threshold: bool = True,
        min_valid_segments_for_dynamic: int = 20,
        otsu_bias: float = -0.05,
        # Mode control
        mode: str = "anomaly",  # "anomaly" or "heuristic"
        # Anomaly config
        anomaly_weight: float = 0.70,  # how much anomaly influences final score (rest is rust prior)
        rust_prior_weight: float = 0.30,
        anomaly_threshold_mode: str = "otsu",  # "otsu" or "percentile"
        anomaly_percentile: float = 92.0,  # used if anomaly_threshold_mode="percentile"
        anomaly_model_path: str = "",  # load/save anomaly model
        clean_dir: str = "",  # directory of CLEAN images to fit anomaly model
    ):
        self.n_segments = int(n_segments)
        self.fast_mode = bool(fast_mode)
        self.verbose = bool(verbose)

        self.rust_threshold_fallback = float(rust_threshold_fallback)
        self.ensure_one_positive = bool(ensure_one_positive)

        self.dynamic_feature_gates = bool(dynamic_feature_gates)
        self.dynamic_score_threshold = bool(dynamic_score_threshold)
        self.min_valid_segments_for_dynamic = int(min_valid_segments_for_dynamic)
        self.otsu_bias = float(otsu_bias)

        self.mode = str(mode).strip().lower()
        if self.mode not in ("heuristic", "anomaly"):
            raise ValueError("mode must be 'heuristic' or 'anomaly'.")

        self.anomaly_weight = float(anomaly_weight)
        self.rust_prior_weight = float(rust_prior_weight)
        s = self.anomaly_weight + self.rust_prior_weight
        if s <= 0:
            self.anomaly_weight, self.rust_prior_weight = 0.7, 0.3
        else:
            self.anomaly_weight /= s
            self.rust_prior_weight /= s

        self.anomaly_threshold_mode = str(anomaly_threshold_mode).strip().lower()
        if self.anomaly_threshold_mode not in ("otsu", "percentile"):
            raise ValueError("anomaly_threshold_mode must be 'otsu' or 'percentile'.")
        self.anomaly_percentile = float(anomaly_percentile)

        self.anomaly_model_path = str(anomaly_model_path)
        self.clean_dir = str(clean_dir)

        # SAM
        self.sam_checkpoint = sam_checkpoint
        self.sam_model = None
        self.mask: Optional[np.ndarray] = None
        self.processed_mask: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None

        # Morphology
        self.kernel_size = 3
        self.erosion_iterations = 2
        self.dilation_iterations = 2

        # Anomaly model
        self.anom_model = RustAnomalyModel(eps=1e-4)

        if self.verbose:
            self._print_backend_info()

        self.load_sam2_model()
        self._maybe_prepare_anomaly_model()

    # ---- logging ----
    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("FastRustDetector initialized:")
        print("  Per-segment feature vectors (SLIC segments)")
        print(f"  Target segments: {self.n_segments}")
        print(f"  Fast mode: {self.fast_mode}")
        print(f"  Mode: {self.mode}")
        print(f"  Dynamic feature gates (heuristic): {self.dynamic_feature_gates}")
        print(f"  Dynamic score threshold (heuristic Otsu): {self.dynamic_score_threshold}")
        print(f"  Fallback rust threshold (heuristic): {self.rust_threshold_fallback:.2f}")
        print(f"  Ensure one positive: {self.ensure_one_positive}")
        print(f"  Min valid segments for dynamic: {self.min_valid_segments_for_dynamic}")
        print(f"  Heuristic Otsu bias: {self.otsu_bias:+.2f}")
        if self.mode == "anomaly":
            print(f"  Anomaly weights: anomaly={self.anomaly_weight:.2f}, rust_prior={self.rust_prior_weight:.2f}")
            print(f"  Anomaly threshold mode: {self.anomaly_threshold_mode}")
            if self.anomaly_threshold_mode == "percentile":
                print(f"  Anomaly percentile: {self.anomaly_percentile:.1f}")
            if self.anomaly_model_path:
                print(f"  Anomaly model path: {self.anomaly_model_path}")
            if self.clean_dir:
                print(f"  Clean directory: {self.clean_dir}")
            print(f"  sklearn LedoitWolf available: {SKLEARN_COV_AVAILABLE}")

    # ---- SAM ----
    def load_sam2_model(self):
        try:
            self._log(f"Loading SAM2 model from {self.sam_checkpoint}...")
            self.sam_model = SAM(self.sam_checkpoint)
            self._log("SAM2 model loaded successfully.")
        except Exception as e:
            self._log(f"Failed to load SAM2 model: {e}")
            self.sam_model = None

    # ---- Mask Utilities ----
    def apply_morphological_operations(self, mask: np.ndarray | None):
        if mask is None:
            return None
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.erosion_iterations > 0:
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations)
        if self.dilation_iterations > 0:
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=self.dilation_iterations)
        return (mask_uint8 > 127).astype(np.uint8)

    def detect_metal_regions(self) -> List[Dict]:
        if self.original_image is None:
            return []
        try:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            _, A, _ = cv2.split(lab)
            metal_mask1 = cv2.inRange(A, 120, 135)

            _, S, V = cv2.split(hsv)
            metal_mask2 = cv2.inRange(S, 30, 100) & cv2.inRange(V, 100, 255)

            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)

            combined = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined = cv2.bitwise_or(combined, edges_dilated)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append(
                        {
                            "id": len(regions),
                            "bbox": [x, y, x + w, y + h],
                            "center": [x + w // 2, y + h // 2],
                            "area": area,
                        }
                    )
            return regions
        except Exception as e:
            self._log(f"Metal detection error: {e}")
            return []

    def _get_red_plastic_mask_and_points(self, image: np.ndarray, max_points: int = 60):
        """Bright red plastic -> negative points for SAM + hard exclusion mask."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            l1, u1 = np.array([0, 140, 100]), np.array([8, 255, 255])
            l2, u2 = np.array([172, 140, 100]), np.array([180, 255, 255])

            m1 = cv2.inRange(hsv, l1, u1)
            m2 = cv2.inRange(hsv, l2, u2)
            plastic_mask = cv2.bitwise_or(m1, m2)

            plastic_mask = cv2.morphologyEx(plastic_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            plastic_mask = cv2.morphologyEx(plastic_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

            inner_mask = cv2.erode(plastic_mask, np.ones((9, 9), np.uint8), iterations=1)
            y_idxs, x_idxs = np.where(inner_mask > 0)

            neg_pts, neg_lbls = [], []
            if len(x_idxs) > 0:
                count = len(x_idxs)
                idxs = np.linspace(0, count - 1, max_points, dtype=int) if count > max_points else np.arange(count)
                for i in idxs:
                    neg_pts.append([float(x_idxs[i]), float(y_idxs[i])])
                    neg_lbls.append(0)

            return plastic_mask, neg_pts, neg_lbls
        except Exception:
            return None, [], []

    def sam2_predict(self, points: List[List[float]], labels: List[int], image: np.ndarray | None = None):
        from tkinter import messagebox

        if self.sam_model is None:
            messagebox.showerror("Error", "SAM 2 model not loaded!")
            return None

        target_img = image if image is not None else self.original_image
        if target_img is None:
            return None

        try:
            plastic_mask, neg_pts, neg_lbls = self._get_red_plastic_mask_and_points(target_img)

            final_points = list(points) + neg_pts
            final_labels = list(labels) + neg_lbls

            image_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            results = self.sam_model.predict(image_rgb, points=final_points, labels=final_labels)

            if results and results[0].masks is not None and len(results[0].masks.data) > 0:
                mask = results[0].masks.data[0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)

                # hard exclude plastic
                if plastic_mask is not None:
                    binary_mask[plastic_mask > 0] = 0

                self.processed_mask = self.apply_morphological_operations(binary_mask)
                return binary_mask

            return None
        except Exception as e:
            self._log(f"SAM 2 prediction error: {e}")
            return None

    # ---- Interactive metal selection ----
    def interactive_metal_segmentation(self) -> bool:
        from tkinter import messagebox

        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return False

        regions = self.detect_metal_regions()
        if regions:
            self._log(f"Found {len(regions)} metal regions - auto mode")
            return self._auto_detection_mode(regions)

        self._log("No regions auto-detected - manual point mode")
        return False

    def _auto_detection_mode(self, regions: List[Dict]) -> bool:
        win = "Click METAL region - Press 's' to save, 'q' quit"
        cv2.namedWindow(win)

        def click_callback(event, x, y, flags, param):
            regs = param["regions"]
            if event == cv2.EVENT_LBUTTONDOWN:
                for r in regs:
                    x1, y1, x2, y2 = map(int, r["bbox"])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        cx, cy = r["center"]
                        pts = [[float(cx), float(cy)]]
                        lbl = [1]
                        self.mask = self.sam2_predict(points=pts, labels=lbl)
                        self._update_interactive_display(win, regions=regs, selected_id=r["id"])
                        break

        self._update_interactive_display(win, regions=regions, selected_id=-1)
        cv2.setMouseCallback(win, click_callback, {"regions": regions})

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                cv2.destroyAllWindows()
                return False
            if key == ord("s"):
                if self.mask is not None and np.any(self.mask):
                    cv2.destroyAllWindows()
                    return True
                print("Warning: Please select a metal region first!")
            if key == ord("q"):
                cv2.destroyAllWindows()
                return False

    def _update_interactive_display(self, win: str, regions: List[Dict] | None = None, selected_id: int = -1):
        if self.original_image is None:
            return
        img = self.original_image.copy()

        if regions:
            for r in regions:
                x1, y1, x2, y2 = map(int, r["bbox"])
                cx, cy = map(int, r["center"])
                is_sel = r["id"] == selected_id
                color = (0, 255, 0) if is_sel else (255, 0, 0)
                thickness = 3 if is_sel else 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, f"M{r['id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if self.mask is not None and np.any(self.mask):
            overlay = img.copy()
            overlay[self.mask == 1] = [0, 255, 0]
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        cv2.putText(
            img,
            "Press 'm' manual | 's' save | 'q' quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, img)

    # ---- Features ----
    def _compute_feature_maps(self, crop: np.ndarray) -> Dict[str, np.ndarray]:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)

        k = 5
        local_mean = cv2.blur(gray, (k, k))
        local_sq_mean = cv2.blur(gray**2, (k, k))
        entropy_proxy = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

        if LBP_AVAILABLE and not self.fast_mode:
            from skimage.feature import local_binary_pattern

            lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method="uniform").astype(np.float32)
        else:
            lbp = np.zeros_like(gray, dtype=np.float32)

        a_centered = lab[:, :, 1] - 128
        b_centered = lab[:, :, 2] - 128
        chroma = np.sqrt(a_centered**2 + b_centered**2)
        redness_ratio = a_centered / (lab[:, :, 0] + 1e-6)

        return {
            "L": lab[:, :, 0],
            "a": lab[:, :, 1],
            "b": lab[:, :, 2],
            "H": hsv[:, :, 0],
            "S": hsv[:, :, 1],
            "V": hsv[:, :, 2],
            "gray": gray,
            "gradient": gradient,
            "entropy": entropy_proxy,
            "lbp": lbp,
            "chroma": chroma,
            "redness_ratio": redness_ratio,
        }

    def _extract_features_vectorized(
        self, feature_maps: Dict[str, np.ndarray], segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          features: (n_seg, 15) per-segment feature vectors (ONE vector per segment)
          segment_ids: the actual segment labels in order
          metal_scores: per-segment mean of metal_mask (for gating)
        """
        unique_segments = np.unique(segments)
        n_seg = len(unique_segments)
        features = np.zeros((n_seg, 15), dtype=np.float32)

        def safe_mean(vals):
            return float(np.mean(vals)) if len(vals) else 0.0

        def safe_std(vals):
            return float(np.std(vals)) if len(vals) else 0.0

        lc = ndimage.labeled_comprehension
        f = feature_maps
        s = segments
        u = unique_segments

        features[:, 0] = lc(f["L"], s, u, safe_mean, np.float32, 0)
        features[:, 1] = lc(f["a"], s, u, safe_mean, np.float32, 0)
        features[:, 2] = lc(f["b"], s, u, safe_mean, np.float32, 0)

        features[:, 3] = lc(f["L"], s, u, safe_std, np.float32, 0)
        features[:, 4] = lc(f["a"], s, u, safe_std, np.float32, 0)
        features[:, 5] = lc(f["b"], s, u, safe_std, np.float32, 0)

        features[:, 6] = lc(f["gray"], s, u, safe_std, np.float32, 0)
        features[:, 7] = lc(f["entropy"], s, u, safe_mean, np.float32, 0)

        features[:, 8] = lc(f["gradient"], s, u, safe_mean, np.float32, 0)
        features[:, 9] = lc(f["gradient"], s, u, safe_std, np.float32, 0)

        features[:, 10] = lc(f["chroma"], s, u, safe_mean, np.float32, 0)
        features[:, 11] = lc(f["redness_ratio"], s, u, safe_mean, np.float32, 0)

        features[:, 12] = lc(f["H"], s, u, safe_mean, np.float32, 0)
        features[:, 13] = lc(f["S"], s, u, safe_mean, np.float32, 0)
        features[:, 14] = lc(f["V"], s, u, safe_mean, np.float32, 0)

        metal_scores = (
            lc(f["metal_mask"], s, u, safe_mean, np.float32, 0)
            if "metal_mask" in f
            else np.ones(n_seg, dtype=np.float32)
        )

        return features, unique_segments, metal_scores

    # --------------------- Dynamic thresholds helpers (heuristic) ---------------------
    @staticmethod
    def _robust_percentile(vals: np.ndarray, q: float, default: float, min_n: int = 10) -> float:
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            return float(default)
        return float(np.percentile(vals, q))

    def _compute_dynamic_gates(self, features_valid: np.ndarray) -> Dict[str, float]:
        a_vals = features_valid[:, 1] if features_valid.size else np.array([])
        b_vals = features_valid[:, 2] if features_valid.size else np.array([])
        s_vals = features_valid[:, 13] if features_valid.size else np.array([])
        v_vals = features_valid[:, 14] if features_valid.size else np.array([])
        rough_vals = features_valid[:, 6] if features_valid.size else np.array([])
        ent_vals = features_valid[:, 7] if features_valid.size else np.array([])

        dyn = {
            "a_warm": self._robust_percentile(a_vals, 55, 126.0),
            "b_warm": self._robust_percentile(b_vals, 55, 124.0),
            "a_red": self._robust_percentile(a_vals, 80, 145.0),
            "a_red_hi": self._robust_percentile(a_vals, 90, 155.0),
            "b_orange": self._robust_percentile(b_vals, 75, 140.0),
            "b_orange_hi": self._robust_percentile(b_vals, 85, 150.0),
            "v_dark": self._robust_percentile(v_vals, 10, 160.0),
            "v_very_dark": self._robust_percentile(v_vals, 5, 55.0),
            "v_hi": self._robust_percentile(v_vals, 95, 240.0),
            "s_low": self._robust_percentile(s_vals, 25, 35.0),
            "s_mid": self._robust_percentile(s_vals, 50, 45.0),
            "s_hi": self._robust_percentile(s_vals, 75, 70.0),
            "rough_hi": self._robust_percentile(rough_vals, 75, 18.0),
            "ent_hi": self._robust_percentile(ent_vals, 75, 10.0),
        }

        dyn["a_warm"] = float(np.clip(dyn["a_warm"], 110.0, 155.0))
        dyn["b_warm"] = float(np.clip(dyn["b_warm"], 110.0, 155.0))
        dyn["a_red"] = float(np.clip(dyn["a_red"], dyn["a_warm"], 175.0))
        dyn["a_red_hi"] = float(np.clip(dyn["a_red_hi"], dyn["a_red"], 185.0))
        dyn["b_orange"] = float(np.clip(dyn["b_orange"], dyn["b_warm"], 180.0))
        dyn["b_orange_hi"] = float(np.clip(dyn["b_orange_hi"], dyn["b_orange"], 190.0))

        dyn["v_very_dark"] = float(np.clip(dyn["v_very_dark"], 25.0, 110.0))
        dyn["v_dark"] = float(np.clip(dyn["v_dark"], min(dyn["v_very_dark"] + 10.0, 140.0), 210.0))
        dyn["v_hi"] = float(np.clip(dyn["v_hi"], 210.0, 255.0))

        dyn["s_low"] = float(np.clip(dyn["s_low"], 5.0, 90.0))
        dyn["s_mid"] = float(np.clip(dyn["s_mid"], dyn["s_low"], 140.0))
        dyn["s_hi"] = float(np.clip(dyn["s_hi"], dyn["s_mid"], 200.0))

        dyn["rough_hi"] = float(np.clip(dyn["rough_hi"], 5.0, 45.0))
        dyn["ent_hi"] = float(np.clip(dyn["ent_hi"], 3.0, 30.0))

        return dyn

    def _compute_dynamic_score_threshold_otsu(self, rust_scores_valid: np.ndarray) -> float:
        scores = rust_scores_valid[np.isfinite(rust_scores_valid)]
        if scores.size < self.min_valid_segments_for_dynamic:
            return float(self.rust_threshold_fallback)

        s255 = np.clip(scores * 255.0, 0, 255).astype(np.uint8)
        if int(s255.max()) - int(s255.min()) < 5:
            return float(self.rust_threshold_fallback)

        t, _ = cv2.threshold(s255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
        thr = float(thr + self.otsu_bias)
        thr = float(np.clip(thr, 0.25, 0.75))
        return thr

    # --------------------- Heuristic per-vector classifier ---------------------
    def _rust_score_for_feature(self, fv: np.ndarray, dyn: Optional[Dict[str, float]] = None) -> float:
        mean_a = float(fv[1])
        mean_b = float(fv[2])
        mean_rough = float(fv[6])
        mean_ent = float(fv[7])
        mean_h = float(fv[12])
        mean_s = float(fv[13])
        mean_v = float(fv[14])

        if dyn is None:
            dyn = {
                "a_warm": 126.0,
                "b_warm": 124.0,
                "a_red": 145.0,
                "a_red_hi": 155.0,
                "b_orange": 140.0,
                "b_orange_hi": 150.0,
                "v_dark": 160.0,
                "v_very_dark": 55.0,
                "v_hi": 240.0,
                "s_low": 35.0,
                "s_mid": 45.0,
                "s_hi": 70.0,
                "rough_hi": 18.0,
                "ent_hi": 10.0,
            }

        very_dark_gate = max(dyn["v_very_dark"], 70.0)
        is_very_dark = (mean_v < very_dark_gate)

        textured_enough = (mean_rough > dyn["rough_hi"] * 0.75) or (mean_ent > dyn["ent_hi"] * 0.75)
        brownish_enough = (mean_b > (dyn["b_warm"] - 6.0))

        if is_very_dark and (textured_enough or brownish_enough):
            return 0.82

        if mean_v > dyn["v_hi"]:
            return 0.0

        dark_relax = mean_v < (dyn["v_dark"] * 0.65)
        a_min = dyn["a_warm"] - (14.0 if dark_relax else 0.0)
        b_min = dyn["b_warm"] - (14.0 if dark_relax else 0.0)
        if mean_a < a_min or mean_b < b_min:
            return 0.0

        if (mean_s < dyn["s_low"]) and (mean_a < (dyn["a_red"] - (10.0 if dark_relax else 0.0))) and (
            mean_v > (dyn["v_dark"] * 0.55)
        ):
            return 0.0

        if (35.0 < mean_h < 95.0) and (mean_s > max(75.0, dyn["s_hi"])) and (mean_v > max(175.0, dyn["v_dark"])) and (
            mean_a < dyn["a_red"]
        ):
            return 0.0

        hsv_score = 0.0
        is_rust_hue = (mean_h < 55.0 or mean_h > 165.0)
        if is_rust_hue and mean_s > max(20.0, dyn["s_low"] * 0.7):
            hsv_score += 1.0

        if mean_s > dyn["s_hi"]:
            hsv_score += 1.0
        elif mean_s > dyn["s_mid"]:
            hsv_score += 0.5
        elif mean_s > dyn["s_low"]:
            hsv_score += 0.25
        else:
            if dark_relax and mean_v < dyn["v_dark"]:
                hsv_score += 0.15

        if mean_v < dyn["v_dark"]:
            hsv_score += 1.0
        elif mean_v < (dyn["v_dark"] + 50.0):
            hsv_score += 0.5

        if mean_a > dyn["a_red_hi"]:
            hsv_score += 2.2
        elif mean_a > dyn["a_red"]:
            hsv_score += 1.6
        elif mean_a > (dyn["a_red"] - (12.0 if dark_relax else 7.0)):
            hsv_score += 1.0

        if mean_b > dyn["b_orange_hi"]:
            hsv_score += 0.8
        elif mean_b > dyn["b_orange"]:
            hsv_score += 0.4
        elif dark_relax and mean_b > dyn["b_warm"]:
            hsv_score += 0.2

        hsv_norm = min(1.0, hsv_score / 4.0)

        tex_rough = min(1.0, mean_rough / max(26.0, dyn["rough_hi"] * 2.0))
        tex_ent = min(1.0, mean_ent / max(16.0, dyn["ent_hi"] * 1.8))
        tex = (tex_rough + tex_ent) / 2.0
        if hsv_norm < 0.18:
            tex = 0.0

        rust_score = 0.80 * hsv_norm + 0.20 * tex

        if (mean_s < dyn["s_mid"]) and (mean_v < dyn["v_dark"]) and (mean_b > (dyn["b_warm"] - 4.0)):
            rust_score = max(rust_score, 0.74 if dark_relax else 0.70)

        if mean_v < (very_dark_gate + 15.0) and (textured_enough or mean_b > dyn["b_warm"]):
            rust_score = max(rust_score, 0.78)

        return float(np.clip(rust_score, 0.0, 1.0))

    # --------------------- Rust prior for anomaly mode ---------------------
    def _rust_prior_for_feature(self, fv: np.ndarray) -> float:
        """
        A mild, lighting-tolerant rust prior from the feature vector.
        Higher when warm/brown/red and not super bright.
        """
        mean_a = float(fv[1])
        mean_b = float(fv[2])
        mean_h = float(fv[12])
        mean_s = float(fv[13])
        mean_v = float(fv[14])
        rough = float(fv[6])
        ent = float(fv[7])

        # Hue: rust-ish (wrap-around reds)
        hue_rust = 1.0 if (mean_h < 55.0 or mean_h > 165.0) else 0.0

        # Warmth and redness (Lab)
        warm = np.clip((mean_b - 118.0) / 35.0, 0.0, 1.0)
        red = np.clip((mean_a - 130.0) / 35.0, 0.0, 1.0)

        # Dark/brown rust can be low S; so just mild saturation use
        sat = np.clip((mean_s - 15.0) / 80.0, 0.0, 1.0)

        # Prefer not-too-bright (but allow dark rust)
        val = 1.0 - np.clip((mean_v - 110.0) / 160.0, 0.0, 1.0)

        # Texture (helps distinguish stains vs flat paint)
        tex = np.clip((rough / 25.0 + ent / 14.0) * 0.5, 0.0, 1.0)

        prior = 0.30 * hue_rust + 0.25 * warm + 0.25 * red + 0.10 * sat + 0.10 * val
        prior = max(prior, 0.55 * tex)  # if texture is strong, allow prior to rise
        return float(np.clip(prior, 0.0, 1.0))

    # --------------------- Anomaly threshold helper ---------------------
    def _threshold_scores(self, scores_valid: np.ndarray, fallback: float) -> float:
        scores = scores_valid[np.isfinite(scores_valid)]
        if scores.size < self.min_valid_segments_for_dynamic:
            return float(fallback)

        if self.anomaly_threshold_mode == "percentile":
            thr = float(np.percentile(scores, self.anomaly_percentile))
            return float(np.clip(thr, 0.25, 0.90))

        # Otsu on 0..1
        s255 = np.clip(scores * 255.0, 0, 255).astype(np.uint8)
        if int(s255.max()) - int(s255.min()) < 5:
            return float(fallback)
        t, _ = cv2.threshold(s255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
        # make it a bit more inclusive for anomalies
        thr = float(np.clip(thr - 0.03, 0.20, 0.85))
        return thr

    # --------------------- Anomaly model preparation ---------------------
    @staticmethod
    def _iter_images_in_dir(folder: str) -> List[str]:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        out = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
        return sorted(out)

    def _maybe_prepare_anomaly_model(self):
        """
        If mode=anomaly:
          - load model from anomaly_model_path if it exists
          - else if clean_dir is provided, fit model from clean images and save if anomaly_model_path given
        """
        if self.mode != "anomaly":
            return

        if self.anomaly_model_path and os.path.exists(self.anomaly_model_path):
            try:
                self.anom_model.load(self.anomaly_model_path)
                self._log(f"Loaded anomaly model: {self.anomaly_model_path}")
                return
            except Exception as e:
                self._log(f"Failed to load anomaly model ({self.anomaly_model_path}): {e}")

        if self.clean_dir and os.path.isdir(self.clean_dir):
            self._log(f"Fitting anomaly model from clean_dir: {self.clean_dir}")
            feats_all = []
            img_paths = self._iter_images_in_dir(self.clean_dir)
            if len(img_paths) == 0:
                self._log("No images found in clean_dir. Will fall back to per-image fit.")
                return

            # Fit on full image (metal mask assumed entire image unless you want SAM here too)
            for p in img_paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                # Use full image as "metal" for model training; you can replace with your own metal mask logic.
                crop = img
                metal_mask = np.ones(crop.shape[:2], dtype=np.uint8)
                seg = slic(
                    img_as_float(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    n_segments=max(2000, self.n_segments // 4),
                    compactness=20,
                    sigma=1,
                    start_label=0,
                    channel_axis=2,
                )
                fmap = self._compute_feature_maps(crop)
                fmap["metal_mask"] = metal_mask
                features, _, metal_scores = self._extract_features_vectorized(fmap, seg)
                valid = metal_scores > 0.5
                if np.any(valid):
                    feats_all.append(features[valid])

            if len(feats_all) == 0:
                self._log("No valid features extracted from clean_dir. Will fall back to per-image fit.")
                return

            X = np.concatenate(feats_all, axis=0)
            try:
                self.anom_model.fit(X)
                self._log(f"Anomaly model fitted on {X.shape[0]} segments (D={X.shape[1]}).")
                if self.anomaly_model_path:
                    self.anom_model.save(self.anomaly_model_path)
                    self._log(f"Saved anomaly model to: {self.anomaly_model_path}")
            except Exception as e:
                self._log(f"Failed to fit anomaly model from clean_dir: {e}")

        else:
            if self.clean_dir:
                self._log(f"clean_dir not found/invalid: {self.clean_dir} (will use per-image fit)")

    # --------------------- Per-image anomaly fit (no dataset) ---------------------
    def _fit_anomaly_model_per_image(self, features_valid: np.ndarray) -> bool:
        """
        Robustly select likely-clean subset then fit.
        Strategy:
          - compute rust prior; take lowest ~70% prior (least rust-looking)
          - fit model
        """
        if features_valid.shape[0] < max(self.min_valid_segments_for_dynamic, 25):
            return False

        priors = np.array([self._rust_prior_for_feature(fv) for fv in features_valid], dtype=np.float32)
        cutoff = float(np.percentile(priors, 70.0))
        clean_idx = np.where(priors <= cutoff)[0]
        if clean_idx.size < max(20, features_valid.shape[1] + 2):
            # fallback: take lowest 80%
            cutoff = float(np.percentile(priors, 80.0))
            clean_idx = np.where(priors <= cutoff)[0]

        if clean_idx.size < max(20, features_valid.shape[1] + 2):
            return False

        try:
            self.anom_model.fit(features_valid[clean_idx])
            return True
        except Exception:
            return False

    # --------------------- Segmentation analysis (shared) ---------------------
    def _perform_segmentation(self, crop: np.ndarray) -> np.ndarray:
        return slic(
            img_as_float(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            n_segments=self.n_segments,
            compactness=20,
            sigma=1,
            start_label=0,
            channel_axis=2,
        )

    def _perform_analysis(self, crop: np.ndarray, metal_mask_crop: np.ndarray) -> Dict:
        segments = self._perform_segmentation(crop)

        fmap = self._compute_feature_maps(crop)
        fmap["metal_mask"] = metal_mask_crop

        features, segment_ids, metal_scores = self._extract_features_vectorized(fmap, segments)
        valid = metal_scores > 0.5
        valid_indices = np.where(valid)[0]
        n_valid = int(len(valid_indices))

        # Maps
        score_map = np.zeros(crop.shape[:2], dtype=np.float32)
        pred_map = np.zeros(crop.shape[:2], dtype=np.uint8)
        final_mask = np.zeros(crop.shape[:2], dtype=np.uint8)

        seg_areas = np.bincount(segments.ravel())
        rust_pixels = 0
        metal_pixels = 0

        # ----------------- MODE: HEURISTIC -----------------
        if self.mode == "heuristic":
            dyn_gates: Optional[Dict[str, float]] = None
            if self.dynamic_feature_gates and n_valid >= self.min_valid_segments_for_dynamic:
                dyn_gates = self._compute_dynamic_gates(features[valid_indices])

            rust_scores = np.zeros(len(features), dtype=np.float32)
            if n_valid > 0:
                for i in valid_indices:
                    rust_scores[i] = self._rust_score_for_feature(features[i], dyn=dyn_gates)

            if self.dynamic_score_threshold and n_valid >= self.min_valid_segments_for_dynamic:
                thr = self._compute_dynamic_score_threshold_otsu(rust_scores[valid_indices])
            else:
                thr = float(self.rust_threshold_fallback)

            rust_pred = np.zeros(len(features), dtype=np.uint8)
            if n_valid > 0:
                rust_pred[valid_indices] = (rust_scores[valid_indices] >= thr).astype(np.uint8)
                if self.ensure_one_positive and int(np.sum(rust_pred[valid_indices])) == 0:
                    best_i = valid_indices[int(np.argmax(rust_scores[valid_indices]))]
                    rust_pred[best_i] = 1

            for idx in valid_indices:
                sid = int(segment_ids[idx])
                area = int(seg_areas[sid]) if sid < len(seg_areas) else 0
                metal_pixels += area
                m = (segments == sid)

                score_map[m] = float(rust_scores[idx])
                pred_map[m] = int(rust_pred[idx])

                if rust_pred[idx] == 1:
                    rust_pixels += area
                    final_mask[m] = 1

            # clean-up
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = final_mask * metal_mask_crop

            pct = (rust_pixels / metal_pixels) if metal_pixels > 0 else 0.0

            if self.verbose:
                self._log(f"[heuristic] Valid metal segments: {n_valid}")
                self._log(f"[heuristic] Threshold used: {thr:.3f}")

            return {
                "mode": "heuristic",
                "segments": segments,
                "features": features,
                "segment_ids": segment_ids,
                "metal_scores": metal_scores,
                "valid_mask": valid,
                "rust_scores": rust_scores,
                "final_scores": rust_scores,
                "rust_pred": rust_pred,
                "score_map": score_map,
                "pred_map": pred_map,
                "crop_mask": final_mask,
                "rust_percentage": pct,
                "threshold_used": thr,
                "dynamic_gates": dyn_gates,
            }

        # ----------------- MODE: ANOMALY -----------------
        # 1) Get features for valid segments
        if n_valid <= 0:
            return {
                "mode": "anomaly",
                "segments": segments,
                "features": features,
                "segment_ids": segment_ids,
                "metal_scores": metal_scores,
                "valid_mask": valid,
                "rust_scores": np.zeros(len(features), dtype=np.float32),
                "final_scores": np.zeros(len(features), dtype=np.float32),
                "rust_pred": np.zeros(len(features), dtype=np.uint8),
                "score_map": score_map,
                "pred_map": pred_map,
                "crop_mask": final_mask,
                "rust_percentage": 0.0,
                "threshold_used": float(self.rust_threshold_fallback),
                "dynamic_gates": None,
                "anomaly_threshold": float(self.rust_threshold_fallback),
            }

        features_valid = features[valid_indices]

        # 2) Ensure anomaly model is ready (global or per-image)
        if not self.anom_model.is_ready():
            ok = self._fit_anomaly_model_per_image(features_valid)
            if self.verbose:
                self._log(f"[anomaly] Per-image model fit: {'OK' if ok else 'FAILED'}")
            if not ok:
                # fallback: use heuristic if anomaly model cannot fit
                self._log("[anomaly] Falling back to heuristic mode (insufficient segments for anomaly fit).")
                prev_mode = self.mode
                self.mode = "heuristic"
                out = self._perform_analysis(crop, metal_mask_crop)
                self.mode = prev_mode
                return out

        # 3) Anomaly score for valid segments
        anom_scores_valid = self.anom_model.score(features_valid)  # 0..1
        rust_priors_valid = np.array([self._rust_prior_for_feature(fv) for fv in features_valid], dtype=np.float32)

        # 4) Combine: anomaly + rust prior
        final_scores_valid = self.anomaly_weight * anom_scores_valid + self.rust_prior_weight * rust_priors_valid
        final_scores = np.zeros(len(features), dtype=np.float32)
        final_scores[valid_indices] = final_scores_valid

        # 5) Threshold final score
        thr = self._threshold_scores(final_scores_valid, fallback=self.rust_threshold_fallback)

        rust_pred = np.zeros(len(features), dtype=np.uint8)
        rust_pred[valid_indices] = (final_scores_valid >= thr).astype(np.uint8)

        if self.ensure_one_positive and int(np.sum(rust_pred[valid_indices])) == 0:
            best_i = valid_indices[int(np.argmax(final_scores_valid))]
            rust_pred[best_i] = 1

        # 6) Build maps and compute rust percentage
        for j, idx in enumerate(valid_indices):
            sid = int(segment_ids[idx])
            area = int(seg_areas[sid]) if sid < len(seg_areas) else 0
            metal_pixels += area
            m = (segments == sid)

            score_map[m] = float(final_scores[idx])
            pred_map[m] = int(rust_pred[idx])

            if rust_pred[idx] == 1:
                rust_pixels += area
                final_mask[m] = 1

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = final_mask * metal_mask_crop

        pct = (rust_pixels / metal_pixels) if metal_pixels > 0 else 0.0

        if self.verbose:
            self._log(f"[anomaly] Valid metal segments: {n_valid}")
            self._log(
                f"[anomaly] Threshold used: {thr:.3f} | weights (anom={self.anomaly_weight:.2f}, prior={self.rust_prior_weight:.2f})"
            )

        rust_scores = np.zeros(len(features), dtype=np.float32)
        rust_scores[valid_indices] = rust_priors_valid  # keep for debugging/visual

        return {
            "mode": "anomaly",
            "segments": segments,
            "features": features,
            "segment_ids": segment_ids,
            "metal_scores": metal_scores,
            "valid_mask": valid,
            "rust_scores": rust_scores,          # rust priors
            "final_scores": final_scores,        # combined score used for decision
            "rust_pred": rust_pred,
            "score_map": score_map,              # combined score on pixels
            "pred_map": pred_map,
            "crop_mask": final_mask,
            "rust_percentage": pct,
            "threshold_used": thr,
            "dynamic_gates": None,
            "anomaly_threshold": thr,
        }

    # ---- Main analysis ----
    def analyze(self, image_path: str, interactive: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        self.original_image = original.copy()
        timings["load"] = time.time() - t0

        t0 = time.time()
        metal_mask = None
        crop = None
        metal_mask_crop = None
        cx = cy = 0
        cw, ch = original.shape[1], original.shape[0]

        if interactive and self.sam_model is not None:
            self._log("Interactive metal segmentation...")
            if self.interactive_metal_segmentation():
                metal_mask = self.processed_mask if self.processed_mask is not None else self.mask

        if metal_mask is not None:
            y_idx, x_idx = np.where(metal_mask > 0)
            if len(y_idx) > 0:
                y_min, y_max = y_idx.min(), y_idx.max()
                x_min, x_max = x_idx.min(), x_idx.max()

                pad = 20
                H, W = original.shape[:2]
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(W, x_max + pad)
                y_max = min(H, y_max + pad)

                crop = original[y_min:y_max, x_min:x_max]
                metal_mask_crop = metal_mask[y_min:y_max, x_min:x_max]
                cx, cy, cw, ch = x_min, y_min, x_max - x_min, y_max - y_min

        if crop is None:
            self._log("No SAM mask; using full image as crop.")
            crop = original
            cx, cy, cw, ch = 0, 0, original.shape[1], original.shape[0]
            metal_mask_crop = np.ones(crop.shape[:2], dtype=np.uint8)

        timings["crop"] = time.time() - t0

        t0 = time.time()
        self._log(f"Running Analysis (mode={self.mode}, SLIC n={self.n_segments})...")
        res = self._perform_analysis(crop, metal_mask_crop)
        timings["analysis"] = time.time() - t0

        full_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        if res["crop_mask"] is not None:
            full_mask[cy : cy + ch, cx : cx + cw] = res["crop_mask"]

        rust_percentage = res["rust_percentage"] * 100.0
        total_time = float(sum(timings.values()))
        self._log(f"Rust coverage (metal): {rust_percentage:.2f}% | Total time: {total_time:.3f}s")

        return dict(
            original=original,
            crop=crop,
            segments=res["segments"],
            full_mask=full_mask,
            crop_mask=res["crop_mask"],
            score_map=res["score_map"],
            pred_map=res["pred_map"],
            features=res["features"],
            segment_ids=res["segment_ids"],
            rust_scores=res["rust_scores"],     # heuristic score or rust prior
            final_scores=res.get("final_scores", res["rust_scores"]),
            rust_pred=res["rust_pred"],
            valid_segments=res["valid_mask"],
            crop_coords=(cx, cy, cw, ch),
            rust_percentage=rust_percentage,
            metal_mask=metal_mask_crop,
            timings=timings,
            threshold_used=float(res.get("threshold_used", self.rust_threshold_fallback)),
            dynamic_gates=res.get("dynamic_gates", None),
            mode=res.get("mode", self.mode),
        )

    # ---- Visualization ----
    @staticmethod
    def _tight_bbox_from_mask(mask_u8: np.ndarray, pad: int = 2) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            h, w = mask_u8.shape[:2]
            return 0, 0, w, h
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        h, w = mask_u8.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return x1, y1, x2, y2

    def visualize_detection(self, results: Dict, save_path: str | None = None) -> plt.Figure:
        crop = results["crop"]
        segments = results["segments"]
        full_mask = results["full_mask"]
        original = results["original"]
        rust_percentage = results["rust_percentage"]
        metal_mask = results.get("metal_mask", np.ones(crop.shape[:2], dtype=np.uint8)).astype(np.uint8)
        score_map = results.get("score_map", np.zeros(crop.shape[:2], dtype=np.float32))
        thr = float(results.get("threshold_used", self.rust_threshold_fallback))
        mode = results.get("mode", "unknown")

        x1, y1, x2, y2 = self._tight_bbox_from_mask(metal_mask, pad=2)
        crop_t = crop[y1:y2, x1:x2]
        mask_t = metal_mask[y1:y2, x1:x2]
        seg_t = segments[y1:y2, x1:x2]
        score_t = score_map[y1:y2, x1:x2]

        metal_input = crop_t.copy()
        metal_input[mask_t == 0] = 255

        vis_segments = mark_boundaries(cv2.cvtColor(metal_input, cv2.COLOR_BGR2RGB), seg_t)

        lab_crop = cv2.cvtColor(crop_t, cv2.COLOR_BGR2Lab)
        lab_a = lab_crop[:, :, 1].copy()
        lab_a[mask_t == 0] = 128

        score_vis = cv2.normalize(score_t, None, 0, 1, cv2.NORM_MINMAX)
        score_vis_rgb = plt.cm.inferno(score_vis)[:, :, :3]
        score_vis_rgb[mask_t == 0] = 1.0

        vis_final = original.copy()
        vis_final[full_mask == 1] = [0, 0, 255]
        vis_final = cv2.addWeighted(original, 0.6, vis_final, 0.4, 0)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            f"Rust Detection [{mode.upper()}] | Segments: {self.n_segments}\n"
            f"Coverage (Metal): {rust_percentage:.1f}% | Threshold used: {thr:.2f}",
            fontsize=12,
        )

        axes[0, 0].imshow(cv2.cvtColor(metal_input, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1. Metal Input (Tight Crop, No Black BG)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(vis_segments)
        axes[0, 1].set_title(f"2. SLIC (tight) ({len(np.unique(seg_t))} segments)")
        axes[0, 1].axis("off")

        im_a = axes[0, 2].imshow(lab_a, cmap="RdYlGn_r", vmin=0, vmax=255)
        axes[0, 2].set_title("3. LAB a* (Green↔Red) [tight]")
        axes[0, 2].axis("off")
        plt.colorbar(im_a, ax=axes[0, 2], fraction=0.046)

        axes[1, 0].imshow(mask_t, cmap="gray")
        axes[1, 0].set_title("4. Metal Mask (tight)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(score_vis_rgb)
        axes[1, 1].set_title("5. Score Heatmap (per-segment) [tight]")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("6. Final Detection Result (Rust=Red)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            self._log(f"Visualization saved to: {save_path}")

        return fig


# ----------------------------- CLI + Main -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rust Detection (heuristic OR anomaly detection over segments)")
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=interactive SAM metal selection; 0=full image.")
    p.add_argument("--sam_checkpoint", type=str, default="sam2.1_b.pt", help="Path to SAM2 checkpoint.")
    p.add_argument("--n_segments", type=int, default=10000)
    p.add_argument("--fast_mode", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    # Mode
    p.add_argument(
        "--mode",
        type=str,
        default="anomaly",
        choices=["anomaly", "heuristic"],
        help="anomaly = anomaly detection + rust prior, heuristic = original color/texture scoring",
    )

    # Heuristic settings
    p.add_argument(
        "--rust_threshold_fallback",
        type=float,
        default=0.60,
        help="Fallback score threshold (heuristic or anomaly fallback).",
    )
    p.add_argument("--dynamic_feature_gates", type=int, default=1)
    p.add_argument("--dynamic_score_threshold", type=int, default=1)
    p.add_argument("--min_valid_segments_for_dynamic", type=int, default=20)
    p.add_argument("--ensure_one_positive", type=int, default=1)
    p.add_argument("--otsu_bias", type=float, default=-0.05)

    # Anomaly settings
    p.add_argument("--anomaly_weight", type=float, default=0.70)
    p.add_argument("--rust_prior_weight", type=float, default=0.30)
    p.add_argument("--anomaly_threshold_mode", type=str, default="otsu", choices=["otsu", "percentile"])
    p.add_argument("--anomaly_percentile", type=float, default=92.0)
    p.add_argument("--anomaly_model_path", type=str, default="", help="Path to .npz anomaly model (load/save).")
    p.add_argument("--clean_dir", type=str, default="", help="Directory of CLEAN images to fit anomaly model.")

    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    detector = FastRustDetector(
        n_segments=args.n_segments,
        fast_mode=bool(args.fast_mode),
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        rust_threshold_fallback=float(args.rust_threshold_fallback),
        ensure_one_positive=bool(args.ensure_one_positive),
        dynamic_feature_gates=bool(args.dynamic_feature_gates),
        dynamic_score_threshold=bool(args.dynamic_score_threshold),
        min_valid_segments_for_dynamic=int(args.min_valid_segments_for_dynamic),
        otsu_bias=float(args.otsu_bias),
        mode=str(args.mode),
        anomaly_weight=float(args.anomaly_weight),
        rust_prior_weight=float(args.rust_prior_weight),
        anomaly_threshold_mode=str(args.anomaly_threshold_mode),
        anomaly_percentile=float(args.anomaly_percentile),
        anomaly_model_path=str(args.anomaly_model_path),
        clean_dir=str(args.clean_dir),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))
    out = f"results/{os.path.splitext(os.path.basename(image_path))[0]}_{results.get('mode','mode')}_result.png"
    detector.visualize_detection(results, save_path=out)


if __name__ == "__main__":
    main()
