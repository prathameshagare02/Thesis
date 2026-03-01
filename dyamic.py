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


# ----------------------------- Core Detector -----------------------------
class FastRustDetector:
    """
    Per-segment feature vectors:
      - Each SLIC segment == one feature vector (your "one cluster -> one vector").
      - Each feature vector is classified rust/no-rust.

    Dynamic thresholds:
      - Dynamic feature gates (Option A): derive feature cutoffs from percentiles of METAL segments in the image.
      - Dynamic score threshold (Option B): derive final score cutoff from Otsu over rust_scores (metal segments).

    Updates in this version:
      1) "Metal Input" visualization is tightly cropped to the metal mask bbox, with NO black background.
      2) Dark brown / near-black rust is more likely to classify as rust:
         - relaxed warm-veto for very dark segments
         - stronger "dark rust" allowance
         - slightly more permissive Otsu threshold (biased lower + wider clamp)
    """

    def __init__(
        self,
        n_segments: int = 10000,
        fast_mode: bool = False,
        verbose: bool = True,
        sam_checkpoint: str = "sam2.1_b.pt",
        # Fallback / default behavior:
        rust_threshold_fallback: float = 0.60,
        ensure_one_positive: bool = True,
        # Dynamic behavior toggles:
        dynamic_feature_gates: bool = True,
        dynamic_score_threshold: bool = True,
        # Safety knobs:
        min_valid_segments_for_dynamic: int = 20,
        # Otsu bias (negative makes it more inclusive):
        otsu_bias: float = -0.05,
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

        if self.verbose:
            self._print_backend_info()

        self.load_sam2_model()

    # ---- logging ----
    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("FastRustDetector initialized:")
        print("  Mode: PER-SEGMENT feature vectors (no clustering)")
        print(f"  Target segments: {self.n_segments}")
        print(f"  Fast mode: {self.fast_mode}")
        print(f"  Dynamic feature gates: {self.dynamic_feature_gates}")
        print(f"  Dynamic score threshold (Otsu): {self.dynamic_score_threshold}")
        print(f"  Fallback rust threshold: {self.rust_threshold_fallback:.2f}")
        print(f"  Ensure one positive: {self.ensure_one_positive}")
        print(f"  Min valid segments for dynamic: {self.min_valid_segments_for_dynamic}")
        print(f"  Otsu bias: {self.otsu_bias:+.2f}")

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

    # --------------------- Dynamic thresholds helpers ---------------------
    @staticmethod
    def _robust_percentile(vals: np.ndarray, q: float, default: float, min_n: int = 10) -> float:
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            return float(default)
        return float(np.percentile(vals, q))

    def _compute_dynamic_gates(self, features_valid: np.ndarray) -> Dict[str, float]:
        """
        Derive dynamic gates from the image itself (metal-only segments).
        Uses percentiles; falls back to the classic constants if data is scarce.
        """
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

        # clamp to sane ranges
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
        """
        Otsu threshold on rust_scores (0..1) for metal segments.
        Slightly biased lower to include darker/brown rust.
        """
        scores = rust_scores_valid[np.isfinite(rust_scores_valid)]
        if scores.size < self.min_valid_segments_for_dynamic:
            return float(self.rust_threshold_fallback)

        s255 = np.clip(scores * 255.0, 0, 255).astype(np.uint8)

        if int(s255.max()) - int(s255.min()) < 5:
            return float(self.rust_threshold_fallback)

        t, _ = cv2.threshold(s255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0

        # Bias + clamp (more permissive than before)
        thr = float(thr + self.otsu_bias)
        thr = float(np.clip(thr, 0.25, 0.75))
        return thr

    # --------------------- Per-vector classifier ---------------------
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

        # ---------------- Stage 0: stronger dark-rust allowance ----------------
        # If very dark, allow rust with texture OR just "not-cold" b* (brownish) even if a* is low.
        very_dark_gate = max(dyn["v_very_dark"], 70.0)  # helps for your dark brown/black
        is_very_dark = (mean_v < very_dark_gate)

        textured_enough = (mean_rough > dyn["rough_hi"] * 0.75) or (mean_ent > dyn["ent_hi"] * 0.75)
        brownish_enough = (mean_b > (dyn["b_warm"] - 6.0))  # allow darker browns

        if is_very_dark and (textured_enough or brownish_enough):
            return 0.82

        # ---------------- Stage 1: vetoes (relaxed for dark segments) ----------------
        if mean_v > dyn["v_hi"]:
            return 0.0

        # Warm gate: for dark segments, relax the warm requirement (black/brown rust can have lower a*)
        dark_relax = mean_v < (dyn["v_dark"] * 0.65)
        a_min = dyn["a_warm"] - (14.0 if dark_relax else 0.0)
        b_min = dyn["b_warm"] - (14.0 if dark_relax else 0.0)
        if mean_a < a_min or mean_b < b_min:
            return 0.0

        # Brown rust can be low S; only veto very low S if not red enough and not dark.
        if (mean_s < dyn["s_low"]) and (mean_a < (dyn["a_red"] - (10.0 if dark_relax else 0.0))) and (mean_v > (dyn["v_dark"] * 0.55)):
            return 0.0

        # Yellow/brass veto
        if (35.0 < mean_h < 95.0) and (mean_s > max(75.0, dyn["s_hi"])) and (mean_v > max(175.0, dyn["v_dark"])) and (mean_a < dyn["a_red"]):
            return 0.0

        # ---------------- Stage 2: scoring ----------------
        hsv_score = 0.0

        is_rust_hue = (mean_h < 55.0 or mean_h > 165.0)
        if is_rust_hue and mean_s > max(20.0, dyn["s_low"] * 0.7):
            hsv_score += 1.0

        # Saturation: partial credit for browns
        if mean_s > dyn["s_hi"]:
            hsv_score += 1.0
        elif mean_s > dyn["s_mid"]:
            hsv_score += 0.5
        elif mean_s > dyn["s_low"]:
            hsv_score += 0.25
        else:
            # extra small credit for very dark segments with low S (dark/brown rust)
            if dark_relax and mean_v < dyn["v_dark"]:
                hsv_score += 0.15

        # Value: darker tends to be rustier
        if mean_v < dyn["v_dark"]:
            hsv_score += 1.0
        elif mean_v < (dyn["v_dark"] + 50.0):
            hsv_score += 0.5

        # Redness (Lab a*)
        if mean_a > dyn["a_red_hi"]:
            hsv_score += 2.2
        elif mean_a > dyn["a_red"]:
            hsv_score += 1.6
        elif mean_a > (dyn["a_red"] - (12.0 if dark_relax else 7.0)):
            hsv_score += 1.0

        # Warmth (Lab b*) — a bit more weight (helps brown/black rust)
        if mean_b > dyn["b_orange_hi"]:
            hsv_score += 0.8
        elif mean_b > dyn["b_orange"]:
            hsv_score += 0.4
        elif dark_relax and mean_b > dyn["b_warm"]:
            hsv_score += 0.2

        hsv_norm = min(1.0, hsv_score / 4.0)

        # Texture
        tex_rough = min(1.0, mean_rough / max(26.0, dyn["rough_hi"] * 2.0))
        tex_ent = min(1.0, mean_ent / max(16.0, dyn["ent_hi"] * 1.8))
        tex = (tex_rough + tex_ent) / 2.0
        if hsv_norm < 0.18:
            tex = 0.0

        rust_score = 0.80 * hsv_norm + 0.20 * tex

        # Brown-rust allowance: low S but dark + warm-ish
        if (mean_s < dyn["s_mid"]) and (mean_v < dyn["v_dark"]) and (mean_b > (dyn["b_warm"] - 4.0)):
            rust_score = max(rust_score, 0.74 if dark_relax else 0.70)

        # Very dark boost
        if mean_v < (very_dark_gate + 15.0) and (textured_enough or mean_b > dyn["b_warm"]):
            rust_score = max(rust_score, 0.78)

        return float(np.clip(rust_score, 0.0, 1.0))

    # ---- Segmentation analysis ----
    def _perform_segmentation_analysis(self, crop: np.ndarray, metal_mask_crop: np.ndarray) -> Dict:
        segments = slic(
            img_as_float(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            n_segments=self.n_segments,
            compactness=20,
            sigma=1,
            start_label=0,
            channel_axis=2,
        )

        fmap = self._compute_feature_maps(crop)
        fmap["metal_mask"] = metal_mask_crop

        features, segment_ids, metal_scores = self._extract_features_vectorized(fmap, segments)

        valid = metal_scores > 0.5
        valid_indices = np.where(valid)[0]
        n_valid = int(len(valid_indices))

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

        seg_areas = np.bincount(segments.ravel())
        rust_pixels = 0
        metal_pixels = 0

        final_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        score_map = np.zeros(crop.shape[:2], dtype=np.float32)
        pred_map = np.zeros(crop.shape[:2], dtype=np.uint8)

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

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = final_mask * metal_mask_crop

        pct = (rust_pixels / metal_pixels) if metal_pixels > 0 else 0.0

        if self.verbose:
            self._log(f"Valid metal segments: {n_valid}")
            self._log(f"Final score threshold used: {thr:.3f} (dynamic={self.dynamic_score_threshold})")
            if dyn_gates is not None:
                gate_str = (
                    f"a_warm={dyn_gates['a_warm']:.1f}, b_warm={dyn_gates['b_warm']:.1f}, "
                    f"a_red={dyn_gates['a_red']:.1f}, b_orange={dyn_gates['b_orange']:.1f}, "
                    f"v_dark={dyn_gates['v_dark']:.1f}, v_very_dark={dyn_gates['v_very_dark']:.1f}, "
                    f"s_low={dyn_gates['s_low']:.1f}, s_mid={dyn_gates['s_mid']:.1f}, s_hi={dyn_gates['s_hi']:.1f}"
                )
                self._log(f"Dynamic gates: {gate_str}")
            else:
                self._log("Dynamic gates: OFF / insufficient segments (using defaults)")

        return {
            "segments": segments,
            "features": features,
            "segment_ids": segment_ids,
            "metal_scores": metal_scores,
            "valid_mask": valid,
            "rust_scores": rust_scores,
            "rust_pred": rust_pred,
            "score_map": score_map,
            "pred_map": pred_map,
            "crop_mask": final_mask,
            "rust_percentage": pct,
            "threshold_used": thr,
            "dynamic_gates": dyn_gates,
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
            self._log("Cropping (robust)...")
            crop = original
            cx, cy, cw, ch = 0, 0, original.shape[1], original.shape[0]
            self._log("Robust crop removed - using full image.")
            metal_mask_crop = np.ones(crop.shape[:2], dtype=np.uint8)

        timings["crop"] = time.time() - t0

        t0 = time.time()
        self._log(f"Running Analysis (SLIC n={self.n_segments})...")
        res = self._perform_segmentation_analysis(crop, metal_mask_crop)
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
            rust_scores=res["rust_scores"],
            rust_pred=res["rust_pred"],
            valid_segments=res["valid_mask"],
            crop_coords=(cx, cy, cw, ch),
            rust_percentage=rust_percentage,
            metal_mask=metal_mask_crop,
            timings=timings,
            threshold_used=res["threshold_used"],
            dynamic_gates=res["dynamic_gates"],
        )

    # ---- Visualization ----
    @staticmethod
    def _tight_bbox_from_mask(mask_u8: np.ndarray, pad: int = 2) -> Tuple[int, int, int, int]:
        """Return tight bbox (x1,y1,x2,y2) inside mask image coordinates."""
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

        # ---- Tight crop to metal bbox for subplots 1-5 (no black background) ----
        x1, y1, x2, y2 = self._tight_bbox_from_mask(metal_mask, pad=2)
        crop_t = crop[y1:y2, x1:x2]
        mask_t = metal_mask[y1:y2, x1:x2]
        seg_t = segments[y1:y2, x1:x2]
        score_t = score_map[y1:y2, x1:x2]

        # 1) Metal Input: set non-metal pixels to white (NO black)
        metal_input = crop_t.copy()
        metal_input[mask_t == 0] = 255

        # 2) SLIC boundaries drawn on the same "white background" metal input
        vis_segments = mark_boundaries(cv2.cvtColor(metal_input, cv2.COLOR_BGR2RGB), seg_t)

        # 3) LAB a* (show only metal; outside metal -> neutral 128)
        lab_crop = cv2.cvtColor(crop_t, cv2.COLOR_BGR2Lab)
        lab_a = lab_crop[:, :, 1].copy()
        lab_a[mask_t == 0] = 128

        # 5) Rust-score heatmap inside metal (outside -> 0)
        score_vis = cv2.normalize(score_t, None, 0, 1, cv2.NORM_MINMAX)
        score_vis_rgb = plt.cm.inferno(score_vis)[:, :, :3]
        score_vis_rgb[mask_t == 0] = 1.0  # white outside metal

        # 6) Final result stays on full original (as before)
        vis_final = original.copy()
        vis_final[full_mask == 1] = [0, 0, 255]  # red in BGR
        vis_final = cv2.addWeighted(original, 0.6, vis_final, 0.4, 0)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            f"Fast Rust Detection [PER-SEGMENT] | Segments: {self.n_segments}\n"
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
        axes[1, 1].set_title("5. Per-Segment Rust Score (Heatmap) [tight]")
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
    p = argparse.ArgumentParser(description="Fast Rust Detection (per-segment, dynamic thresholds)")
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=interactive SAM metal selection; 0=full image.")
    p.add_argument("--sam_checkpoint", type=str, default="sam2.1_b.pt", help="Path to SAM2 checkpoint.")
    p.add_argument("--n_segments", type=int, default=10000)
    p.add_argument("--fast_mode", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    p.add_argument(
        "--rust_threshold_fallback",
        type=float,
        default=0.60,
        help="Fallback score threshold when dynamic score thresholding is off/insufficient data.",
    )
    p.add_argument(
        "--dynamic_feature_gates",
        type=int,
        default=1,
        help="1=derive feature gates from per-image percentiles (metal segments); 0=use defaults.",
    )
    p.add_argument(
        "--dynamic_score_threshold",
        type=int,
        default=1,
        help="1=derive final score threshold from Otsu on rust scores; 0=use fallback threshold.",
    )
    p.add_argument(
        "--min_valid_segments_for_dynamic",
        type=int,
        default=20,
        help="Minimum number of metal segments required to enable dynamic thresholds.",
    )
    p.add_argument(
        "--ensure_one_positive",
        type=int,
        default=1,
        help="If 1 and nothing passes threshold, mark best-scoring segment as rust.",
    )
    p.add_argument(
        "--otsu_bias",
        type=float,
        default=-0.05,
        help="Bias applied to Otsu-derived threshold (negative => more inclusive).",
    )
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
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))
    out = f"results/{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    detector.visualize_detection(results, save_path=out)


if __name__ == "__main__":
    main()