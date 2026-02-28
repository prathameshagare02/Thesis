from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

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
    Pipeline (SAM3 ROI -> SLIC -> per-segment scoring):

      1) SAM3 text prompt ("metal") on the FULL image.
      2) Select the detection with MAX metal confidence score.
      3) Crop the FULL IMAGE using that MAX-score BOX (optionally padded).
         IMPORTANT: further processing runs on the WHOLE BOX region (not just the metal mask).
      4) Run SLIC superpixels in the cropped ROI.
      5) Compute per-segment features and rust scores.
      6) Threshold using Otsu (dynamic) or fallback.
      7) Map rust mask back to original coords for final overlay.

    Visualization:
      Shows multiple stages (input, best box, crop ROI, superpixels, rust-in-ROI, final overlay).
    """

    def __init__(
        self,
        n_segments: int = 10000,
        fast_mode: bool = False,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        rust_threshold_fallback: float = 0.60,
        ensure_one_positive: bool = True,
        dynamic_feature_gates: bool = True,
        dynamic_score_threshold: bool = True,
        min_valid_segments_for_dynamic: int = 20,
        otsu_bias: float = -0.02,
        roi_pad: int = 0,  # padding around max-score box (still "whole box", just expanded)
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

        self.roi_pad = int(roi_pad)

        # SAM3
        self.sam_checkpoint = sam_checkpoint
        self.sam3_predictor = None
        self.prompt_metal = "metal"

        self.original_image: Optional[np.ndarray] = None

        if self.verbose:
            self._print_backend_info()

        self.load_sam3_model()

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
        print(f"  SAM3 prompt_metal: {self.prompt_metal!r}")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  ROI padding: {self.roi_pad}px")

    # ---- SAM3 ----
    def load_sam3_model(self):
        """
        Loads SAM3SemanticPredictor from Ultralytics.
        If it fails, SAM3 is disabled and pipeline falls back to full-image box.
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception:
            self._log(
                "Failed to import SAM3SemanticPredictor. SAM3 will be disabled.\n"
                "Try: pip install -U ultralytics"
            )
            self.sam3_predictor = None
            return

        if not os.path.exists(self.sam_checkpoint):
            self._log(f"SAM3 checkpoint not found: {self.sam_checkpoint} (SAM3 disabled)")
            self.sam3_predictor = None
            return

        try:
            self._log(f"Loading SAM3SemanticPredictor from {self.sam_checkpoint}...")
            overrides = dict(task="segment", mode="predict", model=self.sam_checkpoint, verbose=self.verbose)
            overrides["half"] = True
            self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_predictor.setup_model()
            self._log("SAM3 model loaded successfully.")
        except Exception as e:
            self._log(f"Failed to load SAM3 model: {e}")
            self.sam3_predictor = None

    # ---- Mask Utilities ----
    @staticmethod
    def _bbox_from_mask(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return (x1, y1, x2, y2)

    def apply_morphological_operations(self, mask: np.ndarray | None):
        if mask is None:
            return None
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=2)
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
        return (mask_uint8 > 127).astype(np.uint8)

    def _sam3_best_detection_for_prompt(
        self, image_path: str, prompt: str
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        Returns:
          best_mask (H,W) uint8 in full-image coords
          best_box  (x1,y1,x2,y2) in full-image coords
          best_score float (confidence if available)
        """
        if self.sam3_predictor is None:
            return None, None, 0.0

        try:
            self.sam3_predictor.set_image(image_path)
            results = self.sam3_predictor(text=[prompt])
            if not results:
                return None, None, 0.0
            r0 = results[0]

            if getattr(r0, "masks", None) is None or r0.masks is None or r0.masks.data is None:
                return None, None, 0.0
            if len(r0.masks.data) == 0:
                return None, None, 0.0

            masks = r0.masks.data  # torch [N,H,W]
            n = len(masks)

            best_i = 0
            best_score = 0.0

            # prefer max confidence if boxes.conf exists
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "conf", None) is not None
            ):
                conf = r0.boxes.conf.detach().cpu().numpy().astype(float)
                if conf.size >= n:
                    best_i = int(np.argmax(conf[:n]))
                    best_score = float(conf[best_i])
                else:
                    best_i = int(np.argmax(conf))
                    best_score = float(conf[best_i])
            else:
                # fallback: choose largest mask
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = 1.0

            best_mask = (masks[best_i].detach().cpu().numpy() > 0.5).astype(np.uint8)
            processed = self.apply_morphological_operations(best_mask)
            if processed is not None:
                best_mask = processed

            best_box = None
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "xyxy", None) is not None
            ):
                xyxy = r0.boxes.xyxy.detach().cpu().numpy()
                if xyxy.shape[0] > best_i:
                    x1, y1, x2, y2 = xyxy[best_i]
                    best_box = (int(x1), int(y1), int(x2), int(y2))

            if best_box is None:
                best_box = self._bbox_from_mask(best_mask)

            return best_mask, best_box, float(best_score)
        except Exception as e:
            self._log(f"SAM3 best-detection error for prompt={prompt!r}: {e}")
            return None, None, 0.0

    def get_best_metal_roi(
        self, image_path: str, original: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Returns:
          metal_mask_full (HxW uint8 0/1),
          metal_box_full (x1,y1,x2,y2) from MAX-score detection,
          metal_score float

        If SAM3 fails -> full image box, mask=ones, score=0.0
        """
        mask, box, score = self._sam3_best_detection_for_prompt(image_path, self.prompt_metal)
        if mask is None or box is None or not np.any(mask):
            H, W = original.shape[:2]
            return np.ones((H, W), dtype=np.uint8), (0, 0, W, H), 0.0
        return mask.astype(np.uint8), box, float(score)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        return features, unique_segments

    # --------------------- Dynamic thresholds helpers ---------------------
    @staticmethod
    def _robust_percentile(vals: np.ndarray, q: float, default: float, min_n: int = 10) -> float:
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            return float(default)
        return float(np.percentile(vals, q))

    def _compute_dynamic_gates(self, features: np.ndarray) -> Dict[str, float]:
        # use ALL segments inside ROI box (since ROI box is what user asked to process)
        a_vals = features[:, 1] if features.size else np.array([])
        b_vals = features[:, 2] if features.size else np.array([])
        s_vals = features[:, 13] if features.size else np.array([])
        v_vals = features[:, 14] if features.size else np.array([])
        rough_vals = features[:, 6] if features.size else np.array([])
        ent_vals = features[:, 7] if features.size else np.array([])

        dyn = {
            "a_warm": self._robust_percentile(a_vals, 35, 126.0),
            "b_warm": self._robust_percentile(b_vals, 35, 124.0),
            "a_red": self._robust_percentile(a_vals, 65, 140.0),
            "a_red_hi": self._robust_percentile(a_vals, 75, 150.0),
            "b_orange": self._robust_percentile(b_vals, 60, 135.0),
            "b_orange_hi": self._robust_percentile(b_vals, 70, 145.0),
            "v_dark": self._robust_percentile(v_vals, 20, 155.0),
            "v_very_dark": self._robust_percentile(v_vals, 10, 55.0),
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

    def _compute_dynamic_score_threshold_otsu(self, rust_scores: np.ndarray) -> float:
        scores = rust_scores[np.isfinite(rust_scores)]
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

    # ---- Segmentation analysis (inside ROI box) ----
    def _perform_segmentation_analysis(self, crop: np.ndarray) -> Dict:
        segments = slic(
            img_as_float(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            n_segments=self.n_segments,
            compactness=20,
            sigma=1,
            start_label=0,
            channel_axis=2,
        )

        fmap = self._compute_feature_maps(crop)
        features, segment_ids = self._extract_features_vectorized(fmap, segments)

        n_seg = int(features.shape[0])
        dyn_gates: Optional[Dict[str, float]] = None
        if self.dynamic_feature_gates and n_seg >= self.min_valid_segments_for_dynamic:
            dyn_gates = self._compute_dynamic_gates(features)

        rust_scores = np.zeros(n_seg, dtype=np.float32)
        for i in range(n_seg):
            rust_scores[i] = self._rust_score_for_feature(features[i], dyn=dyn_gates)

        if self.dynamic_score_threshold and n_seg >= self.min_valid_segments_for_dynamic:
            thr = self._compute_dynamic_score_threshold_otsu(rust_scores)
        else:
            thr = float(self.rust_threshold_fallback)

        rust_pred = (rust_scores >= thr).astype(np.uint8)
        if self.ensure_one_positive and int(np.sum(rust_pred)) == 0 and n_seg > 0:
            rust_pred[int(np.argmax(rust_scores))] = 1

        seg_areas = np.bincount(segments.ravel())
        rust_pixels = 0
        total_pixels = 0

        final_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        score_map = np.zeros(crop.shape[:2], dtype=np.float32)
        pred_map = np.zeros(crop.shape[:2], dtype=np.uint8)

        for i in range(n_seg):
            sid = int(segment_ids[i])
            area = int(seg_areas[sid]) if sid < len(seg_areas) else 0
            total_pixels += area

            m = (segments == sid)
            score_map[m] = float(rust_scores[i])
            pred_map[m] = int(rust_pred[i])

            if rust_pred[i] == 1:
                rust_pixels += area
                final_mask[m] = 1

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        pct = (rust_pixels / total_pixels) if total_pixels > 0 else 0.0

        if self.verbose:
            self._log(f"ROI segments: {n_seg}")
            self._log(f"Final score threshold used: {thr:.3f} (dynamic={self.dynamic_score_threshold})")

        return {
            "segments": segments,
            "features": features,
            "segment_ids": segment_ids,
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

        # 1) SAM3 -> best metal box (max score)
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log(f"SAM3: selecting MAX-score box for prompt={self.prompt_metal!r} ...")
            metal_mask_full, metal_box_full, metal_box_score = self.get_best_metal_roi(image_path, original)
        else:
            H, W = original.shape[:2]
            metal_mask_full = np.ones((H, W), dtype=np.uint8)
            metal_box_full = (0, 0, W, H)
            metal_box_score = 0.0
        timings["sam3"] = time.time() - t0

        # 2) Crop using the FULL BOX (NOT just metal mask)
        t0 = time.time()
        x1, y1, x2, y2 = metal_box_full
        H, W = original.shape[:2]
        pad = int(self.roi_pad)

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        crop = original[y1p:y2p, x1p:x2p].copy()
        crop_coords = (x1p, y1p, x2p - x1p, y2p - y1p)
        metal_mask_crop = metal_mask_full[y1p:y2p, x1p:x2p].astype(np.uint8)
        timings["crop"] = time.time() - t0

        # 3) SLIC + per-seg scoring inside crop box
        t0 = time.time()
        self._log(f"Running analysis INSIDE max-score box (SLIC n={self.n_segments})...")
        res = self._perform_segmentation_analysis(crop)
        timings["analysis"] = time.time() - t0

        # 4) Map rust mask back to full
        full_rust_mask = np.zeros((H, W), dtype=np.uint8)
        full_rust_mask[y1p:y2p, x1p:x2p] = res["crop_mask"]

        rust_percentage = res["rust_percentage"] * 100.0
        total_time = float(sum(timings.values()))
        self._log(
            f"Metal box score: {metal_box_score:.2f} | "
            f"Rust coverage (within box): {rust_percentage:.2f}% | Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            metal_mask_full=metal_mask_full,
            metal_box_full=metal_box_full,
            metal_box_score=float(metal_box_score),
            crop=crop,
            crop_coords=crop_coords,
            metal_mask_crop=metal_mask_crop,
            segments=res["segments"],
            full_mask=full_rust_mask,
            crop_mask=res["crop_mask"],
            score_map=res["score_map"],
            pred_map=res["pred_map"],
            threshold_used=res["threshold_used"],
            dynamic_gates=res["dynamic_gates"],
            rust_percentage=rust_percentage,
            timings=timings,
        )

    # ---- Visualization (MULTI-STAGE) ----
    def visualize_detection(self, results: Dict, save_path: str | None = None) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]
        segments = results["segments"]
        full_mask = results["full_mask"]
        crop_mask = results["crop_mask"]
        score_map = results["score_map"]
        metal_mask_crop = results["metal_mask_crop"]

        (bx1, by1, bx2, by2) = results["metal_box_full"]
        bscore = float(results["metal_box_score"])
        rust_percentage = float(results["rust_percentage"])
        thr = float(results.get("threshold_used", self.rust_threshold_fallback))

        # Stage 1: input image
        stage1 = original.copy()

        # Stage 2: input + best box (max metal score)
        stage2 = original.copy()
        cv2.rectangle(stage2, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        cv2.putText(
            stage2,
            f"best metal box {bscore:.2f}",
            (bx1, max(0, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # Stage 3: cropped ROI (whole box), show metal mask (optional sanity)
        stage3 = crop.copy()
        # blue overlay where SAM3 thinks metal is (still processing whole box regardless)
        if metal_mask_crop is not None and metal_mask_crop.size > 0:
            overlay = stage3.copy()
            overlay[metal_mask_crop == 1] = [255, 0, 0]  # blue in BGR
            stage3 = cv2.addWeighted(stage3, 0.7, overlay, 0.3, 0)

        # Stage 4: superpixels boundaries on cropped ROI
        # mark_boundaries expects RGB float image
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        spx = mark_boundaries(crop_rgb, segments, color=(1, 1, 0), mode="thick")  # yellow-ish boundaries
        stage4 = (spx * 255).astype(np.uint8)
        stage4 = cv2.cvtColor(stage4, cv2.COLOR_RGB2BGR)

        # Stage 5: rust mask overlay on cropped ROI
        stage5 = crop.copy()
        rust_overlay_roi = stage5.copy()
        rust_overlay_roi[crop_mask == 1] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.6, rust_overlay_roi, 0.4, 0)

        # Stage 6: final overlay on full original
        stage6 = original.copy()
        rust_overlay_full = stage6.copy()
        rust_overlay_full[full_mask == 1] = [0, 0, 255]
        stage6 = cv2.addWeighted(stage6, 0.6, rust_overlay_full, 0.4, 0)
        # draw box again
        cv2.rectangle(stage6, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Rust Detection Stages (SAM3 max-score box -> crop box -> SLIC superpixels -> rust)\n"
            f"Metal box score={bscore:.2f} | Segments={self.n_segments} | "
            f"Coverage (within box)={rust_percentage:.1f}% | Threshold={thr:.2f}",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6]
        titles = [
            "1) Input image",
            "2) Input + best metal box (max score)",
            "3) Cropped ROI (whole box) + metal mask (blue)",
            "4) Superpixels (SLIC boundaries) in ROI",
            "5) Rust result in ROI (red)",
            "6) Final rust overlay on input (red) + box",
        ]

        for ax, img, title in zip(axes.ravel(), imgs, titles):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            self._log(f"Visualization saved to: {save_path}")

        return fig


# ----------------------------- CLI + Main -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rust Detection: SAM3 max-score box crop -> SLIC superpixels -> Otsu threshold")
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3 to find max-score metal box; 0=use full image.")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around the max-score box before cropping.")

    p.add_argument("--n_segments", type=int, default=10000)
    p.add_argument("--fast_mode", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    p.add_argument("--rust_threshold_fallback", type=float, default=0.60)
    p.add_argument("--dynamic_feature_gates", type=int, default=1)
    p.add_argument("--dynamic_score_threshold", type=int, default=1)
    p.add_argument("--min_valid_segments_for_dynamic", type=int, default=20)
    p.add_argument("--ensure_one_positive", type=int, default=1)
    p.add_argument("--otsu_bias", type=float, default=-0.02)
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
        roi_pad=int(args.roi_pad),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))
    out = f"results/New/{os.path.splitext(os.path.basename(image_path))[0]}_stages.png"
    detector.visualize_detection(results, save_path=out)


if __name__ == "__main__":
    main()