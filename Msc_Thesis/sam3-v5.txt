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
class FastRustDetectorSAM3TwoPrompts:
    """
    UPDATED PIPELINE (SAM3 two prompts -> ROI -> SLIC -> per-segment probability):

      1) Run SAM3 text prompt for BOTH:
           - prompt_clean: "clean shiny metal"
           - prompt_rust:  "rusty metal"
         on the FULL image.

      2) For each prompt, select its best detection (highest conf if available, else largest mask).
         This yields:
           - mask_clean_full, box_clean_full, score_clean
           - mask_rust_full,  box_rust_full,  score_rust

      3) Compute prompt-level probabilities:
           p_rust_prompt = softmax([score_clean, score_rust])[rust]
           p_clean_prompt = 1 - p_rust_prompt
         (fallback if scores missing: derive from mask areas)

      4) ROI: use union of the two best boxes (with optional padding).
         Further processing runs on the WHOLE ROI BOX.

      5) Run SLIC superpixels in cropped ROI.

      6) For each segment:
           - compute handcrafted rust_likelihood from LAB/HSV/texture (your original scorer)
           - compute SAM3 segment prior from overlaps with prompt masks + prompt-level probability
           - fuse => per-segment rust probability in [0,1]

      7) Map prob map + hard mask back to original coords.

    Outputs:
      - prob_map_full (float32 0..1)
      - mask_full (uint8 0/1) from threshold on probability (dynamic Otsu or fixed fallback)
      - prompt probabilities and best boxes

    Visualization:
      2x4 grid:
        1) input
        2) input + best boxes (clean=green, rust=red)
        3) cropped ROI + prompt masks overlay
        4) superpixels boundaries
        5) rust probability heatmap in ROI
        6) rust mask in ROI
        7) full-image probability heatmap overlay
        8) final mask overlay + boxes
    """

    def __init__(
        self,
        n_segments: int = 8000,
        fast_mode: bool = False,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        roi_pad: int = 0,
        # probability fusion controls
        prior_weight: float = 0.45,  # SAM3 prior weight
        lik_weight: float = 0.55,  # handcrafted likelihood weight
        overlap_weight: float = 0.75,  # weight of overlap-based prior vs prompt-level prob
        # thresholding
        prob_threshold_fallback: float = 0.55,
        dynamic_prob_threshold: bool = True,
        min_valid_segments_for_dynamic: int = 20,
        otsu_bias: float = -0.02,
        ensure_one_positive: bool = True,
        # prompts
        prompt_clean: str = "clean shiny metal",
        prompt_rust: str = "rusty metal",
    ):
        self.n_segments = int(n_segments)
        self.fast_mode = bool(fast_mode)
        self.verbose = bool(verbose)

        self.sam_checkpoint = sam_checkpoint
        self.sam3_predictor = None

        self.roi_pad = int(roi_pad)

        self.prior_weight = float(prior_weight)
        self.lik_weight = float(lik_weight)
        s = self.prior_weight + self.lik_weight
        if s <= 1e-9:
            self.prior_weight, self.lik_weight = 0.5, 0.5
        else:
            self.prior_weight /= s
            self.lik_weight /= s

        self.overlap_weight = float(np.clip(overlap_weight, 0.0, 1.0))

        self.prob_threshold_fallback = float(prob_threshold_fallback)
        self.dynamic_prob_threshold = bool(dynamic_prob_threshold)
        self.min_valid_segments_for_dynamic = int(min_valid_segments_for_dynamic)
        self.otsu_bias = float(otsu_bias)
        self.ensure_one_positive = bool(ensure_one_positive)

        self.prompt_clean = str(prompt_clean)
        self.prompt_rust = str(prompt_rust)

        self.original_image: Optional[np.ndarray] = None

        if self.verbose:
            self._print_backend_info()

        self.load_sam3_model()

    # ---- logging ----
    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("FastRustDetectorSAM3TwoPrompts initialized:")
        print("  SAM3 two-prompts -> per-segment probability fusion")
        print(f"  Target segments: {self.n_segments}")
        print(f"  Fast mode: {self.fast_mode}")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  Prompts: clean={self.prompt_clean!r}, rust={self.prompt_rust!r}")
        print(f"  ROI padding: {self.roi_pad}px")
        print(f"  Fusion weights: prior={self.prior_weight:.2f}, likelihood={self.lik_weight:.2f}")
        print(f"  Prior overlap weight: {self.overlap_weight:.2f}")
        print(f"  Dynamic prob threshold (Otsu): {self.dynamic_prob_threshold}")
        print(f"  Fallback prob threshold: {self.prob_threshold_fallback:.2f}")
        print(f"  Ensure one positive: {self.ensure_one_positive}")
        print(f"  Min valid segments for dynamic: {self.min_valid_segments_for_dynamic}")
        print(f"  Otsu bias: {self.otsu_bias:+.2f}")

    # ---- SAM3 ----
    def load_sam3_model(self):
        """
        Loads SAM3SemanticPredictor from Ultralytics.
        If it fails, SAM3 is disabled and pipeline falls back to full-image ROI + no prompt prior.
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

    @staticmethod
    def _union_boxes(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]):
        if a is None:
            return b
        if b is None:
            return a
        x1 = min(a[0], b[0])
        y1 = min(a[1], b[1])
        x2 = max(a[2], b[2])
        y2 = max(a[3], b[3])
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
          best_score float (confidence if available, else derived from area)
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

            # Prefer confidence if available
            conf_arr = None
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "conf", None) is not None
            ):
                try:
                    conf_arr = r0.boxes.conf.detach().cpu().numpy().astype(float)
                except Exception:
                    conf_arr = None

            if conf_arr is not None and conf_arr.size > 0:
                if conf_arr.size >= n:
                    best_i = int(np.argmax(conf_arr[:n]))
                    best_score = float(conf_arr[best_i])
                else:
                    best_i = int(np.argmax(conf_arr))
                    best_score = float(conf_arr[best_i])
            else:
                # Fallback: choose largest mask and use normalized area as score
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = float(areas[best_i])  # raw area for now

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
                try:
                    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
                    if xyxy.shape[0] > best_i:
                        x1, y1, x2, y2 = xyxy[best_i]
                        best_box = (int(x1), int(y1), int(x2), int(y2))
                except Exception:
                    best_box = None

            if best_box is None:
                best_box = self._bbox_from_mask(best_mask)

            return best_mask.astype(np.uint8), best_box, float(best_score)
        except Exception as e:
            self._log(f"SAM3 best-detection error for prompt={prompt!r}: {e}")
            return None, None, 0.0

    @staticmethod
    def _softmax2(a: float, b: float, temp: float = 1.0) -> Tuple[float, float]:
        t = max(1e-6, float(temp))
        x = np.array([a, b], dtype=np.float32) / t
        x = x - float(np.max(x))
        e = np.exp(x)
        s = float(np.sum(e)) if float(np.sum(e)) > 0 else 1.0
        p0 = float(e[0] / s)
        p1 = float(e[1] / s)
        return p0, p1

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
    def _compute_dynamic_prob_threshold_otsu(self, probs: np.ndarray) -> float:
        p = probs[np.isfinite(probs)]
        if p.size < self.min_valid_segments_for_dynamic:
            return float(self.prob_threshold_fallback)

        p255 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
        if int(p255.max()) - int(p255.min()) < 5:
            return float(self.prob_threshold_fallback)

        t, _ = cv2.threshold(p255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
        thr = float(thr + self.otsu_bias)
        thr = float(np.clip(thr, 0.25, 0.85))
        return thr

    # --------------------- Handcrafted likelihood (your scorer) ---------------------
    def _rust_likelihood_for_feature(self, fv: np.ndarray) -> float:
        """
        Returns a "rust-likelihood" in [0,1] from your prior rules.
        This is essentially your previous rust_score.
        """
        mean_a = float(fv[1])
        mean_b = float(fv[2])
        mean_rough = float(fv[6])
        mean_ent = float(fv[7])
        mean_h = float(fv[12])
        mean_s = float(fv[13])
        mean_v = float(fv[14])

        # Conservative-ish defaults (from your updated gates)
        dyn = {
            "a_warm": 140.0,
            "b_warm": 140.0,
            "a_red": 155.0,
            "a_red_hi": 170.0,
            "b_orange": 155.0,
            "b_orange_hi": 170.0,
            "v_dark": 175.0,
            "v_very_dark": 80.0,
            "v_hi": 245.0,
            "s_low": 55.0,
            "s_mid": 75.0,
            "s_hi": 100.0,
            "rough_hi": 20.0,
            "ent_hi": 12.0,
        }

        very_dark_gate = max(dyn["v_very_dark"], 80.0)
        is_very_dark = (mean_v < very_dark_gate)

        textured_enough = (mean_rough > dyn["rough_hi"] * 0.75) or (mean_ent > dyn["ent_hi"] * 0.75)
        brownish_enough = (mean_b > (dyn["b_warm"] - 6.0))

        if is_very_dark and (textured_enough or brownish_enough):
            return 0.82

        if mean_v > dyn["v_hi"]:
            return 0.0

        dark_relax = mean_v < (dyn["v_dark"] * 0.65)
        a_min = dyn["a_warm"] - (10.0 if dark_relax else 0.0)
        b_min = dyn["b_warm"] - (10.0 if dark_relax else 0.0)
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

        # Rust hue gate (OpenCV H 0..179)
        is_rust_hue = (mean_h < 40.0 or mean_h > 170.0)

        if is_rust_hue and mean_s > max(25.0, dyn["s_low"] * 0.75):
            hsv_score += 1.0

        if mean_s > dyn["s_hi"]:
            hsv_score += 1.0
        elif mean_s > dyn["s_mid"]:
            hsv_score += 0.5
        elif mean_s > dyn["s_low"]:
            hsv_score += 0.25
        else:
            if dark_relax and mean_v < dyn["v_dark"]:
                hsv_score += 0.05

        if mean_v < dyn["v_dark"]:
            hsv_score += 1.0
        elif mean_v < (dyn["v_dark"] + 50.0):
            hsv_score += 0.5

        if mean_a > dyn["a_red_hi"]:
            hsv_score += 2.2
        elif mean_a > dyn["a_red"]:
            hsv_score += 1.6
        elif mean_a > (dyn["a_red"] - (10.0 if dark_relax else 6.0)):
            hsv_score += 1.0

        if mean_b > dyn["b_orange_hi"]:
            hsv_score += 0.8
        elif mean_b > dyn["b_orange"]:
            hsv_score += 0.4
        elif dark_relax and mean_b > dyn["b_warm"]:
            hsv_score += 0.15

        hsv_norm = min(1.0, hsv_score / 4.0)

        tex_rough = min(1.0, mean_rough / max(28.0, dyn["rough_hi"] * 2.0))
        tex_ent = min(1.0, mean_ent / max(18.0, dyn["ent_hi"] * 1.8))
        tex = (tex_rough + tex_ent) / 2.0
        if hsv_norm < 0.20:
            tex = 0.0

        rust_score = 0.82 * hsv_norm + 0.18 * tex

        if (mean_s < dyn["s_mid"]) and (mean_v < dyn["v_dark"]) and (mean_b > (dyn["b_warm"] - 4.0)):
            rust_score = max(rust_score, 0.72 if dark_relax else 0.68)

        if mean_v < (very_dark_gate + 15.0) and (textured_enough or mean_b > dyn["b_warm"]):
            rust_score = max(rust_score, 0.78)

        return float(np.clip(rust_score, 0.0, 1.0))

    # --------------------- Segmentation + probability fusion ---------------------
    def _perform_segmentation_probability(
        self,
        crop: np.ndarray,
        clean_mask_crop: Optional[np.ndarray],
        rust_mask_crop: Optional[np.ndarray],
        p_rust_prompt: float,
    ) -> Dict:
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

        # segment-wise likelihood (handcrafted)
        lik = np.zeros(n_seg, dtype=np.float32)
        for i in range(n_seg):
            lik[i] = self._rust_likelihood_for_feature(features[i])

        # segment-wise SAM3 prior (from overlaps + prompt-level prob)
        prior = np.full(n_seg, float(p_rust_prompt), dtype=np.float32)
        if clean_mask_crop is not None and rust_mask_crop is not None:
            clean_u8 = (clean_mask_crop > 0).astype(np.uint8)
            rust_u8 = (rust_mask_crop > 0).astype(np.uint8)

            # compute per-segment overlap fractions efficiently
            seg_flat = segments.ravel()
            max_sid = int(seg_flat.max()) if seg_flat.size else 0
            counts = np.bincount(seg_flat, minlength=max_sid + 1).astype(np.float32) + 1e-6

            rust_counts = np.bincount(seg_flat, weights=rust_u8.ravel().astype(np.float32), minlength=max_sid + 1)
            clean_counts = np.bincount(seg_flat, weights=clean_u8.ravel().astype(np.float32), minlength=max_sid + 1)

            rust_frac = rust_counts / counts
            clean_frac = clean_counts / counts

            # map to our segment_ids list
            rf = rust_frac[segment_ids.astype(int)]
            cf = clean_frac[segment_ids.astype(int)]

            # overlap-based rust prior: normalize rf vs (rf+cf)
            ov = rf / (rf + cf + 1e-6)

            # fuse overlap prior with prompt-level prior
            prior = (self.overlap_weight * ov + (1.0 - self.overlap_weight) * float(p_rust_prompt)).astype(np.float32)

        # final prob fusion
        prob = (self.prior_weight * prior + self.lik_weight * lik).astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)

        # threshold
        if self.dynamic_prob_threshold and n_seg >= self.min_valid_segments_for_dynamic:
            thr = self._compute_dynamic_prob_threshold_otsu(prob)
        else:
            thr = float(self.prob_threshold_fallback)

        pred = (prob >= thr).astype(np.uint8)
        if self.ensure_one_positive and int(np.sum(pred)) == 0 and n_seg > 0:
            pred[int(np.argmax(prob))] = 1

        # build maps
        prob_map = np.zeros(crop.shape[:2], dtype=np.float32)
        pred_map = np.zeros(crop.shape[:2], dtype=np.uint8)

        for i in range(n_seg):
            sid = int(segment_ids[i])
            m = (segments == sid)
            prob_map[m] = float(prob[i])
            pred_map[m] = int(pred[i])

        # smooth mask a bit
        kernel = np.ones((3, 3), np.uint8)
        pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_OPEN, kernel)
        pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_CLOSE, kernel)

        rust_pct = float(pred_map.mean()) * 100.0

        if self.verbose:
            self._log(f"ROI segments: {n_seg}")
            self._log(f"Final probability threshold used: {thr:.3f} (dynamic={self.dynamic_prob_threshold})")
            self._log(f"Rust coverage (ROI mask pixels): {rust_pct:.2f}%")

        return {
            "segments": segments,
            "features": features,
            "segment_ids": segment_ids,
            "likelihood": lik,
            "prior": prior,
            "prob": prob,
            "prob_map": prob_map,
            "pred_map": pred_map,
            "threshold_used": thr,
            "rust_percentage": rust_pct,
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
        H, W = original.shape[:2]
        timings["load"] = time.time() - t0

        # 1) SAM3 two prompts
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log("SAM3: running two prompts (clean vs rust) ...")
            mask_clean_full, box_clean_full, score_clean = self._sam3_best_detection_for_prompt(image_path, self.prompt_clean)
            mask_rust_full, box_rust_full, score_rust = self._sam3_best_detection_for_prompt(image_path, self.prompt_rust)

            # If scores were derived from area (very large numbers), normalize by image area
            img_area = float(H * W) if H * W > 0 else 1.0
            if score_clean > 1.0:
                score_clean = float(score_clean / img_area)
            if score_rust > 1.0:
                score_rust = float(score_rust / img_area)

            # If both missing, fall back
            if mask_clean_full is None and mask_rust_full is None:
                mask_clean_full = np.ones((H, W), dtype=np.uint8)
                mask_rust_full = np.zeros((H, W), dtype=np.uint8)
                box_clean_full = (0, 0, W, H)
                box_rust_full = (0, 0, W, H)
                score_clean = 0.0
                score_rust = 0.0
        else:
            mask_clean_full = np.ones((H, W), dtype=np.uint8)
            mask_rust_full = np.zeros((H, W), dtype=np.uint8)
            box_clean_full = (0, 0, W, H)
            box_rust_full = (0, 0, W, H)
            score_clean = 0.0
            score_rust = 0.0

        timings["sam3"] = time.time() - t0

        # prompt probabilities
        if interactive and self.sam3_predictor is not None:
            # softmax on scores (temperature=1)
            p_clean, p_rust = self._softmax2(float(score_clean), float(score_rust), temp=1.0)
        else:
            p_clean, p_rust = 0.5, 0.5

        # 2) ROI union box
        t0 = time.time()
        roi_box = self._union_boxes(box_clean_full, box_rust_full)
        if roi_box is None:
            roi_box = (0, 0, W, H)

        x1, y1, x2, y2 = roi_box
        pad = int(self.roi_pad)
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        crop = original[y1p:y2p, x1p:x2p].copy()
        crop_coords = (x1p, y1p, x2p - x1p, y2p - y1p)

        clean_mask_crop = None
        rust_mask_crop = None
        if mask_clean_full is not None:
            clean_mask_crop = mask_clean_full[y1p:y2p, x1p:x2p].astype(np.uint8)
        if mask_rust_full is not None:
            rust_mask_crop = mask_rust_full[y1p:y2p, x1p:x2p].astype(np.uint8)

        timings["crop"] = time.time() - t0

        # 3) SLIC + per-seg probability
        t0 = time.time()
        self._log(f"Running probability analysis INSIDE ROI (SLIC n={self.n_segments})...")
        res = self._perform_segmentation_probability(
            crop=crop,
            clean_mask_crop=clean_mask_crop,
            rust_mask_crop=rust_mask_crop,
            p_rust_prompt=float(p_rust),
        )
        timings["analysis"] = time.time() - t0

        # 4) Map back to full
        t0 = time.time()
        prob_map_full = np.zeros((H, W), dtype=np.float32)
        mask_full = np.zeros((H, W), dtype=np.uint8)

        prob_map_full[y1p:y2p, x1p:x2p] = res["prob_map"]
        mask_full[y1p:y2p, x1p:x2p] = res["pred_map"]
        timings["map_back"] = time.time() - t0

        total_time = float(sum(timings.values()))
        self._log(
            f"SAM3 prompt probs: P(clean)={p_clean:.3f}, P(rust)={p_rust:.3f} | "
            f"ROI rust%={res['rust_percentage']:.2f}% | Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            # SAM3 prompt outputs
            mask_clean_full=mask_clean_full,
            mask_rust_full=mask_rust_full,
            box_clean_full=box_clean_full,
            box_rust_full=box_rust_full,
            score_clean=float(score_clean),
            score_rust=float(score_rust),
            p_clean=float(p_clean),
            p_rust=float(p_rust),
            # ROI
            roi_box_full=(x1p, y1p, x2p, y2p),
            crop=crop,
            crop_coords=crop_coords,
            clean_mask_crop=clean_mask_crop,
            rust_mask_crop=rust_mask_crop,
            # SLIC outputs
            segments=res["segments"],
            prob_map_crop=res["prob_map"],
            mask_crop=res["pred_map"],
            threshold_used=float(res["threshold_used"]),
            rust_percentage=float(res["rust_percentage"]),
            # full maps
            prob_map_full=prob_map_full,
            mask_full=mask_full,
            timings=timings,
        )

    # ---- Visualization ----
    @staticmethod
    def _overlay_heatmap_bgr(base_bgr: np.ndarray, prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Apply a JET heatmap from prob (0..1) onto base_bgr.
        """
        p = np.clip(prob, 0.0, 1.0)
        hm = (p * 255.0).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base_bgr, 1.0 - float(alpha), hm_color, float(alpha), 0)
        return out

    def visualize_detection(self, results: Dict, save_path: str | None = None) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]
        segments = results["segments"]

        mask_crop = results["mask_crop"]
        prob_crop = results["prob_map_crop"]

        mask_full = results["mask_full"]
        prob_full = results["prob_map_full"]

        clean_mask_crop = results["clean_mask_crop"]
        rust_mask_crop = results["rust_mask_crop"]

        box_clean = results["box_clean_full"]
        box_rust = results["box_rust_full"]
        roi_box = results["roi_box_full"]

        p_clean = float(results["p_clean"])
        p_rust = float(results["p_rust"])
        thr = float(results["threshold_used"])
        rust_pct = float(results["rust_percentage"])

        # 1) input
        stage1 = original.copy()

        # 2) input + boxes
        stage2 = original.copy()
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(stage2, f"clean score", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(stage2, f"rust score", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # draw ROI union
        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(stage2, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(stage2, "ROI", (rx1, max(0, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # 3) cropped ROI + prompt masks overlay
        stage3 = crop.copy()
        if clean_mask_crop is not None:
            ov = stage3.copy()
            ov[clean_mask_crop > 0] = [0, 255, 0]  # green
            stage3 = cv2.addWeighted(stage3, 0.75, ov, 0.25, 0)
        if rust_mask_crop is not None:
            ov = stage3.copy()
            ov[rust_mask_crop > 0] = [0, 0, 255]  # red
            stage3 = cv2.addWeighted(stage3, 0.70, ov, 0.30, 0)

        # 4) superpixels boundaries
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        spx = mark_boundaries(crop_rgb, segments, color=(1, 1, 0), mode="thick")
        stage4 = (spx * 255).astype(np.uint8)
        stage4 = cv2.cvtColor(stage4, cv2.COLOR_RGB2BGR)

        # 5) probability heatmap in ROI
        stage5 = self._overlay_heatmap_bgr(crop.copy(), prob_crop, alpha=0.50)

        # 6) rust mask overlay in ROI
        stage6 = crop.copy()
        ov = stage6.copy()
        ov[mask_crop > 0] = [0, 0, 255]
        stage6 = cv2.addWeighted(stage6, 0.60, ov, 0.40, 0)

        # 7) full probability heatmap overlay
        stage7 = self._overlay_heatmap_bgr(original.copy(), prob_full, alpha=0.45)
        # boxes again
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage7, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage7, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(stage7, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        # 8) final mask overlay on full
        stage8 = original.copy()
        ov = stage8.copy()
        ov[mask_full > 0] = [0, 0, 255]
        stage8 = cv2.addWeighted(stage8, 0.60, ov, 0.40, 0)
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage8, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage8, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(stage8, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.suptitle(
            "Rust Detection (SAM3 two prompts -> SLIC -> probability)\n"
            f"P(rust|prompt)={p_rust:.3f}  P(clean|prompt)={p_clean:.3f} | "
            f"Segments={self.n_segments} | ROI rust%={rust_pct:.1f}% | Thr={thr:.2f}",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8]
        titles = [
            "1) Input image",
            "2) Input + boxes (clean=green, rust=red) + ROI (blue)",
            "3) Cropped ROI + prompt masks (clean green, rust red)",
            "4) Superpixels (SLIC boundaries) in ROI",
            "5) Rust probability heatmap in ROI",
            "6) Rust mask in ROI",
            "7) Full-image probability heatmap overlay",
            "8) Final rust mask overlay + boxes",
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
    p = argparse.ArgumentParser(
        description="Rust Detection: SAM3(clean vs rust) -> union ROI -> SLIC superpixels -> probability + Otsu threshold"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3; 0=disable SAM3 and use full image.")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around the union ROI box before cropping.")
    p.add_argument("--n_segments", type=int, default=8000)
    p.add_argument("--fast_mode", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    # fusion controls
    p.add_argument("--prior_weight", type=float, default=0.45, help="Weight for SAM3 prior in final probability.")
    p.add_argument("--lik_weight", type=float, default=0.55, help="Weight for handcrafted likelihood in final probability.")
    p.add_argument(
        "--overlap_weight",
        type=float,
        default=0.75,
        help="Weight for overlap-based prior vs prompt-level probability (0..1).",
    )

    # thresholding
    p.add_argument("--prob_threshold_fallback", type=float, default=0.55, help="Fallback probability threshold.")
    p.add_argument("--dynamic_prob_threshold", type=int, default=1, help="1=Otsu threshold on segment probabilities.")
    p.add_argument("--min_valid_segments_for_dynamic", type=int, default=20)
    p.add_argument("--otsu_bias", type=float, default=-0.02)
    p.add_argument("--ensure_one_positive", type=int, default=1)

    # prompts
    p.add_argument("--prompt_clean", type=str, default="clean shiny metal")
    p.add_argument("--prompt_rust", type=str, default="rusty metal")

    p.add_argument("--res_dir", type=str, default="TwoPromptProb")
    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    detector = FastRustDetectorSAM3TwoPrompts(
        n_segments=args.n_segments,
        fast_mode=bool(args.fast_mode),
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        roi_pad=int(args.roi_pad),
        prior_weight=float(args.prior_weight),
        lik_weight=float(args.lik_weight),
        overlap_weight=float(args.overlap_weight),
        prob_threshold_fallback=float(args.prob_threshold_fallback),
        dynamic_prob_threshold=bool(args.dynamic_prob_threshold),
        min_valid_segments_for_dynamic=int(args.min_valid_segments_for_dynamic),
        otsu_bias=float(args.otsu_bias),
        ensure_one_positive=bool(args.ensure_one_positive),
        prompt_clean=str(args.prompt_clean),
        prompt_rust=str(args.prompt_rust),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))

    out = f"results/{args.res_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_prob_stages.png"
    detector.visualize_detection(results, save_path=out)


if __name__ == "__main__":
    main()