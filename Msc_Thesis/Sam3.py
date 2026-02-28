from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic
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


# ----------------------------- Checkpoint Auto-Download -----------------------------
DEFAULT_SAM3_FILENAME = "sam3.pt"


def _ensure_checkpoint(
    ckpt_path: str,
    *,
    repo_id: str,
    filename: str = DEFAULT_SAM3_FILENAME,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Ensure checkpoint exists locally. If missing, download from Hugging Face Hub.

    Notes:
      - SAM3 weights are typically gated; download will only work if:
          (a) you have access to the repo, and
          (b) you are authenticated (hf auth login) OR you pass a token / set HF_TOKEN.
    """
    ckpt_path = str(ckpt_path or "").strip() or filename
    p = Path(ckpt_path)

    # If user gave a directory, store filename inside it.
    if ckpt_path.endswith(os.sep) or (p.exists() and p.is_dir()):
        p = Path(ckpt_path) / filename

    if p.exists() and p.is_file():
        if verbose:
            print(f"✓ Using checkpoint: {p}")
        return str(p)

    if not repo_id:
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            "Auto-download needs --hf_repo.\n"
            "Either:\n"
            "  - place sam3.pt locally and pass --sam_checkpoint /path/to/sam3.pt\n"
            "  - OR pass --hf_repo <org/name> so the code can download it.\n"
        )

    if verbose:
        print(f"⚠ Checkpoint not found at: {p}")
        print(f"  Attempting download from Hugging Face: repo_id={repo_id}, filename={filename}")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for auto-download.\n"
            "Install it with:\n"
            "  pip install -U huggingface_hub\n"
        ) from e

    # Token priority: CLI -> env var -> None
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    try:
        downloaded_cache_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,             # required for gated repos
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "Auto-download failed.\n"
            "Common causes:\n"
            "  1) You don't have access to the gated SAM3 repo.\n"
            "  2) You're not authenticated (run: hf auth login) and no HF_TOKEN provided.\n"
            "  3) repo_id or filename is wrong.\n\n"
            "Fix:\n"
            "  - Request access to the SAM3 repo on Hugging Face\n"
            "  - Then run: hf auth login\n"
            "  - Or set env var: HF_TOKEN=your_token\n"
            "  - Or pass: --hf_token your_token\n"
            "  - Or manually download sam3.pt and pass --sam_checkpoint\n"
        ) from e

    # Copy from HF cache to desired local path
    p.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(downloaded_cache_path, p)

    if verbose:
        print(f"✓ Downloaded checkpoint to: {p}")

    return str(p)


# ----------------------------- Core Detector (SAM3 text prompts) -----------------------------
class FastRustDetector:
    """
    Uses SAM3 *text prompts* ("rusted metal" vs "clean metal") to produce a METAL ROI mask,
    then runs your existing SLIC-per-segment rust scoring inside that ROI.

    Metal ROI mask strategy:
      metal_mask = union( mask("rusted metal"), mask("clean metal") )
      fallback: mask("metal")
      final fallback: full image

    Requirements:
      - ultralytics with SAM3 support (SAM3SemanticPredictor)
      - sam3.pt available (local or auto-downloaded via Hugging Face)
    """

    def __init__(
        self,
        n_segments: int = 10000,
        fast_mode: bool = False,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        device: str = "",  # "", "cpu", "cuda", "mps", "0" etc.
        bpe_path: str = "",  # optional tokenizer path if required by your environment

        prompt_rust: str = "rusted metal",
        prompt_clean: str = "clean metal",
        prompt_fallback_metal: str = "metal",

        rust_threshold_fallback: float = 0.60,
        ensure_one_positive: bool = True,
        dynamic_feature_gates: bool = True,
        dynamic_score_threshold: bool = True,
        min_valid_segments_for_dynamic: int = 20,
        otsu_bias: float = -0.02,
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

        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.bpe_path = bpe_path

        self.prompt_rust = prompt_rust
        self.prompt_clean = prompt_clean
        self.prompt_fallback_metal = prompt_fallback_metal

        self.sam3_predictor = None

        self.processed_mask: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None

        # Morphology
        self.kernel_size = 3
        self.erosion_iterations = 2
        self.dilation_iterations = 2

        if self.verbose:
            self._print_backend_info()

        self.load_sam3_text_model()

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
        print("  SAM3 text prompts:")
        print(f"    rust: {self.prompt_rust!r}")
        print(f"    clean: {self.prompt_clean!r}")
        print(f"    fallback metal: {self.prompt_fallback_metal!r}")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  Device: {self.device or '(auto)'}")
        if self.bpe_path:
            print(f"  BPE path: {self.bpe_path}")

    # ---- SAM3 (text prompting) ----
    def load_sam3_text_model(self):
        """
        Loads SAM3SemanticPredictor for concept segmentation with text prompts.
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception as e:
            raise RuntimeError(
                "Failed to import SAM3SemanticPredictor.\n"
                "Please upgrade Ultralytics to a version that includes SAM3 support.\n"
                "Try:\n"
                "  pip install -U ultralytics\n"
            ) from e

        if not os.path.exists(self.sam_checkpoint):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {self.sam_checkpoint}\n"
                "Pass --sam_checkpoint to an existing file, or enable auto-download with --hf_repo."
            )

        self._log(f"Loading SAM3SemanticPredictor from {self.sam_checkpoint} ...")

        overrides = dict(
            task="segment",
            mode="predict",
            model=self.sam_checkpoint,
            verbose=self.verbose,
        )
        if self.device:
            overrides["device"] = self.device

        # half=True is good on CUDA; if you use CPU and see issues, set half=False here.
        overrides["half"] = True

        try:
            if self.bpe_path:
                self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path=self.bpe_path)
            else:
                self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_predictor.setup_model()
            self._log("SAM3SemanticPredictor loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize SAM3SemanticPredictor.\n"
                "If you see tokenizer/BPE related errors, provide --bpe_path.\n"
                f"Original error: {e}"
            ) from e

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

    def _sam3_text_predict_union(self, image_path: str, prompts: List[str]) -> Optional[np.ndarray]:
        """
        Returns union mask of all instances detected for the given prompts.
        """
        if self.sam3_predictor is None:
            return None

        try:
            self.sam3_predictor.set_image(image_path)
            results = self.sam3_predictor(text=prompts)

            if not results:
                return None
            r0 = results[0]

            if getattr(r0, "masks", None) is None or r0.masks is None or r0.masks.data is None:
                return None
            if len(r0.masks.data) == 0:
                return None

            masks = r0.masks.data  # torch tensor [N,H,W]
            m = masks[0].detach().cpu().numpy()
            for i in range(1, len(masks)):
                m = np.maximum(m, masks[i].detach().cpu().numpy())

            binary = (m > 0.5).astype(np.uint8)
            processed = self.apply_morphological_operations(binary)
            return processed if processed is not None else binary
        except Exception as e:
            self._log(f"SAM3 text prediction error for prompts={prompts}: {e}")
            return None

    def get_metal_mask_from_text(self, image_path: str) -> Optional[np.ndarray]:
        """
        New requirement: use text prompts "rusted metal" vs "clean metal".
        Metal ROI is union(rust, clean). If both fail, try "metal".
        """
        rust_m = self._sam3_text_predict_union(image_path, [self.prompt_rust])
        clean_m = self._sam3_text_predict_union(image_path, [self.prompt_clean])

        if rust_m is not None and clean_m is not None:
            return np.clip(rust_m + clean_m, 0, 1).astype(np.uint8)
        if rust_m is not None:
            return rust_m.astype(np.uint8)
        if clean_m is not None:
            return clean_m.astype(np.uint8)

        return self._sam3_text_predict_union(image_path, [self.prompt_fallback_metal])

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
        a_vals = features_valid[:, 1] if features_valid.size else np.array([])
        b_vals = features_valid[:, 2] if features_valid.size else np.array([])
        s_vals = features_valid[:, 13] if features_valid.size else np.array([])
        v_vals = features_valid[:, 14] if features_valid.size else np.array([])
        rough_vals = features_valid[:, 6] if features_valid.size else np.array([])
        ent_vals = features_valid[:, 7] if features_valid.size else np.array([])

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

        for idx in valid_indices:
            sid = int(segment_ids[idx])
            area = int(seg_areas[sid]) if sid < len(seg_areas) else 0
            metal_pixels += area

            m = (segments == sid)
            score_map[m] = float(rust_scores[idx])

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
                self._log(
                    "Dynamic gates: "
                    f"a_warm={dyn_gates['a_warm']:.1f}, b_warm={dyn_gates['b_warm']:.1f}, "
                    f"a_red={dyn_gates['a_red']:.1f}, b_orange={dyn_gates['b_orange']:.1f}, "
                    f"v_dark={dyn_gates['v_dark']:.1f}, v_very_dark={dyn_gates['v_very_dark']:.1f}, "
                    f"s_low={dyn_gates['s_low']:.1f}, s_mid={dyn_gates['s_mid']:.1f}, s_hi={dyn_gates['s_hi']:.1f}"
                )
            else:
                self._log("Dynamic gates: OFF / insufficient segments (using defaults)")

        return {
            "segments": segments,
            "score_map": score_map,
            "crop_mask": final_mask,
            "rust_percentage": pct,
            "threshold_used": thr,
            "dynamic_gates": dyn_gates,
        }

    # ---- Main analysis ----
    def analyze(self, image_path: str) -> Dict:
        timings: Dict[str, float] = {}

        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        self.original_image = original.copy()
        timings["load"] = time.time() - t0

        t0 = time.time()
        self._log("SAM3 text-prompt segmentation for metal ROI (rusted/clean metal)...")
        metal_mask_full = self.get_metal_mask_from_text(image_path)
        timings["sam3_text"] = time.time() - t0

        if metal_mask_full is None or not np.any(metal_mask_full):
            self._log("Warning: SAM3 did not return a usable mask. Falling back to full image as metal ROI.")
            metal_mask_full = np.ones(original.shape[:2], dtype=np.uint8)

        # Crop tightly around metal ROI
        t0 = time.time()
        y_idx, x_idx = np.where(metal_mask_full > 0)
        if len(y_idx) > 0:
            y_min, y_max = int(y_idx.min()), int(y_idx.max())
            x_min, x_max = int(x_idx.min()), int(x_idx.max())

            pad = 20
            H, W = original.shape[:2]
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(W, x_max + pad)
            y_max = min(H, y_max + pad)

            crop = original[y_min:y_max, x_min:x_max]
            metal_mask_crop = metal_mask_full[y_min:y_max, x_min:x_max]
            cx, cy, cw, ch = x_min, y_min, x_max - x_min, y_max - y_min
        else:
            crop = original
            metal_mask_crop = np.ones(crop.shape[:2], dtype=np.uint8)
            cx, cy, cw, ch = 0, 0, original.shape[1], original.shape[0]
        timings["crop"] = time.time() - t0

        t0 = time.time()
        self._log(f"Running Analysis (SLIC n={self.n_segments})...")
        res = self._perform_segmentation_analysis(crop, metal_mask_crop)
        timings["analysis"] = time.time() - t0

        full_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        full_mask[cy : cy + ch, cx : cx + cw] = res["crop_mask"]

        rust_percentage = res["rust_percentage"] * 100.0
        total_time = float(sum(timings.values()))
        self._log(f"Rust coverage (metal): {rust_percentage:.2f}% | Total time: {total_time:.3f}s")

        return dict(
            original=original,
            crop=crop,
            full_mask=full_mask,
            crop_mask=res["crop_mask"],
            score_map=res["score_map"],
            crop_coords=(cx, cy, cw, ch),
            rust_percentage=rust_percentage,
            metal_mask=metal_mask_crop,
            timings=timings,
            threshold_used=res["threshold_used"],
            dynamic_gates=res["dynamic_gates"],
        )

    # ---- Visualization (ONLY 2 PANELS) ----
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
        """
        Shows ONLY:
          1) Metal Input (tight crop, non-metal -> white)
          2) Final Detection Result (Rust=Red) on full original
        """
        crop = results["crop"]
        full_mask = results["full_mask"]
        original = results["original"]
        rust_percentage = results["rust_percentage"]
        metal_mask = results.get("metal_mask", np.ones(crop.shape[:2], dtype=np.uint8)).astype(np.uint8)
        thr = float(results.get("threshold_used", self.rust_threshold_fallback))

        # Tight crop for Metal Input
        x1, y1, x2, y2 = self._tight_bbox_from_mask(metal_mask, pad=2)
        crop_t = crop[y1:y2, x1:x2]
        mask_t = metal_mask[y1:y2, x1:x2]

        metal_input = crop_t.copy()
        metal_input[mask_t == 0] = 255  # outside metal -> white

        # Final overlay on full original
        vis_final = original.copy()
        vis_final[full_mask == 1] = [0, 0, 255]  # red in BGR
        vis_final = cv2.addWeighted(original, 0.6, vis_final, 0.4, 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Fast Rust Detection [SAM3 text prompts] | Segments: {self.n_segments}\n"
            f"Coverage (Metal): {rust_percentage:.1f}% | Threshold used: {thr:.2f}",
            fontsize=12,
        )

        axes[0].imshow(cv2.cvtColor(metal_input, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Metal Input (Tight Crop)")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Final Detection Result (Rust=Red)")
        axes[1].axis("off")

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
        description="Fast Rust Detection (SAM3 text prompts, per-segment, dynamic thresholds + auto-download weights)"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")

    # SAM3 checkpoint + auto-download
    p.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM3_FILENAME, help="Path to SAM3 checkpoint.")
    p.add_argument(
        "--hf_repo",
        type=str,
        default="",
        help="Hugging Face repo_id that contains sam3.pt (required for auto-download if missing).",
    )
    p.add_argument("--hf_token", type=str, default="", help="HF token (or set env var HF_TOKEN).")
    p.add_argument("--hf_cache_dir", type=str, default="", help="Optional HF cache dir.")

    # SAM3 predictor options
    p.add_argument("--device", type=str, default="", help="Device: ''(auto), 'cpu', 'cuda', 'mps', '0', etc.")
    p.add_argument("--bpe_path", type=str, default="", help="Optional BPE tokenizer path if your env requires it.")

    # Text prompts (new requirement)
    p.add_argument("--prompt_rust", type=str, default="rusted metal")
    p.add_argument("--prompt_clean", type=str, default="clean metal")
    p.add_argument("--prompt_fallback_metal", type=str, default="metal")

    # pipeline params
    p.add_argument("--n_segments", type=int, default=10000)
    p.add_argument("--fast_mode", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    # thresholding
    p.add_argument("--rust_threshold_fallback", type=float, default=0.60)
    p.add_argument("--dynamic_feature_gates", type=int, default=1)
    p.add_argument("--dynamic_score_threshold", type=int, default=1)
    p.add_argument("--min_valid_segments_for_dynamic", type=int, default=20)
    p.add_argument("--ensure_one_positive", type=int, default=1)
    p.add_argument("--otsu_bias", type=float, default=-0.02)

    return p


def main():
    args = build_argparser().parse_args()
    verbose = bool(args.verbose)

    # pick image
    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    # ensure checkpoint exists (auto-download if missing and hf_repo provided)
    try:
        ckpt_path = _ensure_checkpoint(
            args.sam_checkpoint,
            repo_id=args.hf_repo.strip(),
            filename=DEFAULT_SAM3_FILENAME,
            token=(args.hf_token.strip() or None),
            cache_dir=(args.hf_cache_dir.strip() or None),
            verbose=verbose,
        )
    except Exception as e:
        print(str(e))
        sys.exit(2)

    detector = FastRustDetector(
        n_segments=args.n_segments,
        fast_mode=bool(args.fast_mode),
        verbose=bool(args.verbose),
        sam_checkpoint=ckpt_path,
        device=args.device,
        bpe_path=args.bpe_path,
        prompt_rust=args.prompt_rust,
        prompt_clean=args.prompt_clean,
        prompt_fallback_metal=args.prompt_fallback_metal,
        rust_threshold_fallback=float(args.rust_threshold_fallback),
        ensure_one_positive=bool(args.ensure_one_positive),
        dynamic_feature_gates=bool(args.dynamic_feature_gates),
        dynamic_score_threshold=bool(args.dynamic_score_threshold),
        min_valid_segments_for_dynamic=int(args.min_valid_segments_for_dynamic),
        otsu_bias=float(args.otsu_bias),
    )

    results = detector.analyze(image_path)
    out = f"results/New/{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    detector.visualize_detection(results, save_path=out)


if __name__ == "__main__":
    main()