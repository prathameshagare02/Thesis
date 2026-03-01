from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
class SAM3PixelRustKMeansDetector:
    """
    Updated pipeline (dual text prompts + per-pixel SAM3-response features + k-means):

      1) Run SAM3SemanticPredictor with text prompt A: "clean shiny metal"
      2) Run SAM3SemanticPredictor with text prompt B: "rusty metal"
      3) Convert SAM3 detections into dense per-pixel prompt-response maps
         (union, max-conf, mean-conf, count)
      4) Build a crop ROI from the UNION of both prompt masks (optionally padded)
      5) Build per-pixel feature vectors (SAM3-response features; optional color cues)
      6) Run K-means directly on per-pixel vectors (no SLIC)
      7) Choose which cluster is "rust" based on prompt-response statistics
      8) Map result back to full image and visualize/save

    NOTE:
      Ultralytics SAM3SemanticPredictor public results typically expose masks/boxes/conf.
      This implementation uses dense per-pixel prompt-response maps derived from SAM3 outputs
      as the per-pixel "SAM3 features" for clustering.
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        clean_prompt: str = "clean shiny metal",
        rust_prompt: str = "rusty metal",
        roi_pad: int = 0,
        use_color_features: bool = True,
        use_prompt_support_only: bool = True,
        kmeans_k: int = 2,
        kmeans_attempts: int = 5,
        kmeans_max_iter: int = 100,
        kmeans_eps: float = 1e-3,
        min_pixels_for_kmeans: int = 64,
        morph_kernel: int = 3,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = str(sam_checkpoint)

        self.clean_prompt = str(clean_prompt)
        self.rust_prompt = str(rust_prompt)
        self.roi_pad = int(roi_pad)

        self.use_color_features = bool(use_color_features)
        self.use_prompt_support_only = bool(use_prompt_support_only)

        self.kmeans_k = int(kmeans_k)
        self.kmeans_attempts = int(kmeans_attempts)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.kmeans_eps = float(kmeans_eps)
        self.min_pixels_for_kmeans = int(min_pixels_for_kmeans)

        self.morph_kernel = max(1, int(morph_kernel))
        self.sam3_predictor = None
        self.original_image: Optional[np.ndarray] = None

        if self.verbose:
            self._print_backend_info()
        self.load_sam3_model()

    # ---- logging ----
    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("SAM3PixelRustKMeansDetector initialized:")
        print("  Mode: per-pixel SAM3 prompt-response features + K-means (no SLIC)")
        print(f"  clean_prompt: {self.clean_prompt!r}")
        print(f"  rust_prompt:  {self.rust_prompt!r}")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  ROI padding: {self.roi_pad}px")
        print(f"  use_color_features: {self.use_color_features}")
        print(f"  use_prompt_support_only: {self.use_prompt_support_only}")
        print(
            f"  K-means: K={self.kmeans_k}, attempts={self.kmeans_attempts}, "
            f"max_iter={self.kmeans_max_iter}, eps={self.kmeans_eps}"
        )
        print(f"  min_pixels_for_kmeans: {self.min_pixels_for_kmeans}")

    # ---- SAM3 ----
    def load_sam3_model(self):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception:
            self._log(
                "Failed to import SAM3SemanticPredictor. Install/upgrade ultralytics:\n"
                "    pip install -U ultralytics"
            )
            self.sam3_predictor = None
            return

        if not os.path.exists(self.sam_checkpoint):
            self._log(f"SAM3 checkpoint not found: {self.sam_checkpoint}")
            self.sam3_predictor = None
            return

        try:
            self._log(f"Loading SAM3SemanticPredictor from {self.sam_checkpoint} ...")
            overrides = dict(task="segment", mode="predict", model=self.sam_checkpoint, verbose=self.verbose)
            overrides["half"] = True
            self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_predictor.setup_model()
            self._log("SAM3 model loaded successfully.")
        except Exception as e:
            self._log(f"Failed to load SAM3 model: {e}")
            self.sam3_predictor = None

    # ---- Utilities ----
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
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return (int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def _clip_box(box: Tuple[int, int, int, int], H: int, W: int, pad: int = 0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)
        if x2 <= x1:
            x2 = min(W, x1 + 1)
        if y2 <= y1:
            y2 = min(H, y1 + 1)
        return (int(x1), int(y1), int(x2), int(y2))

    def _morph_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        if mask_u8 is None:
            return mask_u8
        k = max(1, self.morph_kernel)
        kernel = np.ones((k, k), np.uint8)
        out = (mask_u8 > 0).astype(np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        return out.astype(np.uint8)

    def _sam3_prompt_response(self, image_path: str, prompt: str, H: int, W: int) -> Dict[str, np.ndarray]:
        """
        Runs SAM3 for one text prompt and converts detections into dense prompt-response maps.

        Returns dict with:
          union_mask: uint8 HxW (0/1)
          max_conf:   float32 HxW
          mean_conf:  float32 HxW
          sum_conf:   float32 HxW
          count:      float32 HxW
          best_box:   tuple or None
          best_score: float
        """
        zeros_u8 = np.zeros((H, W), dtype=np.uint8)
        zeros_f = np.zeros((H, W), dtype=np.float32)

        if self.sam3_predictor is None:
            return dict(
                union_mask=zeros_u8,
                max_conf=zeros_f,
                mean_conf=zeros_f,
                sum_conf=zeros_f,
                count=zeros_f,
                best_box=None,
                best_score=0.0,
                n_masks=0,
                prompt=prompt,
            )

        try:
            self.sam3_predictor.set_image(image_path)
            results = self.sam3_predictor(text=[prompt])
            if not results:
                return dict(
                    union_mask=zeros_u8,
                    max_conf=zeros_f,
                    mean_conf=zeros_f,
                    sum_conf=zeros_f,
                    count=zeros_f,
                    best_box=None,
                    best_score=0.0,
                    n_masks=0,
                    prompt=prompt,
                )

            r0 = results[0]
            has_masks = (
                getattr(r0, "masks", None) is not None
                and r0.masks is not None
                and getattr(r0.masks, "data", None) is not None
            )
            if not has_masks or len(r0.masks.data) == 0:
                return dict(
                    union_mask=zeros_u8,
                    max_conf=zeros_f,
                    mean_conf=zeros_f,
                    sum_conf=zeros_f,
                    count=zeros_f,
                    best_box=None,
                    best_score=0.0,
                    n_masks=0,
                    prompt=prompt,
                )

            masks_t = r0.masks.data  # [N,H,W]
            n = len(masks_t)

            # confidences if available
            conf = None
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "conf", None) is not None
            ):
                try:
                    conf = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
                except Exception:
                    conf = None
            if conf is None or conf.size == 0:
                conf = np.ones((n,), dtype=np.float32)
            if conf.size < n:
                conf2 = np.ones((n,), dtype=np.float32)
                conf2[: conf.size] = conf[: conf.size]
                conf = conf2
            else:
                conf = conf[:n]

            union_mask = np.zeros((H, W), dtype=np.uint8)
            max_conf_map = np.zeros((H, W), dtype=np.float32)
            sum_conf_map = np.zeros((H, W), dtype=np.float32)
            count_map = np.zeros((H, W), dtype=np.float32)

            best_i = int(np.argmax(conf))
            best_score = float(conf[best_i])

            xyxy = None
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "xyxy", None) is not None
            ):
                try:
                    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
                except Exception:
                    xyxy = None

            best_box = None
            if xyxy is not None and xyxy.shape[0] > best_i:
                x1, y1, x2, y2 = xyxy[best_i]
                best_box = (int(x1), int(y1), int(x2), int(y2))

            # rasterize all detections into dense maps
            for i in range(n):
                try:
                    mi = masks_t[i].detach().cpu().numpy()
                except Exception:
                    mi = np.array(masks_t[i])
                m = (mi > 0.5).astype(np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    m = (m > 0).astype(np.uint8)
                m = self._morph_mask(m)
                if not np.any(m):
                    continue

                c = float(conf[i])

                union_mask[m == 1] = 1
                max_conf_map[m == 1] = np.maximum(max_conf_map[m == 1], c)
                sum_conf_map[m == 1] += c
                count_map[m == 1] += 1.0

            valid = count_map > 0
            mean_conf_map = np.zeros_like(sum_conf_map, dtype=np.float32)
            mean_conf_map[valid] = sum_conf_map[valid] / np.maximum(count_map[valid], 1e-6)

            if best_box is None and np.any(union_mask):
                best_box = self._bbox_from_mask(union_mask)

            return dict(
                union_mask=union_mask.astype(np.uint8),
                max_conf=max_conf_map.astype(np.float32),
                mean_conf=mean_conf_map.astype(np.float32),
                sum_conf=sum_conf_map.astype(np.float32),
                count=count_map.astype(np.float32),
                best_box=best_box,
                best_score=best_score,
                n_masks=int(n),
                prompt=prompt,
            )

        except Exception as e:
            self._log(f"SAM3 prompt-response error for prompt={prompt!r}: {e}")
            return dict(
                union_mask=zeros_u8,
                max_conf=zeros_f,
                mean_conf=zeros_f,
                sum_conf=zeros_f,
                count=zeros_f,
                best_box=None,
                best_score=0.0,
                n_masks=0,
                prompt=prompt,
            )

    def _build_roi_from_dual_prompts(
        self, clean_resp: Dict, rust_resp: Dict, H: int, W: int
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        union = ((clean_resp["union_mask"] > 0) | (rust_resp["union_mask"] > 0)).astype(np.uint8)
        box = self._bbox_from_mask(union)

        if box is None:
            cand = []
            if clean_resp.get("best_box") is not None:
                cand.append(clean_resp["best_box"])
            if rust_resp.get("best_box") is not None:
                cand.append(rust_resp["best_box"])
            box = self._merge_boxes(cand)

        if box is None:
            union = np.ones((H, W), dtype=np.uint8)
            box = (0, 0, W, H)

        box = self._clip_box(box, H, W, pad=self.roi_pad)
        return union.astype(np.uint8), box

    # ---- Per-pixel feature extraction ----
    def _compute_pixel_feature_tensor(
        self,
        crop_bgr: np.ndarray,
        clean_maps: Dict[str, np.ndarray],
        rust_maps: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build HxWxD per-pixel feature tensor.

        Core SAM3-derived features:
          clean_max, clean_mean, clean_count,
          rust_max, rust_mean, rust_count,
          delta_max, delta_mean, ratio_mean, support_strength

        Optional color/texture helpers (still per-pixel, no SLIC):
          Lab L,a,b, HSV H,S,V, gradient
        """
        eps = 1e-6

        cmax = clean_maps["max_conf"].astype(np.float32)
        cmean = clean_maps["mean_conf"].astype(np.float32)
        ccount = clean_maps["count"].astype(np.float32)

        rmax = rust_maps["max_conf"].astype(np.float32)
        rmean = rust_maps["mean_conf"].astype(np.float32)
        rcount = rust_maps["count"].astype(np.float32)

        ccount_s = ccount / (ccount + 1.0)
        rcount_s = rcount / (rcount + 1.0)

        delta_max = rmax - cmax
        delta_mean = rmean - cmean
        ratio_mean = rmean / (cmean + eps)
        ratio_mean = np.clip(ratio_mean, 0.0, 5.0) / 5.0
        support_strength = np.clip(np.maximum(cmax, rmax) + 0.5 * np.maximum(cmean, rmean), 0.0, 1.5) / 1.5

        feats = [
            cmax,
            cmean,
            ccount_s,
            rmax,
            rmean,
            rcount_s,
            delta_max,
            delta_mean,
            ratio_mean,
            support_strength,
        ]
        names = [
            "clean_max_conf",
            "clean_mean_conf",
            "clean_count_scaled",
            "rust_max_conf",
            "rust_mean_conf",
            "rust_count_scaled",
            "delta_max_rust_minus_clean",
            "delta_mean_rust_minus_clean",
            "ratio_rust_over_clean_mean",
            "support_strength",
        ]

        if self.use_color_features:
            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(sobelx**2 + sobely**2)
            g99 = float(np.percentile(grad, 99)) if grad.size else 1.0
            g99 = max(g99, 1.0)
            grad_n = np.clip(grad / g99, 0.0, 1.0)

            L = np.clip(lab[:, :, 0] / 255.0, 0.0, 1.0)
            a = np.clip(lab[:, :, 1] / 255.0, 0.0, 1.0)
            b = np.clip(lab[:, :, 2] / 255.0, 0.0, 1.0)
            Hh = np.clip(hsv[:, :, 0] / 179.0, 0.0, 1.0)
            S = np.clip(hsv[:, :, 1] / 255.0, 0.0, 1.0)
            V = np.clip(hsv[:, :, 2] / 255.0, 0.0, 1.0)

            feats.extend([L, a, b, Hh, S, V, grad_n])
            names.extend(["lab_L", "lab_a", "lab_b", "hsv_H", "hsv_S", "hsv_V", "gradient"])

        feat_tensor = np.stack(feats, axis=-1).astype(np.float32)
        return feat_tensor, names

    # ---- K-means ----
    def _normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sigma = np.nanstd(X, axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        Xn = (X - mu) / sigma
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return Xn, mu.astype(np.float32), sigma.astype(np.float32)

    def _kmeans_labels(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("K-means input must be [N,D] with N>0.")

        Xn, _, _ = self._normalize_features(X)
        if Xn.shape[0] < self.min_pixels_for_kmeans:
            labels = np.zeros((Xn.shape[0], 1), dtype=np.int32)
            centers = np.zeros((1, Xn.shape[1]), dtype=np.float32)
            return labels, centers

        K = min(max(2, self.kmeans_k), Xn.shape[0])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.kmeans_max_iter, self.kmeans_eps)
        compactness, labels, centers = cv2.kmeans(
            Xn.astype(np.float32),
            K,
            None,
            criteria,
            self.kmeans_attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        _ = compactness
        return labels.astype(np.int32), centers.astype(np.float32)

    def _select_rust_cluster(
        self,
        labels: np.ndarray,
        rust_max: np.ndarray,
        clean_max: np.ndarray,
        rust_mean: np.ndarray,
        clean_mean: np.ndarray,
        crop_bgr_valid: Optional[np.ndarray] = None,
    ) -> int:
        """
        Cluster labeling is arbitrary; choose rust cluster using prompt-response dominance.
        Adds a small orange/brown prior from color if crop_bgr_valid is provided.
        """
        labels_1d = labels.reshape(-1)
        K = int(labels_1d.max()) + 1 if labels_1d.size else 1

        best_k = 0
        best_score = -1e9

        color_prior = np.zeros_like(rust_max, dtype=np.float32)
        if crop_bgr_valid is not None and len(crop_bgr_valid) == len(labels_1d):
            lab = cv2.cvtColor(crop_bgr_valid.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
            hsv = cv2.cvtColor(crop_bgr_valid.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            a_n = np.clip((lab[:, 1] - 128.0) / 60.0, -1.0, 1.0)
            b_n = np.clip((lab[:, 2] - 128.0) / 60.0, -1.0, 1.0)
            s_n = hsv[:, 1] / 255.0
            v_n = hsv[:, 2] / 255.0
            color_prior = (
                0.45 * np.maximum(a_n, 0.0)
                + 0.35 * np.maximum(b_n, 0.0)
                + 0.20 * s_n
                - 0.10 * np.maximum(v_n - 0.9, 0.0)
            ).astype(np.float32)

        for k in range(K):
            idx = labels_1d == k
            if not np.any(idx):
                continue

            s = 0.0
            s += 2.0 * float(np.mean(rust_max[idx] - clean_max[idx]))
            s += 1.5 * float(np.mean(rust_mean[idx] - clean_mean[idx]))
            s += 0.8 * float(np.mean(rust_max[idx]))
            s -= 0.5 * float(np.mean(clean_max[idx]))

            if color_prior.size:
                s += 0.25 * float(np.mean(color_prior[idx]))

            if s > best_score:
                best_score = s
                best_k = int(k)

        return best_k

    def _classify_pixels_in_crop(
        self,
        crop_bgr: np.ndarray,
        feat_tensor: np.ndarray,
        clean_maps_crop: Dict[str, np.ndarray],
        rust_maps_crop: Dict[str, np.ndarray],
    ) -> Dict:
        Hc, Wc = crop_bgr.shape[:2]

        clean_max = clean_maps_crop["max_conf"]
        rust_max = rust_maps_crop["max_conf"]
        clean_mean = clean_maps_crop["mean_conf"]
        rust_mean = rust_maps_crop["mean_conf"]

        if self.use_prompt_support_only:
            support = ((clean_max > 0) | (rust_max > 0)).astype(np.uint8)
            if int(np.sum(support)) == 0:
                support = np.ones((Hc, Wc), dtype=np.uint8)
        else:
            support = np.ones((Hc, Wc), dtype=np.uint8)

        valid_idx = np.where(support.ravel() > 0)[0]
        X = feat_tensor.reshape(-1, feat_tensor.shape[-1])[valid_idx]
        crop_valid = crop_bgr.reshape(-1, 3)[valid_idx]

        labels_valid, centers = self._kmeans_labels(X)

        if labels_valid.size == 0:
            labels_valid = np.zeros((len(valid_idx), 1), dtype=np.int32)
            rust_cluster = 0
        else:
            rust_cluster = self._select_rust_cluster(
                labels_valid,
                rust_max.ravel()[valid_idx].astype(np.float32),
                clean_max.ravel()[valid_idx].astype(np.float32),
                rust_mean.ravel()[valid_idx].astype(np.float32),
                clean_mean.ravel()[valid_idx].astype(np.float32),
                crop_bgr_valid=crop_valid,
            )

        labels_full = np.full((Hc * Wc,), -1, dtype=np.int32)
        labels_full[valid_idx] = labels_valid.reshape(-1)

        rust_mask = np.zeros((Hc * Wc,), dtype=np.uint8)
        clean_mask = np.zeros((Hc * Wc,), dtype=np.uint8)

        rust_mask[valid_idx] = (labels_valid.reshape(-1) == rust_cluster).astype(np.uint8)
        clean_mask[valid_idx] = (labels_valid.reshape(-1) != rust_cluster).astype(np.uint8)

        rust_mask = rust_mask.reshape(Hc, Wc)
        clean_mask = clean_mask.reshape(Hc, Wc)
        label_map = labels_full.reshape(Hc, Wc)

        rust_mask = self._morph_mask(rust_mask)
        clean_mask = ((support > 0) & (rust_mask == 0)).astype(np.uint8)

        delta_map = (rust_max - clean_max).astype(np.float32)
        delta_vis = np.clip((delta_map + 1.0) / 2.0, 0.0, 1.0)

        denom = int(np.sum(support))
        rust_pct_support = (100.0 * float(np.sum(rust_mask)) / denom) if denom > 0 else 0.0

        return dict(
            support_mask=support.astype(np.uint8),
            rust_mask=rust_mask.astype(np.uint8),
            clean_mask=clean_mask.astype(np.uint8),
            label_map=label_map.astype(np.int32),
            delta_map=delta_map.astype(np.float32),
            delta_vis=delta_vis.astype(np.float32),
            centers=centers,
            rust_cluster=int(rust_cluster),
            rust_percentage_support=float(rust_pct_support),
        )

    # ---- Main analysis ----
    def analyze(self, image_path: str, use_sam3: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        self.original_image = original.copy()
        H, W = original.shape[:2]
        timings["load"] = time.time() - t0

        # 1) SAM3 dual prompts -> dense response maps
        t0 = time.time()
        if use_sam3 and self.sam3_predictor is not None:
            self._log(f'Running SAM3 prompt: {self.clean_prompt!r}')
            clean_resp = self._sam3_prompt_response(image_path, self.clean_prompt, H, W)
            self._log(f'Running SAM3 prompt: {self.rust_prompt!r}')
            rust_resp = self._sam3_prompt_response(image_path, self.rust_prompt, H, W)
        else:
            zeros_u8 = np.zeros((H, W), dtype=np.uint8)
            zeros_f = np.zeros((H, W), dtype=np.float32)
            clean_resp = dict(
                union_mask=zeros_u8,
                max_conf=zeros_f,
                mean_conf=zeros_f,
                sum_conf=zeros_f,
                count=zeros_f,
                best_box=None,
                best_score=0.0,
                n_masks=0,
                prompt=self.clean_prompt,
            )
            rust_resp = dict(
                union_mask=zeros_u8,
                max_conf=zeros_f,
                mean_conf=zeros_f,
                sum_conf=zeros_f,
                count=zeros_f,
                best_box=None,
                best_score=0.0,
                n_masks=0,
                prompt=self.rust_prompt,
            )
        timings["sam3_dual_prompts"] = time.time() - t0

        if self.verbose:
            self._log(
                f'clean prompt detections: {clean_resp.get("n_masks", 0)} | '
                f'best score={clean_resp.get("best_score", 0.0):.3f}'
            )
            self._log(
                f'rust  prompt detections: {rust_resp.get("n_masks", 0)} | '
                f'best score={rust_resp.get("best_score", 0.0):.3f}'
            )

        # 2) ROI from union of both prompts
        t0 = time.time()
        prompt_union_full, roi_box = self._build_roi_from_dual_prompts(clean_resp, rust_resp, H, W)
        x1, y1, x2, y2 = roi_box
        crop = original[y1:y2, x1:x2].copy()
        timings["roi_crop"] = time.time() - t0

        def crop_resp(resp: Dict) -> Dict[str, np.ndarray]:
            return dict(
                union_mask=resp["union_mask"][y1:y2, x1:x2].astype(np.uint8),
                max_conf=resp["max_conf"][y1:y2, x1:x2].astype(np.float32),
                mean_conf=resp["mean_conf"][y1:y2, x1:x2].astype(np.float32),
                sum_conf=resp["sum_conf"][y1:y2, x1:x2].astype(np.float32),
                count=resp["count"][y1:y2, x1:x2].astype(np.float32),
            )

        clean_crop = crop_resp(clean_resp)
        rust_crop = crop_resp(rust_resp)

        # 3) Per-pixel feature tensor (inside ROI)
        t0 = time.time()
        feat_tensor, feat_names = self._compute_pixel_feature_tensor(crop, clean_crop, rust_crop)
        timings["feature_tensor"] = time.time() - t0

        # 4) K-means classification directly on per-pixel vectors
        t0 = time.time()
        kmeans_res = self._classify_pixels_in_crop(crop, feat_tensor, clean_crop, rust_crop)
        timings["kmeans"] = time.time() - t0

        # 5) Map masks back to full image
        t0 = time.time()
        full_rust_mask = np.zeros((H, W), dtype=np.uint8)
        full_clean_mask = np.zeros((H, W), dtype=np.uint8)
        full_support_mask = np.zeros((H, W), dtype=np.uint8)

        full_rust_mask[y1:y2, x1:x2] = kmeans_res["rust_mask"]
        full_clean_mask[y1:y2, x1:x2] = kmeans_res["clean_mask"]
        full_support_mask[y1:y2, x1:x2] = kmeans_res["support_mask"]
        timings["map_back"] = time.time() - t0

        total_time = float(sum(timings.values()))
        self._log(
            f"ROI box: ({x1},{y1})-({x2},{y2}) | "
            f"Rust coverage (support pixels in ROI): {kmeans_res['rust_percentage_support']:.2f}% | "
            f"Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            image_path=image_path,
            roi_box=roi_box,
            crop=crop,
            crop_coords=(x1, y1, x2 - x1, y2 - y1),
            clean_prompt=self.clean_prompt,
            rust_prompt=self.rust_prompt,
            clean_resp_full=clean_resp,
            rust_resp_full=rust_resp,
            prompt_union_full=prompt_union_full,
            clean_resp_crop=clean_crop,
            rust_resp_crop=rust_crop,
            pixel_features=feat_tensor,  # Hc x Wc x D
            pixel_feature_names=feat_names,
            support_mask_crop=kmeans_res["support_mask"],
            rust_mask_crop=kmeans_res["rust_mask"],
            clean_mask_crop=kmeans_res["clean_mask"],
            label_map_crop=kmeans_res["label_map"],
            delta_map_crop=kmeans_res["delta_map"],
            delta_vis_crop=kmeans_res["delta_vis"],
            full_rust_mask=full_rust_mask,
            full_clean_mask=full_clean_mask,
            full_support_mask=full_support_mask,
            rust_cluster=int(kmeans_res["rust_cluster"]),
            rust_percentage_support=float(kmeans_res["rust_percentage_support"]),
            timings=timings,
        )

    # ---- Visualization ----
    def visualize_detection(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        original = results["original"].copy()
        crop = results["crop"].copy()
        (x1, y1, x2, y2) = results["roi_box"]

        clean_max_full = results["clean_resp_full"]["max_conf"]
        rust_max_full = results["rust_resp_full"]["max_conf"]
        prompt_union = results["prompt_union_full"]

        delta_vis_crop = results["delta_vis_crop"]

        rust_mask_crop = results["rust_mask_crop"]
        clean_mask_crop = results["clean_mask_crop"]
        support_mask_crop = results["support_mask_crop"]
        full_rust_mask = results["full_rust_mask"]

        rust_pct = float(results["rust_percentage_support"])
        clean_best = float(results["clean_resp_full"].get("best_score", 0.0))
        rust_best = float(results["rust_resp_full"].get("best_score", 0.0))

        # Stage 1: input + ROI box
        stage1 = original.copy()
        cv2.rectangle(stage1, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            stage1,
            "ROI from dual-prompt union",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )

        # Stage 2: prompt union overlay on full image
        stage2 = original.copy()
        overlay2 = stage2.copy()
        overlay2[prompt_union > 0] = [255, 255, 0]  # cyan-ish in BGR
        stage2 = cv2.addWeighted(stage2, 0.65, overlay2, 0.35, 0)
        cv2.rectangle(stage2, (x1, y1), (x2, y2), (255, 0, 0), 2)

        def heat_bgr(img_bgr: np.ndarray, map01: np.ndarray, title_text: str) -> np.ndarray:
            m = np.clip(map01, 0.0, 1.0)
            hm = (m * 255).astype(np.uint8)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            base = cv2.addWeighted(img_bgr, 0.55, hm, 0.45, 0)
            cv2.putText(base, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return base

        stage3 = heat_bgr(original.copy(), clean_max_full, f"clean max-conf (best={clean_best:.2f})")
        stage4 = heat_bgr(original.copy(), rust_max_full, f"rust max-conf (best={rust_best:.2f})")

        # Stage 5: crop with K-means labels
        stage5 = crop.copy()
        ov5 = stage5.copy()
        ov5[(support_mask_crop > 0) & (clean_mask_crop > 0)] = [0, 255, 0]
        ov5[(support_mask_crop > 0) & (rust_mask_crop > 0)] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.60, ov5, 0.40, 0)

        # Stage 6: delta heatmap in ROI (rust-clean)
        delta_hm = (np.clip(delta_vis_crop, 0.0, 1.0) * 255).astype(np.uint8)
        delta_hm = cv2.applyColorMap(delta_hm, cv2.COLORMAP_TURBO)
        stage6 = cv2.addWeighted(crop.copy(), 0.55, delta_hm, 0.45, 0)

        # Stage 7: final rust overlay full image
        stage7 = original.copy()
        ov7 = stage7.copy()
        ov7[full_rust_mask > 0] = [0, 0, 255]
        stage7 = cv2.addWeighted(stage7, 0.65, ov7, 0.35, 0)
        cv2.rectangle(stage7, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Stage 8: prompt competition map
        comp = rust_max_full - clean_max_full
        comp_vis = np.clip((comp + 1.0) / 2.0, 0.0, 1.0)
        stage8 = heat_bgr(original.copy(), comp_vis, "prompt delta: rusty - clean")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            "SAM3 Dual-Prompt Rust Detection (per-pixel feature vectors + K-means)\n"
            f'Prompts: clean="{results["clean_prompt"]}" | rust="{results["rust_prompt"]}" | '
            f"Rust coverage on support pixels={rust_pct:.1f}%",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8]
        titles = [
            "1) Input + ROI box",
            "2) Dual-prompt union support (full)",
            "3) SAM3 clean prompt response (full)",
            "4) SAM3 rust prompt response (full)",
            "5) ROI K-means labels (green=clean, red=rust)",
            "6) ROI prompt delta heatmap",
            "7) Final rust overlay (full)",
            "8) Full prompt competition map",
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
            fig.savefig(save_path, dpi=250)
            self._log(f"Visualization saved to: {save_path}")

        return fig


# ----------------------------- CLI + Main -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rust Detection: SAM3 dual prompts -> per-pixel feature vectors -> K-means (no SLIC)"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--use_sam3", type=int, default=1, help="1=use SAM3 text prompts, 0=skip (not recommended).")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--clean_prompt", type=str, default="clean shiny metal")
    p.add_argument("--rust_prompt", type=str, default="rusty metal")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around dual-prompt union ROI box.")

    p.add_argument("--use_color_features", type=int, default=1, help="1=append per-pixel color/gradient features.")
    p.add_argument(
        "--use_prompt_support_only",
        type=int,
        default=1,
        help="1=cluster only pixels touched by either prompt.",
    )

    p.add_argument("--kmeans_k", type=int, default=2)
    p.add_argument("--kmeans_attempts", type=int, default=5)
    p.add_argument("--kmeans_max_iter", type=int, default=100)
    p.add_argument("--kmeans_eps", type=float, default=1e-3)
    p.add_argument("--min_pixels_for_kmeans", type=int, default=64)
    p.add_argument("--morph_kernel", type=int, default=3)

    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--res_dir", type=str, default="SAM3PixelKMeans")
    p.add_argument(
        "--save_features",
        type=int,
        default=1,
        help="1=save per-pixel feature tensor as .npy + feature names json",
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

    detector = SAM3PixelRustKMeansDetector(
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        clean_prompt=args.clean_prompt,
        rust_prompt=args.rust_prompt,
        roi_pad=int(args.roi_pad),
        use_color_features=bool(args.use_color_features),
        use_prompt_support_only=bool(args.use_prompt_support_only),
        kmeans_k=int(args.kmeans_k),
        kmeans_attempts=int(args.kmeans_attempts),
        kmeans_max_iter=int(args.kmeans_max_iter),
        kmeans_eps=float(args.kmeans_eps),
        min_pixels_for_kmeans=int(args.min_pixels_for_kmeans),
        morph_kernel=int(args.morph_kernel),
    )

    results = detector.analyze(image_path, use_sam3=bool(args.use_sam3))

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("results", args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig_path = os.path.join(out_dir, f"{stem}_sam3_pixel_kmeans_stages.png")
    detector.visualize_detection(results, save_path=fig_path)

    if bool(args.save_features):
        x1, y1, w, h = results["crop_coords"]
        feat_path = os.path.join(out_dir, f"{stem}_pixel_features_roi.npy")
        meta_path = os.path.join(out_dir, f"{stem}_pixel_features_roi_meta.json")
        np.save(feat_path, results["pixel_features"])
        meta = {
            "image_path": results["image_path"],
            "roi_box_xyxy": [int(x1), int(y1), int(x1 + w), int(y1 + h)],
            "feature_shape": list(map(int, results["pixel_features"].shape)),
            "feature_names": results["pixel_feature_names"],
            "clean_prompt": results["clean_prompt"],
            "rust_prompt": results["rust_prompt"],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved per-pixel feature tensor: {feat_path}")
        print(f"Saved feature metadata:       {meta_path}")

    print("\nDone.")
    print(f"Rust coverage on prompt support pixels: {results['rust_percentage_support']:.2f}%")
    print("Timings (s):")
    for k, v in results["timings"].items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
