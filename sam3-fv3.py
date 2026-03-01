from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Union
import argparse
import os
import time
import tempfile

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
class FastRustDetector:
    """
    Updated pipeline (SAM3 twice + per-pixel KMeans, no SLIC):

      PASS 1 (full image):
        1) SAM3 text prompt "metal"
        2) Select detection with MAX confidence
        3) Crop full image using that best metal box (optional padding)

      PASS 2 (cropped ROI):
        4) SAM3 prompt "clean shiny metal" -> per-pixel evidence map
        5) SAM3 prompt "rusty metal"       -> per-pixel evidence map
        6) Build per-pixel feature vectors (image features + SAM3 evidence features)
        7) K-means cluster pixels directly (K=2 by default)
        8) Select rust cluster using prompt-guided cluster scoring
        9) Map ROI rust mask back to full image

    Outputs:
      - per-pixel feature tensor (H_roi, W_roi, F)
      - SAM3 clean/rust evidence maps
      - K-means labels map
      - final rust mask (ROI + full image)

    Visualization:
      Shows 6 stages:
        1) Input
        2) Best metal box
        3) Cropped ROI + first-pass metal mask overlay
        4) Second-pass prompt evidence (green=clean, red=rust)
        5) KMeans cluster result in ROI
        6) Final rust overlay on original
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        roi_pad: int = 0,
        # Prompts
        metal_prompt: str = "metal",
        clean_prompt: str = "clean shiny metal",
        rust_prompt: str = "rusty metal",
        # KMeans
        kmeans_k: int = 2,
        kmeans_attempts: int = 5,
        kmeans_max_iter: int = 50,
        kmeans_eps: float = 0.2,
        kmeans_train_samples: int = 80000,
        # Mask / gating
        sam_evidence_thresh: float = 0.02,
        ensure_one_positive: bool = True,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = sam_checkpoint
        self.roi_pad = int(roi_pad)

        self.prompt_metal = str(metal_prompt)
        self.prompt_clean = str(clean_prompt)
        self.prompt_rust = str(rust_prompt)

        self.kmeans_k = max(2, int(kmeans_k))
        self.kmeans_attempts = max(1, int(kmeans_attempts))
        self.kmeans_max_iter = max(5, int(kmeans_max_iter))
        self.kmeans_eps = float(kmeans_eps)
        self.kmeans_train_samples = max(1000, int(kmeans_train_samples))

        self.sam_evidence_thresh = float(sam_evidence_thresh)
        self.ensure_one_positive = bool(ensure_one_positive)

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
        print("FastRustDetector initialized (UPDATED: SAM3 x2 + per-pixel KMeans)")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  ROI padding: {self.roi_pad}px")
        print(f"  Prompts: metal={self.prompt_metal!r}, clean={self.prompt_clean!r}, rust={self.prompt_rust!r}")
        print(f"  KMeans: K={self.kmeans_k}, attempts={self.kmeans_attempts}, max_iter={self.kmeans_max_iter}")
        print(f"  KMeans train samples: {self.kmeans_train_samples}")
        print(f"  SAM evidence threshold (valid pixels): {self.sam_evidence_thresh:.3f}")
        print(f"  Ensure one positive: {self.ensure_one_positive}")

    # ---- SAM3 ----
    def load_sam3_model(self):
        """
        Loads SAM3SemanticPredictor from Ultralytics.
        If it fails, SAM3 is disabled.
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception:
            self._log(
                "Failed to import SAM3SemanticPredictor. SAM3 disabled.\n"
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

    # ---- Utility ----
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
    def _safe_resize_mask(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        h, w = hw
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _norm_percentile(arr: np.ndarray, q: float = 99.0, eps: float = 1e-6) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        denom = float(np.percentile(finite, q))
        denom = max(denom, eps)
        return np.clip(arr / denom, 0.0, 1.0).astype(np.float32)

    def apply_morphological_operations(self, mask: np.ndarray | None):
        if mask is None:
            return None
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        return (mask_uint8 > 127).astype(np.uint8)

    def _run_sam3_text_prompt(
        self, image_source: Union[str, np.ndarray], prompt: str
    ):
        """
        Robust wrapper: supports path or ndarray by saving ndarray to a temporary image.
        """
        if self.sam3_predictor is None:
            return None

        tmp_path = None
        try:
            src = image_source
            if isinstance(image_source, np.ndarray):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = tmp.name
                tmp.close()
                ok = cv2.imwrite(tmp_path, image_source)
                if not ok:
                    raise RuntimeError("Failed to write temp image for SAM3 inference.")
                src = tmp_path

            self.sam3_predictor.set_image(src)
            results = self.sam3_predictor(text=[prompt])
            return results
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _sam3_best_detection_for_prompt(
        self, image_source: Union[str, np.ndarray], prompt: str, expected_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        Returns:
          best_mask (H,W) uint8
          best_box  (x1,y1,x2,y2)
          best_score float
        """
        if self.sam3_predictor is None:
            return None, None, 0.0

        try:
            results = self._run_sam3_text_prompt(image_source, prompt)
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

            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "conf", None) is not None
            ):
                conf = r0.boxes.conf.detach().cpu().numpy().astype(float)
                if conf.size >= n:
                    best_i = int(np.argmax(conf[:n]))
                    best_score = float(conf[best_i])
                elif conf.size > 0:
                    best_i = int(np.argmax(conf))
                    best_score = float(conf[best_i])
                else:
                    best_i = 0
                    best_score = 1.0
            else:
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = 1.0

            best_mask = masks[best_i].detach().cpu().numpy().astype(np.float32)
            best_mask = np.clip(best_mask, 0.0, 1.0)
            if expected_hw is not None:
                best_mask = self._safe_resize_mask(best_mask, expected_hw)

            best_mask_u8 = (best_mask > 0.5).astype(np.uint8)
            processed = self.apply_morphological_operations(best_mask_u8)
            if processed is not None:
                best_mask_u8 = processed

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
                best_box = self._bbox_from_mask(best_mask_u8)

            return best_mask_u8, best_box, float(best_score)
        except Exception as e:
            self._log(f"SAM3 best-detection error for prompt={prompt!r}: {e}")
            return None, None, 0.0

    def _sam3_prompt_evidence_map(
        self,
        image_source: Union[str, np.ndarray],
        prompt: str,
        expected_hw: Tuple[int, int],
    ) -> Dict:
        """
        Builds a per-pixel evidence map from all detections for a prompt.
        Evidence is max(conf_i * mask_i) over detections.

        Returns dict:
          {
            "evidence_map": float32 HxW in [0,1],
            "best_mask": uint8 HxW,
            "best_box": (x1,y1,x2,y2) or None,
            "best_score": float,
            "num_detections": int,
          }
        """
        h, w = expected_hw
        zeros_f = np.zeros((h, w), dtype=np.float32)
        zeros_u = np.zeros((h, w), dtype=np.uint8)

        if self.sam3_predictor is None:
            return dict(
                evidence_map=zeros_f,
                best_mask=zeros_u,
                best_box=None,
                best_score=0.0,
                num_detections=0,
            )

        try:
            results = self._run_sam3_text_prompt(image_source, prompt)
            if not results:
                return dict(
                    evidence_map=zeros_f,
                    best_mask=zeros_u,
                    best_box=None,
                    best_score=0.0,
                    num_detections=0,
                )
            r0 = results[0]

            if getattr(r0, "masks", None) is None or r0.masks is None or r0.masks.data is None:
                return dict(
                    evidence_map=zeros_f,
                    best_mask=zeros_u,
                    best_box=None,
                    best_score=0.0,
                    num_detections=0,
                )

            masks = r0.masks.data
            n = int(len(masks))
            if n == 0:
                return dict(
                    evidence_map=zeros_f,
                    best_mask=zeros_u,
                    best_box=None,
                    best_score=0.0,
                    num_detections=0,
                )

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

            xyxy = None
            if (
                getattr(r0, "boxes", None) is not None
                and r0.boxes is not None
                and getattr(r0.boxes, "xyxy", None) is not None
            ):
                try:
                    xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                except Exception:
                    xyxy = None

            evidence = np.zeros((h, w), dtype=np.float32)
            best_i = 0
            best_score = -1.0

            for i in range(n):
                mi = masks[i].detach().cpu().numpy().astype(np.float32)
                mi = np.clip(mi, 0.0, 1.0)
                if mi.shape[:2] != (h, w):
                    mi = self._safe_resize_mask(mi, (h, w))

                ci = float(conf[i]) if (conf is not None and i < len(conf)) else 1.0
                ci = float(np.clip(ci, 0.0, 1.0))

                evidence = np.maximum(evidence, mi * ci)

                if ci > best_score:
                    best_score = ci
                    best_i = i

            best_mask_f = masks[best_i].detach().cpu().numpy().astype(np.float32)
            best_mask_f = np.clip(best_mask_f, 0.0, 1.0)
            if best_mask_f.shape[:2] != (h, w):
                best_mask_f = self._safe_resize_mask(best_mask_f, (h, w))
            best_mask_u8 = (best_mask_f > 0.5).astype(np.uint8)
            processed = self.apply_morphological_operations(best_mask_u8)
            if processed is not None:
                best_mask_u8 = processed

            best_box = None
            if xyxy is not None and xyxy.shape[0] > best_i:
                x1, y1, x2, y2 = xyxy[best_i]
                best_box = (int(x1), int(y1), int(x2), int(y2))
            if best_box is None:
                best_box = self._bbox_from_mask(best_mask_u8)

            return dict(
                evidence_map=np.clip(evidence, 0.0, 1.0).astype(np.float32),
                best_mask=best_mask_u8,
                best_box=best_box,
                best_score=float(max(best_score, 0.0)),
                num_detections=n,
            )

        except Exception as e:
            self._log(f"SAM3 evidence-map error for prompt={prompt!r}: {e}")
            return dict(
                evidence_map=zeros_f,
                best_mask=zeros_u,
                best_box=None,
                best_score=0.0,
                num_detections=0,
            )

    def get_best_metal_roi(
        self, image_path: str, original: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Returns:
          metal_mask_full (HxW uint8 0/1),
          metal_box_full (x1,y1,x2,y2),
          metal_score float

        If SAM3 fails -> full image box, mask=ones, score=0.0
        """
        h, w = original.shape[:2]
        mask, box, score = self._sam3_best_detection_for_prompt(image_path, self.prompt_metal, expected_hw=(h, w))
        if mask is None or box is None or not np.any(mask):
            return np.ones((h, w), dtype=np.uint8), (0, 0, w, h), 0.0
        return mask.astype(np.uint8), box, float(score)

    # ---- Per-pixel features (ROI) ----
    def _compute_pixel_feature_tensor(
        self,
        crop: np.ndarray,
        clean_evidence: np.ndarray,
        rust_evidence: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Returns feature tensor HxWxF (float32), plus feature names.
        """
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)

        k = 5
        local_mean = cv2.blur(gray, (k, k))
        local_sq_mean = cv2.blur(gray**2, (k, k))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0.0))

        # Normalize image-space features
        L_norm = (lab[:, :, 0] / 255.0).astype(np.float32)
        a_norm = (lab[:, :, 1] / 255.0).astype(np.float32)
        b_norm = (lab[:, :, 2] / 255.0).astype(np.float32)

        H_norm = (hsv[:, :, 0] / 179.0).astype(np.float32)
        S_norm = (hsv[:, :, 1] / 255.0).astype(np.float32)
        V_norm = (hsv[:, :, 2] / 255.0).astype(np.float32)

        grad_norm = self._norm_percentile(grad, q=99.0)
        texture_norm = self._norm_percentile(local_std, q=99.0)

        a_centered = lab[:, :, 1] - 128.0
        b_centered = lab[:, :, 2] - 128.0
        chroma = np.sqrt(a_centered**2 + b_centered**2).astype(np.float32)
        chroma_norm = self._norm_percentile(chroma, q=99.0)

        # Rust-oriented color priors (normalized [0,1]-ish)
        # Higher when red/brown-ish; reduced by strong "clean" evidence later via features.
        redness = np.clip((a_centered + 10.0) / 80.0, 0.0, 1.0).astype(np.float32)
        brownness = np.clip((0.6 * a_centered + 0.8 * b_centered + 20.0) / 120.0, 0.0, 1.0).astype(np.float32)

        # SAM3-derived evidence channels
        clean_ev = np.clip(clean_evidence.astype(np.float32), 0.0, 1.0)
        rust_ev = np.clip(rust_evidence.astype(np.float32), 0.0, 1.0)
        ev_union = np.maximum(clean_ev, rust_ev).astype(np.float32)
        ev_delta = (rust_ev - clean_ev).astype(np.float32)
        ev_ratio = (rust_ev / (rust_ev + clean_ev + 1e-6)).astype(np.float32)

        # Feature tensor (HxWxF)
        features = np.stack(
            [
                L_norm,
                a_norm,
                b_norm,
                H_norm,
                S_norm,
                V_norm,
                grad_norm,
                texture_norm,
                chroma_norm,
                redness,
                brownness,
                clean_ev,
                rust_ev,
                ev_union,
                ev_delta,
                ev_ratio,
            ],
            axis=2,
        ).astype(np.float32)

        feature_names = [
            "L_norm",
            "a_norm",
            "b_norm",
            "H_norm",
            "S_norm",
            "V_norm",
            "grad_norm",
            "texture_norm",
            "chroma_norm",
            "redness",
            "brownness",
            "sam_clean_evidence",
            "sam_rust_evidence",
            "sam_union_evidence",
            "sam_rust_minus_clean",
            "sam_rust_ratio",
        ]

        return features, feature_names

    def _build_valid_pixel_mask(
        self,
        metal_mask_crop: np.ndarray,
        clean_evidence: np.ndarray,
        rust_evidence: np.ndarray,
    ) -> np.ndarray:
        """
        Pixels considered for KMeans. Prefer union of:
          - first-pass metal mask inside crop
          - second-pass prompt evidence
        """
        metal_mask = (metal_mask_crop > 0).astype(np.uint8)
        ev_union = np.maximum(clean_evidence, rust_evidence)
        ev_mask = (ev_union >= self.sam_evidence_thresh).astype(np.uint8)

        valid = ((metal_mask > 0) | (ev_mask > 0)).astype(np.uint8)

        # Fallback if too sparse
        h, w = valid.shape
        min_pixels = max(100, int(0.01 * h * w))
        if int(valid.sum()) < min_pixels:
            valid = np.ones((h, w), dtype=np.uint8)

        return valid

    def _kmeans_pixel_classification(
        self,
        feature_tensor: np.ndarray,
        feature_names: List[str],
        valid_mask: np.ndarray,
    ) -> Dict:
        """
        KMeans on per-pixel feature vectors (valid pixels only).
        Returns labels map and rust mask.
        """
        h, w, fdim = feature_tensor.shape
        valid = valid_mask.astype(bool)
        n_valid = int(valid.sum())

        cluster_map = np.full((h, w), fill_value=-1, dtype=np.int16)
        rust_mask = np.zeros((h, w), dtype=np.uint8)

        if n_valid == 0:
            return dict(
                cluster_map=cluster_map,
                rust_mask=rust_mask,
                rust_cluster_id=-1,
                clean_cluster_id=-1,
                centers=None,
                cluster_scores={},
                n_valid=0,
            )

        X_raw = feature_tensor[valid].reshape(-1, fdim).astype(np.float32)

        # Standardize for KMeans
        mu = X_raw.mean(axis=0, keepdims=True)
        sigma = X_raw.std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        X = (X_raw - mu) / sigma
        X = X.astype(np.float32)

        # Subsample for training if too many pixels
        n = X.shape[0]
        if n > self.kmeans_train_samples:
            rng = np.random.default_rng(0)
            idx_train = rng.choice(n, size=self.kmeans_train_samples, replace=False)
            X_train = X[idx_train]
        else:
            X_train = X

        K = min(self.kmeans_k, max(2, X_train.shape[0]))
        if X_train.shape[0] < 2:
            # Fallback: direct rust-vs-clean evidence rule
            idx = {name: i for i, name in enumerate(feature_names)}
            rust_i = idx["sam_rust_evidence"]
            clean_i = idx["sam_clean_evidence"]
            rule = (X_raw[:, rust_i] > X_raw[:, clean_i]).astype(np.uint8)
            tmp = np.zeros(n_valid, dtype=np.int16)
            tmp[:] = rule.astype(np.int16)
            cluster_map[valid] = tmp
            rust_mask[valid] = rule
            return dict(
                cluster_map=cluster_map,
                rust_mask=rust_mask,
                rust_cluster_id=1,
                clean_cluster_id=0,
                centers=None,
                cluster_scores={"fallback_rule": True},
                n_valid=n_valid,
            )

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(self.kmeans_max_iter),
            float(self.kmeans_eps),
        )
        compactness, labels_train, centers = cv2.kmeans(
            X_train,
            K,
            None,
            criteria,
            int(self.kmeans_attempts),
            cv2.KMEANS_PP_CENTERS,
        )

        # Assign ALL valid pixels to nearest center
        # Distances in standardized feature space
        # X: [N,F], centers: [K,F]
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_all = np.argmin(d2, axis=1).astype(np.int16)

        # Rust cluster selection (prompt-guided + color/texture prior)
        idx = {name: i for i, name in enumerate(feature_names)}

        # Score each cluster
        cluster_scores: Dict[int, float] = {}
        for c in range(K):
            m = labels_all == c
            if not np.any(m):
                cluster_scores[c] = -1e9
                continue

            meanv = X_raw[m].mean(axis=0)

            # Strongly prioritize "rusty metal" prompt evidence over "clean shiny metal"
            score = (
                2.6 * float(meanv[idx["sam_rust_evidence"]])
                - 2.1 * float(meanv[idx["sam_clean_evidence"]])
                + 0.45 * float(meanv[idx["redness"]])
                + 0.40 * float(meanv[idx["brownness"]])
                + 0.18 * float(meanv[idx["texture_norm"]])
                + 0.10 * float(meanv[idx["grad_norm"]])
                + 0.10 * float(meanv[idx["S_norm"]])
                - 0.05 * float(meanv[idx["V_norm"]])
            )
            cluster_scores[c] = float(score)

        rust_cluster_id = int(max(cluster_scores, key=cluster_scores.get))
        clean_cluster_id = None
        if K == 2:
            clean_cluster_id = 1 - rust_cluster_id
        else:
            # choose second-best as "clean" for visualization
            ordered = sorted(cluster_scores.items(), key=lambda kv: kv[1], reverse=True)
            clean_cluster_id = int(ordered[1][0]) if len(ordered) > 1 else rust_cluster_id

        # Initial rust mask from cluster id
        rust_pred = (labels_all == rust_cluster_id).astype(np.uint8)

        # Optional consistency filter using SAM3 evidence features
        rust_i = idx["sam_rust_evidence"]
        clean_i = idx["sam_clean_evidence"]
        delta_i = idx["sam_rust_minus_clean"]
        redness_i = idx["redness"]
        brown_i = idx["brownness"]

        evidence_support = (
            (X_raw[:, rust_i] >= (X_raw[:, clean_i] - 0.02))
            | (X_raw[:, delta_i] > -0.02)
            | ((X_raw[:, redness_i] > 0.55) & (X_raw[:, brown_i] > 0.45))
        )
        rust_pred = (rust_pred & evidence_support.astype(np.uint8)).astype(np.uint8)

        # Ensure at least one positive if requested
        if self.ensure_one_positive and int(rust_pred.sum()) == 0 and n_valid > 0:
            # choose the pixel with max rust-vs-clean evidence, then warmest as tiebreak
            rank = (
                2.0 * X_raw[:, rust_i]
                - 1.5 * X_raw[:, clean_i]
                + 0.5 * X_raw[:, redness_i]
                + 0.4 * X_raw[:, brown_i]
            )
            rust_pred[int(np.argmax(rank))] = 1

        # Write back maps
        cluster_map[valid] = labels_all
        rust_mask[valid] = rust_pred

        # Morphology on ROI rust mask
        kernel = np.ones((3, 3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)

        return dict(
            cluster_map=cluster_map,
            rust_mask=rust_mask.astype(np.uint8),
            rust_cluster_id=int(rust_cluster_id),
            clean_cluster_id=int(clean_cluster_id),
            centers=centers,
            cluster_scores=cluster_scores,
            n_valid=n_valid,
            compactness=float(compactness),
        )

    def _perform_pixel_kmeans_analysis(
        self,
        crop: np.ndarray,
        metal_mask_crop: np.ndarray,
    ) -> Dict:
        """
        Second-pass SAM3 prompts + per-pixel feature tensor + KMeans classification.
        """
        h, w = crop.shape[:2]

        self._log(f"SAM3 pass #2 in ROI: prompt={self.prompt_clean!r}")
        clean_out = self._sam3_prompt_evidence_map(crop, self.prompt_clean, expected_hw=(h, w))

        self._log(f"SAM3 pass #2 in ROI: prompt={self.prompt_rust!r}")
        rust_out = self._sam3_prompt_evidence_map(crop, self.prompt_rust, expected_hw=(h, w))

        clean_ev = clean_out["evidence_map"]
        rust_ev = rust_out["evidence_map"]

        feature_tensor, feature_names = self._compute_pixel_feature_tensor(crop, clean_ev, rust_ev)
        valid_mask = self._build_valid_pixel_mask(metal_mask_crop, clean_ev, rust_ev)

        km = self._kmeans_pixel_classification(feature_tensor, feature_names, valid_mask)

        rust_mask_roi = km["rust_mask"]
        rust_pixels = int(rust_mask_roi.sum())
        total_pixels_box = int(h * w)
        valid_pixels = int(valid_mask.sum())

        rust_pct_box = (100.0 * rust_pixels / total_pixels_box) if total_pixels_box > 0 else 0.0
        rust_pct_valid = (100.0 * rust_pixels / valid_pixels) if valid_pixels > 0 else 0.0

        # Prompt-guided "score map" for optional diagnostics/visualization
        score_map = np.clip(0.5 + 0.5 * (rust_ev - clean_ev), 0.0, 1.0).astype(np.float32)

        self._log(
            f"ROI pixels={total_pixels_box}, valid={valid_pixels}, rust={rust_pixels} "
            f"({rust_pct_box:.2f}% of box, {rust_pct_valid:.2f}% of valid)"
        )

        return dict(
            sam_clean_evidence=clean_ev,
            sam_rust_evidence=rust_ev,
            sam_clean_best_mask=clean_out["best_mask"],
            sam_rust_best_mask=rust_out["best_mask"],
            sam_clean_best_score=float(clean_out["best_score"]),
            sam_rust_best_score=float(rust_out["best_score"]),
            sam_clean_num_dets=int(clean_out["num_detections"]),
            sam_rust_num_dets=int(rust_out["num_detections"]),
            pixel_feature_tensor=feature_tensor,      # HxWxF
            pixel_feature_names=feature_names,        # len F
            valid_mask=valid_mask.astype(np.uint8),
            cluster_map=km["cluster_map"],            # HxW int16, -1 invalid
            rust_cluster_id=int(km["rust_cluster_id"]),
            clean_cluster_id=int(km["clean_cluster_id"]),
            cluster_scores=km["cluster_scores"],
            crop_mask=rust_mask_roi.astype(np.uint8),
            score_map=score_map,
            rust_percentage_box=rust_pct_box,
            rust_percentage_valid=rust_pct_valid,
            kmeans_meta=km,
        )

    # ---- Main analysis ----
    def analyze(self, image_path: str, interactive: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        # 0) Load
        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        self.original_image = original.copy()
        timings["load"] = time.time() - t0

        # 1) SAM3 pass #1 -> best metal box on full image
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log(f"SAM3 pass #1: selecting MAX-score box for prompt={self.prompt_metal!r} ...")
            metal_mask_full, metal_box_full, metal_box_score = self.get_best_metal_roi(image_path, original)
        else:
            h, w = original.shape[:2]
            metal_mask_full = np.ones((h, w), dtype=np.uint8)
            metal_box_full = (0, 0, w, h)
            metal_box_score = 0.0
        timings["sam3_pass1"] = time.time() - t0

        # Terminate if score is none/zero (same behavior as your original)
        if (metal_box_score is None) or (float(metal_box_score) <= 0.0):
            raise SystemExit("Terminating: metal score is None or 0 (no valid metal detection).")

        # 2) Crop using full box (not mask)
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

        # 3) SAM3 pass #2 (clean/rust prompts) + per-pixel KMeans in ROI
        t0 = time.time()
        self._log("Running per-pixel feature extraction + KMeans (no SLIC) inside ROI...")
        res = self._perform_pixel_kmeans_analysis(crop, metal_mask_crop)
        timings["roi_kmeans"] = time.time() - t0

        # 4) Map ROI rust mask back to full image
        t0 = time.time()
        full_rust_mask = np.zeros((H, W), dtype=np.uint8)
        full_rust_mask[y1p:y2p, x1p:x2p] = res["crop_mask"]
        timings["map_back"] = time.time() - t0

        total_time = float(sum(timings.values()))
        self._log(
            f"Metal box score: {metal_box_score:.2f} | "
            f"ROI rust coverage (box): {res['rust_percentage_box']:.2f}% | "
            f"ROI rust coverage (valid): {res['rust_percentage_valid']:.2f}% | "
            f"Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            metal_mask_full=metal_mask_full,
            metal_box_full=metal_box_full,
            metal_box_score=float(metal_box_score),
            crop=crop,
            crop_coords=crop_coords,
            metal_mask_crop=metal_mask_crop,
            full_mask=full_rust_mask,
            crop_mask=res["crop_mask"],
            score_map=res["score_map"],
            rust_percentage=float(res["rust_percentage_box"]),
            rust_percentage_valid=float(res["rust_percentage_valid"]),
            timings=timings,
            # pass-2 outputs
            sam_clean_evidence=res["sam_clean_evidence"],
            sam_rust_evidence=res["sam_rust_evidence"],
            sam_clean_best_mask=res["sam_clean_best_mask"],
            sam_rust_best_mask=res["sam_rust_best_mask"],
            sam_clean_best_score=res["sam_clean_best_score"],
            sam_rust_best_score=res["sam_rust_best_score"],
            sam_clean_num_dets=res["sam_clean_num_dets"],
            sam_rust_num_dets=res["sam_rust_num_dets"],
            pixel_feature_tensor=res["pixel_feature_tensor"],
            pixel_feature_names=res["pixel_feature_names"],
            valid_mask=res["valid_mask"],
            cluster_map=res["cluster_map"],
            rust_cluster_id=res["rust_cluster_id"],
            clean_cluster_id=res["clean_cluster_id"],
            cluster_scores=res["cluster_scores"],
            kmeans_meta=res["kmeans_meta"],
        )

    # ---- Visualization (updated multi-stage) ----
    def visualize_detection(self, results: Dict, save_path: str | None = None) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]
        full_mask = results["full_mask"]
        crop_mask = results["crop_mask"]
        metal_mask_crop = results["metal_mask_crop"]

        clean_ev = results["sam_clean_evidence"]
        rust_ev = results["sam_rust_evidence"]
        valid_mask = results["valid_mask"]
        cluster_map = results["cluster_map"]
        rust_cluster_id = int(results["rust_cluster_id"])
        clean_cluster_id = int(results["clean_cluster_id"])

        (bx1, by1, bx2, by2) = results["metal_box_full"]
        bscore = float(results["metal_box_score"])
        rust_percentage = float(results["rust_percentage"])
        rust_percentage_valid = float(results.get("rust_percentage_valid", 0.0))

        # Stage 1: input
        stage1 = original.copy()

        # Stage 2: input + best metal box
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

        # Stage 3: cropped ROI + first-pass metal mask (blue)
        stage3 = crop.copy()
        if metal_mask_crop is not None and metal_mask_crop.size > 0:
            overlay = stage3.copy()
            overlay[metal_mask_crop == 1] = [255, 0, 0]  # blue in BGR
            stage3 = cv2.addWeighted(stage3, 0.7, overlay, 0.3, 0)

        # Stage 4: second-pass prompt evidence (clean=green, rust=red)
        stage4 = crop.copy()
        ev_vis = np.zeros_like(crop, dtype=np.uint8)
        ev_vis[:, :, 1] = np.clip(clean_ev * 255.0, 0, 255).astype(np.uint8)  # G
        ev_vis[:, :, 2] = np.clip(rust_ev * 255.0, 0, 255).astype(np.uint8)   # R
        stage4 = cv2.addWeighted(stage4, 0.6, ev_vis, 0.5, 0)

        # Stage 5: KMeans cluster visualization in ROI
        stage5 = crop.copy()
        km_overlay = stage5.copy()

        valid = valid_mask.astype(bool)
        if np.any(valid):
            km_overlay[valid] = (0, 0, 0)  # darken valid region base
            if clean_cluster_id >= 0:
                km_overlay[(cluster_map == clean_cluster_id) & valid] = [0, 255, 0]  # green
            if rust_cluster_id >= 0:
                km_overlay[(cluster_map == rust_cluster_id) & valid] = [0, 0, 255]   # red

        stage5 = cv2.addWeighted(stage5, 0.55, km_overlay, 0.45, 0)

        # highlight final rust mask stronger
        rust_overlay_roi = stage5.copy()
        rust_overlay_roi[crop_mask == 1] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.7, rust_overlay_roi, 0.4, 0)

        # Stage 6: final overlay on full image
        stage6 = original.copy()
        rust_overlay_full = stage6.copy()
        rust_overlay_full[full_mask == 1] = [0, 0, 255]
        stage6 = cv2.addWeighted(stage6, 0.6, rust_overlay_full, 0.4, 0)
        cv2.rectangle(stage6, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Rust Detection Stages (SAM3 metal ROI -> SAM3 clean/rust prompts -> per-pixel features -> KMeans)\n"
            f"Metal box score={bscore:.2f} | Coverage(box)={rust_percentage:.1f}% | Coverage(valid)={rust_percentage_valid:.1f}%",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6]
        titles = [
            "1) Input image",
            "2) Input + best metal box (pass #1)",
            "3) Cropped ROI + first-pass metal mask (blue)",
            "4) SAM3 pass #2 prompt evidence (clean=green, rust=red)",
            "5) KMeans pixel clusters in ROI + final rust mask",
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
    p = argparse.ArgumentParser(
        description="Rust Detection: SAM3 metal ROI (pass1) -> SAM3 clean/rust evidence (pass2) -> per-pixel KMeans"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3 pass #1 for metal ROI; 0=use full image.")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around best metal box before cropping.")

    # Prompts
    p.add_argument("--prompt_metal", type=str, default="metal", help="SAM3 prompt for pass #1 ROI extraction.")
    p.add_argument(
        "--prompt_clean",
        type=str,
        default="clean shiny metal",
        help="SAM3 prompt for pass #2 clean metal evidence.",
    )
    p.add_argument(
        "--prompt_rust",
        type=str,
        default="rusty metal",
        help="SAM3 prompt for pass #2 rust evidence.",
    )

    # KMeans
    p.add_argument("--kmeans_k", type=int, default=2, help="Number of clusters for pixel KMeans (recommended=2).")
    p.add_argument("--kmeans_attempts", type=int, default=5)
    p.add_argument("--kmeans_max_iter", type=int, default=50)
    p.add_argument("--kmeans_eps", type=float, default=0.2)
    p.add_argument(
        "--kmeans_train_samples",
        type=int,
        default=80000,
        help="Max sampled pixels used to fit KMeans; all valid pixels are assigned afterward.",
    )

    # Masks / behavior
    p.add_argument("--sam_evidence_thresh", type=float, default=0.02, help="Threshold for SAM evidence valid pixel mask.")
    p.add_argument("--ensure_one_positive", type=int, default=1)
    p.add_argument("--verbose", type=int, default=1)

    # Outputs
    p.add_argument("--save_features", type=int, default=1, help="Save per-pixel feature tensor and maps as .npz")
    p.add_argument("--res_dir", type=str, default="NewTh")
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
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        roi_pad=int(args.roi_pad),
        metal_prompt=args.prompt_metal,
        clean_prompt=args.prompt_clean,
        rust_prompt=args.prompt_rust,
        kmeans_k=int(args.kmeans_k),
        kmeans_attempts=int(args.kmeans_attempts),
        kmeans_max_iter=int(args.kmeans_max_iter),
        kmeans_eps=float(args.kmeans_eps),
        kmeans_train_samples=int(args.kmeans_train_samples),
        sam_evidence_thresh=float(args.sam_evidence_thresh),
        ensure_one_positive=bool(args.ensure_one_positive),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("results", args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    img_out = os.path.join(out_dir, f"{base}_stages.png")
    detector.visualize_detection(results, save_path=img_out)

    # Save per-pixel feature vectors + key maps
    if bool(args.save_features):
        feat_out = os.path.join(out_dir, f"{base}_pixel_features.npz")

        # Save feature names as fixed-width unicode array for compatibility
        feature_names_arr = np.array(results["pixel_feature_names"], dtype="<U64")

        np.savez_compressed(
            feat_out,
            pixel_feature_tensor=results["pixel_feature_tensor"].astype(np.float32),  # HxWxF
            pixel_feature_names=feature_names_arr,  # [F]
            crop_coords=np.array(results["crop_coords"], dtype=np.int32),  # x,y,w,h
            valid_mask=results["valid_mask"].astype(np.uint8),             # HxW
            cluster_map=results["cluster_map"].astype(np.int16),           # HxW
            rust_cluster_id=np.array([results["rust_cluster_id"]], dtype=np.int32),
            clean_cluster_id=np.array([results["clean_cluster_id"]], dtype=np.int32),
            sam_clean_evidence=results["sam_clean_evidence"].astype(np.float32),
            sam_rust_evidence=results["sam_rust_evidence"].astype(np.float32),
            sam_clean_best_mask=results["sam_clean_best_mask"].astype(np.uint8),
            sam_rust_best_mask=results["sam_rust_best_mask"].astype(np.uint8),
            metal_mask_crop=results["metal_mask_crop"].astype(np.uint8),
            rust_mask_roi=results["crop_mask"].astype(np.uint8),
            rust_mask_full=results["full_mask"].astype(np.uint8),
            metal_box_full=np.array(results["metal_box_full"], dtype=np.int32),
            metal_box_score=np.array([results["metal_box_score"]], dtype=np.float32),
            rust_percentage_box=np.array([results["rust_percentage"]], dtype=np.float32),
            rust_percentage_valid=np.array([results.get("rust_percentage_valid", 0.0)], dtype=np.float32),
        )
        print(f"Saved per-pixel feature tensor + maps to: {feat_out}")
        print(f"Feature tensor shape (H, W, F): {results['pixel_feature_tensor'].shape}")
        print(f"Feature names: {results['pixel_feature_names']}")


if __name__ == "__main__":
    main()
