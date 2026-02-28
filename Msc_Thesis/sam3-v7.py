from __future__ import annotations

from typing import Dict, Tuple, Optional
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- Optional deps -----------------------------
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


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
    filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")),
) -> str:
    from tkinter import filedialog
    root = _get_tk_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise ValueError("No file selected.")
    return path


# ----------------------------- Math Utils -----------------------------
def softmax2(a: float, b: float, temp: float = 1.0) -> Tuple[float, float]:
    t = max(1e-6, float(temp))
    x = np.array([a, b], dtype=np.float32) / t
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(np.sum(e)) if float(np.sum(e)) > 0 else 1.0
    return float(e[0] / s), float(e[1] / s)


def bbox_from_mask(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def union_boxes(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def overlay_heatmap_bgr(base_bgr: np.ndarray, prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    p = np.clip(prob, 0.0, 1.0)
    hm = (p * 255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1.0 - float(alpha), hm_color, float(alpha), 0)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n


# ----------------------------- Numpy MiniBatch KMeans fallback -----------------------------
def kmeans_numpy_minibatch(
    X: np.ndarray,
    k: int = 2,
    iters: int = 30,
    seed: int = 0,
    batch: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple mini-batch kmeans in NumPy.
    Returns:
      centers: (k, D)
      labels:  (N,) 0..k-1
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    if N == 0:
        return np.zeros((k, D), np.float32), np.zeros((0,), np.int32)

    init_idx = rng.choice(N, size=min(k, N), replace=False)
    centers = X[init_idx].astype(np.float32).copy()
    if centers.shape[0] < k:
        centers = np.concatenate([centers, np.repeat(centers[:1], k - centers.shape[0], axis=0)], axis=0)

    for _ in range(iters):
        b = min(batch, N)
        idx = rng.choice(N, size=b, replace=False)
        xb = X[idx].astype(np.float32)

        x2 = np.sum(xb * xb, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        d2 = x2 - 2.0 * (xb @ centers.T) + c2
        lab = np.argmin(d2, axis=1)

        for j in range(k):
            m = lab == j
            if not np.any(m):
                continue
            mu = xb[m].mean(axis=0)
            centers[j] = 0.7 * centers[j] + 0.3 * mu

    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 - 2.0 * (X @ centers.T) + c2
    labels = np.argmin(d2, axis=1).astype(np.int32)
    return centers.astype(np.float32), labels


# ----------------------------- Detector -----------------------------
class RustDetectorSAM3EmbKMeans:
    """
    UPDATED (NO SLIC):
      - Two SAM3 prompts (clean vs rust) for ROI + weak supervision
      - Extract SAM3 dense image embeddings (feature vectors)
      - Build per-pixel features from embeddings (optionally + Lab color)
      - KMeans(2) clustering inside ROI
      - Decide which cluster is rust using overlap with SAM3 prompt masks
      - Produce per-pixel P(rust) + visualization
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        roi_pad: int = 0,
        prompt_clean: str = "clean shiny metal",
        prompt_rust: str = "rusty metal",
        # feature controls
        use_color_features: bool = True,
        color_weight: float = 0.35,
        # kmeans controls
        sample_pixels: int = 150_000,
        kmeans_iters: int = 40,
        seed: int = 0,
        # probability / threshold controls
        dynamic_prob_threshold: bool = True,
        prob_threshold_fallback: float = 0.55,
        otsu_bias: float = -0.02,
        ensure_one_positive: bool = True,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = sam_checkpoint
        self.roi_pad = int(roi_pad)
        self.prompt_clean = str(prompt_clean)
        self.prompt_rust = str(prompt_rust)

        self.use_color_features = bool(use_color_features)
        self.color_weight = float(np.clip(color_weight, 0.0, 2.0))

        self.sample_pixels = int(sample_pixels)
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)

        self.dynamic_prob_threshold = bool(dynamic_prob_threshold)
        self.prob_threshold_fallback = float(prob_threshold_fallback)
        self.otsu_bias = float(otsu_bias)
        self.ensure_one_positive = bool(ensure_one_positive)

        self.sam3_predictor = None
        self._last_image_embedding: Optional[np.ndarray] = None  # (C,h,w) float32

        if self.verbose:
            self._print_info()

        self.load_sam3_model()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_info(self):
        print("RustDetectorSAM3EmbKMeans initialized:")
        print(f"  sam_checkpoint: {self.sam_checkpoint}")
        print(f"  prompts: clean={self.prompt_clean!r}, rust={self.prompt_rust!r}")
        print(f"  roi_pad: {self.roi_pad}px")
        print(f"  use_color_features: {self.use_color_features} (weight={self.color_weight:.2f})")
        print(f"  sample_pixels: {self.sample_pixels}")
        print(f"  kmeans_iters: {self.kmeans_iters} | sklearn={SKLEARN_AVAILABLE}")
        print(f"  dynamic_prob_threshold: {self.dynamic_prob_threshold}")
        print(f"  prob_threshold_fallback: {self.prob_threshold_fallback:.2f}")
        print(f"  otsu_bias: {self.otsu_bias:+.2f}")

    # ----------------- SAM3 load + embedding hook -----------------
    def load_sam3_model(self):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception:
            self._log("Failed to import SAM3SemanticPredictor. Install: pip install -U ultralytics")
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

        if self.sam3_predictor is not None and not TORCH_AVAILABLE:
            self._log("WARNING: torch not available; cannot extract SAM embeddings reliably.")

    def _try_get_image_encoder_module(self):
        if self.sam3_predictor is None:
            return None
        model = getattr(self.sam3_predictor, "model", None)
        if model is None:
            return None

        candidates = []
        for name in ["sam", "model", "net"]:
            m = getattr(model, name, None)
            if m is not None:
                candidates.append(m)
        candidates.append(model)

        for m in candidates:
            for attr in ["image_encoder", "img_encoder", "encoder", "backbone"]:
                enc = getattr(m, attr, None)
                if enc is not None:
                    return enc

        inner = getattr(model, "model", None)
        if inner is not None:
            for attr in ["image_encoder", "img_encoder", "encoder", "backbone"]:
                enc = getattr(inner, attr, None)
                if enc is not None:
                    return enc
        return None

    def _register_embedding_hook(self):
        if self.sam3_predictor is None or not TORCH_AVAILABLE:
            return None

        enc = self._try_get_image_encoder_module()
        if enc is None:
            self._log("Could not locate SAM3 image encoder (embedding hook). Will run without embeddings.")
            return None

        self._last_image_embedding = None

        def _hook(_module, _inputs, output):
            try:
                out = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(out, "detach"):
                    t = out.detach()
                    if t.ndim == 4:
                        t = t[0]
                    self._last_image_embedding = t.float().cpu().numpy()
            except Exception:
                self._last_image_embedding = None

        try:
            return enc.register_forward_hook(_hook)
        except Exception as e:
            self._log(f"Hook register failed: {e}")
            return None

    # ----------------- SAM3 prompt inference -----------------
    def _sam3_best_detection_for_prompt(
        self, image_path: str, prompt: str
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
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
                best_i = int(np.argmax(conf_arr[:n])) if conf_arr.size >= n else int(np.argmax(conf_arr))
                best_score = float(conf_arr[best_i])
            else:
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = float(areas[best_i])

            best_mask = (masks[best_i].detach().cpu().numpy() > 0.5).astype(np.uint8)

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
                best_box = bbox_from_mask(best_mask)

            return best_mask.astype(np.uint8), best_box, float(best_score)
        except Exception as e:
            self._log(f"SAM3 error prompt={prompt!r}: {e}")
            return None, None, 0.0

    # ----------------- Threshold (Otsu) -----------------
    def _otsu_threshold(self, probs: np.ndarray) -> float:
        p = probs[np.isfinite(probs)]
        if p.size < 100:
            return float(self.prob_threshold_fallback)
        p255 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
        if int(p255.max()) - int(p255.min()) < 5:
            return float(self.prob_threshold_fallback)
        t, _ = cv2.threshold(p255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
        return float(np.clip(thr + self.otsu_bias, 0.25, 0.85))

    # ----------------- Feature building -----------------
    def _dense_features_for_roi(self, crop_bgr: np.ndarray, dense_emb_full: np.ndarray) -> np.ndarray:
        """
        Build per-pixel features in ROI from dense embedding.
        NOTE: dense_emb_full is captured for FULL image; we assume the SAM encoder ran on that same image
              and returned a dense embedding map aligned to the predictor's internal image.
              For practicality, we just upsample to ROI size (works well in practice).
        Returns: X (Npix, D)
        """
        C, eh, ew = dense_emb_full.shape
        H, W = crop_bgr.shape[:2]

        emb_hw_c = np.transpose(dense_emb_full, (1, 2, 0))  # (eh,ew,C)
        emb_up = cv2.resize(emb_hw_c, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        emb_flat = emb_up.reshape(-1, C)
        emb_flat = l2_normalize_rows(emb_flat)

        if not self.use_color_features:
            return emb_flat

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_flat = lab.reshape(-1, 3)
        # normalize Lab roughly
        lab_flat[:, 0] /= 255.0
        lab_flat[:, 1] = (lab_flat[:, 1] - 128.0) / 128.0
        lab_flat[:, 2] = (lab_flat[:, 2] - 128.0) / 128.0

        X = np.concatenate([emb_flat, self.color_weight * lab_flat.astype(np.float32)], axis=1).astype(np.float32)
        return X

    # ----------------- KMeans -----------------
    def _fit_kmeans2(self, X_sample: np.ndarray) -> np.ndarray:
        """
        Fit k=2 and return centers (2,D).
        """
        if X_sample.shape[0] == 0:
            return np.zeros((2, X_sample.shape[1]), np.float32)

        if SKLEARN_AVAILABLE:
            km = MiniBatchKMeans(
                n_clusters=2,
                random_state=self.seed,
                batch_size=min(50_000, max(10_000, X_sample.shape[0] // 5)),
                n_init="auto",
                max_iter=self.kmeans_iters,
                reassignment_ratio=0.01,
            )
            km.fit(X_sample)
            return km.cluster_centers_.astype(np.float32)

        centers, _ = kmeans_numpy_minibatch(X_sample, k=2, iters=self.kmeans_iters, seed=self.seed)
        return centers.astype(np.float32)

    @staticmethod
    def _assign_labels(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        d2 = x2 - 2.0 * (X @ centers.T) + c2
        return np.argmin(d2, axis=1).astype(np.int32)

    @staticmethod
    def _prob_from_centers(X: np.ndarray, centers: np.ndarray, rust_center_idx: int) -> np.ndarray:
        """
        Probability from distance margin:
          p(rust) = sigmoid((d_other - d_rust) / scale)
        """
        d0 = np.sum((X - centers[0][None, :]) ** 2, axis=1)
        d1 = np.sum((X - centers[1][None, :]) ** 2, axis=1)

        if rust_center_idx == 0:
            d_r = d0
            d_o = d1
        else:
            d_r = d1
            d_o = d0

        margin = (d_o - d_r).astype(np.float32)
        scale = float(np.std(margin)) + 1e-6
        z = margin / (2.0 * scale)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    # ----------------- Main analyze -----------------
    def analyze(self, image_path: str, interactive: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        t0 = time.time()
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        H, W = original.shape[:2]
        timings["load"] = time.time() - t0

        # Hook for embeddings
        hook = None
        if interactive and self.sam3_predictor is not None and TORCH_AVAILABLE:
            hook = self._register_embedding_hook()

        # Run SAM3 prompts (weak supervision)
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log("SAM3: running two prompts (clean vs rust) ...")
            mask_clean_full, box_clean, score_clean = self._sam3_best_detection_for_prompt(image_path, self.prompt_clean)
            mask_rust_full, box_rust, score_rust = self._sam3_best_detection_for_prompt(image_path, self.prompt_rust)

            # normalize area-scores if needed
            img_area = float(H * W) if H * W > 0 else 1.0
            if score_clean > 1.0:
                score_clean = float(score_clean / img_area)
            if score_rust > 1.0:
                score_rust = float(score_rust / img_area)

            p_clean, p_rust = softmax2(float(score_clean), float(score_rust), temp=1.0)
        else:
            mask_clean_full = np.ones((H, W), dtype=np.uint8)
            mask_rust_full = np.zeros((H, W), dtype=np.uint8)
            box_clean = (0, 0, W, H)
            box_rust = (0, 0, W, H)
            score_clean = 0.0
            score_rust = 0.0
            p_clean, p_rust = 0.5, 0.5

        timings["sam3"] = time.time() - t0

        # remove hook
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass

        dense_emb = self._last_image_embedding
        has_emb = isinstance(dense_emb, np.ndarray) and dense_emb is not None and dense_emb.ndim == 3
        if not has_emb:
            self._log("WARNING: SAM3 dense embeddings not captured. Will fallback to prompt masks as probabilities.")
            # probability = rust mask * p_rust
            prob_map_full = (mask_rust_full.astype(np.float32) * float(p_rust)).astype(np.float32)
            thr = self._otsu_threshold(prob_map_full.ravel()) if self.dynamic_prob_threshold else self.prob_threshold_fallback
            mask_full = (prob_map_full >= float(thr)).astype(np.uint8)
            if self.ensure_one_positive and mask_full.sum() == 0 and prob_map_full.size > 0:
                mask_full.ravel()[int(np.argmax(prob_map_full.ravel()))] = 1
            return dict(
                original=original,
                mask_clean_full=mask_clean_full,
                mask_rust_full=mask_rust_full,
                box_clean_full=box_clean,
                box_rust_full=box_rust,
                score_clean=float(score_clean),
                score_rust=float(score_rust),
                p_clean=float(p_clean),
                p_rust=float(p_rust),
                roi_box_full=(0, 0, W, H),
                crop=original.copy(),
                crop_coords=(0, 0, W, H),
                prob_map_crop=prob_map_full.copy(),
                mask_crop=mask_full.copy(),
                prob_map_full=prob_map_full,
                mask_full=mask_full,
                threshold_used=float(thr),
                rust_percentage=float(mask_full.mean() * 100.0),
                has_dense_embedding=False,
                dense_embedding_shape=None,
                timings=timings,
            )

        # ROI = union boxes
        roi = union_boxes(box_clean, box_rust)
        if roi is None:
            roi = (0, 0, W, H)

        x1, y1, x2, y2 = roi
        pad = self.roi_pad
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        crop = original[y1p:y2p, x1p:x2p].copy()
        clean_crop = mask_clean_full[y1p:y2p, x1p:x2p].astype(np.uint8) if mask_clean_full is not None else None
        rust_crop = mask_rust_full[y1p:y2p, x1p:x2p].astype(np.uint8) if mask_rust_full is not None else None

        # build features in ROI
        t0 = time.time()
        X = self._dense_features_for_roi(crop, dense_emb.astype(np.float32))
        timings["features"] = time.time() - t0

        # sample pixels for kmeans fit
        t0 = time.time()
        N = X.shape[0]
        rng = np.random.default_rng(self.seed)
        if N > self.sample_pixels:
            idx = rng.choice(N, size=self.sample_pixels, replace=False)
            Xs = X[idx]
        else:
            idx = np.arange(N)
            Xs = X

        centers = self._fit_kmeans2(Xs)
        labels = self._assign_labels(X, centers)  # 0/1 per pixel
        timings["kmeans"] = time.time() - t0

        # decide which cluster is "rust" using overlap with SAM3 rust mask
        rust_center_idx = 1
        if rust_crop is not None:
            rust_flat = (rust_crop.reshape(-1) > 0)
            if rust_flat.any():
                # compute hit ratio per cluster inside rust mask
                hit0 = float(np.mean(labels[rust_flat] == 0))
                hit1 = float(np.mean(labels[rust_flat] == 1))
                rust_center_idx = 0 if hit0 > hit1 else 1
            else:
                # if no rust pixels, use prompt prob: if p_rust small, pick opposite of clean overlap
                rust_center_idx = 1 if p_rust >= 0.5 else 0

        # probability per pixel from distances to centers
        t0 = time.time()
        prob_flat = self._prob_from_centers(X, centers, rust_center_idx=rust_center_idx)

        # optional: mix in prompt-level prior slightly so probabilities don't go crazy
        # (keeps things stable if KMeans splits weirdly)
        prob_flat = (0.85 * prob_flat + 0.15 * float(p_rust)).astype(np.float32)

        prob_crop = prob_flat.reshape(crop.shape[:2]).astype(np.float32)
        timings["prob"] = time.time() - t0

        # threshold
        t0 = time.time()
        if self.dynamic_prob_threshold:
            thr = self._otsu_threshold(prob_flat)
        else:
            thr = float(self.prob_threshold_fallback)

        mask_crop = (prob_crop >= float(thr)).astype(np.uint8)
        if self.ensure_one_positive and mask_crop.sum() == 0:
            mask_crop.ravel()[int(np.argmax(prob_flat))] = 1

        kernel = np.ones((3, 3), np.uint8)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)
        timings["threshold"] = time.time() - t0

        # map back
        prob_map_full = np.zeros((H, W), dtype=np.float32)
        mask_full = np.zeros((H, W), dtype=np.uint8)
        prob_map_full[y1p:y2p, x1p:x2p] = prob_crop
        mask_full[y1p:y2p, x1p:x2p] = mask_crop

        rust_pct = float(mask_crop.mean() * 100.0)
        total_time = float(sum(timings.values()))
        self._log(
            f"P(rust|prompt)={p_rust:.3f} | embeddings={has_emb} {dense_emb.shape} | "
            f"ROI rust%={rust_pct:.2f}% | thr={thr:.2f} | time={total_time:.3f}s"
        )

        return dict(
            original=original,
            # sam outputs
            mask_clean_full=mask_clean_full,
            mask_rust_full=mask_rust_full,
            box_clean_full=box_clean,
            box_rust_full=box_rust,
            score_clean=float(score_clean),
            score_rust=float(score_rust),
            p_clean=float(p_clean),
            p_rust=float(p_rust),
            # roi
            roi_box_full=(x1p, y1p, x2p, y2p),
            crop=crop,
            crop_coords=(x1p, y1p, x2p - x1p, y2p - y1p),
            clean_mask_crop=clean_crop,
            rust_mask_crop=rust_crop,
            # probabilities
            prob_map_crop=prob_crop,
            mask_crop=mask_crop,
            prob_map_full=prob_map_full,
            mask_full=mask_full,
            threshold_used=float(thr),
            rust_percentage=float(rust_pct),
            # embedding info
            has_dense_embedding=True,
            dense_embedding_shape=tuple(dense_emb.shape),
            # clustering info
            centers=centers,
            rust_center_idx=int(rust_center_idx),
            timings=timings,
        )

    # ----------------- Visualization -----------------
    def visualize(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]

        prob_crop = results["prob_map_crop"]
        mask_crop = results["mask_crop"]

        prob_full = results["prob_map_full"]
        mask_full = results["mask_full"]

        clean_crop = results.get("clean_mask_crop", None)
        rust_crop = results.get("rust_mask_crop", None)

        box_clean = results.get("box_clean_full", None)
        box_rust = results.get("box_rust_full", None)
        roi_box = results.get("roi_box_full", (0, 0, original.shape[1], original.shape[0]))

        p_clean = float(results.get("p_clean", 0.5))
        p_rust = float(results.get("p_rust", 0.5))
        thr = float(results.get("threshold_used", self.prob_threshold_fallback))
        rust_pct = float(results.get("rust_percentage", 0.0))
        emb_ok = bool(results.get("has_dense_embedding", False))
        emb_shape = results.get("dense_embedding_shape", None)

        # stage1 input
        stage1 = original.copy()

        # stage2 boxes
        stage2 = original.copy()
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(stage2, "clean", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(stage2, "rust", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(stage2, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(stage2, "ROI", (rx1, max(0, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # stage3 ROI with prompt masks overlay
        stage3 = crop.copy()
        if clean_crop is not None:
            ov = stage3.copy()
            ov[clean_crop > 0] = [0, 255, 0]
            stage3 = cv2.addWeighted(stage3, 0.75, ov, 0.25, 0)
        if rust_crop is not None:
            ov = stage3.copy()
            ov[rust_crop > 0] = [0, 0, 255]
            stage3 = cv2.addWeighted(stage3, 0.70, ov, 0.30, 0)

        # stage4 prob heatmap ROI
        stage4 = overlay_heatmap_bgr(crop.copy(), prob_crop, alpha=0.50)

        # stage5 mask overlay ROI
        stage5 = crop.copy()
        ov = stage5.copy()
        ov[mask_crop > 0] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.60, ov, 0.40, 0)

        # stage6 full prob heatmap
        stage6 = overlay_heatmap_bgr(original.copy(), prob_full, alpha=0.45)

        # stage7 full mask overlay
        stage7 = original.copy()
        ov = stage7.copy()
        ov[mask_full > 0] = [0, 0, 255]
        stage7 = cv2.addWeighted(stage7, 0.60, ov, 0.40, 0)
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage7, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage7, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(stage7, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.suptitle(
            "Rust Detection (SAM3 dense embeddings + KMeans(2) clustering)\n"
            f"P(rust|prompt)={p_rust:.3f} P(clean|prompt)={p_clean:.3f} | "
            f"embeddings={emb_ok} {emb_shape} | ROI rust%={rust_pct:.1f}% | thr={thr:.2f}",
            fontsize=12,
        )

        imgs = [
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            stage6,
            stage7,
            original,  # keep last as raw for reference
        ]
        titles = [
            "1) Input image",
            "2) Prompt boxes + ROI",
            "3) ROI + prompt masks overlay",
            "4) P(rust) heatmap (ROI)",
            "5) Rust mask (ROI)",
            "6) P(rust) heatmap (full)",
            "7) Final mask overlay (full)",
            "8) Input (reference)",
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
    p = argparse.ArgumentParser(description="Rust Detection: SAM3 embeddings + KMeans(2) -> per-pixel probabilities")
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3; 0=disable SAM3.")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around union ROI box before cropping.")

    p.add_argument("--verbose", type=int, default=1)

    # prompts
    p.add_argument("--prompt_clean", type=str, default="clean shiny metal")
    p.add_argument("--prompt_rust", type=str, default="rusty metal")

    # features
    p.add_argument("--use_color_features", type=int, default=1, help="Append Lab color to embedding features.")
    p.add_argument("--color_weight", type=float, default=0.35, help="Weight for appended Lab features.")

    # kmeans
    p.add_argument("--sample_pixels", type=int, default=150000)
    p.add_argument("--kmeans_iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)

    # thresholding
    p.add_argument("--dynamic_prob_threshold", type=int, default=1)
    p.add_argument("--prob_threshold_fallback", type=float, default=0.55)
    p.add_argument("--otsu_bias", type=float, default=-0.02)
    p.add_argument("--ensure_one_positive", type=int, default=1)

    p.add_argument("--res_dir", type=str, default="SAM3EmbKMeans")
    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    detector = RustDetectorSAM3EmbKMeans(
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        roi_pad=int(args.roi_pad),
        prompt_clean=str(args.prompt_clean),
        prompt_rust=str(args.prompt_rust),
        use_color_features=bool(args.use_color_features),
        color_weight=float(args.color_weight),
        sample_pixels=int(args.sample_pixels),
        kmeans_iters=int(args.kmeans_iters),
        seed=int(args.seed),
        dynamic_prob_threshold=bool(args.dynamic_prob_threshold),
        prob_threshold_fallback=float(args.prob_threshold_fallback),
        otsu_bias=float(args.otsu_bias),
        ensure_one_positive=bool(args.ensure_one_positive),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))
    out = f"results/{args.res_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_sam3_kmeans_prob.png"
    detector.visualize(results, save_path=out)


if __name__ == "__main__":
    main()