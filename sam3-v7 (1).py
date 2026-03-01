from __future__ import annotations

from typing import Dict, Tuple, Optional, List
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
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


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


# ----------------------------- Utils -----------------------------
def softmax2(a: float, b: float, temp: float = 1.0) -> Tuple[float, float]:
    t = max(1e-6, float(temp))
    x = np.array([a, b], dtype=np.float32) / t
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(np.sum(e)) if float(np.sum(e)) > 0 else 1.0
    return float(e[0] / s), float(e[1] / s)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-9
    bn = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (an * bn))


def overlay_heatmap_bgr(base_bgr: np.ndarray, prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    p = np.clip(prob, 0.0, 1.0)
    hm = (p * 255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1.0 - float(alpha), hm_color, float(alpha), 0)


# ----------------------------- Core Detector -----------------------------
class SAM3EmbeddingRustDetector:
    """
    SAM3 TWO PROMPTS + SAM3 DENSE EMBEDDINGS -> per-segment embeddings -> prototype classifier.

    Prompts:
      - clean: "clean shiny metal"
      - rust:  "rusty metal"

    Steps:
      1) SAM3 inference for clean and rust -> best masks, boxes, scores.
      2) Extract SAM image embedding feature map E (C x h x w) from image encoder.
      3) ROI = union of best boxes (pad optional).
      4) SLIC on ROI.
      5) For each segment, mean-pool E under that segment -> segment embedding vector.
      6) Build rust and clean prototypes by averaging embeddings of segments strongly overlapping prompt masks.
      7) Segment P(rust) via softmax of cosine similarities to prototypes.
      8) Make prob map + threshold -> final mask.
    """

    def __init__(
        self,
        n_segments: int = 8000,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        roi_pad: int = 0,
        prompt_clean: str = "clean shiny metal",
        prompt_rust: str = "rusty metal",
        # prototype selection thresholds (overlap fraction)
        proto_overlap_thr: float = 0.55,
        # thresholding
        prob_threshold_fallback: float = 0.55,
        dynamic_prob_threshold: bool = True,
        min_valid_segments_for_dynamic: int = 20,
        otsu_bias: float = -0.02,
        ensure_one_positive: bool = True,
    ):
        self.n_segments = int(n_segments)
        self.verbose = bool(verbose)
        self.sam_checkpoint = sam_checkpoint
        self.roi_pad = int(roi_pad)

        self.prompt_clean = str(prompt_clean)
        self.prompt_rust = str(prompt_rust)

        self.proto_overlap_thr = float(np.clip(proto_overlap_thr, 0.05, 0.95))

        self.prob_threshold_fallback = float(prob_threshold_fallback)
        self.dynamic_prob_threshold = bool(dynamic_prob_threshold)
        self.min_valid_segments_for_dynamic = int(min_valid_segments_for_dynamic)
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
        print("SAM3EmbeddingRustDetector initialized:")
        print(f"  n_segments: {self.n_segments}")
        print(f"  sam_checkpoint: {self.sam_checkpoint}")
        print(f"  prompts: clean={self.prompt_clean!r}, rust={self.prompt_rust!r}")
        print(f"  roi_pad: {self.roi_pad}px")
        print(f"  proto_overlap_thr: {self.proto_overlap_thr:.2f}")
        print(f"  dynamic_prob_threshold: {self.dynamic_prob_threshold}")
        print(f"  prob_threshold_fallback: {self.prob_threshold_fallback:.2f}")

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
            self._log("SAM3 loaded.")
        except Exception as e:
            self._log(f"Failed to load SAM3: {e}")
            self.sam3_predictor = None

        if self.sam3_predictor is not None and not TORCH_AVAILABLE:
            self._log("WARNING: torch not available; cannot extract SAM embeddings. Will fallback to masks/scores only.")

    def _try_get_image_encoder_module(self):
        """
        Ultralytics internal structure can vary.
        We try a few common paths to find a module whose forward output is the dense image embedding.
        """
        if self.sam3_predictor is None:
            return None

        model = getattr(self.sam3_predictor, "model", None)
        if model is None:
            return None

        # Try common attribute names
        candidates = []
        for name in ["sam", "model", "net"]:
            m = getattr(model, name, None)
            if m is not None:
                candidates.append(m)
        candidates.append(model)

        # Try direct fields
        for m in candidates:
            for attr in ["image_encoder", "img_encoder", "encoder", "backbone"]:
                enc = getattr(m, attr, None)
                if enc is not None:
                    return enc

        # Try nested .model.something
        inner = getattr(model, "model", None)
        if inner is not None:
            for attr in ["image_encoder", "img_encoder", "encoder", "backbone"]:
                enc = getattr(inner, attr, None)
                if enc is not None:
                    return enc

        return None

    def _register_embedding_hook(self):
        """
        Registers a forward hook on the image encoder to capture dense embedding.
        Returns a removable hook handle, or None if failed.
        """
        if self.sam3_predictor is None or not TORCH_AVAILABLE:
            return None

        enc = self._try_get_image_encoder_module()
        if enc is None:
            self._log("Could not locate SAM3 image encoder module (for embeddings). Will fallback.")
            return None

        self._last_image_embedding = None

        def _hook(_module, _inputs, output):
            # output could be tensor (B,C,h,w) or dict; handle common cases
            try:
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    out = output[0]
                else:
                    out = output
                if hasattr(out, "detach"):
                    t = out.detach()
                    # expected (B,C,h,w)
                    if t.ndim == 4:
                        t = t[0]
                    self._last_image_embedding = t.float().cpu().numpy()
            except Exception:
                self._last_image_embedding = None

        try:
            handle = enc.register_forward_hook(_hook)
            self._log("Registered SAM3 embedding hook.")
            return handle
        except Exception as e:
            self._log(f"Failed to register embedding hook: {e}")
            return None

    # ----------------- Masks / boxes -----------------
    @staticmethod
    def _bbox_from_mask(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    @staticmethod
    def _union_boxes(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]):
        if a is None:
            return b
        if b is None:
            return a
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

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
                # fallback largest mask
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = 1.0

            best_mask = (masks[best_i].detach().cpu().numpy() > 0.5).astype(np.uint8)

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

            return best_mask.astype(np.uint8), best_box, float(best_score)
        except Exception as e:
            self._log(f"SAM3 error for prompt={prompt!r}: {e}")
            return None, None, 0.0

    # ----------------- Thresholding -----------------
    def _compute_dynamic_prob_threshold_otsu(self, probs: np.ndarray) -> float:
        p = probs[np.isfinite(probs)]
        if p.size < self.min_valid_segments_for_dynamic:
            return float(self.prob_threshold_fallback)

        p255 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
        if int(p255.max()) - int(p255.min()) < 5:
            return float(self.prob_threshold_fallback)

        t, _ = cv2.threshold(p255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
        thr = float(np.clip(thr + self.otsu_bias, 0.25, 0.85))
        return thr

    # ----------------- Embedding pooling -----------------
    @staticmethod
    def _resize_labels_to_embedding(seg_labels: np.ndarray, emb_hw: Tuple[int, int]) -> np.ndarray:
        # nearest neighbor resize for labels
        h, w = emb_hw
        return cv2.resize(seg_labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def _segment_embeddings_from_dense(
        self, dense_emb: np.ndarray, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        dense_emb: (C, h, w)
        segments:  (H, W) labels 0..K-1 (ROI resolution)
        Returns:
          seg_emb: (K, C) mean pooled embedding per segment
          seg_ids: (K,) segment ids
        """
        C, h, w = dense_emb.shape
        seg_small = self._resize_labels_to_embedding(segments, (h, w))
        seg_ids = np.unique(seg_small)
        K = len(seg_ids)

        seg_emb = np.zeros((K, C), dtype=np.float32)
        counts = np.zeros((K,), dtype=np.float32) + 1e-6

        # vectorized accumulation
        flat_ids = seg_small.ravel()
        flat_emb = dense_emb.reshape(C, -1).T  # (h*w, C)

        # map seg ids to 0..K-1
        id_to_k = {int(sid): i for i, sid in enumerate(seg_ids)}
        k_idx = np.array([id_to_k[int(s)] for s in flat_ids], dtype=np.int32)

        # sum embeddings per segment
        for c in range(C):
            seg_emb[:, c] = np.bincount(k_idx, weights=flat_emb[:, c].astype(np.float32), minlength=K)
        counts[:] = np.bincount(k_idx, minlength=K).astype(np.float32) + 1e-6
        seg_emb /= counts[:, None]

        return seg_emb, seg_ids.astype(np.int32)

    def _overlap_fraction_per_segment(self, segments: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        segments: (H,W) ROI labels
        mask:     (H,W) 0/1 mask
        returns overlap fraction for each segment id in np.unique(segments) order
        """
        seg_ids = np.unique(segments)
        flat = segments.ravel().astype(np.int32)
        m = (mask.ravel() > 0).astype(np.float32)

        max_sid = int(flat.max()) if flat.size else 0
        counts = np.bincount(flat, minlength=max_sid + 1).astype(np.float32) + 1e-6
        hits = np.bincount(flat, weights=m, minlength=max_sid + 1).astype(np.float32)

        frac = hits / counts
        return frac[seg_ids.astype(int)]

    # ----------------- Main analyze -----------------
    def analyze(self, image_path: str, interactive: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        H, W = original.shape[:2]
        timings["load"] = time.time() - t0

        # Hook for embeddings (optional)
        hook_handle = None
        if interactive and self.sam3_predictor is not None and TORCH_AVAILABLE:
            hook_handle = self._register_embedding_hook()

        # SAM3 prompts
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log("SAM3: running prompts for clean and rust ...")
            mask_clean_full, box_clean, score_clean = self._sam3_best_detection_for_prompt(image_path, self.prompt_clean)
            mask_rust_full, box_rust, score_rust = self._sam3_best_detection_for_prompt(image_path, self.prompt_rust)
        else:
            mask_clean_full = np.ones((H, W), dtype=np.uint8)
            mask_rust_full = np.zeros((H, W), dtype=np.uint8)
            box_clean = (0, 0, W, H)
            box_rust = (0, 0, W, H)
            score_clean, score_rust = 0.0, 0.0
        timings["sam3"] = time.time() - t0

        # remove hook (avoid leaks)
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass

        # prompt probs (if confidences exist)
        p_clean, p_rust = (0.5, 0.5)
        if interactive and self.sam3_predictor is not None:
            p_clean, p_rust = softmax2(float(score_clean), float(score_rust), temp=1.0)

        # ROI union
        roi = self._union_boxes(box_clean, box_rust)
        if roi is None:
            roi = (0, 0, W, H)

        x1, y1, x2, y2 = roi
        pad = self.roi_pad
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        crop = original[y1p:y2p, x1p:x2p].copy()
        clean_crop = None if mask_clean_full is None else mask_clean_full[y1p:y2p, x1p:x2p].astype(np.uint8)
        rust_crop = None if mask_rust_full is None else mask_rust_full[y1p:y2p, x1p:x2p].astype(np.uint8)

        # SLIC segments on ROI
        t0 = time.time()
        segments = slic(
            img_as_float(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            n_segments=self.n_segments,
            compactness=20,
            sigma=1,
            start_label=0,
            channel_axis=2,
        )
        timings["slic"] = time.time() - t0

        # Get dense embedding captured from SAM3
        dense_emb = self._last_image_embedding  # (C,h,w) or None
        using_embeddings = dense_emb is not None and isinstance(dense_emb, np.ndarray) and dense_emb.ndim == 3

        if not using_embeddings:
            # Fallback: overlap-based probability only (still gives probabilities)
            self._log("No SAM3 dense embedding found. Falling back to overlap+prompt probability prior.")
            seg_ids = np.unique(segments)
            K = len(seg_ids)
            rust_frac = self._overlap_fraction_per_segment(segments, rust_crop if rust_crop is not None else np.zeros_like(segments))
            clean_frac = self._overlap_fraction_per_segment(segments, clean_crop if clean_crop is not None else np.zeros_like(segments))
            ov = rust_frac / (rust_frac + clean_frac + 1e-6)
            prob_seg = (0.75 * ov + 0.25 * float(p_rust)).astype(np.float32)
        else:
            # Segment embeddings from dense SAM embedding
            seg_emb, seg_ids_small = self._segment_embeddings_from_dense(dense_emb.astype(np.float32), segments)

            # Overlap fractions in ROI-res space (segments)
            if rust_crop is None:
                rust_crop = np.zeros_like(segments, dtype=np.uint8)
            if clean_crop is None:
                clean_crop = np.zeros_like(segments, dtype=np.uint8)

            rust_frac = self._overlap_fraction_per_segment(segments, rust_crop)
            clean_frac = self._overlap_fraction_per_segment(segments, clean_crop)

            # Pick prototype segments
            rust_sel = rust_frac >= self.proto_overlap_thr
            clean_sel = clean_frac >= self.proto_overlap_thr

            # If too few, relax (best-effort)
            if rust_sel.sum() < 2:
                rust_sel = rust_frac >= max(0.2, self.proto_overlap_thr * 0.6)
            if clean_sel.sum() < 2:
                clean_sel = clean_frac >= max(0.2, self.proto_overlap_thr * 0.6)

            # Compute prototypes; if still empty, fallback to global mean
            if rust_sel.sum() > 0:
                proto_rust = seg_emb[rust_sel].mean(axis=0)
            else:
                proto_rust = seg_emb.mean(axis=0)

            if clean_sel.sum() > 0:
                proto_clean = seg_emb[clean_sel].mean(axis=0)
            else:
                proto_clean = seg_emb.mean(axis=0)

            # Segment probability via cosine similarity softmax
            K = seg_emb.shape[0]
            prob_seg = np.zeros((K,), dtype=np.float32)
            for i in range(K):
                sr = cosine_sim(seg_emb[i], proto_rust)
                sc = cosine_sim(seg_emb[i], proto_clean)
                # softmax -> P(rust)
                pc, pr = softmax2(sc, sr, temp=0.25)  # lower temp => sharper separation
                prob_seg[i] = float(pr)

        # Build prob map
        t0 = time.time()
        prob_map_crop = np.zeros(segments.shape, dtype=np.float32)
        for i, sid in enumerate(np.unique(segments)):
            prob_map_crop[segments == sid] = float(prob_seg[i])
        timings["prob_map"] = time.time() - t0

        # Threshold to mask
        if self.dynamic_prob_threshold and prob_seg.size >= self.min_valid_segments_for_dynamic:
            thr = self._compute_dynamic_prob_threshold_otsu(prob_seg)
        else:
            thr = float(self.prob_threshold_fallback)

        mask_crop = (prob_map_crop >= thr).astype(np.uint8)
        if self.ensure_one_positive and mask_crop.sum() == 0:
            # ensure best segment is positive
            sid_best = int(np.unique(segments)[int(np.argmax(prob_seg))])
            mask_crop[segments == sid_best] = 1

        # Morph cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)

        # Map back to full
        prob_map_full = np.zeros((H, W), dtype=np.float32)
        mask_full = np.zeros((H, W), dtype=np.uint8)
        prob_map_full[y1p:y2p, x1p:x2p] = prob_map_crop
        mask_full[y1p:y2p, x1p:x2p] = mask_crop

        rust_pct = float(mask_crop.mean()) * 100.0
        total_time = float(sum(timings.values()))
        self._log(
            f"P(rust|prompt)={p_rust:.3f}  |  embeddings={using_embeddings}  |  "
            f"ROI rust%={rust_pct:.2f}%  |  thr={thr:.2f}  |  time={total_time:.3f}s"
        )

        return dict(
            original=original,
            crop=crop,
            crop_coords=(x1p, y1p, x2p - x1p, y2p - y1p),
            roi_box_full=(x1p, y1p, x2p, y2p),
            segments=segments,
            # sam outputs
            mask_clean_full=mask_clean_full,
            mask_rust_full=mask_rust_full,
            box_clean_full=box_clean,
            box_rust_full=box_rust,
            score_clean=float(score_clean),
            score_rust=float(score_rust),
            p_clean=float(p_clean),
            p_rust=float(p_rust),
            # embedding info
            has_dense_embedding=bool(using_embeddings),
            dense_embedding_shape=None if dense_emb is None else tuple(dense_emb.shape),
            # probabilities
            prob_map_crop=prob_map_crop,
            prob_map_full=prob_map_full,
            threshold_used=float(thr),
            mask_crop=mask_crop,
            mask_full=mask_full,
            rust_percentage=float(rust_pct),
            timings=timings,
        )

    # ----------------- Visualization -----------------
    def visualize(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]
        segments = results["segments"]
        prob_crop = results["prob_map_crop"]
        prob_full = results["prob_map_full"]
        mask_crop = results["mask_crop"]
        mask_full = results["mask_full"]

        box_clean = results["box_clean_full"]
        box_rust = results["box_rust_full"]
        roi_box = results["roi_box_full"]

        p_clean = float(results["p_clean"])
        p_rust = float(results["p_rust"])
        thr = float(results["threshold_used"])
        rust_pct = float(results["rust_percentage"])
        emb_ok = bool(results["has_dense_embedding"])
        emb_shape = results["dense_embedding_shape"]

        # stage1 input
        stage1 = original.copy()

        # stage2 boxes
        stage2 = original.copy()
        if box_clean is not None:
            x1, y1, x2, y2 = box_clean
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(stage2, "clean", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if box_rust is not None:
            x1, y1, x2, y2 = box_rust
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(stage2, "rust", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(stage2, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(stage2, "ROI", (rx1, max(0, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # stage3 superpixels
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        spx = mark_boundaries(crop_rgb, segments, color=(1, 1, 0), mode="thick")
        stage3 = cv2.cvtColor((spx * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # stage4 prob heatmap roi
        stage4 = overlay_heatmap_bgr(crop.copy(), prob_crop, alpha=0.50)

        # stage5 mask overlay roi
        stage5 = crop.copy()
        ov = stage5.copy()
        ov[mask_crop > 0] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.60, ov, 0.40, 0)

        # stage6 full prob heatmap overlay
        stage6 = overlay_heatmap_bgr(original.copy(), prob_full, alpha=0.45)

        # stage7 final mask overlay
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
            "Rust Probabilities from SAM3 Embeddings (Two Prompts)\n"
            f"P(rust|prompt)={p_rust:.3f}, P(clean|prompt)={p_clean:.3f} | "
            f"embeddings={emb_ok} {emb_shape} | ROI rust%={rust_pct:.1f}% | thr={thr:.2f}",
            fontsize=12,
        )

        imgs = [stage1, stage2, crop, stage3, stage4, stage5, stage6, stage7]
        titles = [
            "1) Input",
            "2) Prompt boxes + ROI",
            "3) ROI crop",
            "4) SLIC boundaries (ROI)",
            "5) P(rust) heatmap (ROI)",
            "6) Mask (ROI)",
            "7) P(rust) heatmap (full)",
            "8) Final mask overlay (full)",
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
            self._log(f"Saved: {save_path}")

        return fig


# ----------------------------- CLI -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM3 embeddings -> segment prototypes -> rust probabilities")
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3; 0=disable SAM3.")
    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt")
    p.add_argument("--roi_pad", type=int, default=0)
    p.add_argument("--n_segments", type=int, default=8000)
    p.add_argument("--verbose", type=int, default=1)

    p.add_argument("--prompt_clean", type=str, default="clean shiny metal")
    p.add_argument("--prompt_rust", type=str, default="rusty metal")
    p.add_argument("--proto_overlap_thr", type=float, default=0.55)

    p.add_argument("--prob_threshold_fallback", type=float, default=0.55)
    p.add_argument("--dynamic_prob_threshold", type=int, default=1)
    p.add_argument("--min_valid_segments_for_dynamic", type=int, default=20)
    p.add_argument("--otsu_bias", type=float, default=-0.02)
    p.add_argument("--ensure_one_positive", type=int, default=1)

    p.add_argument("--res_dir", type=str, default="SAM3EmbeddingsProb")
    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    det = SAM3EmbeddingRustDetector(
        n_segments=args.n_segments,
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        roi_pad=int(args.roi_pad),
        prompt_clean=str(args.prompt_clean),
        prompt_rust=str(args.prompt_rust),
        proto_overlap_thr=float(args.proto_overlap_thr),
        prob_threshold_fallback=float(args.prob_threshold_fallback),
        dynamic_prob_threshold=bool(args.dynamic_prob_threshold),
        min_valid_segments_for_dynamic=int(args.min_valid_segments_for_dynamic),
        otsu_bias=float(args.otsu_bias),
        ensure_one_positive=bool(args.ensure_one_positive),
    )

    results = det.analyze(image_path, interactive=bool(args.interactive))
    out = f"results/{args.res_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_sam3embed_prob.png"
    det.visualize(results, save_path=out)


if __name__ == "__main__":
    main()