from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import argparse
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
class FastRustDetector:
    """
    SAM3-only dual-prompt rust detection (NO SLIC):

      1) Run SAM3 on full image with prompt_clean  = "clean shiny metal"
      2) Run SAM3 on full image with prompt_rust   = "rust of metal surface"
      3) Build per-pixel evidence maps from SAM3 outputs:
           evidence(prompt) = max_i(conf_i * mask_i)
      4) Compare rust vs clean evidence to create final rust mask
      5) Morphological cleanup + final overlay

    Notes:
      - No SLIC
      - No per-segment features/scoring
      - Direct SAM3 evidence fusion
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        prompt_clean: str = "clean shiny metal",
        prompt_rust: str = "rust of metal surface",
        min_rust_evidence: float = 0.18,
        delta_margin: float = 0.03,
        rust_ratio_threshold: float = 0.55,
        ensure_one_positive: bool = False,
        morph_kernel: int = 3,
        morph_open_iters: int = 1,
        morph_close_iters: int = 1,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = str(sam_checkpoint)

        # Dual prompts (the "2 embeddings" / text prompts)
        self.prompt_clean = str(prompt_clean)
        self.prompt_rust = str(prompt_rust)

        # Fusion thresholds
        self.min_rust_evidence = float(min_rust_evidence)
        self.delta_margin = float(delta_margin)
        self.rust_ratio_threshold = float(rust_ratio_threshold)
        self.ensure_one_positive = bool(ensure_one_positive)

        # Morphology
        self.morph_kernel = max(1, int(morph_kernel))
        self.morph_open_iters = max(0, int(morph_open_iters))
        self.morph_close_iters = max(0, int(morph_close_iters))

        # SAM3
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
        print("FastRustDetector initialized:")
        print("  Mode: SAM3-only dual-prompt rust detection (NO SLIC)")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  Prompt clean: {self.prompt_clean!r}")
        print(f"  Prompt rust:  {self.prompt_rust!r}")
        print(f"  min_rust_evidence: {self.min_rust_evidence:.3f}")
        print(f"  delta_margin: {self.delta_margin:.3f}")
        print(f"  rust_ratio_threshold: {self.rust_ratio_threshold:.3f}")
        print(f"  ensure_one_positive: {self.ensure_one_positive}")
        print(f"  morphology: kernel={self.morph_kernel}, open={self.morph_open_iters}, close={self.morph_close_iters}")

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

            # Safer default: let Ultralytics choose device if not specified.
            # FP16 can break on CPU/MPS depending on ops, so keep False here.
            overrides["half"] = False

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
    def _safe_resize_mask(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        h, w = hw
        if mask.shape[:2] == (h, w):
            return mask.astype(np.float32, copy=False)
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_morphological_operations(self, mask: np.ndarray | None):
        if mask is None:
            return None
        if mask.size == 0:
            return mask
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        k = max(1, int(self.morph_kernel))
        kernel = np.ones((k, k), np.uint8)

        if self.morph_open_iters > 0:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iters)
        if self.morph_close_iters > 0:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iters)

        return (mask_uint8 > 127).astype(np.uint8)

    # ---- SAM3 prompt evidence ----
    def _run_sam3_text_prompt(self, image_path: str, prompt: str):
        if self.sam3_predictor is None:
            return None
        self.sam3_predictor.set_image(image_path)
        return self.sam3_predictor(text=[prompt])

    def _sam3_prompt_evidence_map(
        self,
        image_path: str,
        prompt: str,
        expected_hw: Tuple[int, int],
    ) -> Dict:
        """
        Build per-pixel evidence map from SAM3 prompt detections:
          evidence = max_i(conf_i * mask_i)

        Returns dict:
          evidence_map: float32 HxW in [0,1]
          best_mask:    uint8 HxW (best detection mask)
          best_box:     (x1,y1,x2,y2) or None
          best_score:   float
          num_detections: int
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
            results = self._run_sam3_text_prompt(image_path, prompt)
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

            # Best mask
            best_mask_f = masks[best_i].detach().cpu().numpy().astype(np.float32)
            best_mask_f = np.clip(best_mask_f, 0.0, 1.0)
            if best_mask_f.shape[:2] != (h, w):
                best_mask_f = self._safe_resize_mask(best_mask_f, (h, w))

            best_mask_u8 = (best_mask_f > 0.5).astype(np.uint8)
            processed = self.apply_morphological_operations(best_mask_u8)
            if processed is not None:
                best_mask_u8 = processed

            # Best box
            best_box = None
            if xyxy is not None and xyxy.shape[0] > best_i:
                x1, y1, x2, y2 = xyxy[best_i]
                best_box = (int(x1), int(y1), int(x2), int(y2))
            if best_box is None:
                best_box = self._bbox_from_mask(best_mask_u8)

            return dict(
                evidence_map=np.clip(evidence, 0.0, 1.0).astype(np.float32),
                best_mask=best_mask_u8.astype(np.uint8),
                best_box=best_box,
                best_score=float(max(best_score, 0.0)),
                num_detections=n,
            )

        except Exception as e:
            self._log(f"SAM3 prompt evidence-map error for prompt={prompt!r}: {e}")
            return dict(
                evidence_map=zeros_f,
                best_mask=zeros_u,
                best_box=None,
                best_score=0.0,
                num_detections=0,
            )

    # ---- Fusion (clean vs rust evidence) ----
    def _fuse_clean_rust_evidence(
        self,
        clean_ev: np.ndarray,
        rust_ev: np.ndarray,
    ) -> Dict:
        """
        Fuse two SAM3 evidence maps into a final rust mask.

        Rules (pixel-wise):
          - rust evidence must exceed min_rust_evidence
          - rust must beat clean by delta_margin OR rust ratio must exceed threshold
        """
        clean_ev = np.clip(clean_ev.astype(np.float32), 0.0, 1.0)
        rust_ev = np.clip(rust_ev.astype(np.float32), 0.0, 1.0)

        delta = rust_ev - clean_ev
        ratio = rust_ev / (rust_ev + clean_ev + 1e-6)

        rust_mask = (
            (rust_ev >= self.min_rust_evidence)
            & (
                (delta >= self.delta_margin)
                | (ratio >= self.rust_ratio_threshold)
            )
        ).astype(np.uint8)

        rust_mask = self.apply_morphological_operations(rust_mask)
        if rust_mask is None:
            rust_mask = np.zeros_like(rust_ev, dtype=np.uint8)

        # Optional fallback: keep one pixel if user wants at least one positive
        if self.ensure_one_positive and int(rust_mask.sum()) == 0 and rust_ev.size > 0:
            rank = (1.8 * rust_ev - 1.2 * clean_ev + 0.5 * ratio).reshape(-1)
            idx = int(np.argmax(rank))
            rust_mask_flat = rust_mask.reshape(-1)
            rust_mask_flat[idx] = 1
            rust_mask = rust_mask_flat.reshape(rust_mask.shape)

        # Score map for visualization (0..1): green-dominant clean -> low, red-dominant rust -> high
        score_map = np.clip(0.5 + 0.5 * delta, 0.0, 1.0).astype(np.float32)

        rust_pixels = int(rust_mask.sum())
        total_pixels = int(rust_mask.size)
        rust_pct = (100.0 * rust_pixels / total_pixels) if total_pixels > 0 else 0.0

        return dict(
            rust_mask=rust_mask.astype(np.uint8),
            delta_map=delta.astype(np.float32),
            ratio_map=np.clip(ratio, 0.0, 1.0).astype(np.float32),
            score_map=score_map,
            rust_percentage=rust_pct,
        )

    # ---- Main analysis ----
    def analyze(self, image_path: str, interactive: bool = True) -> Dict:
        timings: Dict[str, float] = {}

        # 0) Load image
        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")

        self.original_image = original.copy()
        H, W = original.shape[:2]
        timings["load"] = time.time() - t0

        # If image huge, warn
        mega_pixels = (H * W) / 1e6
        if mega_pixels > 12.0:
            self._log(f"WARNING: Large image ({mega_pixels:.1f} MP). SAM3 may be slow.")

        # 1) SAM3 clean prompt
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log(f"SAM3 prompt #1 (clean): {self.prompt_clean!r}")
            clean_out = self._sam3_prompt_evidence_map(image_path, self.prompt_clean, expected_hw=(H, W))
        else:
            clean_out = dict(
                evidence_map=np.zeros((H, W), dtype=np.float32),
                best_mask=np.zeros((H, W), dtype=np.uint8),
                best_box=None,
                best_score=0.0,
                num_detections=0,
            )
        timings["sam3_clean"] = time.time() - t0

        # 2) SAM3 rust prompt
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log(f"SAM3 prompt #2 (rust): {self.prompt_rust!r}")
            rust_out = self._sam3_prompt_evidence_map(image_path, self.prompt_rust, expected_hw=(H, W))
        else:
            rust_out = dict(
                evidence_map=np.zeros((H, W), dtype=np.float32),
                best_mask=np.zeros((H, W), dtype=np.uint8),
                best_box=None,
                best_score=0.0,
                num_detections=0,
            )
        timings["sam3_rust"] = time.time() - t0

        # 3) Fuse evidence maps -> rust mask
        t0 = time.time()
        fused = self._fuse_clean_rust_evidence(clean_out["evidence_map"], rust_out["evidence_map"])
        timings["fuse"] = time.time() - t0

        total_time = float(sum(timings.values()))
        self._log(
            f"Clean score={clean_out['best_score']:.2f} ({clean_out['num_detections']} dets) | "
            f"Rust score={rust_out['best_score']:.2f} ({rust_out['num_detections']} dets) | "
            f"Rust coverage={fused['rust_percentage']:.2f}% | "
            f"Total time={total_time:.3f}s"
        )

        return dict(
            original=original,

            # SAM3 clean
            sam_clean_evidence=clean_out["evidence_map"],
            sam_clean_best_mask=clean_out["best_mask"],
            sam_clean_best_box=clean_out["best_box"],
            sam_clean_best_score=float(clean_out["best_score"]),
            sam_clean_num_dets=int(clean_out["num_detections"]),

            # SAM3 rust
            sam_rust_evidence=rust_out["evidence_map"],
            sam_rust_best_mask=rust_out["best_mask"],
            sam_rust_best_box=rust_out["best_box"],
            sam_rust_best_score=float(rust_out["best_score"]),
            sam_rust_num_dets=int(rust_out["num_detections"]),

            # Fusion / output
            full_mask=fused["rust_mask"],
            score_map=fused["score_map"],
            delta_map=fused["delta_map"],
            ratio_map=fused["ratio_map"],
            rust_percentage=float(fused["rust_percentage"]),
            timings=timings,
        )

    # ---- Visualization ----
    def visualize_detection(self, results: Dict, save_path: str | None = None) -> plt.Figure:
        original = results["original"]

        clean_ev = results["sam_clean_evidence"]
        rust_ev = results["sam_rust_evidence"]
        clean_best_mask = results["sam_clean_best_mask"]
        rust_best_mask = results["sam_rust_best_mask"]

        clean_box = results.get("sam_clean_best_box", None)
        rust_box = results.get("sam_rust_best_box", None)

        clean_score = float(results.get("sam_clean_best_score", 0.0))
        rust_score = float(results.get("sam_rust_best_score", 0.0))
        rust_percentage = float(results.get("rust_percentage", 0.0))

        full_mask = results["full_mask"]
        score_map = results["score_map"]

        # Stage 1: input image
        stage1 = original.copy()

        # Stage 2: clean evidence overlay (green)
        stage2 = original.copy()
        clean_vis = np.zeros_like(stage2, dtype=np.uint8)
        clean_vis[:, :, 1] = np.clip(clean_ev * 255.0, 0, 255).astype(np.uint8)
        stage2 = cv2.addWeighted(stage2, 0.65, clean_vis, 0.5, 0)
        if clean_box is not None:
            x1, y1, x2, y2 = clean_box
            cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                stage2,
                f"clean {clean_score:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Stage 3: rust evidence overlay (red)
        stage3 = original.copy()
        rust_vis = np.zeros_like(stage3, dtype=np.uint8)
        rust_vis[:, :, 2] = np.clip(rust_ev * 255.0, 0, 255).astype(np.uint8)
        stage3 = cv2.addWeighted(stage3, 0.65, rust_vis, 0.5, 0)
        if rust_box is not None:
            x1, y1, x2, y2 = rust_box
            cv2.rectangle(stage3, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                stage3,
                f"rust {rust_score:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # Stage 4: dual-evidence comparison map (clean=green, rust=red)
        stage4 = original.copy()
        cmp_vis = np.zeros_like(stage4, dtype=np.uint8)
        cmp_vis[:, :, 1] = np.clip(clean_ev * 255.0, 0, 255).astype(np.uint8)  # green clean
        cmp_vis[:, :, 2] = np.clip(rust_ev * 255.0, 0, 255).astype(np.uint8)   # red rust
        stage4 = cv2.addWeighted(stage4, 0.55, cmp_vis, 0.55, 0)

        # Stage 5: best masks (clean + rust) overlay for sanity
        stage5 = original.copy()
        tmp5 = stage5.copy()
        tmp5[clean_best_mask == 1] = [0, 255, 0]
        tmp5[rust_best_mask == 1] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.65, tmp5, 0.45, 0)

        # Stage 6: final rust overlay on full image
        stage6 = original.copy()
        rust_overlay = stage6.copy()
        rust_overlay[full_mask == 1] = [0, 0, 255]
        stage6 = cv2.addWeighted(stage6, 0.6, rust_overlay, 0.4, 0)

        # Optional: draw both boxes on final view
        if clean_box is not None:
            x1, y1, x2, y2 = clean_box
            cv2.rectangle(stage6, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if rust_box is not None:
            x1, y1, x2, y2 = rust_box
            cv2.rectangle(stage6, (x1, y1), (x2, y2), (255, 0, 0), 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Rust Detection Stages (SAM3 dual prompts only: clean vs rust, NO SLIC)\n"
            f"Clean score={clean_score:.2f} | Rust score={rust_score:.2f} | Rust coverage={rust_percentage:.1f}%",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6]
        titles = [
            "1) Input image",
            "2) SAM3 clean evidence (green)",
            "3) SAM3 rust evidence (red)",
            "4) Dual evidence overlay (clean=green, rust=red)",
            "5) Best masks (clean=green, rust=red)",
            "6) Final rust overlay (red)",
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
        description="Rust Detection: SAM3 dual prompts (clean shiny metal vs rust of metal surface), NO SLIC"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=run SAM3 prompts; 0=skip SAM3 (debug fallback).")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--prompt_clean", type=str, default="clean shiny metal", help='SAM3 clean-metal prompt.')
    p.add_argument("--prompt_rust", type=str, default="rust of metal surface", help='SAM3 rust prompt.')

    # Fusion thresholds
    p.add_argument("--min_rust_evidence", type=float, default=0.18, help="Minimum rust evidence to consider a rust pixel.")
    p.add_argument("--delta_margin", type=float, default=0.03, help="Require rust_evidence - clean_evidence >= this margin (or pass ratio gate).")
    p.add_argument("--rust_ratio_threshold", type=float, default=0.55, help="Require rust/(rust+clean) >= this ratio (alternative gate).")
    p.add_argument("--ensure_one_positive", type=int, default=0, help="Force at least one positive pixel if no rust pixel survives fusion.")

    # Morphology
    p.add_argument("--morph_kernel", type=int, default=3)
    p.add_argument("--morph_open_iters", type=int, default=1)
    p.add_argument("--morph_close_iters", type=int, default=1)

    p.add_argument("--verbose", type=int, default=1)
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
        prompt_clean=args.prompt_clean,
        prompt_rust=args.prompt_rust,
        min_rust_evidence=float(args.min_rust_evidence),
        delta_margin=float(args.delta_margin),
        rust_ratio_threshold=float(args.rust_ratio_threshold),
        ensure_one_positive=bool(args.ensure_one_positive),
        morph_kernel=int(args.morph_kernel),
        morph_open_iters=int(args.morph_open_iters),
        morph_close_iters=int(args.morph_close_iters),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))

    out = f"results/{args.res_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_stages.png"
    detector.visualize_detection(results, save_path=out)

    print("\nDone.")
    print("Pipeline: SAM3 clean prompt + SAM3 rust prompt -> evidence fusion -> final rust mask")
    print(f"Rust coverage: {results['rust_percentage']:.2f}%")
    print(f"Saved visualization: {out}")


if __name__ == "__main__":
    main()