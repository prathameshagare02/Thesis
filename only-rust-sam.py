from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Union
import argparse
import os
import time
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


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
class RustOnlyDetector:
    """
    Simplified rust detection using ONLY rust prompts with max pooling.
    
    No clean prompt, no background prompt - just rust detection.
    
    Pipeline:
    1) Run multiple SAM3 rust prompts (e.g., "rust", "rusted area", "brown stain", etc.)
    2) Combine all rust evidence maps via max pooling
    3) Threshold the combined evidence: is_rust = (rust_ev >= threshold)
    
    Pros:
    - Faster (fewer SAM3 calls)
    - Simpler logic
    - Direct rust detection
    
    Cons:
    - No context from clean metal
    - Potential false positives on non-metal areas
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        # Multiple rust prompts for different rust types
        rust_prompts: Optional[List[str]] = None,
        # Classification - simple threshold
        rust_threshold: float = 0.3,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = sam_checkpoint
        self.rust_threshold = float(rust_threshold)
        
        # Multiple rust prompts to catch light + heavy rust
        if rust_prompts:
            self.rust_prompts = list(rust_prompts)
        else:
            # Default: multiple prompts for different rust severities
            self.rust_prompts = [
                "rust",                    # General rust
                "rusted area",             # Heavy rust
                "brown stain",             # Light rust / discoloration
                "oxidation",               # Early stage rust
                "corrosion",               # Another term for rust
            ]

        # Detect available device
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        self.sam3_predictor = None
        self.original_image: Optional[np.ndarray] = None

        if self.verbose:
            self._print_backend_info()

        self.load_sam3_model()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("RustOnlyDetector initialized (rust-only, max pooling)")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  Torch device: {self.device}")
        print(f"  Rust prompts ({len(self.rust_prompts)}): {self.rust_prompts}")
        print(f"  Rust threshold: {self.rust_threshold}")

    def load_sam3_model(self):
        """Load SAM3SemanticPredictor from Ultralytics."""
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
            overrides = dict(
                task="segment",
                mode="predict",
                model=self.sam_checkpoint,
                verbose=self.verbose,
                device=self.device,
                half=True if self.device == "cuda" else False,
                save=False,
            )
            self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_predictor.setup_model()
            
            try:
                model_device = next(self.sam3_predictor.model.parameters()).device
                self._log(f"SAM3 model loaded on device: {model_device}")
            except Exception:
                self._log("SAM3 model loaded successfully.")
        except Exception as e:
            self._log(f"Failed to load SAM3 model: {e}")
            self.sam3_predictor = None

    @staticmethod
    def _safe_resize_mask(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        h, w = hw
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def _run_sam3_text_prompt(self, image_source: Union[str, np.ndarray], prompt: str):
        """Wrapper for SAM3 inference."""
        if self.sam3_predictor is None:
            return None

        tmp_path = None
        try:
            src = image_source
            if isinstance(image_source, np.ndarray):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = tmp.name
                tmp.close()
                cv2.imwrite(tmp_path, image_source)
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

    def _get_evidence_map(
        self,
        image_source: Union[str, np.ndarray],
        prompt: str,
        expected_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, int]:
        """
        Get per-pixel evidence map for a prompt.
        Evidence = max(conf_i * mask_i) over all detections.
        
        Returns:
            evidence_map: HxW float32 array
            num_detections: number of detections found
        """
        h, w = expected_hw
        zeros = np.zeros((h, w), dtype=np.float32)

        if self.sam3_predictor is None:
            return zeros, 0

        try:
            results = self._run_sam3_text_prompt(image_source, prompt)
            if not results:
                return zeros, 0

            r0 = results[0]
            if getattr(r0, "masks", None) is None or r0.masks is None or r0.masks.data is None:
                return zeros, 0
            
            masks = r0.masks.data
            n = int(len(masks))
            if n == 0:
                return zeros, 0

            # Get confidence scores
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

            # Combine all detections: evidence = max(mask_i * conf_i)
            evidence = np.zeros((h, w), dtype=np.float32)
            for i in range(n):
                mi = masks[i].detach().cpu().numpy().astype(np.float32)
                mi = np.clip(mi, 0.0, 1.0)
                if mi.shape[:2] != (h, w):
                    mi = self._safe_resize_mask(mi, (h, w))

                ci = float(conf[i]) if (conf is not None and i < len(conf)) else 1.0
                ci = float(np.clip(ci, 0.0, 1.0))

                evidence = np.maximum(evidence, mi * ci)

            return np.clip(evidence, 0.0, 1.0).astype(np.float32), n

        except Exception as e:
            self._log(f"SAM3 error for prompt={prompt!r}: {e}")
            return zeros, 0

    def analyze(self, image_path: str) -> Dict:
        """Run the rust-only analysis pipeline."""
        timings: Dict[str, float] = {}

        # Load image
        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        
        self.original_image = original.copy()
        h, w = original.shape[:2]
        timings["load"] = time.time() - t0

        # Run ALL rust prompts and combine evidence (max pooling)
        t0 = time.time()
        rust_ev = np.zeros((h, w), dtype=np.float32)
        total_rust_detections = 0
        per_prompt_evidence = {}  # Store individual prompt evidence for visualization
        
        for i, rust_prompt in enumerate(self.rust_prompts):
            self._log(f"SAM3 rust prompt {i+1}/{len(self.rust_prompts)}: {rust_prompt!r}")
            ev, n = self._get_evidence_map(image_path, rust_prompt, (h, w))
            per_prompt_evidence[rust_prompt] = ev.copy()
            rust_ev = np.maximum(rust_ev, ev)  # Max pooling: union of all rust detections
            total_rust_detections += n
            self._log(f"  → {n} detection(s), max_ev={ev.max():.3f}")
        
        timings["sam3_rust"] = time.time() - t0
        self._log(f"  → Combined rust: {total_rust_detections} total detection(s), max_ev={rust_ev.max():.3f}")

        # Simple threshold classification
        t0 = time.time()
        self._log(f"Thresholding at {self.rust_threshold}...")
        rust_mask = (rust_ev >= self.rust_threshold).astype(np.uint8)
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        timings["classify"] = time.time() - t0

        rust_pixels = int(rust_mask.sum())
        total_pixels = h * w
        rust_percentage = 100.0 * rust_pixels / total_pixels if total_pixels > 0 else 0.0

        total_time = sum(timings.values())
        self._log(
            f"Rust pixels: {rust_pixels}/{total_pixels} ({rust_percentage:.2f}%) | "
            f"Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            rust_evidence=rust_ev,
            per_prompt_evidence=per_prompt_evidence,
            rust_mask=rust_mask,
            rust_percentage=rust_percentage,
            total_detections=total_rust_detections,
            timings=timings,
        )

    def visualize_detection(
        self, 
        results: Dict, 
        save_path: Optional[str] = None, 
        show: bool = True
    ) -> plt.Figure:
        """Create visualization of detection stages."""
        original = results["original"]
        rust_ev = results["rust_evidence"]
        rust_mask = results["rust_mask"]
        rust_percentage = results["rust_percentage"]

        # Stage 1: Input image
        stage1 = original.copy()

        # Stage 2: Rust evidence (red overlay on original)
        stage2 = original.copy()
        overlay2 = np.zeros_like(original)
        overlay2[:, :, 2] = (rust_ev * 255).astype(np.uint8)  # red channel
        stage2 = cv2.addWeighted(stage2, 0.6, overlay2, 0.4, 0)

        # Stage 3: Final result with percentage (only rust pixels are red)
        stage3 = original.copy()
        stage3[rust_mask == 1] = [0, 0, 255]  # Only rust pixels turn red
        cv2.putText(
            stage3,
            f"Rust: {rust_percentage:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        # Create figure (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        fig.suptitle(
            f"Rust-Only Detection ({len(self.rust_prompts)} prompts, max pooling)\n"
            f"Threshold: {self.rust_threshold} | Rust coverage: {rust_percentage:.1f}%",
            fontsize=14,
        )

        # Row 1: Input + Evidence overlay
        axes[0, 0].imshow(cv2.cvtColor(stage1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1) Input image", fontsize=10)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"2) Rust evidence overlay ({len(self.rust_prompts)} prompts)", fontsize=10)
        axes[0, 1].axis("off")

        # Row 2: Heatmap + Final result
        im3 = axes[1, 0].imshow(rust_ev, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1, 0].set_title(f"3) Rust evidence heatmap (max={rust_ev.max():.2f})", fontsize=10)
        axes[1, 0].axis("off")
        cbar = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar.set_label('Evidence', fontsize=10)

        axes[1, 1].imshow(cv2.cvtColor(stage3, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"4) Final result ({rust_percentage:.1f}% rust)", fontsize=10, fontweight='bold')
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            self._log(f"Visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


# ----------------------------- CLI -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rust-Only Detection: Multiple SAM3 rust prompts with max pooling"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")

    # Prompts
    p.add_argument("--rust_prompts", type=str, default="",
                   help="Comma-separated rust prompts. Default: 'rust,rusted area,brown stain,oxidation,corrosion'")

    # Classification
    p.add_argument("--rust_threshold", type=float, default=0.3,
                   help="Minimum rust evidence to classify pixel as rust.")

    # Output
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--res_dir", type=str, default="rust_only")
    p.add_argument("--no_show", action="store_true", help="Don't show matplotlib window.")
    
    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    # Parse rust prompts
    rust_prompts = None
    if args.rust_prompts:
        rust_prompts = [p.strip() for p in args.rust_prompts.split(',') if p.strip()]
    
    detector = RustOnlyDetector(
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        rust_prompts=rust_prompts,
        rust_threshold=args.rust_threshold,
    )

    results = detector.analyze(image_path)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("results", args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    img_out = os.path.join(out_dir, f"{base}_rust_only.png")
    detector.visualize_detection(results, save_path=img_out, show=not args.no_show)

    print(f"\nResults saved to: {img_out}")
    print(f"Rust coverage: {results['rust_percentage']:.2f}%")
    print(f"Total detections: {results['total_detections']}")
    print(f"Time: {sum(results['timings'].values()):.2f}s")


if __name__ == "__main__":
    main()
