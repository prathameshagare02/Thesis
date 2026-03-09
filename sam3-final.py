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


def pick_image_file(title: str = "Select an image") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        root.destroy()
        if not path:
            raise ValueError("No file selected.")
        return path
    except ImportError:
        raise RuntimeError("tkinter not available")


class RustDetector:
    """
    Simple SAM3-based rust detection.
    
    Pipeline:
    1. SAM3 text prompts → per-pixel evidence maps
    2. P(rust) = rust_ev / (rust_ev + clean_ev + ε)
    3. rust_mask = rust_ev >= threshold
    4. rust_% = rust_pixels / total_pixels × 100
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        clean_prompt: str = "clean shiny metal",
        rust_prompts: Optional[List[str]] = None,
        evidence_threshold: float = 0.4,
    ):
        self.verbose = verbose
        self.sam_checkpoint = sam_checkpoint
        self.prompt_clean = clean_prompt
        self.evidence_threshold = evidence_threshold
        
        self.rust_prompts = rust_prompts or [
            "rust",
            "rusted area", 
            "brown stain",
            "oxidation",
            "corrosion",
        ]

        # Device detection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.sam3_predictor = None
        
        if verbose:
            print("=" * 50)
            print("RustDetector - Pipeline")
            print("=" * 50)
            print(f"Device: {self.device}")
            print(f"Prompts:")
            print(f"  Clean: '{self.prompt_clean}'")
            print(f"  Rust: {self.rust_prompts}")
            print(f"Evidence Threshold: {self.evidence_threshold}")
            print("=" * 50)

        self._load_sam3()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _load_sam3(self):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            self._log("SAM3 not available. Install: pip install -U ultralytics")
            return

        if not os.path.exists(self.sam_checkpoint):
            self._log(f"Checkpoint not found: {self.sam_checkpoint}")
            return

        try:
            self._log(f"Loading SAM3...")
            self.sam3_predictor = SAM3SemanticPredictor(overrides={
                "task": "segment", "mode": "predict",
                "model": self.sam_checkpoint, "device": self.device,
                "verbose": False, "save": False,
            })
            self.sam3_predictor.setup_model()
            self._log("SAM3 loaded.")
        except Exception as e:
            self._log(f"Failed: {e}")

    def _get_evidence(self, image_path: str, prompt: str, hw: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """Get per-pixel evidence map for a text prompt."""
        h, w = hw
        if self.sam3_predictor is None:
            return np.zeros((h, w), np.float32), 0

        try:
            self.sam3_predictor.set_image(image_path)
            results = self.sam3_predictor(text=[prompt])
            
            if not results or results[0].masks is None:
                return np.zeros((h, w), np.float32), 0

            masks = results[0].masks.data
            n = len(masks)
            
            # Get confidences
            conf = None
            if results[0].boxes is not None and results[0].boxes.conf is not None:
                conf = results[0].boxes.conf.cpu().numpy()

            # Evidence = max(mask_i × conf_i)
            evidence = np.zeros((h, w), np.float32)
            for i in range(n):
                m = masks[i].cpu().numpy().astype(np.float32)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h))
                c = float(conf[i]) if conf is not None and i < len(conf) else 1.0
                evidence = np.maximum(evidence, m * c)

            return np.clip(evidence, 0, 1), n
        except Exception as e:
            self._log(f"Error for '{prompt}': {e}")
            return np.zeros((h, w), np.float32), 0

    def analyze(self, image_path: str) -> Dict:
        """Run analysis pipeline."""
        self._log(f"Analyzing: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        h, w = img.shape[:2]

        # Get evidence maps
        self._log(f"Clean prompt: '{self.prompt_clean}'")
        clean_ev, n1 = self._get_evidence(image_path, self.prompt_clean, (h, w))
        self._log(f"  {n1} detections, max={clean_ev.max():.3f}")

        # Rust (multiple prompts, max pooled)
        rust_ev = np.zeros((h, w), np.float32)
        total_n = 0
        for rp in self.rust_prompts:
            self._log(f"Rust prompt: '{rp}'")
            ev, n = self._get_evidence(image_path, rp, (h, w))
            rust_ev = np.maximum(rust_ev, ev)
            total_n += n
            self._log(f"  {n} detections, max={ev.max():.3f}")
        self._log(f"Combined rust: max={rust_ev.max():.3f}")

        # Compute probability: P(rust) = rust / (rust + clean + ε)
        total = rust_ev + clean_ev + 1e-6
        rust_prob = rust_ev / total
        clean_prob = clean_ev / total

        # Binary mask (threshold on rust evidence)
        rust_mask = (rust_ev >= self.evidence_threshold).astype(np.uint8)
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)

        # Rust percentage
        rust_pixels = int(rust_mask.sum())
        total_pixels = h * w
        rust_pct = 100.0 * rust_pixels / total_pixels

        self._log(f"")
        self._log(f"{'='*40}")
        self._log(f"RUST PERCENTAGE: {rust_pct:.2f}%")
        self._log(f"Rust pixels: {rust_pixels:,} / {total_pixels:,}")
        self._log(f"{'='*40}")

        return {
            "original": img,
            "clean_evidence": clean_ev,
            "rust_evidence": rust_ev,
            "rust_probability": rust_prob,
            "rust_mask": rust_mask,
            "rust_percentage": rust_pct,
            "rust_pixels": rust_pixels,
            "total_pixels": total_pixels,
        }

    def visualize(self, results: Dict, save_path: Optional[str] = None, show: bool = True):
        """Create visualization."""
        img = results["original"]
        clean_ev = results["clean_evidence"]
        rust_ev = results["rust_evidence"]
        rust_prob = results["rust_probability"]
        rust_mask = results["rust_mask"]
        rust_pct = results["rust_percentage"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Rust Detection | RUST: {rust_pct:.1f}% | Evidence Threshold: {self.evidence_threshold}",
            fontsize=14, fontweight='bold'
        )

        # Row 1: Input + Evidence maps
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1) Input")

        # Clean (green overlay)
        overlay = img.copy()
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + (clean_ev * 100).astype(np.uint8), 0, 255)
        axes[0, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"2) Clean Evidence (max={clean_ev.max():.2f})")

        # Rust (red overlay)
        overlay = img.copy()
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + (rust_ev * 100).astype(np.uint8), 0, 255)
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"3) Rust Evidence (max={rust_ev.max():.2f})")

        # Row 2: Probability + Mask + Final
        im = axes[1, 0].imshow(rust_prob, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title(f"4) P(rust) (mean={rust_prob.mean():.3f})")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

        # Mask overlay
        overlay = img.copy()
        overlay[rust_mask == 1] = [0, 0, 255]
        axes[1, 1].imshow(cv2.cvtColor(cv2.addWeighted(img, 0.5, overlay, 0.5, 0), cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5) Rust Mask Overlay")

        # Final
        final = img.copy()
        final[rust_mask == 1] = [0, 0, 255]
        cv2.putText(final, f"Rust: {rust_pct:.1f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        axes[1, 2].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f"6) RESULT: {rust_pct:.1f}% Rust", fontweight='bold')

        for ax in axes.ravel():
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=300)
            self._log(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def main():
    parser = argparse.ArgumentParser(description="SAM3 Rust Detection")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--sam_checkpoint", type=str, default="sam3.pt")
    parser.add_argument("--prompt_clean", type=str, default="clean shiny metal")
    parser.add_argument("--rust_prompts", type=str, default="",
                        help="Comma-separated rust prompts")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Evidence threshold for rust detection")
    parser.add_argument("--res_dir", type=str, default="final")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    image_path = args.image if args.image else pick_image_file()
    if not os.path.exists(image_path):
        raise SystemExit(f"Not found: {image_path}")

    rust_prompts = None
    if args.rust_prompts:
        rust_prompts = [p.strip() for p in args.rust_prompts.split(',') if p.strip()]

    detector = RustDetector(
        verbose=bool(args.verbose),
        sam_checkpoint=args.sam_checkpoint,
        clean_prompt=args.prompt_clean,
        rust_prompts=rust_prompts,
        evidence_threshold=args.threshold,
    )

    results = detector.analyze(image_path)

    # Save results
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("results", args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    img_out = os.path.join(out_dir, f"{base}_result.png")
    detector.visualize(results, save_path=img_out, show=not args.no_show)

    print(f"\nResults: {img_out}")
    print(f"\n{'='*40}")
    print(f"RUST PERCENTAGE: {results['rust_percentage']:.2f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
