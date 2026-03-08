"""
Extract and visualize per-pixel rust probability from SAM3.

Outputs:
  - rust_probability.png: Heatmap visualization
  - rust_probability.npz: Raw probability map (H, W) with values [0, 1]
"""

from __future__ import annotations
import os
import tempfile
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_sam3_predictor(checkpoint: str = "sam3.pt"):
    """Load SAM3 predictor."""
    from ultralytics.models.sam import SAM3SemanticPredictor
    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    
    # Detect device
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Loading SAM3 on device: {device}")
    
    overrides = dict(
        task="segment",
        mode="predict", 
        model=checkpoint,
        verbose=False,
        device=device,
        half=(device == "cuda"),
        save=False,
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.setup_model()
    return predictor


def run_sam3(predictor, image_source, prompt: str):
    """Run SAM3 with a text prompt."""
    tmp_path = None
    try:
        if isinstance(image_source, np.ndarray):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            cv2.imwrite(tmp_path, image_source)
            src = tmp_path
        else:
            src = image_source
        
        predictor.set_image(src)
        return predictor(text=[prompt])
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except:
                pass


def build_evidence_map(results, expected_hw):
    """
    Build per-pixel evidence map from all detections.
    evidence[pixel] = max(mask_i[pixel] * conf_i) over all detections
    """
    h, w = expected_hw
    evidence = np.zeros((h, w), dtype=np.float32)
    
    if not results or len(results) == 0:
        return evidence
    
    r0 = results[0]
    if r0.masks is None or r0.masks.data is None:
        return evidence
    
    masks = r0.masks.data
    n = len(masks)
    
    # Get confidence scores
    conf = None
    if r0.boxes is not None and hasattr(r0.boxes, 'conf') and r0.boxes.conf is not None:
        conf = r0.boxes.conf.detach().cpu().numpy()
    
    for i in range(n):
        mask = masks[i].detach().cpu().numpy().astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)
        
        # Resize if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Get confidence (default 1.0 if not available)
        c = float(conf[i]) if (conf is not None and i < len(conf)) else 1.0
        
        # Max pooling over detections
        evidence = np.maximum(evidence, mask * c)
    
    return np.clip(evidence, 0.0, 1.0)


def get_rust_probability(
    image_path: str,
    checkpoint: str = "sam3.pt",
    prompt_rust: str = "rusty metal",
    prompt_clean: str = "clean shiny metal",
    output_dir: str = "results",
):
    """
    Compute per-pixel rust probability.
    
    Returns:
        rust_prob: ndarray (H, W) with values in [0, 1]
            - 1.0 = definitely rust
            - 0.0 = definitely clean/not rust
            - 0.5 = uncertain
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load: {image_path}")
    h, w = image.shape[:2]
    print(f"Image: {image_path} ({w}x{h})")
    
    # Load SAM3
    predictor = load_sam3_predictor(checkpoint)
    
    # Get rust evidence
    print(f"Running SAM3 with prompt: '{prompt_rust}'")
    results_rust = run_sam3(predictor, image_path, prompt_rust)
    rust_evidence = build_evidence_map(results_rust, (h, w))
    print(f"  Rust evidence: min={rust_evidence.min():.4f}, max={rust_evidence.max():.4f}, mean={rust_evidence.mean():.4f}")
    
    # Get clean evidence
    print(f"Running SAM3 with prompt: '{prompt_clean}'")
    results_clean = run_sam3(predictor, image_path, prompt_clean)
    clean_evidence = build_evidence_map(results_clean, (h, w))
    print(f"  Clean evidence: min={clean_evidence.min():.4f}, max={clean_evidence.max():.4f}, mean={clean_evidence.mean():.4f}")
    
    # =========================================================================
    # Per-pixel rust probability: P(rust | rust_evidence, clean_evidence)
    # =========================================================================
    
    # Method 1: Ratio (normalized)
    # P(rust) = rust_ev / (rust_ev + clean_ev)
    rust_prob_ratio = rust_evidence / (rust_evidence + clean_evidence + 1e-6)
    
    # Method 2: Raw rust evidence (if you only care about rust, not relative to clean)
    rust_prob_raw = rust_evidence
    
    # Method 3: Difference (signed, centered at 0.5)
    # 0.5 + 0.5*(rust - clean) maps to [0, 1] where 0.5 = equal evidence
    rust_prob_diff = np.clip(0.5 + 0.5 * (rust_evidence - clean_evidence), 0.0, 1.0)
    
    # Choose the primary method (ratio is most interpretable as probability)
    rust_prob = rust_prob_ratio
    
    print(f"\nPer-pixel rust probability:")
    print(f"  Min:  {rust_prob.min():.4f}")
    print(f"  Max:  {rust_prob.max():.4f}")
    print(f"  Mean: {rust_prob.mean():.4f}")
    print(f"  Pixels > 0.5 (more rust than clean): {(rust_prob > 0.5).sum()} ({100*(rust_prob > 0.5).mean():.2f}%)")
    print(f"  Pixels > 0.7 (likely rust): {(rust_prob > 0.7).sum()} ({100*(rust_prob > 0.7).mean():.2f}%)")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Evidence maps
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")
    
    im1 = axes[0, 1].imshow(rust_evidence, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title(f"Rust Evidence\n(prompt: '{prompt_rust}')")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(clean_evidence, cmap='Greens', vmin=0, vmax=1)
    axes[0, 2].set_title(f"Clean Evidence\n(prompt: '{prompt_clean}')")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Probability maps
    im3 = axes[1, 0].imshow(rust_prob_ratio, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title("Rust Probability (Ratio)\nP(rust) = rust/(rust+clean)")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(rust_prob_diff, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 1].set_title("Rust Probability (Diff)\n0.5 + 0.5*(rust-clean)")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Thresholded binary mask
    threshold = 0.5
    binary_mask = (rust_prob > threshold).astype(np.uint8)
    overlay = image.copy()
    overlay[binary_mask == 1] = [0, 0, 255]  # Red overlay
    overlay = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"Binary Mask (threshold={threshold})\nRust pixels marked red")
    axes[1, 2].axis("off")
    
    plt.suptitle(f"Per-Pixel Rust Probability: {base}", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"{base}_rust_probability.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved visualization: {fig_path}")
    
    # Save raw data
    npz_path = os.path.join(output_dir, f"{base}_rust_probability.npz")
    np.savez_compressed(
        npz_path,
        rust_probability=rust_prob.astype(np.float32),          # Primary output [0,1]
        rust_evidence=rust_evidence.astype(np.float32),         # Raw rust evidence
        clean_evidence=clean_evidence.astype(np.float32),       # Raw clean evidence
        rust_prob_ratio=rust_prob_ratio.astype(np.float32),     # Method 1
        rust_prob_diff=rust_prob_diff.astype(np.float32),       # Method 2
    )
    print(f"Saved raw data: {npz_path}")
    print(f"  Load with: data = np.load('{npz_path}'); rust_prob = data['rust_probability']")
    
    plt.show()
    
    return rust_prob, rust_evidence, clean_evidence


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    import argparse
    
    def pick_image_file():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
            )
            root.destroy()
            if not path:
                raise ValueError("No file selected.")
            return path
        except ImportError:
            raise RuntimeError("tkinter not available. Use --image argument.")
    
    parser = argparse.ArgumentParser(description="Get per-pixel rust probability from SAM3")
    parser.add_argument("--image", type=str, default="", help="Path to image")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM3 checkpoint")
    parser.add_argument("--prompt_rust", type=str, default="rusty metal", help="Prompt for rust detection")
    parser.add_argument("--prompt_clean", type=str, default="clean shiny metal", help="Prompt for clean metal")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    image_path = args.image if args.image else pick_image_file()
    
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")
    
    rust_prob, rust_ev, clean_ev = get_rust_probability(
        image_path,
        checkpoint=args.checkpoint,
        prompt_rust=args.prompt_rust,
        prompt_clean=args.prompt_clean,
        output_dir=args.output_dir,
    )
