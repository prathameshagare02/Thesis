"""
Debug script to inspect SAM3 outputs for Pass 1 and Pass 2.
This helps understand what SAM3 returns at each stage.
"""

from __future__ import annotations
import os
import tempfile
import cv2
import numpy as np
import torch


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def inspect_tensor(name: str, tensor, indent: int = 2):
    """Helper to inspect a tensor or numpy array."""
    prefix = " " * indent
    if tensor is None:
        print(f"{prefix}{name}: None")
        return
    
    if hasattr(tensor, 'detach'):  # PyTorch tensor
        arr = tensor.detach().cpu().numpy()
        print(f"{prefix}{name}: Tensor -> shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    elif isinstance(tensor, np.ndarray):
        print(f"{prefix}{name}: ndarray -> shape={tensor.shape}, dtype={tensor.dtype}, "
              f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
    else:
        print(f"{prefix}{name}: {type(tensor).__name__} = {tensor}")


def inspect_result_object(result, prefix: str = ""):
    """Recursively inspect a SAM3 result object."""
    print(f"{prefix}Result object type: {type(result).__name__}")
    
    # List all attributes
    attrs = [a for a in dir(result) if not a.startswith('_')]
    print(f"{prefix}Available attributes: {attrs}")
    
    for attr in attrs:
        try:
            val = getattr(result, attr)
            if callable(val):
                continue  # Skip methods
            
            print(f"\n{prefix}  .{attr}:")
            
            if val is None:
                print(f"{prefix}    None")
            elif hasattr(val, 'data'):  # Has .data attribute (like masks, boxes)
                print(f"{prefix}    Type: {type(val).__name__}")
                if hasattr(val, 'shape'):
                    print(f"{prefix}    Shape: {val.shape}")
                if hasattr(val.data, 'shape'):
                    data = val.data
                    if hasattr(data, 'detach'):
                        data = data.detach().cpu().numpy()
                    print(f"{prefix}    .data shape: {data.shape}, dtype: {data.dtype}")
                    if data.size > 0:
                        print(f"{prefix}    .data stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                
                # Check for xyxy (bounding boxes)
                if hasattr(val, 'xyxy'):
                    xyxy = val.xyxy
                    if hasattr(xyxy, 'detach'):
                        xyxy = xyxy.detach().cpu().numpy()
                    print(f"{prefix}    .xyxy shape: {xyxy.shape}")
                    if xyxy.size > 0:
                        print(f"{prefix}    .xyxy values:\n{xyxy}")
                
                # Check for conf (confidence)
                if hasattr(val, 'conf'):
                    conf = val.conf
                    if hasattr(conf, 'detach'):
                        conf = conf.detach().cpu().numpy()
                    print(f"{prefix}    .conf shape: {conf.shape}")
                    if conf.size > 0:
                        print(f"{prefix}    .conf values: {conf}")
                
                # Check for cls (class)
                if hasattr(val, 'cls'):
                    cls = val.cls
                    if hasattr(cls, 'detach'):
                        cls = cls.detach().cpu().numpy()
                    print(f"{prefix}    .cls shape: {cls.shape}")
                    if cls.size > 0:
                        print(f"{prefix}    .cls values: {cls}")
                        
            elif hasattr(val, 'shape'):  # Tensor or array
                inspect_tensor(attr, val, indent=len(prefix) + 4)
            elif isinstance(val, (list, tuple)):
                print(f"{prefix}    Type: {type(val).__name__}, len={len(val)}")
                if len(val) > 0 and len(val) <= 5:
                    for i, item in enumerate(val):
                        print(f"{prefix}      [{i}]: {type(item).__name__}")
            elif isinstance(val, dict):
                print(f"{prefix}    Type: dict, keys={list(val.keys())}")
            else:
                print(f"{prefix}    Type: {type(val).__name__}, value: {val}")
        except Exception as e:
            print(f"{prefix}    Error accessing {attr}: {e}")


def load_sam3_predictor(checkpoint: str = "sam3.pt", device: str = None):
    """Load SAM3 predictor."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError:
        raise RuntimeError("Failed to import SAM3SemanticPredictor. Install ultralytics: pip install -U ultralytics")
    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    
    # Detect device
    if device is None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Loading SAM3 with device: {device}")
    
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


def run_sam3_prompt(predictor, image_source, prompt: str):
    """Run SAM3 with a text prompt and return raw results."""
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
        results = predictor(text=[prompt])
        return results
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def debug_sam3_outputs(image_path: str, checkpoint: str = "sam3.pt"):
    """Main debug function to inspect SAM3 outputs."""
    
    # Load image
    print_separator("Loading Image")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"Image path: {image_path}")
    print(f"Image shape: {image.shape} (H, W, C)")
    print(f"Image dtype: {image.dtype}")
    
    # Load SAM3
    print_separator("Loading SAM3 Model")
    predictor = load_sam3_predictor(checkpoint)
    
    # =====================================================================
    # PASS 1: Metal detection (for ROI extraction)
    # =====================================================================
    print_separator("PASS 1: SAM3 with prompt='metal'")
    
    prompt_metal = "metal"
    print(f"Running SAM3 with prompt: '{prompt_metal}'")
    
    results_pass1 = run_sam3_prompt(predictor, image_path, prompt_metal)
    
    print(f"\nRaw results type: {type(results_pass1)}")
    print(f"Number of results: {len(results_pass1)}")
    
    if results_pass1 and len(results_pass1) > 0:
        r0 = results_pass1[0]
        print("\n--- Inspecting results[0] ---")
        inspect_result_object(r0, prefix="  ")
        
        # Extract key information
        print("\n--- Key Information Extracted ---")
        
        if hasattr(r0, 'masks') and r0.masks is not None:
            masks = r0.masks.data.detach().cpu().numpy()
            print(f"  Number of masks: {len(masks)}")
            for i, mask in enumerate(masks):
                coverage = (mask > 0.5).sum() / mask.size * 100
                print(f"    Mask {i}: shape={mask.shape}, coverage={coverage:.2f}%")
        
        if hasattr(r0, 'boxes') and r0.boxes is not None:
            if hasattr(r0.boxes, 'xyxy') and r0.boxes.xyxy is not None:
                xyxy = r0.boxes.xyxy.detach().cpu().numpy()
                print(f"\n  Bounding boxes (xyxy): {xyxy.shape[0]} boxes")
                for i, box in enumerate(xyxy):
                    print(f"    Box {i}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}")
            
            if hasattr(r0.boxes, 'conf') and r0.boxes.conf is not None:
                conf = r0.boxes.conf.detach().cpu().numpy()
                print(f"\n  Confidence scores: {conf}")
                best_idx = np.argmax(conf)
                print(f"  Best detection index: {best_idx} (conf={conf[best_idx]:.4f})")
    
    # Get best metal box for cropping
    if results_pass1 and len(results_pass1) > 0:
        r0 = results_pass1[0]
        if hasattr(r0, 'boxes') and r0.boxes is not None and hasattr(r0.boxes, 'conf'):
            conf = r0.boxes.conf.detach().cpu().numpy()
            best_idx = int(np.argmax(conf))
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            best_box = xyxy[best_idx].astype(int)
            x1, y1, x2, y2 = best_box
            
            # Crop for Pass 2
            crop = image[y1:y2, x1:x2].copy()
            print(f"\n  Cropped ROI for Pass 2: shape={crop.shape}")
        else:
            # Fallback to full image
            crop = image.copy()
            print("\n  No boxes found, using full image for Pass 2")
    else:
        crop = image.copy()
        print("\n  No results, using full image for Pass 2")
    
    # =====================================================================
    # PASS 2: Clean metal detection
    # =====================================================================
    print_separator("PASS 2a: SAM3 with prompt='clean shiny metal' (on cropped ROI)")
    
    prompt_clean = "clean shiny metal"
    print(f"Running SAM3 with prompt: '{prompt_clean}'")
    print(f"Input crop shape: {crop.shape}")
    
    results_pass2_clean = run_sam3_prompt(predictor, crop, prompt_clean)
    
    print(f"\nRaw results type: {type(results_pass2_clean)}")
    print(f"Number of results: {len(results_pass2_clean)}")
    
    if results_pass2_clean and len(results_pass2_clean) > 0:
        r0 = results_pass2_clean[0]
        print("\n--- Inspecting results[0] ---")
        inspect_result_object(r0, prefix="  ")
        
        print("\n--- Key Information Extracted ---")
        if hasattr(r0, 'masks') and r0.masks is not None:
            masks = r0.masks.data.detach().cpu().numpy()
            print(f"  Number of masks: {len(masks)}")
            for i, mask in enumerate(masks):
                coverage = (mask > 0.5).sum() / mask.size * 100
                print(f"    Mask {i}: shape={mask.shape}, coverage={coverage:.2f}%, "
                      f"min={mask.min():.4f}, max={mask.max():.4f}")
    
    # =====================================================================
    # PASS 2: Rusty metal detection  
    # =====================================================================
    print_separator("PASS 2b: SAM3 with prompt='rusty metal' (on cropped ROI)")
    
    prompt_rust = "rusty metal"
    print(f"Running SAM3 with prompt: '{prompt_rust}'")
    print(f"Input crop shape: {crop.shape}")
    
    results_pass2_rust = run_sam3_prompt(predictor, crop, prompt_rust)
    
    print(f"\nRaw results type: {type(results_pass2_rust)}")
    print(f"Number of results: {len(results_pass2_rust)}")
    
    if results_pass2_rust and len(results_pass2_rust) > 0:
        r0 = results_pass2_rust[0]
        print("\n--- Inspecting results[0] ---")
        inspect_result_object(r0, prefix="  ")
        
        print("\n--- Key Information Extracted ---")
        if hasattr(r0, 'masks') and r0.masks is not None:
            masks = r0.masks.data.detach().cpu().numpy()
            print(f"  Number of masks: {len(masks)}")
            for i, mask in enumerate(masks):
                coverage = (mask > 0.5).sum() / mask.size * 100
                print(f"    Mask {i}: shape={mask.shape}, coverage={coverage:.2f}%, "
                      f"min={mask.min():.4f}, max={mask.max():.4f}")
    
    # =====================================================================
    # Summary
    # =====================================================================
    print_separator("SUMMARY: SAM3 Output Structure")
    print("""
SAM3 returns a list of Result objects. Each Result object contains:

  .masks - Masks object with:
      .data    : Tensor of shape (N, H, W) - N binary/soft masks
                 Values typically in [0, 1], threshold at 0.5 for binary
      .shape   : Original shape info
      
  .boxes - Boxes object with:
      .xyxy    : Tensor of shape (N, 4) - bounding boxes [x1, y1, x2, y2]
      .conf    : Tensor of shape (N,) - confidence scores per detection
      .cls     : Tensor of shape (N,) - class IDs (usually 0 for single-class)
      
  .orig_shape  : Original image shape (H, W)
  .orig_img    : Original image (if kept)
  .path        : Path to input image
  .speed       : Dict with timing info (preprocess, inference, postprocess)

Key observations:
  - Pass 1 ("metal"): Used to find best metal region for cropping
  - Pass 2 ("clean shiny metal" / "rusty metal"): Used to build per-pixel evidence maps
  - Confidence (.boxes.conf) indicates detection quality
  - Masks (.masks.data) are soft masks that can be binarized at 0.5
""")
    
    return {
        "image": image,
        "crop": crop,
        "results_pass1": results_pass1,
        "results_pass2_clean": results_pass2_clean,
        "results_pass2_rust": results_pass2_rust,
    }


# =====================================================================
# Main entry point
# =====================================================================
if __name__ == "__main__":
    import argparse
    
    # File picker fallback
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
            raise RuntimeError("tkinter not available. Please provide --image argument.")
    
    parser = argparse.ArgumentParser(description="Debug SAM3 outputs for Pass 1 and Pass 2")
    parser.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint")
    
    args = parser.parse_args()
    
    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"Image not found: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()
    
    print(f"\nUsing image: {image_path}")
    print(f"Using checkpoint: {args.checkpoint}\n")
    
    results = debug_sam3_outputs(image_path, checkpoint=args.checkpoint)
    
    print("\n" + "=" * 80)
    print(" Debug complete!")
    print("=" * 80)
