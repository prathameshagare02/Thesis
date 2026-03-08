"""
Quick diagnostic to see what SAM3 is returning for rust/clean prompts.
"""
import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_sam3_evidence.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Import detector
    import importlib.util
    spec = importlib.util.spec_from_file_location("sam3_exp", "sam3-exp-2-stage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    FastRustDetector = module.FastRustDetector
    
    # Create detector with verbose output
    print("="*60)
    print("Initializing detector...")
    print("="*60)
    detector = FastRustDetector(verbose=True)
    
    print("\n" + "="*60)
    print(f"Analyzing: {image_path}")
    print("="*60)
    
    try:
        results = detector.analyze(image_path, interactive=True)
    except SystemExit:
        print("Detector exited (likely no metal found)")
        return
    
    # Print diagnostic info
    print("\n" + "="*60)
    print("DIAGNOSTIC RESULTS")
    print("="*60)
    
    clean_ev = results.get('sam_clean_evidence')
    rust_ev = results.get('sam_rust_evidence')
    
    print(f"\nSAM3 Clean Evidence Map:")
    print(f"  Shape: {clean_ev.shape}")
    print(f"  Dtype: {clean_ev.dtype}")
    print(f"  Min: {clean_ev.min():.6f}")
    print(f"  Max: {clean_ev.max():.6f}")
    print(f"  Mean: {clean_ev.mean():.6f}")
    print(f"  Nonzero pixels: {(clean_ev > 0).sum()} / {clean_ev.size}")
    
    print(f"\nSAM3 Rust Evidence Map:")
    print(f"  Shape: {rust_ev.shape}")
    print(f"  Dtype: {rust_ev.dtype}")
    print(f"  Min: {rust_ev.min():.6f}")
    print(f"  Max: {rust_ev.max():.6f}")
    print(f"  Mean: {rust_ev.mean():.6f}")
    print(f"  Nonzero pixels: {(rust_ev > 0).sum()} / {rust_ev.size}")
    
    # Additional detection stats
    print(f"\nSAM3 Detection Counts:")
    print(f"  Clean prompt detections: {results.get('sam_clean_num_dets', 'N/A')}")
    print(f"  Rust prompt detections: {results.get('sam_rust_num_dets', 'N/A')}")
    print(f"  Clean best score: {results.get('sam_clean_best_score', 'N/A')}")
    print(f"  Rust best score: {results.get('sam_rust_best_score', 'N/A')}")
    
    # Compute probability
    prob = rust_ev / (rust_ev + clean_ev + 1e-6)
    print(f"\nRaw Rust Probability:")
    print(f"  Min: {prob.min():.6f}")
    print(f"  Max: {prob.max():.6f}")
    print(f"  Mean: {prob.mean():.6f}")
    
    # Check the prompts being used
    print(f"\nPrompts being used:")
    print(f"  Clean prompt: {detector.prompt_clean!r}")
    print(f"  Rust prompt: {detector.prompt_rust!r}")
    print(f"  Metal prompt: {detector.prompt_roi!r}")
    
    # Try running SAM3 directly to see if it detects anything
    print("\n" + "="*60)
    print("RAW SAM3 OUTPUT TEST")
    print("="*60)
    
    crop = results['crop']
    
    # Test with different prompts
    test_prompts = [
        detector.prompt_rust,
        detector.prompt_clean,
        "corrosion",
        "rust stains",
        "brown spots",
        "metal surface",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")
        try:
            raw_results = detector._run_sam3_text_prompt(crop, prompt)
            if raw_results and len(raw_results) > 0:
                r0 = raw_results[0]
                n_masks = len(r0.masks.data) if r0.masks is not None and r0.masks.data is not None else 0
                confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None and r0.boxes.conf is not None else []
                print(f"  Detections: {n_masks}")
                if len(confs) > 0:
                    print(f"  Confidence scores: {confs}")
            else:
                print(f"  Detections: 0 (no results)")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
