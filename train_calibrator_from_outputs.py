"""
Train probability calibrator using pipeline outputs as pseudo-ground-truth.

This script:
1. Runs the detector on selected images
2. Uses the KMeans-based rust masks as pseudo-labels
3. Trains a calibrator to map raw scores → calibrated probabilities

Usage:
    python train_calibrator_from_outputs.py --images data/New/1.JPG,data/New/2.JPG,data/New/3.JPG
    
Or run on all images in a folder:
    python train_calibrator_from_outputs.py --image_dir data/New --max_images 10
"""

from __future__ import annotations
import argparse
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import from main script
from importlib import import_module


def get_detector_and_calibrator():
    """Import classes from sam3-exp-2-stage.py"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import the module (handle the hyphen in filename)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sam3_exp_2_stage", "sam3-exp-2-stage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.FastRustDetector, module.RustProbabilityCalibrator


def collect_training_data(
    image_paths: list,
    detector,
    sample_fraction: float = 0.1,
    visualize: bool = True,
    output_dir: str = "calibration_data",
):
    """
    Run detector on images and collect (raw_score, pseudo_label) pairs.
    
    The pseudo-labels come from the KMeans-based rust mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_scores = []
    all_labels = []
    image_stats = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_path}")
        
        try:
            results = detector.analyze(img_path, interactive=True)
        except SystemExit as e:
            print(f"  Skipped (no metal detected): {e}")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        # Get raw rust probability
        rust_ev = results['sam_rust_evidence']
        clean_ev = results['sam_clean_evidence']
        raw_prob = (rust_ev / (rust_ev + clean_ev + 1e-6)).astype(np.float32)
        
        # Get pseudo-labels from KMeans rust mask
        pseudo_labels = results['crop_mask'].astype(np.uint8)
        
        # Sample pixels (stratified to balance rust/clean)
        n_pixels = raw_prob.size
        rust_indices = np.where(pseudo_labels.ravel() == 1)[0]
        clean_indices = np.where(pseudo_labels.ravel() == 0)[0]
        
        n_rust = len(rust_indices)
        n_clean = len(clean_indices)
        
        # Sample equally from both classes (up to sample_fraction)
        n_sample_per_class = max(500, int(n_pixels * sample_fraction / 2))
        
        if n_rust > 0:
            rust_sample = np.random.choice(rust_indices, size=min(n_sample_per_class, n_rust), replace=False)
        else:
            rust_sample = np.array([], dtype=int)
            
        if n_clean > 0:
            clean_sample = np.random.choice(clean_indices, size=min(n_sample_per_class, n_clean), replace=False)
        else:
            clean_sample = np.array([], dtype=int)
        
        indices = np.concatenate([rust_sample, clean_sample])
        
        if len(indices) == 0:
            print(f"  Skipped (no valid pixels)")
            continue
        
        sampled_scores = raw_prob.ravel()[indices]
        sampled_labels = pseudo_labels.ravel()[indices]
        
        all_scores.append(sampled_scores)
        all_labels.append(sampled_labels)
        
        rust_ratio = sampled_labels.mean()
        stats = {
            'image': os.path.basename(img_path),
            'n_samples': len(indices),
            'rust_ratio': rust_ratio,
            'mean_score_rust': sampled_scores[sampled_labels == 1].mean() if rust_ratio > 0 else 0,
            'mean_score_clean': sampled_scores[sampled_labels == 0].mean() if rust_ratio < 1 else 0,
        }
        image_stats.append(stats)
        
        print(f"  Sampled {len(indices)} pixels | Rust ratio: {rust_ratio:.3f}")
        print(f"  Mean score (rust): {stats['mean_score_rust']:.3f} | Mean score (clean): {stats['mean_score_clean']:.3f}")
        
        # Save visualization
        if visualize:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            crop = results['crop']
            axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Cropped ROI')
            axes[0].axis('off')
            
            im1 = axes[1].imshow(raw_prob, cmap='RdYlGn_r', vmin=0, vmax=1)
            axes[1].set_title('Raw Rust Probability')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            
            axes[2].imshow(pseudo_labels, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title('Pseudo-Label (KMeans mask)')
            axes[2].axis('off')
            
            # Overlay
            overlay = crop.copy()
            overlay[pseudo_labels == 1] = [0, 0, 255]
            overlay = cv2.addWeighted(crop, 0.6, overlay, 0.4, 0)
            axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[3].set_title(f'Overlay (rust={rust_ratio*100:.1f}%)')
            axes[3].axis('off')
            
            plt.suptitle(f'{os.path.basename(img_path)}', fontsize=12)
            plt.tight_layout()
            
            base = os.path.splitext(os.path.basename(img_path))[0]
            fig.savefig(os.path.join(output_dir, f'{base}_calibration_data.png'), dpi=100)
            plt.close(fig)
    
    if len(all_scores) == 0:
        raise RuntimeError("No valid images processed!")
    
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    print(f"\n{'='*60}")
    print(f"Total samples collected: {len(labels)}")
    print(f"Overall rust ratio: {labels.mean():.3f}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"{'='*60}")
    
    return scores, labels, image_stats


def main():
    parser = argparse.ArgumentParser(description="Train calibrator from pipeline outputs")
    parser.add_argument("--images", type=str, default="",
                       help="Comma-separated list of image paths")
    parser.add_argument("--image_dir", type=str, default="",
                       help="Directory containing images (alternative to --images)")
    parser.add_argument("--max_images", type=int, default=15,
                       help="Maximum number of images to process from --image_dir")
    parser.add_argument("--sample_fraction", type=float, default=0.1,
                       help="Fraction of pixels to sample from each image")
    parser.add_argument("--output", type=str, default="calibrator.pkl",
                       help="Output path for calibrator")
    parser.add_argument("--method", type=str, default="isotonic", choices=['isotonic', 'sigmoid'],
                       help="Calibration method")
    parser.add_argument("--sam_checkpoint", type=str, default="sam3.pt")
    parser.add_argument("--output_dir", type=str, default="calibration_data",
                       help="Directory for calibration visualizations")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Skip saving visualizations")
    
    args = parser.parse_args()
    
    # Get image paths
    if args.images:
        image_paths = [p.strip() for p in args.images.split(',')]
    elif args.image_dir:
        patterns = ['*.JPG', '*.jpg', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, pattern)))
        image_paths = sorted(image_paths)[:args.max_images]
    else:
        raise ValueError("Provide either --images or --image_dir")
    
    if not image_paths:
        raise ValueError("No images found!")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Load detector and calibrator classes
    FastRustDetector, RustProbabilityCalibrator = get_detector_and_calibrator()
    
    # Initialize detector (without calibrator)
    print("\nInitializing detector...")
    detector = FastRustDetector(
        verbose=True,
        sam_checkpoint=args.sam_checkpoint,
    )
    
    # Collect training data
    print("\nCollecting training data from pipeline outputs...")
    scores, labels, stats = collect_training_data(
        image_paths,
        detector,
        sample_fraction=args.sample_fraction,
        visualize=not args.no_visualize,
        output_dir=args.output_dir,
    )
    
    # Train calibrator
    print(f"\nTraining calibrator (method={args.method})...")
    calibrator = RustProbabilityCalibrator(method=args.method)
    calibrator.fit(scores, labels)
    
    # Evaluate
    ece = calibrator.compute_calibration_error(scores, labels)
    print(f"\nCalibration Error (ECE):")
    print(f"  Raw:        {ece['raw_ece']:.4f}")
    print(f"  Calibrated: {ece['calibrated_ece']:.4f}")
    print(f"  Improvement: {ece['improvement']:.4f} ({100*ece['improvement']/max(ece['raw_ece'], 1e-6):.1f}%)")
    
    # Save calibrator
    calibrator.save(args.output)
    
    # Plot and save calibration curve
    print("\nGenerating calibration curve...")
    fig = calibrator.plot_calibration_curve(
        scores, labels,
        save_path=args.output.replace('.pkl', '_curve.png'),
        title=f"Calibration Curve ({args.method}) - {len(image_paths)} images"
    )
    plt.close(fig)
    
    # Save statistics
    stats_path = args.output.replace('.pkl', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Calibration Training Statistics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Overall rust ratio: {labels.mean():.4f}\n")
        f.write(f"Raw ECE: {ece['raw_ece']:.4f}\n")
        f.write(f"Calibrated ECE: {ece['calibrated_ece']:.4f}\n")
        f.write(f"Improvement: {ece['improvement']:.4f}\n\n")
        f.write("Per-image statistics:\n")
        f.write("-"*50 + "\n")
        for s in stats:
            f.write(f"{s['image']}: n={s['n_samples']}, rust={s['rust_ratio']:.3f}, "
                   f"score_rust={s['mean_score_rust']:.3f}, score_clean={s['mean_score_clean']:.3f}\n")
    print(f"Saved statistics to: {stats_path}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nCalibrator saved to: {args.output}")
    print(f"\nTo use it:")
    print(f"  python sam3-exp-2-stage.py --image <img> --calibrator_path {args.output} --save_probability_map 1")


if __name__ == "__main__":
    main()
