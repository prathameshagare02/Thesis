from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Union
import argparse
import os
import time
import tempfile
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Scikit-learn calibration imports
try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import BaseEstimator, ClassifierMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Probability calibration disabled.")
    print("Install with: pip install scikit-learn")


# ----------------------------- Probability Calibration -----------------------------
class RustProbabilityCalibrator:
    """
    Calibrates raw rust scores to well-calibrated probabilities.
    
    Supports two methods:
    - 'isotonic': Non-parametric isotonic regression (recommended for rust detection)
    - 'sigmoid': Platt scaling (logistic regression on scores)
    
    Usage:
        # Training (requires labeled data)
        calibrator = RustProbabilityCalibrator(method='isotonic')
        calibrator.fit(raw_scores, ground_truth_labels)  # labels: 0=clean, 1=rust
        calibrator.save('calibrator.pkl')
        
        # Inference
        calibrator = RustProbabilityCalibrator.load('calibrator.pkl')
        calibrated_probs = calibrator.predict_proba(raw_scores)
    """
    
    def __init__(self, method: str = 'isotonic', clip_range: Tuple[float, float] = (0.001, 0.999)):
        """
        Args:
            method: 'isotonic' or 'sigmoid'
            clip_range: Clip raw scores to this range before calibration
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for probability calibration")
        
        self.method = method
        self.clip_range = clip_range
        self.calibrator = None
        self.is_fitted = False
        
        # Statistics from training data
        self.train_stats = {}
    
    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> 'RustProbabilityCalibrator':
        """
        Fit calibrator on labeled data.
        
        Args:
            raw_scores: Raw rust scores (e.g., rust_ev / (rust_ev + clean_ev))
                       Shape: (n_samples,) with values in [0, 1]
            labels: Ground truth binary labels (0=clean, 1=rust)
                   Shape: (n_samples,)
        
        Returns:
            self
        """
        raw_scores = np.asarray(raw_scores).ravel().astype(np.float64)
        labels = np.asarray(labels).ravel().astype(np.int32)
        
        assert len(raw_scores) == len(labels), "Scores and labels must have same length"
        assert set(np.unique(labels)).issubset({0, 1}), "Labels must be 0 or 1"
        
        # Clip scores
        scores_clipped = np.clip(raw_scores, self.clip_range[0], self.clip_range[1])
        
        # Store training statistics
        self.train_stats = {
            'n_samples': len(labels),
            'n_rust': int((labels == 1).sum()),
            'n_clean': int((labels == 0).sum()),
            'rust_ratio': float((labels == 1).mean()),
            'score_mean': float(raw_scores.mean()),
            'score_std': float(raw_scores.std()),
            'score_min': float(raw_scores.min()),
            'score_max': float(raw_scores.max()),
        }
        
        if self.method == 'isotonic':
            # Isotonic regression: non-parametric, monotonic
            self.calibrator = IsotonicRegression(
                y_min=0.0, 
                y_max=1.0, 
                out_of_bounds='clip'
            )
            self.calibrator.fit(scores_clipped, labels)
            
        elif self.method == 'sigmoid':
            # Platt scaling: logistic regression on log-odds
            # Transform scores to log-odds for better fitting
            eps = 1e-7
            log_odds = np.log((scores_clipped + eps) / (1 - scores_clipped + eps))
            
            self.calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.calibrator.fit(log_odds.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'isotonic' or 'sigmoid'")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Transform raw scores to calibrated probabilities.
        
        Args:
            raw_scores: Raw rust scores, shape (n_samples,) or (H, W)
        
        Returns:
            Calibrated probabilities with same shape as input, values in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first or load a saved model.")
        
        original_shape = raw_scores.shape
        scores_flat = np.asarray(raw_scores).ravel().astype(np.float64)
        scores_clipped = np.clip(scores_flat, self.clip_range[0], self.clip_range[1])
        
        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(scores_clipped)
        elif self.method == 'sigmoid':
            eps = 1e-7
            log_odds = np.log((scores_clipped + eps) / (1 - scores_clipped + eps))
            calibrated = self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        
        calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        return calibrated.reshape(original_shape)
    
    def predict(self, raw_scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels from raw scores.
        
        Args:
            raw_scores: Raw rust scores
            threshold: Probability threshold for rust classification
        
        Returns:
            Binary predictions (0=clean, 1=rust)
        """
        probs = self.predict_proba(raw_scores)
        return (probs >= threshold).astype(np.uint8)
    
    def save(self, path: str):
        """Save calibrator to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'clip_range': self.clip_range,
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted,
                'train_stats': self.train_stats,
            }, f)
        print(f"Calibrator saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RustProbabilityCalibrator':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(method=data['method'], clip_range=data['clip_range'])
        obj.calibrator = data['calibrator']
        obj.is_fitted = data['is_fitted']
        obj.train_stats = data.get('train_stats', {})
        return obj
    
    def plot_calibration_curve(
        self, 
        raw_scores: np.ndarray, 
        labels: np.ndarray, 
        n_bins: int = 10,
        save_path: Optional[str] = None,
        title: str = "Calibration Curve"
    ) -> plt.Figure:
        """
        Plot calibration curve comparing raw vs calibrated probabilities.
        
        A well-calibrated model has points close to the diagonal.
        """
        raw_scores = np.asarray(raw_scores).ravel()
        labels = np.asarray(labels).ravel()
        
        # Get calibrated probabilities
        if self.is_fitted:
            calibrated_probs = self.predict_proba(raw_scores)
        else:
            calibrated_probs = raw_scores  # Use raw if not fitted
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Calibration curves
        ax1 = axes[0]
        
        # Raw scores calibration curve
        prob_true_raw, prob_pred_raw = calibration_curve(labels, raw_scores, n_bins=n_bins, strategy='uniform')
        ax1.plot(prob_pred_raw, prob_true_raw, 's-', label='Raw scores', color='red', alpha=0.7)
        
        # Calibrated probabilities curve
        if self.is_fitted:
            prob_true_cal, prob_pred_cal = calibration_curve(labels, calibrated_probs, n_bins=n_bins, strategy='uniform')
            ax1.plot(prob_pred_cal, prob_true_cal, 'o-', label='Calibrated', color='blue', alpha=0.7)
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives (Rust)')
        ax1.set_title(title)
        ax1.legend(loc='lower right')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Right: Histograms of predictions
        ax2 = axes[1]
        ax2.hist(raw_scores[labels == 0], bins=30, alpha=0.5, label='Clean (raw)', color='green', density=True)
        ax2.hist(raw_scores[labels == 1], bins=30, alpha=0.5, label='Rust (raw)', color='red', density=True)
        if self.is_fitted:
            ax2.hist(calibrated_probs[labels == 1], bins=30, alpha=0.3, label='Rust (calibrated)', 
                    color='blue', density=True, histtype='step', linewidth=2)
        ax2.set_xlabel('Probability Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Score Distribution by Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Calibration curve saved to: {save_path}")
        
        return fig
    
    def compute_calibration_error(self, raw_scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Compute Expected Calibration Error (ECE) for raw and calibrated scores.
        
        ECE measures how well the predicted probabilities match actual outcomes.
        Lower is better.
        """
        raw_scores = np.asarray(raw_scores).ravel()
        labels = np.asarray(labels).ravel()
        
        def ece(probs, labels, n_bins):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece_sum = 0.0
            for i in range(n_bins):
                mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
                if mask.sum() > 0:
                    bin_acc = labels[mask].mean()
                    bin_conf = probs[mask].mean()
                    bin_size = mask.sum() / len(probs)
                    ece_sum += bin_size * abs(bin_acc - bin_conf)
            return float(ece_sum)
        
        raw_ece = ece(raw_scores, labels, n_bins)
        
        if self.is_fitted:
            calibrated_probs = self.predict_proba(raw_scores)
            calibrated_ece = ece(calibrated_probs, labels, n_bins)
        else:
            calibrated_ece = None
        
        return {
            'raw_ece': raw_ece,
            'calibrated_ece': calibrated_ece,
            'improvement': (raw_ece - calibrated_ece) if calibrated_ece is not None else None,
        }


def create_calibration_training_data(
    image_paths: List[str],
    mask_paths: List[str],
    detector: 'FastRustDetector',
    sample_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to create training data for calibration from labeled images.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of corresponding ground truth mask paths (binary: 0=clean, 255=rust)
        detector: FastRustDetector instance
        sample_fraction: Fraction of pixels to sample from each image
    
    Returns:
        raw_scores: Concatenated raw rust scores from all images
        labels: Concatenated ground truth labels
    """
    all_scores = []
    all_labels = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        print(f"Processing: {img_path}")
        
        # Run detection
        results = detector.analyze(img_path, interactive=True)
        
        # Get raw rust probability (ratio method)
        rust_ev = results['sam_rust_evidence']
        clean_ev = results['sam_clean_evidence']
        raw_prob = rust_ev / (rust_ev + clean_ev + 1e-6)
        
        # Load ground truth mask
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"  Warning: Could not load mask {mask_path}, skipping")
            continue
        
        # Resize GT mask to match ROI if needed
        crop_h, crop_w = raw_prob.shape
        if gt_mask.shape != (crop_h, crop_w):
            # Extract ROI from GT mask
            x, y, w, h = results['crop_coords']
            gt_mask = gt_mask[y:y+h, x:x+w]
            if gt_mask.shape != (crop_h, crop_w):
                gt_mask = cv2.resize(gt_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary labels
        gt_labels = (gt_mask > 127).astype(np.uint8)
        
        # Sample pixels
        n_pixels = raw_prob.size
        n_samples = max(1000, int(n_pixels * sample_fraction))
        indices = np.random.choice(n_pixels, size=min(n_samples, n_pixels), replace=False)
        
        all_scores.append(raw_prob.ravel()[indices])
        all_labels.append(gt_labels.ravel()[indices])
        
        print(f"  Sampled {len(indices)} pixels, rust_ratio={gt_labels.ravel()[indices].mean():.3f}")
    
    return np.concatenate(all_scores), np.concatenate(all_labels)


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
    CORRECTED pipeline (SAM3 twice + per-pixel KMeans, NO SLIC):

      PASS 1 (full image):
        1) SAM3 prompt "metal"
        2) Select detection with MAX confidence
        3) Crop FULL IMAGE using that best metal box (optionally padded)

      PASS 2 (cropped ROI only):
        4) SAM3 prompt "clean shiny metal" -> per-pixel evidence map in ROI
        5) SAM3 prompt "rusty metal"       -> per-pixel evidence map in ROI
        6) Build per-pixel feature vectors ONLY from:
             - ROI image pixels
             - PASS-2 SAM3 evidence (clean/rust)
        7) K-means on ALL PIXELS in the ROI box
        8) Select rust cluster using prompt-guided cluster scoring
        9) Map ROI rust mask back to original image coordinates

    IMPORTANT:
      - Pass #1 is ONLY for cropping (best metal box).
      - Pass #1 mask is NOT used for rust classification, NOT used for feature vectors,
        and NOT used to gate pass #2.
      - Pass #2 receives the whole cropped ROI box image.
    """

    def __init__(
        self,
        verbose: bool = True,
        sam_checkpoint: str = "sam3.pt",
        roi_pad: int = 0,
        # Prompts
        metal_prompt: str = "metal",
        clean_prompt: str = "clean shiny metal",
        rust_prompt: str = "rusted area",
        # KMeans
        kmeans_k: int = 2,
        kmeans_attempts: int = 5,
        kmeans_max_iter: int = 50,
        kmeans_eps: float = 0.2,
        kmeans_train_samples: int = 80000,
        # Behavior
        ensure_one_positive: bool = True,
        # Calibration
        calibrator_path: Optional[str] = None,
        # Probability-based mask (for consistency with probability maps)
        use_probability_mask: bool = False,
        probability_threshold: float = 0.5,
    ):
        self.verbose = bool(verbose)
        self.sam_checkpoint = sam_checkpoint
        self.roi_pad = int(roi_pad)
        
        # Use probability threshold instead of KMeans for final mask
        self.use_probability_mask = bool(use_probability_mask)
        self.probability_threshold = float(probability_threshold)
        
        # Probability calibration
        self.calibrator: Optional[RustProbabilityCalibrator] = None
        if calibrator_path and os.path.exists(calibrator_path):
            if SKLEARN_AVAILABLE:
                try:
                    self.calibrator = RustProbabilityCalibrator.load(calibrator_path)
                    if self.verbose:
                        print(f"  Loaded calibrator from: {calibrator_path}")
                except Exception as e:
                    print(f"  Warning: Failed to load calibrator: {e}")
            else:
                print("  Warning: scikit-learn not available, calibration disabled")

        self.prompt_metal = str(metal_prompt)
        self.prompt_clean = str(clean_prompt)
        self.prompt_rust = str(rust_prompt)

        self.kmeans_k = max(2, int(kmeans_k))
        self.kmeans_attempts = max(1, int(kmeans_attempts))
        self.kmeans_max_iter = max(5, int(kmeans_max_iter))
        self.kmeans_eps = float(kmeans_eps)
        self.kmeans_train_samples = max(1000, int(kmeans_train_samples))

        self.ensure_one_positive = bool(ensure_one_positive)

        # Detect available device (prefer MPS on Apple silicon, then CUDA)
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

    # ---- logging ----
    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    def _print_backend_info(self):
        print("FastRustDetector initialized (PASS1 crop only, PASS2 features/classification)")
        print(f"  SAM3 checkpoint: {self.sam_checkpoint}")
        print(f"  ROI padding: {self.roi_pad}px")
        print(f"  Torch device: {getattr(self, 'device', 'unknown')}")
        print(f"  Prompts: metal={self.prompt_metal!r}, clean={self.prompt_clean!r}, rust={self.prompt_rust!r}")
        print(f"  KMeans: K={self.kmeans_k}, attempts={self.kmeans_attempts}, max_iter={self.kmeans_max_iter}")
        print(f"  KMeans train samples: {self.kmeans_train_samples}")
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
            # prefer device detected earlier (mps on Apple silicon, cuda if available)
            overrides["device"] = getattr(self, "device", "cpu")
            # enable FP16 only on CUDA; neither CPU nor MPS reliably support half for all ops
            overrides["half"] = True if overrides["device"] == "cuda" else False
            # Disable saving predictions to runs/segment/predict*
            overrides["save"] = False
            self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam3_predictor.setup_model()
            
            # Verify the device of the loaded model
            try:
                model_device = next(self.sam3_predictor.model.parameters()).device
                self._log(f"SAM3 model loaded successfully on device: {model_device}")
            except Exception:
                self._log("SAM3 model loaded successfully (device check failed).")
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

    def _run_sam3_text_prompt(self, image_source: Union[str, np.ndarray], prompt: str):
        """
        Wrapper for SAM3 inference. Supports path or ndarray.
        ndarray is written to a temporary PNG and passed to SAM3.
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

    # ---- PASS #1: Best box + mask ----
    def _sam3_best_box_for_prompt(
        self, image_source: Union[str, np.ndarray], prompt: str, expected_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, Optional[np.ndarray]]:
        """
        Returns:
          best_box  (x1,y1,x2,y2)
          best_score float
          best_mask uint8 HxW or None

        Note:
          Returns the segmentation mask to filter background pixels.
        """
        if self.sam3_predictor is None:
            return None, 0.0, None

        try:
            results = self._run_sam3_text_prompt(image_source, prompt)
            if not results:
                return None, 0.0, None
            r0 = results[0]

            if getattr(r0, "masks", None) is None or r0.masks is None or r0.masks.data is None:
                return None, 0.0, None
            if len(r0.masks.data) == 0:
                return None, 0.0, None

            masks = r0.masks.data
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
                # fallback: choose largest mask
                areas = []
                for i in range(n):
                    mi = masks[i].detach().cpu().numpy()
                    areas.append(float((mi > 0.5).sum()))
                best_i = int(np.argmax(np.array(areas)))
                best_score = 1.0

            # prefer detector bbox from best detection
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

            # IMPORTANT: Create UNION of ALL metal detection masks
            # This ensures rusty metal parts are included in foreground
            h_target, w_target = expected_hw if expected_hw is not None else (None, None)
            union_mask = None
            
            for i in range(n):
                mi = masks[i].detach().cpu().numpy().astype(np.float32)
                mi = np.clip(mi, 0.0, 1.0)
                if h_target is not None and w_target is not None and mi.shape[:2] != (h_target, w_target):
                    mi = self._safe_resize_mask(mi, (h_target, w_target))
                mi_binary = (mi > 0.5).astype(np.uint8)
                
                if union_mask is None:
                    union_mask = mi_binary
                else:
                    union_mask = np.maximum(union_mask, mi_binary)
            
            # Apply morphological operations to clean up union mask
            if union_mask is not None:
                processed_mask = self.apply_morphological_operations(union_mask)
                if processed_mask is not None:
                    union_mask = processed_mask
            else:
                # Fallback: use best mask only
                best_mask_raw = masks[best_i].detach().cpu().numpy().astype(np.float32)
                best_mask_raw = np.clip(best_mask_raw, 0.0, 1.0)
                if expected_hw is not None:
                    best_mask_raw = self._safe_resize_mask(best_mask_raw, expected_hw)
                union_mask = (best_mask_raw > 0.5).astype(np.uint8)
                processed_mask = self.apply_morphological_operations(union_mask)
                if processed_mask is not None:
                    union_mask = processed_mask

            # fallback bbox from union mask
            if best_box is None:
                best_box = self._bbox_from_mask(union_mask)

            self._log(f"Pass #1: Combined {n} metal detection(s) into union foreground mask")
            return best_box, float(best_score), union_mask

        except Exception as e:
            self._log(f"SAM3 best-box error for prompt={prompt!r}: {e}")
            return None, 0.0, None

    def get_best_metal_box(
        self, image_path: str, original: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]:
        """
        Returns:
          metal_box_full (x1,y1,x2,y2)
          metal_score float
          metal_mask uint8 HxW (foreground mask to filter background)

        If SAM3 fails -> full image box, score=0.0, full mask
        """
        h, w = original.shape[:2]
        box, score, mask = self._sam3_best_box_for_prompt(image_path, self.prompt_metal, expected_hw=(h, w))
        if box is None:
            return (0, 0, w, h), 0.0, np.ones((h, w), dtype=np.uint8)
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8)
        return box, float(score), mask

    # ---- PASS #2: Prompt evidence maps (ROI) ----
    def _sam3_prompt_evidence_map(
        self,
        image_source: Union[str, np.ndarray],
        prompt: str,
        expected_hw: Tuple[int, int],
    ) -> Dict:
        """
        Builds a per-pixel evidence map from all detections for a prompt in the ROI.
        Evidence = max(conf_i * mask_i) over detections.

        Returns:
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

    # ---- Per-pixel features (ROI only, pass2 evidence only) ----
    def _compute_pixel_feature_tensor(
        self,
        crop: np.ndarray,
        clean_evidence: np.ndarray,
        rust_evidence: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Per-pixel feature tensor for the ROI.
        Uses:
          - ROI image features
          - PASS #2 SAM3 evidence channels (clean/rust)
        Does NOT use pass #1 metal mask/features.
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

        # Normalized image features
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

        # Rust-oriented color priors
        redness = np.clip((a_centered + 10.0) / 80.0, 0.0, 1.0).astype(np.float32)
        brownness = np.clip((0.6 * a_centered + 0.8 * b_centered + 20.0) / 120.0, 0.0, 1.0).astype(np.float32)

        # PASS #2 SAM3 evidence channels only
        clean_ev = np.clip(clean_evidence.astype(np.float32), 0.0, 1.0)
        rust_ev = np.clip(rust_evidence.astype(np.float32), 0.0, 1.0)
        ev_union = np.maximum(clean_ev, rust_ev).astype(np.float32)
        ev_delta = (rust_ev - clean_ev).astype(np.float32)
        ev_ratio = (rust_ev / (rust_ev + clean_ev + 1e-6)).astype(np.float32)

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
            "sam2_clean_evidence",
            "sam2_rust_evidence",
            "sam2_union_evidence",
            "sam2_rust_minus_clean",
            "sam2_rust_ratio",
        ]
        return features, feature_names

    # ---- KMeans on whole ROI box ----
    def _kmeans_pixel_classification(
        self,
        feature_tensor: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        KMeans on ALL pixels of the ROI box.
        No pass #1 mask gating.
        """
        h, w, fdim = feature_tensor.shape
        n_pixels = h * w

        cluster_map = np.full((h, w), fill_value=-1, dtype=np.int16)
        rust_mask = np.zeros((h, w), dtype=np.uint8)

        if n_pixels == 0:
            return dict(
                cluster_map=cluster_map,
                rust_mask=rust_mask,
                rust_cluster_id=-1,
                clean_cluster_id=-1,
                centers=None,
                cluster_scores={},
                n_pixels=0,
            )

        X_raw = feature_tensor.reshape(-1, fdim).astype(np.float32)

        # Standardize for KMeans
        mu = X_raw.mean(axis=0, keepdims=True)
        sigma = X_raw.std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        X = ((X_raw - mu) / sigma).astype(np.float32)

        # Subsample for fitting, assign all later
        n = X.shape[0]
        if n > self.kmeans_train_samples:
            rng = np.random.default_rng(0)
            idx_train = rng.choice(n, size=self.kmeans_train_samples, replace=False)
            X_train = X[idx_train]
        else:
            X_train = X

        K = min(self.kmeans_k, max(2, X_train.shape[0]))
        if X_train.shape[0] < 2:
            idx = {name: i for i, name in enumerate(feature_names)}
            rust_i = idx["sam2_rust_evidence"]
            clean_i = idx["sam2_clean_evidence"]
            rule = (X_raw[:, rust_i] > X_raw[:, clean_i]).astype(np.uint8)
            cluster_map[:] = rule.reshape(h, w).astype(np.int16)
            rust_mask[:] = rule.reshape(h, w).astype(np.uint8)
            return dict(
                cluster_map=cluster_map,
                rust_mask=rust_mask,
                rust_cluster_id=1,
                clean_cluster_id=0,
                centers=None,
                cluster_scores={"fallback_rule": True},
                n_pixels=n_pixels,
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

        # Assign all pixels to nearest center
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_all = np.argmin(d2, axis=1).astype(np.int16)

        # Prompt-guided rust cluster selection (uses pass2 evidence only)
        idx = {name: i for i, name in enumerate(feature_names)}
        cluster_scores: Dict[int, float] = {}

        for c in range(K):
            m = labels_all == c
            if not np.any(m):
                cluster_scores[c] = -1e9
                continue

            meanv = X_raw[m].mean(axis=0)

            score = (
                2.8 * float(meanv[idx["sam2_rust_evidence"]])
                - 2.2 * float(meanv[idx["sam2_clean_evidence"]])
                + 0.45 * float(meanv[idx["redness"]])
                + 0.40 * float(meanv[idx["brownness"]])
                + 0.18 * float(meanv[idx["texture_norm"]])
                + 0.10 * float(meanv[idx["grad_norm"]])
                + 0.10 * float(meanv[idx["S_norm"]])
                - 0.05 * float(meanv[idx["V_norm"]])
            )
            cluster_scores[c] = float(score)

        rust_cluster_id = int(max(cluster_scores, key=cluster_scores.get))
        if K == 2:
            clean_cluster_id = 1 - rust_cluster_id
        else:
            ordered = sorted(cluster_scores.items(), key=lambda kv: kv[1], reverse=True)
            clean_cluster_id = int(ordered[1][0]) if len(ordered) > 1 else rust_cluster_id

        rust_pred = (labels_all == rust_cluster_id).astype(np.uint8)

        # Light refinement (pass2 evidence + color only)
        rust_i = idx["sam2_rust_evidence"]
        clean_i = idx["sam2_clean_evidence"]
        delta_i = idx["sam2_rust_minus_clean"]
        redness_i = idx["redness"]
        brown_i = idx["brownness"]
        a_i = idx["a_norm"]
        b_i = idx["b_norm"]
        S_i = idx["S_norm"]
        H_i = idx["H_norm"]
        chroma_i = idx["chroma_norm"]
        L_i = idx["L_norm"]

        # Exclude blue/cyan background pixels more aggressively
        # Blue/cyan characteristics in 12.JPG:
        # - a_norm: mean=0.513, max=0.592
        # - b_norm: mean=0.377, max=0.475
        # - H_norm: mean=0.603 (blue-cyan range ~0.45-0.75)
        # Blue is characterized by being in the blue hue range with low b*
        is_blue_background = (
            (X_raw[:, b_i] < 0.50) &                 # low b* (blue direction in Lab)
            (X_raw[:, H_i] > 0.45) & (X_raw[:, H_i] < 0.75)  # hue in blue-cyan range
        )

        support = (
            (X_raw[:, rust_i] >= (X_raw[:, clean_i] - 0.02))
            | (X_raw[:, delta_i] > -0.02)
            | ((X_raw[:, redness_i] > 0.55) & (X_raw[:, brown_i] > 0.45))
        )
        rust_pred = (rust_pred & support.astype(np.uint8)).astype(np.uint8)
        
        # Directly mask out blue background pixels from rust prediction
        rust_pred = rust_pred & (~is_blue_background).astype(np.uint8)
        
        # Require minimum color evidence for rust
        # 12.JPG non-rust has redness=0.123, brownness=0.190
        # 1.JPG rust has redness=0.278 mean (10th percentile=0.100)
        # Use higher thresholds to filter more strictly
        has_rust_color = (X_raw[:, redness_i] > 0.12) | (X_raw[:, brown_i] > 0.20)
        rust_pred = rust_pred & has_rust_color.astype(np.uint8)

        if self.ensure_one_positive and int(rust_pred.sum()) == 0 and n > 0:
            rank = (
                2.0 * X_raw[:, rust_i]
                - 1.5 * X_raw[:, clean_i]
                + 0.5 * X_raw[:, redness_i]
                + 0.4 * X_raw[:, brown_i]
            )
            rust_pred[int(np.argmax(rank))] = 1

        cluster_map = labels_all.reshape(h, w).astype(np.int16)
        rust_mask = rust_pred.reshape(h, w).astype(np.uint8)

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
            n_pixels=n_pixels,
            compactness=float(compactness),
        )

    def _perform_pixel_kmeans_analysis(self, crop: np.ndarray) -> Dict:
        """
        PASS #2 on the WHOLE ROI crop:
          - clean/rust SAM3 evidence maps
          - per-pixel feature vectors
          - KMeans on all ROI pixels
        """
        h, w = crop.shape[:2]

        self._log(f"SAM3 pass #2 (whole ROI) prompt={self.prompt_clean!r}")
        t0 = time.time()
        clean_out = self._sam3_prompt_evidence_map(crop, self.prompt_clean, expected_hw=(h, w))
        self._log(f"  → Clean prompt done in {time.time() - t0:.3f}s")

        self._log(f"SAM3 pass #2 (whole ROI) prompt={self.prompt_rust!r}")
        t0 = time.time()
        rust_out = self._sam3_prompt_evidence_map(crop, self.prompt_rust, expected_hw=(h, w))
        self._log(f"  → Rust prompt done in {time.time() - t0:.3f}s")

        clean_ev = clean_out["evidence_map"]
        rust_ev = rust_out["evidence_map"]

        t0 = time.time()
        feature_tensor, feature_names = self._compute_pixel_feature_tensor(crop, clean_ev, rust_ev)
        self._log(f"  → Pixel feature tensor (CPU-heavy) done in {time.time() - t0:.3f}s")
        self._log(f"    Feature tensor shape: {feature_tensor.shape}")

        # KMeans on whole ROI box
        t0 = time.time()
        km = self._kmeans_pixel_classification(feature_tensor, feature_names)
        self._log(f"  → Pixel KMeans (CPU-heavy) done in {time.time() - t0:.3f}s")

        rust_mask_roi = km["rust_mask"]
        rust_pixels = int(rust_mask_roi.sum())
        total_pixels_box = int(h * w)
        rust_pct_box = (100.0 * rust_pixels / total_pixels_box) if total_pixels_box > 0 else 0.0

        # Compute raw rust probability (ratio method)
        raw_rust_prob = (rust_ev / (rust_ev + clean_ev + 1e-6)).astype(np.float32)
        
        # Apply calibration if available
        if self.calibrator is not None and self.calibrator.is_fitted:
            calibrated_rust_prob = self.calibrator.predict_proba(raw_rust_prob)
            self._log(f"Applied probability calibration (method={self.calibrator.method})")
        else:
            calibrated_rust_prob = raw_rust_prob
            if self.verbose:
                self._log("No calibrator loaded, using raw probability scores")
        
        # Option to use probability-based mask for consistency with probability maps
        if self.use_probability_mask:
            rust_mask_roi = (calibrated_rust_prob >= self.probability_threshold).astype(np.uint8)
            # Apply morphological cleaning
            kernel = np.ones((3, 3), np.uint8)
            rust_mask_roi = cv2.morphologyEx(rust_mask_roi, cv2.MORPH_OPEN, kernel)
            rust_mask_roi = cv2.morphologyEx(rust_mask_roi, cv2.MORPH_CLOSE, kernel)
            rust_pixels = int(rust_mask_roi.sum())
            rust_pct_box = (100.0 * rust_pixels / total_pixels_box) if total_pixels_box > 0 else 0.0
            self._log(f"Using probability-based mask (threshold={self.probability_threshold:.2f})")
        
        # diagnostic score map (now uses calibrated probabilities if available)
        score_map = np.clip(calibrated_rust_prob, 0.0, 1.0).astype(np.float32)

        self._log(f"ROI pixels={total_pixels_box}, rust={rust_pixels} ({rust_pct_box:.2f}% of ROI box)")

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
            valid_mask=np.ones((h, w), dtype=np.uint8),  # whole ROI is valid by design
            cluster_map=km["cluster_map"],            # HxW int16
            rust_cluster_id=int(km["rust_cluster_id"]),
            clean_cluster_id=int(km["clean_cluster_id"]),
            cluster_scores=km["cluster_scores"],
            crop_mask=rust_mask_roi.astype(np.uint8),
            score_map=score_map,
            rust_percentage_box=rust_pct_box,
            kmeans_meta=km,
            # Probability outputs
            raw_rust_probability=raw_rust_prob,
            calibrated_rust_probability=calibrated_rust_prob,
            is_calibrated=(self.calibrator is not None and self.calibrator.is_fitted),
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
        
        # Warn if image is very large (e.g. > 12MP)
        mega_pixels = (original.shape[0] * original.shape[1]) / 1e6
        if mega_pixels > 12.0:
            self._log(f"WARNING: Image is very large ({mega_pixels:.1f} MP). CPU steps (KMeans, features) will be slow.")

        self.original_image = original.copy()
        timings["load"] = time.time() - t0

        # 1) PASS #1: SAM3 -> best metal box only (crop-only logic)
        t0 = time.time()
        if interactive and self.sam3_predictor is not None:
            self._log(f"SAM3 pass #1: selecting MAX-score box for prompt={self.prompt_metal!r} (crop only)")
            metal_box_full, metal_box_score, _ = self.get_best_metal_box(image_path, original)
        else:
            h, w = original.shape[:2]
            metal_box_full = (0, 0, w, h)
            metal_box_score = 0.0
        timings["sam3_pass1"] = time.time() - t0

        # Keep your original behavior
        if (metal_box_score is None) or (float(metal_box_score) <= 0.0):
            raise SystemExit("Terminating: metal score is None or 0 (no valid metal detection).")

        # 2) Crop WHOLE box from original (no mask usage)
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
        timings["crop"] = time.time() - t0

        self._log("PASS #1 complete: only ROI box was used. No pass1 mask/features are used downstream.")

        # 3) PASS #2 + per-pixel KMeans on WHOLE ROI box
        t0 = time.time()
        self._log("Running PASS #2 (clean/rust prompts) + per-pixel KMeans on whole ROI box...")
        res = self._perform_pixel_kmeans_analysis(crop)
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
            f"Total time: {total_time:.3f}s"
        )

        return dict(
            original=original,
            metal_box_full=metal_box_full,
            metal_box_score=float(metal_box_score),
            crop=crop,
            crop_coords=crop_coords,
            full_mask=full_rust_mask,
            crop_mask=res["crop_mask"],
            score_map=res["score_map"],
            rust_percentage=float(res["rust_percentage_box"]),
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
            valid_mask=res["valid_mask"],  # all ones (whole ROI)
            cluster_map=res["cluster_map"],
            rust_cluster_id=res["rust_cluster_id"],
            clean_cluster_id=res["clean_cluster_id"],
            cluster_scores=res["cluster_scores"],
            kmeans_meta=res["kmeans_meta"],
            # Probability outputs
            raw_rust_probability=res["raw_rust_probability"],
            calibrated_rust_probability=res["calibrated_rust_probability"],
            is_calibrated=res["is_calibrated"],
        )

    # ---- Visualization ----
    def visualize_detection(self, results: Dict, save_path: str | None = None, show: bool = True) -> plt.Figure:
        original = results["original"]
        crop = results["crop"]
        full_mask = results["full_mask"]
        crop_mask = results["crop_mask"]

        clean_ev = results["sam_clean_evidence"]
        rust_ev = results["sam_rust_evidence"]
        cluster_map = results["cluster_map"]
        rust_cluster_id = int(results["rust_cluster_id"])
        clean_cluster_id = int(results["clean_cluster_id"])

        (bx1, by1, bx2, by2) = results["metal_box_full"]
        bscore = float(results["metal_box_score"])
        rust_percentage = float(results["rust_percentage"])

        # Stage 1: input
        stage1 = original.copy()

        # Stage 2: input + best metal box (pass1)
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

        # Stage 3: CROPPED ROI ONLY (whole box from pass1) -- no pass1 mask overlay
        stage3 = crop.copy()

        # Stage 4: pass2 prompt evidence (clean=green, rust=red)
        stage4 = crop.copy()
        ev_vis = np.zeros_like(crop, dtype=np.uint8)
        ev_vis[:, :, 1] = np.clip(clean_ev * 255.0, 0, 255).astype(np.uint8)  # green
        ev_vis[:, :, 2] = np.clip(rust_ev * 255.0, 0, 255).astype(np.uint8)   # red
        stage4 = cv2.addWeighted(stage4, 0.6, ev_vis, 0.5, 0)

        # Stage 5: KMeans clusters + final rust mask in ROI
        stage5 = crop.copy()
        km_overlay = stage5.copy()
        if clean_cluster_id >= 0:
            km_overlay[(cluster_map == clean_cluster_id)] = [0, 255, 0]  # green
        if rust_cluster_id >= 0:
            km_overlay[(cluster_map == rust_cluster_id)] = [0, 0, 255]   # red
        stage5 = cv2.addWeighted(stage5, 0.55, km_overlay, 0.45, 0)

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
            "Rust Detection Stages (SAM3 pass1 crop-only -> SAM3 pass2 clean/rust on whole ROI -> per-pixel KMeans)\n"
            f"Metal box score={bscore:.2f} | Coverage within ROI box={rust_percentage:.1f}%",
            fontsize=12,
        )

        imgs = [stage1, stage2, stage3, stage4, stage5, stage6]
        titles = [
            "1) Input image",
            "2) Input + best metal box (SAM3 pass #1, crop only)",
            "3) Cropped ROI (whole max-score box only)",
            "4) SAM3 pass #2 evidence on whole ROI (clean=green, rust=red)",
            "5) KMeans pixel clusters in ROI + final rust mask",
            "6) Final rust overlay on input (red) + box",
        ]

        for ax, img, title in zip(axes.ravel(), imgs, titles):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

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

    def visualize_with_calibration(
        self, 
        results: Dict, 
        save_path: str | None = None, 
        show: bool = True
    ) -> plt.Figure:
        """
        Extended visualization including probability maps and calibration info.
        Creates an 8-panel figure (2 rows x 4 cols).
        """
        original = results["original"]
        crop = results["crop"]
        full_mask = results["full_mask"]
        crop_mask = results["crop_mask"]

        clean_ev = results["sam_clean_evidence"]
        rust_ev = results["sam_rust_evidence"]
        
        raw_prob = results.get("raw_rust_probability", rust_ev / (rust_ev + clean_ev + 1e-6))
        cal_prob = results.get("calibrated_rust_probability", raw_prob)
        is_calibrated = results.get("is_calibrated", False)

        (bx1, by1, bx2, by2) = results["metal_box_full"]
        bscore = float(results["metal_box_score"])
        rust_percentage = float(results["rust_percentage"])

        # Stage 1: input
        stage1 = original.copy()

        # Stage 2: input + best metal box
        stage2 = original.copy()
        cv2.rectangle(stage2, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        # Stage 3: CROPPED ROI
        stage3 = crop.copy()

        # Stage 4: pass2 prompt evidence
        stage4 = crop.copy()
        ev_vis = np.zeros_like(crop, dtype=np.uint8)
        ev_vis[:, :, 1] = np.clip(clean_ev * 255.0, 0, 255).astype(np.uint8)
        ev_vis[:, :, 2] = np.clip(rust_ev * 255.0, 0, 255).astype(np.uint8)
        stage4 = cv2.addWeighted(stage4, 0.6, ev_vis, 0.5, 0)

        # Stage 5: final rust mask overlay
        stage5 = crop.copy()
        rust_overlay_roi = stage5.copy()
        rust_overlay_roi[crop_mask == 1] = [0, 0, 255]
        stage5 = cv2.addWeighted(stage5, 0.6, rust_overlay_roi, 0.4, 0)

        # Stage 6: final overlay on full image
        stage6 = original.copy()
        rust_overlay_full = stage6.copy()
        rust_overlay_full[full_mask == 1] = [0, 0, 255]
        stage6 = cv2.addWeighted(stage6, 0.6, rust_overlay_full, 0.4, 0)
        cv2.rectangle(stage6, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        cal_status = "Calibrated" if is_calibrated else "Uncalibrated"
        fig.suptitle(
            f"Rust Detection with Probability Calibration ({cal_status})\n"
            f"Metal box score={bscore:.2f} | Rust coverage={rust_percentage:.1f}%",
            fontsize=12,
        )

        # Row 1: Detection stages
        axes[0, 0].imshow(cv2.cvtColor(stage1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1) Input Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("2) Metal Detection (SAM3 Pass #1)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(cv2.cvtColor(stage3, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("3) Cropped ROI")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(cv2.cvtColor(stage4, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title("4) SAM3 Evidence (green=clean, red=rust)")
        axes[0, 3].axis("off")

        # Row 2: Probability maps + final output
        im1 = axes[1, 0].imshow(raw_prob, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1, 0].set_title("5) Raw Rust Probability")
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im2 = axes[1, 1].imshow(cal_prob, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1, 1].set_title(f"6) {cal_status} Probability")
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        axes[1, 2].imshow(cv2.cvtColor(stage5, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("7) Final Rust Mask (ROI)")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(cv2.cvtColor(stage6, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title("8) Final Result (Full Image)")
        axes[1, 3].axis("off")

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

    def save_calibration_outputs(
        self,
        results: Dict,
        output_dir: str,
        base_name: str,
    ) -> Dict[str, str]:
        """
        Save all calibration-related outputs:
        - Probability maps (grayscale PNG + NPZ)
        - Calibration statistics (JSON)
        - Probability histogram
        
        Returns dict of output paths.
        """
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = {}
        
        raw_prob = results.get("raw_rust_probability")
        cal_prob = results.get("calibrated_rust_probability")
        is_calibrated = results.get("is_calibrated", False)
        
        if raw_prob is None:
            return output_paths
        
        # 1. Save probability maps as grayscale PNG
        prob_img = (cal_prob * 255).astype(np.uint8)
        prob_path = os.path.join(output_dir, f"{base_name}_probability.png")
        cv2.imwrite(prob_path, prob_img)
        output_paths["probability_png"] = prob_path
        
        # 2. Save probability data as NPZ (both raw and calibrated)
        npz_path = os.path.join(output_dir, f"{base_name}_probability.npz")
        np.savez_compressed(
            npz_path,
            raw_probability=raw_prob.astype(np.float32),
            calibrated_probability=cal_prob.astype(np.float32),
            is_calibrated=np.array([is_calibrated]),
            rust_mask=results["crop_mask"].astype(np.uint8),
        )
        output_paths["probability_npz"] = npz_path
        
        # 3. Save statistics as JSON
        stats = {
            "is_calibrated": is_calibrated,
            "calibration_method": self.calibrator.method if self.calibrator else None,
            "rust_percentage": float(results["rust_percentage"]),
            "metal_box_score": float(results["metal_box_score"]),
            "raw_probability": {
                "min": float(raw_prob.min()),
                "max": float(raw_prob.max()),
                "mean": float(raw_prob.mean()),
                "std": float(raw_prob.std()),
            },
            "calibrated_probability": {
                "min": float(cal_prob.min()),
                "max": float(cal_prob.max()),
                "mean": float(cal_prob.mean()),
                "std": float(cal_prob.std()),
            },
            "thresholded_rust_pixels": {
                "threshold_0.3": int((cal_prob > 0.3).sum()),
                "threshold_0.5": int((cal_prob > 0.5).sum()),
                "threshold_0.7": int((cal_prob > 0.7).sum()),
            },
            "total_pixels": int(cal_prob.size),
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_calibration_stats.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        output_paths["stats_json"] = json_path
        
        # 4. Save probability histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(raw_prob.ravel(), bins=50, alpha=0.7, color='red', label='Raw')
        axes[0].hist(cal_prob.ravel(), bins=50, alpha=0.5, color='blue', label='Calibrated')
        axes[0].set_xlabel('Probability')
        axes[0].set_ylabel('Pixel Count')
        axes[0].set_title('Probability Distribution (All Pixels)')
        axes[0].legend()
        axes[0].set_xlim([0, 1])
        
        # Cumulative distribution
        sorted_probs = np.sort(cal_prob.ravel())
        cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        axes[1].plot(sorted_probs, cdf, color='blue')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Probability Threshold')
        axes[1].set_ylabel('Fraction of Pixels Below Threshold')
        axes[1].set_title('Cumulative Distribution')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = os.path.join(output_dir, f"{base_name}_probability_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        output_paths["histogram_png"] = hist_path
        
        return output_paths


# ----------------------------- CLI + Main -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rust Detection: SAM3 pass1 metal ROI crop-only -> SAM3 pass2 clean/rust evidence on whole ROI -> per-pixel KMeans"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--interactive", type=int, default=1, help="1=use SAM3 pass #1 for metal ROI; 0=use full image.")

    p.add_argument("--sam_checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint.")
    p.add_argument("--roi_pad", type=int, default=0, help="Padding around best metal box before cropping.")

    # Prompts
    p.add_argument("--prompt_metal", type=str, default="metal", help="SAM3 prompt for pass #1 ROI extraction.")
    p.add_argument("--prompt_clean", type=str, default="clean shiny metal", help="SAM3 prompt for pass #2 clean evidence.")
    p.add_argument("--prompt_rust", type=str, default="rusted area", help="SAM3 prompt for pass #2 rust evidence.")

    # KMeans
    p.add_argument("--kmeans_k", type=int, default=2, help="Number of clusters for pixel KMeans (recommended=2).")
    p.add_argument("--kmeans_attempts", type=int, default=5)
    p.add_argument("--kmeans_max_iter", type=int, default=50)
    p.add_argument("--kmeans_eps", type=float, default=0.2)
    p.add_argument(
        "--kmeans_train_samples",
        type=int,
        default=80000,
        help="Max sampled pixels used to fit KMeans; all ROI pixels are assigned afterward.",
    )

    # Behavior
    p.add_argument("--ensure_one_positive", type=int, default=1)
    p.add_argument("--verbose", type=int, default=1)

    # Outputs
    p.add_argument("--save_features", type=int, default=0, help="Save per-pixel feature tensor and maps as .npz")
    p.add_argument("--res_dir", type=str, default="DEBUG")
    p.add_argument("--no_show", action="store_true", help="Don't show matplotlib window (for batch processing)")
    
    # Calibration
    p.add_argument("--calibrator_path", type=str, default="", help="Path to saved calibrator (.pkl) for probability calibration")
    p.add_argument("--save_probability_map", type=int, default=0, help="Save calibrated probability map as image")
    p.add_argument("--save_calibration_outputs", type=int, default=0, help="Save all calibration outputs (probability maps, stats, histograms)")
    
    # Probability-based mask (for consistent visualization)
    p.add_argument("--use_probability_mask", type=int, default=0, 
                   help="Use probability threshold for rust mask instead of KMeans (makes mask consistent with probability maps)")
    p.add_argument("--probability_threshold", type=float, default=0.5,
                   help="Threshold for probability-based mask (pixels with P(rust) >= threshold are marked as rust)")
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
        ensure_one_positive=bool(args.ensure_one_positive),
        calibrator_path=args.calibrator_path if args.calibrator_path else None,
        use_probability_mask=bool(args.use_probability_mask),
        probability_threshold=float(args.probability_threshold),
    )

    results = detector.analyze(image_path, interactive=bool(args.interactive))

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("results", args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Use extended visualization with probability maps when calibrator is loaded
    if args.calibrator_path or args.save_calibration_outputs or args.save_probability_map:
        img_out = os.path.join(out_dir, f"{base}_calibrated_stages.png")
        detector.visualize_with_calibration(results, save_path=img_out, show=not args.no_show)
    else:
        img_out = os.path.join(out_dir, f"{base}_stages.png")
        detector.visualize_detection(results, save_path=img_out, show=not args.no_show)
    
    # Save all calibration outputs
    if args.save_calibration_outputs:
        output_paths = detector.save_calibration_outputs(results, out_dir, base)
        print(f"\nCalibration outputs saved:")
        for key, path in output_paths.items():
            print(f"  {key}: {path}")

    # Save per-pixel feature vectors + maps
    if bool(args.save_features):
        feat_out = os.path.join(out_dir, f"{base}_pixel_features.npz")

        feature_names_arr = np.array(results["pixel_feature_names"], dtype="<U64")

        np.savez_compressed(
            feat_out,
            # ROI / geometry
            crop_coords=np.array(results["crop_coords"], dtype=np.int32),  # x,y,w,h
            metal_box_full=np.array(results["metal_box_full"], dtype=np.int32),
            metal_box_score=np.array([results["metal_box_score"]], dtype=np.float32),

            # PASS #2 SAM evidence (used for classification)
            sam2_clean_evidence=results["sam_clean_evidence"].astype(np.float32),
            sam2_rust_evidence=results["sam_rust_evidence"].astype(np.float32),
            sam2_clean_best_mask=results["sam_clean_best_mask"].astype(np.uint8),
            sam2_rust_best_mask=results["sam_rust_best_mask"].astype(np.uint8),

            # Per-pixel features (whole ROI)
            pixel_feature_tensor=results["pixel_feature_tensor"].astype(np.float32),  # HxWxF
            pixel_feature_names=feature_names_arr,  # [F]

            # KMeans / output maps (whole ROI classification)
            valid_mask=results["valid_mask"].astype(np.uint8),  # all ones (whole ROI)
            cluster_map=results["cluster_map"].astype(np.int16),
            rust_cluster_id=np.array([results["rust_cluster_id"]], dtype=np.int32),
            clean_cluster_id=np.array([results["clean_cluster_id"]], dtype=np.int32),
            rust_mask_roi=results["crop_mask"].astype(np.uint8),
            rust_mask_full=results["full_mask"].astype(np.uint8),

            # Summary
            rust_percentage_box=np.array([results["rust_percentage"]], dtype=np.float32),
        )

        print(f"Saved per-pixel feature tensor + maps to: {feat_out}")
        print(f"Feature tensor shape (H, W, F): {results['pixel_feature_tensor'].shape}")
        print(f"Feature names: {results['pixel_feature_names']}")
        print("CONFIRMED: pass #1 only crops ROI; pass #2 runs on the whole cropped ROI box.")
    
    # Save probability map if requested
    if bool(args.save_probability_map):
        prob_map = results['calibrated_rust_probability']
        
        # Save as grayscale image (0-255)
        prob_img = (prob_map * 255).astype(np.uint8)
        prob_out = os.path.join(out_dir, f"{base}_rust_probability.png")
        cv2.imwrite(prob_out, prob_img)
        print(f"Saved probability map to: {prob_out}")
        
        # Save as heatmap visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original crop
        axes[0].imshow(cv2.cvtColor(results['crop'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('Cropped ROI')
        axes[0].axis('off')
        
        # Raw probability
        im1 = axes[1].imshow(results['raw_rust_probability'], cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1].set_title('Raw Rust Probability')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Calibrated probability
        im2 = axes[2].imshow(results['calibrated_rust_probability'], cmap='RdYlGn_r', vmin=0, vmax=1)
        cal_label = '(Calibrated)' if results['is_calibrated'] else '(Uncalibrated)'
        axes[2].set_title(f'Rust Probability {cal_label}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        prob_vis_out = os.path.join(out_dir, f"{base}_probability_heatmap.png")
        fig.savefig(prob_vis_out, dpi=150)
        plt.close(fig)
        print(f"Saved probability heatmap to: {prob_vis_out}")


# ----------------------------- Calibration Training Script -----------------------------
def train_calibrator_cli():
    """
    CLI script to train a probability calibrator from labeled data.
    
    Usage:
        python sam3-exp-2-stage.py --train_calibrator \
            --train_images img1.jpg,img2.jpg \
            --train_masks mask1.png,mask2.png \
            --calibrator_output calibrator.pkl
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train probability calibrator")
    parser.add_argument("--train_images", type=str, required=True, 
                       help="Comma-separated list of image paths")
    parser.add_argument("--train_masks", type=str, required=True,
                       help="Comma-separated list of ground truth mask paths (binary: 0=clean, 255=rust)")
    parser.add_argument("--calibrator_output", type=str, default="calibrator.pkl",
                       help="Output path for calibrator")
    parser.add_argument("--method", type=str, default="isotonic", choices=['isotonic', 'sigmoid'],
                       help="Calibration method")
    parser.add_argument("--sample_fraction", type=float, default=0.1,
                       help="Fraction of pixels to sample from each image")
    parser.add_argument("--sam_checkpoint", type=str, default="sam3.pt")
    
    args = parser.parse_args()
    
    image_paths = [p.strip() for p in args.train_images.split(',')]
    mask_paths = [p.strip() for p in args.train_masks.split(',')]
    
    assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
    
    # Initialize detector
    detector = FastRustDetector(
        verbose=True,
        sam_checkpoint=args.sam_checkpoint,
    )
    
    # Collect training data
    print("\nCollecting training data...")
    raw_scores, labels = create_calibration_training_data(
        image_paths, mask_paths, detector, sample_fraction=args.sample_fraction
    )
    
    print(f"\nTotal samples: {len(labels)}")
    print(f"Rust ratio: {labels.mean():.3f}")
    
    # Train calibrator
    print(f"\nTraining calibrator (method={args.method})...")
    calibrator = RustProbabilityCalibrator(method=args.method)
    calibrator.fit(raw_scores, labels)
    
    # Evaluate
    ece = calibrator.compute_calibration_error(raw_scores, labels)
    print(f"\nCalibration Error (ECE):")
    print(f"  Raw: {ece['raw_ece']:.4f}")
    print(f"  Calibrated: {ece['calibrated_ece']:.4f}")
    print(f"  Improvement: {ece['improvement']:.4f}")
    
    # Save
    calibrator.save(args.calibrator_output)
    
    # Plot calibration curve
    fig = calibrator.plot_calibration_curve(
        raw_scores, labels,
        save_path=args.calibrator_output.replace('.pkl', '_curve.png'),
        title=f"Calibration Curve ({args.method})"
    )
    plt.show()
    
    print("\nDone! Use the calibrator with:")
    print(f"  python sam3-exp-2-stage.py --image <img> --calibrator_path {args.calibrator_output}")


if __name__ == "__main__":
    import sys
    if '--train_calibrator' in sys.argv:
        sys.argv.remove('--train_calibrator')
        train_calibrator_cli()
    else:
        main()
