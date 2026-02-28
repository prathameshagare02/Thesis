"""
Smart Rust Analyzer - Professional Edition
A modern, presentation-ready application for rust detection and growth prediction
using SAM (Segment Anything Model) and unsupervised machine learning.

Author: MSc Thesis Project
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Dict, Any
import threading

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from scipy import ndimage
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from ultralytics import SAM

# Check if LBP is available for texture features
try:
    from skimage.feature import local_binary_pattern as _lbp_check
    LBP_AVAILABLE = True
    del _lbp_check
except ImportError:
    LBP_AVAILABLE = False


# ==========================================
# THEME & STYLING CONFIGURATION
# ==========================================

class Theme:
    """Modern dark theme color palette"""
    # Primary colors
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    BG_CARD = "#1f2940"

    # Accent colors
    ACCENT_PRIMARY = "#e94560"
    ACCENT_SECONDARY = "#0ea5e9"
    ACCENT_SUCCESS = "#10b981"
    ACCENT_WARNING = "#f59e0b"
    ACCENT_INFO = "#8b5cf6"

    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_MUTED = "#64748b"

    # Status colors
    STATUS_READY = "#10b981"
    STATUS_PROCESSING = "#f59e0b"
    STATUS_ERROR = "#ef4444"

    # Gradients (for reference)
    GRADIENT_START = "#e94560"
    GRADIENT_END = "#0ea5e9"

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 24, "bold")
    FONT_SUBTITLE = (FONT_FAMILY, 14, "bold")
    FONT_HEADING = (FONT_FAMILY, 12, "bold")
    FONT_BODY = (FONT_FAMILY, 11)
    FONT_SMALL = (FONT_FAMILY, 10)
    FONT_MONO = ("Consolas", 10)


def configure_styles():
    """Configure ttk styles for modern appearance"""
    style = ttk.Style()

    # Try to use clam theme as base (works well for customization)
    try:
        style.theme_use('clam')
    except:
        pass

    # Configure main styles
    style.configure(".",
        background=Theme.BG_DARK,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_BODY
    )

    # Frame styles
    style.configure("Card.TFrame",
        background=Theme.BG_CARD,
        relief="flat"
    )

    style.configure("Dark.TFrame",
        background=Theme.BG_DARK
    )

    # Label styles
    style.configure("Title.TLabel",
        background=Theme.BG_DARK,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_TITLE
    )

    style.configure("Subtitle.TLabel",
        background=Theme.BG_DARK,
        foreground=Theme.TEXT_SECONDARY,
        font=Theme.FONT_SUBTITLE
    )

    style.configure("Heading.TLabel",
        background=Theme.BG_CARD,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_HEADING
    )

    style.configure("Body.TLabel",
        background=Theme.BG_CARD,
        foreground=Theme.TEXT_SECONDARY,
        font=Theme.FONT_BODY
    )

    style.configure("Status.TLabel",
        background=Theme.BG_DARK,
        foreground=Theme.ACCENT_SUCCESS,
        font=Theme.FONT_SMALL
    )

    style.configure("Info.TLabel",
        background=Theme.BG_CARD,
        foreground=Theme.ACCENT_INFO,
        font=Theme.FONT_BODY
    )

    # Button styles
    style.configure("Primary.TButton",
        background=Theme.ACCENT_PRIMARY,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_HEADING,
        padding=(20, 12)
    )
    style.map("Primary.TButton",
        background=[("active", "#d63d56"), ("disabled", Theme.BG_LIGHT)]
    )

    style.configure("Secondary.TButton",
        background=Theme.BG_LIGHT,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_BODY,
        padding=(15, 8)
    )
    style.map("Secondary.TButton",
        background=[("active", Theme.ACCENT_SECONDARY)]
    )

    style.configure("Action.TButton",
        background=Theme.ACCENT_SUCCESS,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_HEADING,
        padding=(30, 15)
    )
    style.map("Action.TButton",
        background=[("active", "#0d9668"), ("disabled", Theme.BG_LIGHT)]
    )

    # Notebook styles
    style.configure("Custom.TNotebook",
        background=Theme.BG_DARK,
        borderwidth=0,
        tabmargins=[0, 0, 0, 0]
    )

    style.configure("Custom.TNotebook.Tab",
        background=Theme.BG_MEDIUM,
        foreground=Theme.TEXT_SECONDARY,
        font=Theme.FONT_HEADING,
        padding=[25, 12],
        borderwidth=0
    )
    style.map("Custom.TNotebook.Tab",
        background=[("selected", Theme.BG_CARD)],
        foreground=[("selected", Theme.TEXT_PRIMARY)]
    )

    # Progress bar
    style.configure("Custom.Horizontal.TProgressbar",
        background=Theme.ACCENT_PRIMARY,
        troughcolor=Theme.BG_LIGHT,
        borderwidth=0,
        lightcolor=Theme.ACCENT_PRIMARY,
        darkcolor=Theme.ACCENT_PRIMARY
    )

    # LabelFrame
    style.configure("Card.TLabelframe",
        background=Theme.BG_CARD,
        foreground=Theme.TEXT_PRIMARY,
        font=Theme.FONT_HEADING,
        borderwidth=2,
        relief="flat"
    )
    style.configure("Card.TLabelframe.Label",
        background=Theme.BG_CARD,
        foreground=Theme.ACCENT_SECONDARY,
        font=Theme.FONT_HEADING
    )


# ==========================================
# PART 1: SAM SEGMENTATION ENGINE (Improved)
# ==========================================

class SAMCropper:
    """
    Improved SAM-based metal segmentation with:
    - Metal region pre-detection for better guidance
    - Morphological cleanup on masks
    - Mask-based cropping (not just bounding box)
    - Metal mask output for rust filtering
    """

    def __init__(self):
        self.model = None
        self.predictor = None
        self.loaded = False

        # Morphological parameters
        self.kernel_size = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 2

        # Store last mask for reuse
        self.last_mask = None
        self.last_metal_mask_crop = None

    def load_model(self, callback=None):
        if not self.loaded:
            if callback:
                callback("Loading SAM Model...")
            try:
                self.model = SAM("sam2.1_b.pt")
            except:
                if callback:
                    callback("Fallback to MobileSAM...")
                self.model = SAM("mobile_sam.pt")
            self.loaded = True
            if callback:
                callback("SAM Model Ready")

    def detect_metal_regions(self, image):
        """
        Pre-detect potential metal regions using classical CV.
        This helps guide SAM for better segmentation.

        Uses multiple detection methods:
        1. Edge detection (Canny)
        2. LAB color space - A channel for metallic colors
        3. HSV saturation/value for reflective surfaces
        """
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

            # Edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # LAB A-channel for metal-like colors (neutral to slightly red)
            _, A, _ = cv2.split(lab)
            metal_mask1 = cv2.inRange(A, 115, 145)

            # HSV: low-medium saturation, medium-high value (metallic/reflective)
            _, S, V = cv2.split(hsv)
            metal_mask2 = cv2.inRange(S, 20, 120) & cv2.inRange(V, 80, 255)

            # Combine all detection methods
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)

            combined = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined = cv2.bitwise_or(combined, edges_dilated)

            # Morphological cleanup
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

            # Find contours and filter by area
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            min_area = 500  # Minimum contour area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        "bbox": (x, y, x + w, y + h),
                        "center": (x + w // 2, y + h // 2),
                        "area": area,
                    })

            # Sort by area (largest first)
            regions.sort(key=lambda r: r["area"], reverse=True)
            return regions

        except Exception as e:
            print(f"Metal detection error: {e}")
            return []

    def apply_morphological_cleanup(self, mask):
        """
        Clean up SAM mask with morphological operations.
        Removes noise and fills small holes.
        """
        if mask is None:
            return None

        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.max() > 1:
            mask_uint8 = (mask_uint8 > 127).astype(np.uint8)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        # Erosion removes small noise
        if self.erosion_iterations > 0:
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations)

        # Dilation fills small holes and smooths edges
        if self.dilation_iterations > 0:
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=self.dilation_iterations)

        # Additional closing to fill remaining holes
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        return mask_uint8 * 255

    def predict_mask(self, image, point_coords):
        """
        Generates a mask based on a single click point.
        Includes morphological cleanup for better results.
        """
        if not self.loaded:
            self.load_model()

        results = self.model.predict(
            image, points=[point_coords], labels=[1], verbose=False
        )

        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Apply morphological cleanup
            cleaned_mask = self.apply_morphological_cleanup(mask_uint8)

            self.last_mask = cleaned_mask
            return cleaned_mask

        return None

    def predict_mask_multi_point(self, image, points, labels=None):
        """
        Generates a mask based on multiple click points.
        Useful for complex metal shapes.

        Args:
            image: RGB image
            points: List of [x, y] coordinates
            labels: List of labels (1=foreground, 0=background). Default all foreground.
        """
        if not self.loaded:
            self.load_model()

        if labels is None:
            labels = [1] * len(points)

        results = self.model.predict(
            image, points=points, labels=labels, verbose=False
        )

        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Apply morphological cleanup
            cleaned_mask = self.apply_morphological_cleanup(mask_uint8)

            self.last_mask = cleaned_mask
            return cleaned_mask

        return None

    def apply_crop(self, image, mask):
        """
        Improved cropping that:
        1. Uses mask extent (not just bounding box)
        2. Applies mask to image (blacks out background)
        3. Returns both cropped image and cropped metal mask

        Returns:
            - cropped_image: Image with background blacked out
            - full_mask: Original full-size mask
            - bbox: (x, y, w, h) crop coordinates
            - metal_mask_crop: Cropped metal mask for rust filtering
        """
        if mask is None or not np.any(mask):
            # Fallback to full image
            return image.copy(), None, (0, 0, image.shape[1], image.shape[0]), None

        # Find mask extent using numpy (more accurate than findNonZero)
        mask_binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask

        y_idx, x_idx = np.where(mask_binary > 0)

        if len(y_idx) == 0:
            return image.copy(), mask, (0, 0, image.shape[1], image.shape[0]), None

        y_min, y_max = y_idx.min(), y_idx.max()
        x_min, x_max = x_idx.min(), x_idx.max()

        # Add padding
        pad = 20
        h_img, w_img = image.shape[:2]
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w_img, x_max + pad)
        y_max = min(h_img, y_max + pad)

        # Calculate crop dimensions
        x, y = x_min, y_min
        w, h = x_max - x_min, y_max - y_min

        # Crop image and mask
        cropped_img = image[y:y+h, x:x+w].copy()
        metal_mask_crop = mask_binary[y:y+h, x:x+w].copy()

        # Apply mask to cropped image (black out background)
        cropped_img[metal_mask_crop == 0] = [0, 0, 0]

        # Store for later use
        self.last_metal_mask_crop = metal_mask_crop

        return cropped_img, mask, (x, y, w, h), metal_mask_crop

    def get_suggested_click_point(self, image):
        """
        Get a suggested click point based on metal region detection.
        Returns the center of the largest detected metal region.
        """
        regions = self.detect_metal_regions(image)
        if regions:
            # Return center of largest region
            return regions[0]["center"]
        else:
            # Fallback to image center
            h, w = image.shape[:2]
            return (w // 2, h // 2)


# ==========================================
# PART 2: RUST DETECTION ENGINE (Improved)
# ==========================================

class FastRustDetector:
    """
    Improved unsupervised rust detector with vectorized operations.
    """

    FEATURE_NAMES = [
        "mean_L", "mean_a", "mean_b",
        "std_L", "std_a", "std_b",
        "roughness", "entropy",
        "gradient_mean", "gradient_std",
        "chroma", "redness_ratio",
    ]

    def __init__(self, n_segments=250, n_clusters=3, fast_mode=False, verbose=False):
        self.n_segments = n_segments
        self.n_clusters = n_clusters
        self.fast_mode = fast_mode
        self.verbose = verbose
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _log(self, message, progress=None):
        """Log message and update progress."""
        if self.verbose:
            print(f"  → {message}")
        if self.progress_callback and progress is not None:
            self.progress_callback(message, progress)

    def _compute_feature_maps(self, crop):
        """Precompute all feature maps once (vectorized)."""
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)

        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        local_variance = local_sq_mean - local_mean**2
        entropy_proxy = np.sqrt(np.maximum(local_variance, 0))

        if LBP_AVAILABLE and not self.fast_mode:
            from skimage.feature import local_binary_pattern as lbp_func
            lbp = lbp_func(gray.astype(np.uint8), P=8, R=1, method="uniform").astype(np.float32)
        else:
            lbp = np.zeros_like(gray)

        a_centered = lab[:, :, 1] - 128
        b_centered = lab[:, :, 2] - 128
        chroma = np.sqrt(a_centered**2 + b_centered**2)
        redness_ratio = a_centered / (lab[:, :, 0] + 1e-6)

        return {
            "L": lab[:, :, 0], "a": lab[:, :, 1], "b": lab[:, :, 2],
            "gray": gray, "gradient": gradient, "entropy": entropy_proxy,
            "lbp": lbp, "chroma": chroma, "redness_ratio": redness_ratio,
        }

    def _extract_features_vectorized(self, feature_maps, segments, crop, metal_mask=None):
        """
        Extract per-segment features using vectorized scipy operations.
        Now includes metal mask filtering for better accuracy.
        """
        unique_segments = np.unique(segments)
        n_segments = len(unique_segments)
        n_features = 12
        features = np.zeros((n_segments, n_features), dtype=np.float32)

        def safe_mean(vals):
            return np.mean(vals) if len(vals) > 0 else 0.0

        def safe_std(vals):
            return np.std(vals) if len(vals) > 0 else 0.0

        features[:, 0] = ndimage.labeled_comprehension(feature_maps["L"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 1] = ndimage.labeled_comprehension(feature_maps["a"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 2] = ndimage.labeled_comprehension(feature_maps["b"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 3] = ndimage.labeled_comprehension(feature_maps["L"], segments, unique_segments, safe_std, np.float32, 0)
        features[:, 4] = ndimage.labeled_comprehension(feature_maps["a"], segments, unique_segments, safe_std, np.float32, 0)
        features[:, 5] = ndimage.labeled_comprehension(feature_maps["b"], segments, unique_segments, safe_std, np.float32, 0)
        features[:, 6] = ndimage.labeled_comprehension(feature_maps["gray"], segments, unique_segments, safe_std, np.float32, 0)
        features[:, 7] = ndimage.labeled_comprehension(feature_maps["entropy"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 8] = ndimage.labeled_comprehension(feature_maps["gradient"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 9] = ndimage.labeled_comprehension(feature_maps["gradient"], segments, unique_segments, safe_std, np.float32, 0)
        features[:, 10] = ndimage.labeled_comprehension(feature_maps["chroma"], segments, unique_segments, safe_mean, np.float32, 0)
        features[:, 11] = ndimage.labeled_comprehension(feature_maps["redness_ratio"], segments, unique_segments, safe_mean, np.float32, 0)

        # Compute metal score per segment if metal mask is available
        if metal_mask is not None:
            # Ensure metal_mask has the same shape as segments
            if metal_mask.shape != segments.shape:
                # Resize metal_mask to match segments shape
                metal_mask_resized = cv2.resize(
                    metal_mask.astype(np.uint8),
                    (segments.shape[1], segments.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)
            else:
                metal_mask_resized = metal_mask.astype(np.float32)

            # Normalize to 0-1 range if needed
            if metal_mask_resized.max() > 1:
                metal_mask_resized = metal_mask_resized / 255.0

            metal_scores = ndimage.labeled_comprehension(
                metal_mask_resized, segments, unique_segments, safe_mean, np.float32, 0
            )
            # Valid if segment is mostly on metal (>50%) and not black background
            valid_mask = (metal_scores > 0.5) & (features[:, 0] > 5)
        else:
            # Fallback: valid if L channel > 5 (not black background)
            valid_mask = features[:, 0] > 5

        return features, unique_segments, valid_mask

    def _identify_rust_cluster(self, features, labels):
        """Identify which cluster corresponds to rust."""
        unique_labels = np.unique(labels)
        cluster_scores = {}

        for label in unique_labels:
            if label == -1:
                cluster_scores[label] = -1.0
                continue

            mask = labels == label
            cluster_features = features[mask]

            if len(cluster_features) == 0:
                cluster_scores[label] = 0.0
                continue

            mean_a = np.mean(cluster_features[:, 1])
            mean_b = np.mean(cluster_features[:, 2])
            mean_roughness = np.mean(cluster_features[:, 6])
            mean_entropy = np.mean(cluster_features[:, 7])
            mean_chroma = np.mean(cluster_features[:, 10])
            mean_redness = np.mean(cluster_features[:, 11])

            redness_score = max(0, (mean_a - 128) / 127)
            yellowness_score = max(0, (mean_b - 128) / 127)
            roughness_score = min(1, mean_roughness / 50)
            entropy_score = min(1, mean_entropy / 30)
            chroma_score = min(1, mean_chroma / 50)
            redness_ratio_score = max(0, min(1, mean_redness / 0.5))

            rust_score = (
                0.25 * redness_score + 0.15 * yellowness_score +
                0.15 * roughness_score + 0.10 * entropy_score +
                0.20 * chroma_score + 0.15 * redness_ratio_score
            )
            cluster_scores[label] = float(rust_score)

        valid_scores = {k: v for k, v in cluster_scores.items() if v >= 0}
        rust_cluster = max(valid_scores.keys(), key=lambda k: valid_scores[k]) if valid_scores else 0

        return rust_cluster, cluster_scores

    def analyze(self, crop_image, metal_mask=None):
        """
        Run the full analysis pipeline.

        Args:
            crop_image: RGB image (cropped from SAM)
            metal_mask: Optional metal mask to filter rust detection to metal only
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Validate and resize metal_mask to match crop_image dimensions
        if metal_mask is not None:
            crop_h, crop_w = crop_image.shape[:2]
            mask_h, mask_w = metal_mask.shape[:2]

            if (mask_h, mask_w) != (crop_h, crop_w):
                self._log(f"Resizing metal_mask from {metal_mask.shape} to {crop_image.shape[:2]}", 5)
                metal_mask = cv2.resize(
                    metal_mask.astype(np.uint8),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

            # Ensure binary mask (0 or 1)
            if metal_mask.max() > 1:
                metal_mask = (metal_mask > 127).astype(np.uint8)

        self._log("SLIC segmentation...", 10)
        segments = slic(
            img_as_float(crop_image),
            n_segments=self.n_segments,
            compactness=20, sigma=1, start_label=0, channel_axis=2,
        )

        self._log("Computing feature maps...", 30)
        feature_maps = self._compute_feature_maps(crop_image)

        # Add metal mask to feature maps for filtering
        if metal_mask is not None:
            feature_maps["metal_mask"] = metal_mask.astype(np.float32)

        self._log("Extracting features...", 50)
        features, segment_ids, valid_mask = self._extract_features_vectorized(
            feature_maps, segments, crop_image, metal_mask
        )

        valid_features = features[valid_mask]
        valid_segment_ids = segment_ids[valid_mask]

        if len(valid_features) < self.n_clusters:
            self._log("Warning: Not enough valid segments", 100)
            return {
                "crop": crop_image, "segments": segments,
                "rust_mask": np.zeros(crop_image.shape[:2], dtype=np.uint8),
                "cluster_map": np.full(crop_image.shape[:2], -1, dtype=np.int32),
                "rust_cluster": 0, "pct": 0.0, "features": features,
                "labels": np.zeros(len(features)), "seg_ids": segment_ids,
                "cluster_scores": {}, "metal_mask": metal_mask,
            }

        self._log("Clustering analysis...", 70)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(valid_features)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        self._log("Identifying rust regions...", 85)
        rust_cluster, cluster_scores = self._identify_rust_cluster(valid_features, labels)

        rust_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
        cluster_vis_map = np.full(crop_image.shape[:2], -1, dtype=np.int32)

        for idx, sid in enumerate(valid_segment_ids):
            m = segments == sid
            cluster_vis_map[m] = labels[idx]
            if labels[idx] == rust_cluster:
                rust_mask[m] = 255

        self._log("Finalizing mask...", 95)
        kernel = np.ones((3, 3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)

        # Apply metal mask to ensure rust is only detected on metal
        if metal_mask is not None:
            # Ensure metal_mask has the same shape as rust_mask
            if metal_mask.shape != rust_mask.shape:
                metal_mask_resized = cv2.resize(
                    metal_mask.astype(np.uint8),
                    (rust_mask.shape[1], rust_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                metal_mask_resized = metal_mask.astype(np.uint8)

            rust_mask = rust_mask * (metal_mask_resized > 0).astype(np.uint8)

        # Calculate percentage based on metal area (not total image)
        if metal_mask is not None:
            if metal_mask.shape != crop_image.shape[:2]:
                metal_mask_for_area = cv2.resize(
                    metal_mask.astype(np.uint8),
                    (crop_image.shape[1], crop_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                metal_mask_for_area = metal_mask
            metal_area = np.sum(metal_mask_for_area > 0)
        else:
            metal_area = np.sum(np.any(crop_image > 5, axis=2))

        rust_area = np.sum(rust_mask > 0)
        pct = (rust_area / metal_area) * 100 if metal_area > 0 else 0

        self._log(f"Complete! Rust: {pct:.1f}%", 100)

        full_labels = np.full(len(segment_ids), -1)
        for idx, sid in enumerate(valid_segment_ids):
            orig_idx = np.where(segment_ids == sid)[0]
            if len(orig_idx) > 0:
                full_labels[orig_idx[0]] = labels[idx]

        return {
            "crop": crop_image, "segments": segments, "rust_mask": rust_mask,
            "cluster_map": cluster_vis_map, "rust_cluster": rust_cluster,
            "pct": pct, "features": features, "labels": full_labels,
            "seg_ids": segment_ids, "cluster_scores": cluster_scores,
            "metal_mask": metal_mask,
        }


# ==========================================
# PART 3: GROWTH ENGINE
# ==========================================

class GrowthEngine:
    def align(self, img_ref, img_mov):
        """Align img_mov to img_ref using ORB."""
        gray1 = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img_mov, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(5000)
        k1, d1 = orb.detectAndCompute(gray1, None)
        k2, d2 = orb.detectAndCompute(gray2, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if d1 is None or d2 is None:
            return img_mov
        matches = sorted(matcher.match(d1, d2), key=lambda x: x.distance)
        good = matches[: int(len(matches) * 0.2)]

        if len(good) < 4:
            return img_mov

        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

        h, w = img_ref.shape[:2]
        return cv2.warpPerspective(img_mov, H, (w, h))

    def predict(self, mask_t1, mask_t2):
        """Predict future rust growth."""
        t1_bool = mask_t1 > 0
        t2_bool = mask_t2 > 0
        delta = (t2_bool & ~t1_bool).astype(np.float32)

        dist = cv2.distanceTransform((~t2_bool).astype(np.uint8), cv2.DIST_L2, 5)
        risk = 1.0 / (dist + 2.0)
        risk = cv2.normalize(risk, None, 0, 1, cv2.NORM_MINMAX)

        growth_rate = np.sum(delta) / (np.sum(t2_bool) + 1)
        momentum = growth_rate * 50.0

        future = risk * momentum
        future = cv2.GaussianBlur(future, (21, 21), 0)

        heatmap = t2_bool.astype(np.float32) + future
        return delta, np.clip(heatmap, 0, 1)


# ==========================================
# PART 4: CUSTOM WIDGETS
# ==========================================

class ModernCard(tk.Frame):
    """A modern card-style container with rounded corners effect."""

    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, **kwargs)

        # Title bar
        if title:
            title_frame = tk.Frame(self, bg=Theme.BG_CARD)
            title_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

            tk.Label(
                title_frame, text=title,
                bg=Theme.BG_CARD, fg=Theme.ACCENT_SECONDARY,
                font=Theme.FONT_HEADING
            ).pack(side=tk.LEFT)

        # Content area
        self.content = tk.Frame(self, bg=Theme.BG_CARD)
        self.content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)


class ImagePanel(tk.Frame):
    """Custom image display panel with status indicator."""

    def __init__(self, parent, title="Image", size=(350, 280)):
        super().__init__(parent, bg=Theme.BG_CARD)
        self.size = size
        self.current_image = None

        # Title
        title_frame = tk.Frame(self, bg=Theme.BG_CARD)
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            title_frame, text=title,
            bg=Theme.BG_CARD, fg=Theme.ACCENT_SECONDARY,
            font=Theme.FONT_HEADING
        ).pack(side=tk.LEFT)

        self.status_indicator = tk.Label(
            title_frame, text="○",
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            font=("Segoe UI", 14)
        )
        self.status_indicator.pack(side=tk.RIGHT)

        # Canvas for image
        self.canvas = tk.Canvas(
            self, width=size[0], height=size[1],
            bg=Theme.BG_MEDIUM, highlightthickness=2,
            highlightbackground=Theme.BG_LIGHT,
            cursor="crosshair"
        )
        self.canvas.pack(padx=10, pady=5)

        # Placeholder text
        self.canvas.create_text(
            size[0]//2, size[1]//2,
            text="Click to load image\nor drag & drop",
            fill=Theme.TEXT_MUTED,
            font=Theme.FONT_BODY,
            justify=tk.CENTER,
            tags="placeholder"
        )

        # Status label
        self.status_label = tk.Label(
            self, text="No image loaded",
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SMALL
        )
        self.status_label.pack(pady=(5, 10))

    def set_status(self, status, color=None):
        """Update status indicator and label."""
        self.status_label.config(text=status)
        if color:
            self.status_indicator.config(fg=color)
            self.status_label.config(fg=color)

    def display_image(self, img_arr):
        """Display an image on the canvas."""
        self.canvas.delete("all")

        h, w = img_arr.shape[:2]
        cw, ch = self.size
        ratio = min(cw / w, ch / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        pil_img = Image.fromarray(img_arr)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.current_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(cw//2, ch//2, image=self.current_image)

        return ratio, (cw//2 - new_w//2), (ch//2 - new_h//2)


class ProgressPanel(tk.Frame):
    """Animated progress indicator panel."""

    def __init__(self, parent):
        super().__init__(parent, bg=Theme.BG_DARK)

        # Progress bar container
        self.progress_frame = tk.Frame(self, bg=Theme.BG_DARK)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=10)

        # Status text
        self.status_text = tk.Label(
            self.progress_frame,
            text="Ready",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY,
            font=Theme.FONT_BODY
        )
        self.status_text.pack(anchor=tk.W)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.progress_frame,
            style="Custom.Horizontal.TProgressbar",
            mode='determinate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=(5, 0))

        # Percentage label
        self.percent_label = tk.Label(
            self.progress_frame,
            text="0%",
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT_PRIMARY,
            font=Theme.FONT_SMALL
        )
        self.percent_label.pack(anchor=tk.E)

    def update(self, message, progress):
        """Update progress bar and message."""
        self.status_text.config(text=message)
        self.progress['value'] = progress
        self.percent_label.config(text=f"{int(progress)}%")
        self.update_idletasks()

    def reset(self):
        """Reset progress bar."""
        self.progress['value'] = 0
        self.status_text.config(text="Ready")
        self.percent_label.config(text="0%")


class StatCard(tk.Frame):
    """Statistics display card."""

    def __init__(self, parent, title, value="--", unit="%", color=Theme.ACCENT_PRIMARY):
        super().__init__(parent, bg=Theme.BG_CARD, padx=20, pady=15)

        tk.Label(
            self, text=title,
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SMALL
        ).pack(anchor=tk.W)

        self.value_label = tk.Label(
            self, text=value,
            bg=Theme.BG_CARD, fg=color,
            font=(Theme.FONT_FAMILY, 28, "bold")
        )
        self.value_label.pack(anchor=tk.W)

        self.unit_label = tk.Label(
            self, text=unit,
            bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
            font=Theme.FONT_BODY
        )
        self.unit_label.pack(anchor=tk.W)

    def set_value(self, value):
        """Update the displayed value."""
        self.value_label.config(text=value)


# ==========================================
# PART 5: MAIN APPLICATION
# ==========================================

class RustAnalyzerApp:
    """Main application class with modern professional UI."""

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Rust Analyzer Pro")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=Theme.BG_DARK)

        # Configure styles
        configure_styles()

        # Initialize components
        self.sam = SAMCropper()
        self.detector = FastRustDetector(n_segments=250, n_clusters=3, verbose=True)
        self.growth = GrowthEngine()

        # State data
        self.data = {
            "t1": {"raw": None, "crop": None, "mask": None, "res": None, "scale": 1, "offset_x": 0, "offset_y": 0},
            "t2": {"raw": None, "crop": None, "mask": None, "res": None, "scale": 1, "offset_x": 0, "offset_y": 0},
        }

        self.is_processing = False

        # Build UI
        self._create_header()
        self._create_main_content()
        self._create_footer()

    def _create_header(self):
        """Create the application header."""
        header = tk.Frame(self.root, bg=Theme.BG_DARK, height=80)
        header.pack(fill=tk.X, padx=30, pady=(20, 10))
        header.pack_propagate(False)

        # Logo/Title area
        title_frame = tk.Frame(header, bg=Theme.BG_DARK)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            title_frame,
            text="🔬 Smart Rust Analyzer",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_TITLE
        ).pack(anchor=tk.W)

        tk.Label(
            title_frame,
            text="SAM-powered unsupervised corrosion detection & growth prediction",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY,
            font=Theme.FONT_BODY
        ).pack(anchor=tk.W)

        # Status indicators
        status_frame = tk.Frame(header, bg=Theme.BG_DARK)
        status_frame.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        self.model_status = tk.Label(
            status_frame,
            text="● SAM Model: Not Loaded",
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT_WARNING,
            font=Theme.FONT_SMALL
        )
        self.model_status.pack(anchor=tk.E)

        self.gpu_status = tk.Label(
            status_frame,
            text="● GPU: Checking...",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SMALL
        )
        self.gpu_status.pack(anchor=tk.E)
        self._check_gpu_status()

    def _check_gpu_status(self):
        """Check and display GPU status."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_status.config(text=f"● GPU: {gpu_name}", fg=Theme.ACCENT_SUCCESS)
            else:
                self.gpu_status.config(text="● GPU: CPU Mode", fg=Theme.ACCENT_WARNING)
        except:
            self.gpu_status.config(text="● GPU: CPU Mode", fg=Theme.ACCENT_WARNING)

    def _create_main_content(self):
        """Create the main content area with notebook."""
        # Main container
        main_frame = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Create notebook
        self.notebook = ttk.Notebook(main_frame, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.tab_input = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.tab_analysis = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.tab_comparison = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.tab_prediction = tk.Frame(self.notebook, bg=Theme.BG_DARK)

        self.notebook.add(self.tab_input, text="  📁 Input & Segmentation  ")
        self.notebook.add(self.tab_analysis, text="  🔍 Rust Analysis  ")
        self.notebook.add(self.tab_comparison, text="  📊 T1 vs T2 Comparison  ")
        self.notebook.add(self.tab_prediction, text="  📈 Growth Prediction  ")

        # Setup each tab
        self._setup_input_tab()
        self._setup_analysis_tab()
        self._setup_comparison_tab()
        self._setup_prediction_tab()

    def _setup_input_tab(self):
        """Setup the input and segmentation tab."""
        # Instructions banner
        banner = tk.Frame(self.tab_input, bg=Theme.BG_LIGHT, height=50)
        banner.pack(fill=tk.X, pady=(0, 15))
        banner.pack_propagate(False)

        tk.Label(
            banner,
            text="📌 Step 1: Load images and click on the metal surface to extract it using SAM",
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_BODY
        ).pack(expand=True)

        # Main content - two columns
        content = tk.Frame(self.tab_input, bg=Theme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True)

        # T1 Panel
        t1_frame = tk.Frame(content, bg=Theme.BG_CARD)
        t1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        t1_header = tk.Frame(t1_frame, bg=Theme.BG_CARD)
        t1_header.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            t1_header,
            text="T1 - Initial State (24h)",
            bg=Theme.BG_CARD,
            fg=Theme.ACCENT_SECONDARY,
            font=Theme.FONT_SUBTITLE
        ).pack(side=tk.LEFT)

        self.btn_load_t1 = tk.Button(
            t1_header,
            text="📂 Load Image",
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_BODY,
            relief=tk.FLAT,
            padx=15, pady=5,
            cursor="hand2",
            command=lambda: self.load_image("t1")
        )
        self.btn_load_t1.pack(side=tk.RIGHT)

        self.panel_t1 = ImagePanel(t1_frame, "Click on metal to segment", size=(380, 320))
        self.panel_t1.pack(padx=15, pady=(0, 15))
        self.panel_t1.canvas.bind("<Button-1>", lambda e: self.on_canvas_click(e, "t1"))

        # T2 Panel
        t2_frame = tk.Frame(content, bg=Theme.BG_CARD)
        t2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        t2_header = tk.Frame(t2_frame, bg=Theme.BG_CARD)
        t2_header.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            t2_header,
            text="T2 - Later State (48h)",
            bg=Theme.BG_CARD,
            fg=Theme.ACCENT_INFO,
            font=Theme.FONT_SUBTITLE
        ).pack(side=tk.LEFT)

        self.btn_load_t2 = tk.Button(
            t2_header,
            text="📂 Load Image",
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_BODY,
            relief=tk.FLAT,
            padx=15, pady=5,
            cursor="hand2",
            command=lambda: self.load_image("t2")
        )
        self.btn_load_t2.pack(side=tk.RIGHT)

        self.panel_t2 = ImagePanel(t2_frame, "Click on metal to segment", size=(380, 320))
        self.panel_t2.pack(padx=15, pady=(0, 15))
        self.panel_t2.canvas.bind("<Button-1>", lambda e: self.on_canvas_click(e, "t2"))

        # Bottom action bar
        action_bar = tk.Frame(self.tab_input, bg=Theme.BG_DARK)
        action_bar.pack(fill=tk.X, pady=15)

        self.progress_panel = ProgressPanel(action_bar)
        self.progress_panel.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_analyze = tk.Button(
            action_bar,
            text="▶  RUN ANALYSIS",
            bg=Theme.ACCENT_PRIMARY,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_HEADING,
            relief=tk.FLAT,
            padx=30, pady=12,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.run_analysis
        )
        self.btn_analyze.pack(side=tk.RIGHT, padx=20)

    def _setup_analysis_tab(self):
        """Setup the detailed analysis tab."""
        # This tab will show the 6-panel analysis for selected timepoint

        # Control bar
        control_bar = tk.Frame(self.tab_analysis, bg=Theme.BG_LIGHT, height=50)
        control_bar.pack(fill=tk.X)
        control_bar.pack_propagate(False)

        tk.Label(
            control_bar,
            text="Select Timepoint:",
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            font=Theme.FONT_BODY
        ).pack(side=tk.LEFT, padx=20)

        self.analysis_var = tk.StringVar(value="t1")

        for val, text in [("t1", "T1 (24h)"), ("t2", "T2 (48h)")]:
            rb = tk.Radiobutton(
                control_bar,
                text=text,
                variable=self.analysis_var,
                value=val,
                bg=Theme.BG_LIGHT,
                fg=Theme.TEXT_PRIMARY,
                selectcolor=Theme.BG_MEDIUM,
                activebackground=Theme.BG_LIGHT,
                font=Theme.FONT_BODY,
                command=self._update_analysis_view
            )
            rb.pack(side=tk.LEFT, padx=10)

        # Analysis canvas
        self.analysis_canvas_frame = tk.Frame(self.tab_analysis, bg=Theme.BG_DARK)
        self.analysis_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Placeholder
        tk.Label(
            self.analysis_canvas_frame,
            text="Run analysis to see results",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SUBTITLE
        ).pack(expand=True)

    def _setup_comparison_tab(self):
        """Setup the T1 vs T2 comparison tab."""
        self.comparison_frame = tk.Frame(self.tab_comparison, bg=Theme.BG_DARK)
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.comparison_frame,
            text="Run analysis to see comparison",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SUBTITLE
        ).pack(expand=True)

    def _setup_prediction_tab(self):
        """Setup the growth prediction tab."""
        self.prediction_frame = tk.Frame(self.tab_prediction, bg=Theme.BG_DARK)
        self.prediction_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.prediction_frame,
            text="Run analysis to see predictions",
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SUBTITLE
        ).pack(expand=True)

    def _create_footer(self):
        """Create the application footer."""
        footer = tk.Frame(self.root, bg=Theme.BG_MEDIUM, height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="Smart Rust Analyzer Pro v2.0 | MSc Thesis Project | Powered by SAM & scikit-learn",
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_MUTED,
            font=Theme.FONT_SMALL
        ).pack(expand=True)

    # ==========================================
    # EVENT HANDLERS
    # ==========================================

    def load_image(self, key):
        """Load an image for the specified timepoint."""
        path = filedialog.askopenfilename(
            title=f"Select {key.upper()} Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        # Load image
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Store and display
        self.data[key]["raw"] = img
        self.data[key]["path"] = path

        panel = self.panel_t1 if key == "t1" else self.panel_t2
        scale, off_x, off_y = panel.display_image(img)

        self.data[key]["scale"] = scale
        self.data[key]["offset_x"] = off_x
        self.data[key]["offset_y"] = off_y

        panel.set_status("Click on metal to segment", Theme.ACCENT_WARNING)

        # Load SAM model in background if not loaded
        if not self.sam.loaded:
            def load_sam():
                self.sam.load_model(lambda msg: self.root.after(0, lambda: self.model_status.config(
                    text=f"● SAM: {msg}",
                    fg=Theme.ACCENT_SUCCESS if "Ready" in msg else Theme.ACCENT_WARNING
                )))
            threading.Thread(target=load_sam, daemon=True).start()

    def on_canvas_click(self, event, key):
        """Handle canvas click for SAM segmentation."""
        if self.data[key]["raw"] is None:
            messagebox.showinfo("Info", "Please load an image first")
            return

        if self.is_processing:
            return

        # Convert canvas coords to image coords
        scale = self.data[key]["scale"]
        off_x = self.data[key]["offset_x"]
        off_y = self.data[key]["offset_y"]

        img_x = int((event.x - off_x) / scale)
        img_y = int((event.y - off_y) / scale)

        panel = self.panel_t1 if key == "t1" else self.panel_t2
        panel.set_status("Segmenting with SAM...", Theme.ACCENT_WARNING)
        self.root.update_idletasks()

        # Run SAM with improved cropping
        def run_sam():
            raw = self.data[key]["raw"]
            mask_uint8 = self.sam.predict_mask(raw, [img_x, img_y])

            if mask_uint8 is not None:
                # Use improved apply_crop that returns metal_mask_crop
                crop, bin_mask, bbox, metal_mask_crop = self.sam.apply_crop(raw, mask_uint8)
                self.data[key]["crop"] = crop
                self.data[key]["mask"] = bin_mask
                self.data[key]["bbox"] = bbox
                self.data[key]["metal_mask"] = metal_mask_crop

                # Update UI in main thread with metal mask
                self.root.after(0, lambda: self._update_after_sam(key, crop, metal_mask_crop))

        threading.Thread(target=run_sam, daemon=True).start()

    def _update_after_sam(self, key, crop, metal_mask=None):
        """Update UI after SAM segmentation."""
        panel = self.panel_t1 if key == "t1" else self.panel_t2
        panel.display_image(crop)
        panel.set_status("✓ Metal extracted successfully!", Theme.ACCENT_SUCCESS)
        panel.status_indicator.config(fg=Theme.ACCENT_SUCCESS)

        # Store metal mask for rust filtering
        if metal_mask is not None:
            self.data[key]["metal_mask"] = metal_mask

        # Check if ready to analyze
        if self.data["t1"]["crop"] is not None and self.data["t2"]["crop"] is not None:
            self.btn_analyze.config(state=tk.NORMAL, bg=Theme.ACCENT_SUCCESS)

    def run_analysis(self):
        """Run the full analysis pipeline."""
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_analyze.config(state=tk.DISABLED, text="⏳ Processing...")

        def analysis_thread():
            try:
                # Analyze T1 with metal mask
                self.root.after(0, lambda: self.progress_panel.update("Analyzing T1...", 10))
                self.detector.set_progress_callback(
                    lambda msg, prog: self.root.after(0, lambda: self.progress_panel.update(f"T1: {msg}", 10 + prog * 0.3))
                )
                # Pass metal mask if available
                t1_metal_mask = self.data["t1"].get("metal_mask")
                self.data["t1"]["res"] = self.detector.analyze(self.data["t1"]["crop"], t1_metal_mask)

                # Align T2
                self.root.after(0, lambda: self.progress_panel.update("Aligning T2 to T1...", 45))
                t2_aligned = self.growth.align(self.data["t1"]["crop"], self.data["t2"]["crop"])

                # Align metal mask too if available
                t2_metal_mask = self.data["t2"].get("metal_mask")
                if t2_metal_mask is not None:
                    # Convert to 3-channel for alignment, then back
                    t2_mask_3ch = np.stack([t2_metal_mask] * 3, axis=-1).astype(np.uint8) * 255
                    t2_mask_aligned = self.growth.align(self.data["t1"]["crop"], t2_mask_3ch)
                    t2_metal_mask = (t2_mask_aligned[:, :, 0] > 127).astype(np.uint8)

                # Analyze T2
                self.detector.set_progress_callback(
                    lambda msg, prog: self.root.after(0, lambda: self.progress_panel.update(f"T2: {msg}", 50 + prog * 0.3))
                )
                self.data["t2"]["res"] = self.detector.analyze(t2_aligned, t2_metal_mask)

                # Predict growth
                self.root.after(0, lambda: self.progress_panel.update("Predicting growth...", 85))
                delta, heatmap = self.growth.predict(
                    self.data["t1"]["res"]["rust_mask"],
                    self.data["t2"]["res"]["rust_mask"]
                )
                self.data["delta"] = delta
                self.data["heatmap"] = heatmap

                # Update UI
                self.root.after(0, lambda: self._analysis_complete())

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self._reset_after_error())

        threading.Thread(target=analysis_thread, daemon=True).start()

    def _analysis_complete(self):
        """Handle analysis completion."""
        self.is_processing = False
        self.progress_panel.update("Analysis complete!", 100)
        self.btn_analyze.config(state=tk.NORMAL, text="▶  RUN ANALYSIS", bg=Theme.ACCENT_SUCCESS)

        # Update all views
        self._update_analysis_view()
        self._update_comparison_view()
        self._update_prediction_view()

        # Switch to comparison tab
        self.notebook.select(self.tab_comparison)

        # Show summary
        t1_pct = self.data["t1"]["res"]["pct"]
        t2_pct = self.data["t2"]["res"]["pct"]
        growth = t2_pct - t1_pct

        messagebox.showinfo(
            "Analysis Complete",
            f"Results Summary:\n\n"
            f"T1 Rust Coverage: {t1_pct:.1f}%\n"
            f"T2 Rust Coverage: {t2_pct:.1f}%\n"
            f"Growth: {growth:+.1f}%\n\n"
            f"Check the tabs for detailed analysis."
        )

    def _reset_after_error(self):
        """Reset UI after an error."""
        self.is_processing = False
        self.progress_panel.reset()
        self.btn_analyze.config(state=tk.NORMAL, text="▶  RUN ANALYSIS", bg=Theme.ACCENT_PRIMARY)

    def _update_analysis_view(self):
        """Update the analysis tab with 6-panel view."""
        key = self.analysis_var.get()
        res = self.data.get(key, {}).get("res")

        if res is None:
            return

        # Clear previous
        for widget in self.analysis_canvas_frame.winfo_children():
            widget.destroy()

        # Create matplotlib figure
        fig = Figure(figsize=(14, 8), dpi=100, facecolor=Theme.BG_DARK)
        fig.patch.set_facecolor(Theme.BG_DARK)

        axs = fig.subplots(2, 3)

        for ax_row in axs:
            for ax in ax_row:
                ax.set_facecolor(Theme.BG_CARD)
                ax.tick_params(colors=Theme.TEXT_MUTED)
                for spine in ax.spines.values():
                    spine.set_color(Theme.BG_LIGHT)

        # 1. Original crop
        axs[0, 0].imshow(res["crop"])
        axs[0, 0].set_title("1. SAM Extracted Metal", color=Theme.TEXT_PRIMARY, fontsize=11)
        axs[0, 0].axis("off")

        # 2. SLIC segments
        vis_slic = mark_boundaries(res["crop"], res["segments"])
        axs[0, 1].imshow(vis_slic)
        axs[0, 1].set_title("2. SLIC Superpixels", color=Theme.TEXT_PRIMARY, fontsize=11)
        axs[0, 1].axis("off")

        # 3. LAB a* channel (redness)
        lab_img = cv2.cvtColor(res["crop"], cv2.COLOR_RGB2Lab)
        im = axs[0, 2].imshow(lab_img[:, :, 1], cmap="RdYlGn_r", vmin=0, vmax=255)
        axs[0, 2].set_title("3. LAB a* (Redness)", color=Theme.TEXT_PRIMARY, fontsize=11)
        axs[0, 2].axis("off")
        cbar = fig.colorbar(im, ax=axs[0, 2], fraction=0.046)
        cbar.ax.yaxis.set_tick_params(color=Theme.TEXT_MUTED)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=Theme.TEXT_MUTED)

        # 4. Cluster map
        unique_labels = np.unique(res["cluster_map"])
        c_viz = np.zeros((*res["crop"].shape[:2], 3))
        colors = [[0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]

        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            if label == res["rust_cluster"]:
                c_viz[res["cluster_map"] == label] = [1.0, 0.27, 0.38]  # Accent color
            else:
                c_viz[res["cluster_map"] == label] = colors[i % 3]

        axs[1, 0].imshow(c_viz)
        axs[1, 0].set_title(f"4. Clusters (Red = Rust)", color=Theme.TEXT_PRIMARY, fontsize=11)
        axs[1, 0].axis("off")

        # 5. Binary mask
        axs[1, 1].imshow(res["rust_mask"], cmap="gray")
        axs[1, 1].set_title("5. Binary Rust Mask", color=Theme.TEXT_PRIMARY, fontsize=11)
        axs[1, 1].axis("off")

        # 6. Final overlay
        final = res["crop"].copy()
        rust_overlay = np.zeros_like(final)
        rust_overlay[res["rust_mask"] > 0] = [233, 69, 96]  # Theme accent color
        final = cv2.addWeighted(final, 0.7, rust_overlay, 0.3, 0)
        final[res["rust_mask"] > 0] = cv2.addWeighted(
            res["crop"][res["rust_mask"] > 0], 0.5,
            np.array([233, 69, 96], dtype=np.uint8), 0.5, 0
        )

        axs[1, 2].imshow(final)
        axs[1, 2].set_title(f"6. Result: {res['pct']:.1f}% Rust", color=Theme.ACCENT_PRIMARY, fontsize=11, fontweight='bold')
        axs[1, 2].axis("off")

        title = "T1 Analysis (24h)" if key == "t1" else "T2 Analysis (48h)"
        fig.suptitle(title, color=Theme.TEXT_PRIMARY, fontsize=14, fontweight='bold')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.analysis_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_comparison_view(self):
        """Update the comparison tab."""
        res1 = self.data.get("t1", {}).get("res")
        res2 = self.data.get("t2", {}).get("res")

        if res1 is None or res2 is None:
            return

        # Clear previous
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        # Stats bar
        stats_bar = tk.Frame(self.comparison_frame, bg=Theme.BG_DARK)
        stats_bar.pack(fill=tk.X, padx=20, pady=15)

        # Stat cards
        t1_card = StatCard(stats_bar, "T1 Rust Coverage", f"{res1['pct']:.1f}", "%", Theme.ACCENT_SECONDARY)
        t1_card.pack(side=tk.LEFT, padx=10)

        t2_card = StatCard(stats_bar, "T2 Rust Coverage", f"{res2['pct']:.1f}", "%", Theme.ACCENT_INFO)
        t2_card.pack(side=tk.LEFT, padx=10)

        growth = res2['pct'] - res1['pct']
        growth_color = Theme.ACCENT_PRIMARY if growth > 0 else Theme.ACCENT_SUCCESS
        growth_card = StatCard(stats_bar, "Growth", f"{growth:+.1f}", "%", growth_color)
        growth_card.pack(side=tk.LEFT, padx=10)

        rate = (growth / res1['pct'] * 100) if res1['pct'] > 0 else 0
        rate_card = StatCard(stats_bar, "Growth Rate", f"{rate:.0f}", "%", Theme.ACCENT_WARNING)
        rate_card.pack(side=tk.LEFT, padx=10)

        # Comparison figure
        fig_frame = tk.Frame(self.comparison_frame, bg=Theme.BG_DARK)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        fig = Figure(figsize=(14, 6), dpi=100, facecolor=Theme.BG_DARK)
        fig.patch.set_facecolor(Theme.BG_DARK)

        axs = fig.subplots(1, 3)

        for ax in axs:
            ax.set_facecolor(Theme.BG_CARD)

        # T1 result
        final1 = res1["crop"].copy()
        final1[res1["rust_mask"] > 0] = [233, 69, 96]
        axs[0].imshow(final1)
        axs[0].set_title(f"T1 (24h): {res1['pct']:.1f}%", color=Theme.ACCENT_SECONDARY, fontsize=12, fontweight='bold')
        axs[0].axis("off")

        # T2 result
        final2 = res2["crop"].copy()
        final2[res2["rust_mask"] > 0] = [233, 69, 96]
        axs[1].imshow(final2)
        axs[1].set_title(f"T2 (48h): {res2['pct']:.1f}%", color=Theme.ACCENT_INFO, fontsize=12, fontweight='bold')
        axs[1].axis("off")

        # Difference view
        delta = self.data.get("delta")
        if delta is not None:
            axs[2].imshow(delta, cmap="magma")
            axs[2].set_title("New Rust Growth", color=Theme.ACCENT_PRIMARY, fontsize=12, fontweight='bold')
        else:
            axs[2].text(0.5, 0.5, "No delta", ha='center', va='center', color=Theme.TEXT_MUTED)
        axs[2].axis("off")

        fig.suptitle("T1 vs T2 Comparison", color=Theme.TEXT_PRIMARY, fontsize=14, fontweight='bold')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_prediction_view(self):
        """Update the prediction tab."""
        res1 = self.data.get("t1", {}).get("res")
        res2 = self.data.get("t2", {}).get("res")
        heatmap = self.data.get("heatmap")
        delta = self.data.get("delta")

        if res1 is None or res2 is None:
            return

        # Clear previous
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig = Figure(figsize=(14, 7), dpi=100, facecolor=Theme.BG_DARK)
        fig.patch.set_facecolor(Theme.BG_DARK)

        # Create grid: 2 rows, 3 columns
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

        # Row 1: Delta, Heatmap, Prediction Chart
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Row 2: Summary spanning all columns
        ax4 = fig.add_subplot(gs[1, :])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(Theme.BG_CARD)
            ax.tick_params(colors=Theme.TEXT_MUTED)
            for spine in ax.spines.values():
                spine.set_color(Theme.BG_LIGHT)

        # 1. New rust growth (delta)
        if delta is not None:
            ax1.imshow(delta, cmap="magma")
            ax1.set_title("New Rust Growth (T1→T2)", color=Theme.TEXT_PRIMARY, fontsize=11)
        ax1.axis("off")

        # 2. Risk/Prediction heatmap
        if heatmap is not None:
            im = ax2.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            ax2.set_title("Risk Heatmap (Future)", color=Theme.TEXT_PRIMARY, fontsize=11)
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=Theme.TEXT_MUTED)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=Theme.TEXT_MUTED)
        ax2.axis("off")

        # 3. Trend chart
        p1 = res1["pct"]
        p2 = res2["pct"]
        p3 = max(0, min(100, p2 + (p2 - p1) * 1.2))  # Linear extrapolation with momentum

        times = [24, 48, 72]
        vals = [p1, p2, p3]

        ax3.plot(times[:2], vals[:2], 'o-', color=Theme.ACCENT_SECONDARY, lw=2, markersize=8, label='Measured')
        ax3.plot(times[1:], vals[1:], 'o--', color=Theme.ACCENT_PRIMARY, lw=2, markersize=8, label='Predicted')
        ax3.fill_between(times, vals, alpha=0.2, color=Theme.ACCENT_PRIMARY)

        ax3.set_title("Corrosion Trend", color=Theme.TEXT_PRIMARY, fontsize=11)
        ax3.set_xlabel("Time (hours)", color=Theme.TEXT_SECONDARY, fontsize=10)
        ax3.set_ylabel("Rust Coverage (%)", color=Theme.TEXT_SECONDARY, fontsize=10)
        ax3.grid(True, alpha=0.3, color=Theme.TEXT_MUTED)
        ax3.legend(facecolor=Theme.BG_CARD, edgecolor=Theme.BG_LIGHT, labelcolor=Theme.TEXT_PRIMARY)

        for i, v in enumerate(vals):
            color = Theme.ACCENT_SECONDARY if i < 2 else Theme.ACCENT_PRIMARY
            ax3.annotate(f'{v:.1f}%', (times[i], v), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, color=color, fontweight='bold')

        ax3.set_xlim(20, 76)
        ax3.set_ylim(0, max(vals) * 1.3)

        # 4. Summary text panel
        growth = p2 - p1
        growth_rate = (growth / p1 * 100) if p1 > 0 else 0
        predicted_growth = p3 - p2

        severity = "Low" if p2 < 15 else "Moderate" if p2 < 30 else "High" if p2 < 50 else "Critical"
        severity_color = Theme.ACCENT_SUCCESS if severity == "Low" else Theme.ACCENT_WARNING if severity == "Moderate" else Theme.ACCENT_PRIMARY

        summary_text = (
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"                                    ANALYSIS SUMMARY\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n\n"
            f"  📊 CURRENT STATE                          📈 GROWTH ANALYSIS\n"
            f"  ─────────────────                         ──────────────────\n"
            f"  T1 Coverage (24h):  {p1:>6.1f}%               Growth (T1→T2):    {growth:>+6.1f}%\n"
            f"  T2 Coverage (48h):  {p2:>6.1f}%               Growth Rate:       {growth_rate:>6.0f}%\n"
            f"  Severity Level:     {severity:<10}            Predicted (72h):   {p3:>6.1f}%\n\n"
            f"  ⚠️  RECOMMENDATION: {'Immediate attention required!' if severity in ['High', 'Critical'] else 'Monitor regularly.' if severity == 'Moderate' else 'Continue normal maintenance.'}\n"
            f"═══════════════════════════════════════════════════════════════════════════════════"
        )

        ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                color=Theme.TEXT_PRIMARY,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=Theme.BG_MEDIUM, edgecolor=Theme.BG_LIGHT))
        ax4.axis("off")

        fig.suptitle("🔮 Growth Prediction & Risk Analysis", color=Theme.TEXT_PRIMARY, fontsize=14, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, master=self.prediction_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


# ==========================================
# ENTRY POINT
# ==========================================

def main():
    """Main entry point for the application."""
    root = tk.Tk()

    # Set window icon (if available)
    try:
        # For Windows
        root.iconbitmap(default='')
    except:
        pass

    # Center window on screen
    root.update_idletasks()
    width = 1500
    height = 900
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Create and run app
    app = RustAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
