import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import SAM
import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image

# torchvision ResNet + transforms
import torchvision.models as models
import torchvision.transforms as T

# For t-SNE and clustering purely from embeddings
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class MetalSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None               # raw segmentation mask (metal = 1)
        self.processed_mask = None     # mask after morphology
        self.crop_image = None         # cropped image around metal
        self.crop_mask = None          # cropped metal mask
        self.crop_coords = None        # (x, y, w, h) crop in original space
        self.model = None              # SAM model
        self.image_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # morphology parameters
        self.kernel_size = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 1

        # rust output
        self.rust_mask = None
        self.rust_percentage = None

        # ResNet feature extractor
        self.feat_model = None
        self.feat_preprocess = None

        # t-SNE outputs (2D only)
        self.embeddings_2d = None

        print(f"Using device: {self.device}")
        self.load_sam2_model()
        self.load_resnet_model()

    # ------------------------------------------------------------------
    # MODEL / IMAGE LOADING
    # ------------------------------------------------------------------
    def load_sam2_model(self):
        """Load SAM 2 model from Ultralytics."""
        try:
            print("Loading SAM 2 model...")
            self.model = SAM("sam2.1_b.pt")
            print("SAM 2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading SAM 2 model: {e}")
            messagebox.showerror("Error", f"Failed to load SAM 2 model: {e}")
            return False

    def load_resnet_model(self):
        """
        Load a pretrained ResNet model to use as a generic image feature extractor.
        We remove the final classification layer and use the global pooled features.
        """
        try:
            print("Loading ResNet-50 feature extractor...")
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            base_model = models.resnet50(weights=weights)

            # Remove final FC layer: use all layers up to avgpool
            modules = list(base_model.children())[:-1]   # everything except the final fc
            self.feat_model = nn.Sequential(*modules).to(self.device)
            self.feat_model.eval()

            # Preprocessing: resize/crop to 224, normalize with ImageNet stats
            self.feat_preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            print("ResNet-50 feature extractor loaded successfully!")
        except Exception as e:
            print(f"Error loading ResNet model: {e}")
            messagebox.showerror("Error", f"Failed to load ResNet model: {e}")
            self.feat_model = None
            self.feat_preprocess = None

    def load_image(self, image_path=None):
        """Load an image from file (using file dialog if path not given)."""
        if image_path is None:
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select metal object image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
            )
            root.destroy()

        if not image_path:
            return False

        self.image = cv2.imread(image_path)
        if self.image is None:
            messagebox.showerror("Error", "Could not load image!")
            return False

        self.image_path = image_path
        self.original_image = self.image.copy()
        print(f"Image loaded: {self.image.shape}")
        return True

    # ------------------------------------------------------------------
    # BASIC MORPHOLOGY
    # ------------------------------------------------------------------
    def apply_morphological_operations(self, mask):
        """Apply erosion and dilation operations to the mask (internal use)."""
        if mask is None:
            return None

        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.erosion_iterations > 0:
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations)
        else:
            mask_eroded = mask_uint8.copy()

        if self.dilation_iterations > 0:
            mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=self.dilation_iterations)
        else:
            mask_dilated = mask_eroded.copy()

        processed_mask = (mask_dilated > 127).astype(np.uint8)
        return processed_mask

    def _get_active_metal_mask(self):
        return self.processed_mask if self.processed_mask is not None else self.mask

    def _compute_crop_from_mask(self, metal_mask, pad=20):
        if self.original_image is None:
            return None, None, (0, 0, 0, 0)

        h, w = self.original_image.shape[:2]
        if metal_mask is None:
            return self.original_image, np.ones((h, w), dtype=np.uint8), (0, 0, w, h)

        metal_mask = (metal_mask > 0).astype(np.uint8)
        ys, xs = np.where(metal_mask > 0)
        if len(xs) == 0:
            return self.original_image, np.ones((h, w), dtype=np.uint8), (0, 0, w, h)

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        crop = self.original_image[y_min:y_max, x_min:x_max]
        mask_crop = metal_mask[y_min:y_max, x_min:x_max]
        return crop, mask_crop, (x_min, y_min, x_max - x_min, y_max - y_min)

    # ------------------------------------------------------------------
    # PATCH EMBEDDING EXTRACTION (NO TEXT EMBEDDINGS)
    # ------------------------------------------------------------------
    def extract_patch_embeddings(
        self,
        patch_size=224,
        stride=112,
        min_metal_ratio=0.3,
        max_patches=500
    ):
        """
        Extract ResNet embeddings of patches from the METAL REGION ONLY.

        Returns:
            embeddings_np: (N, D) numpy array
            centers: list/array of (x_center, y_center) for each patch
        """
        if self.feat_model is None or self.feat_preprocess is None:
            print("ResNet feature model not available for patch embedding extraction.")
            return None, None

        if self.original_image is None:
            print("No image loaded for patch embedding extraction.")
            return None, None

        # Metal-only mask
        metal_mask = self._get_active_metal_mask()
        if metal_mask is None:
            print("No metal mask available for patch embedding extraction.")
            return None, None

        crop_img, crop_mask, crop_coords = self._compute_crop_from_mask(metal_mask, pad=20)
        if crop_img is None or crop_mask is None:
            print("No valid crop available for patch embedding extraction.")
            return None, None

        self.crop_image = crop_img
        self.crop_mask = crop_mask
        self.crop_coords = crop_coords

        metal_mask = (crop_mask > 0).astype(np.uint8)
        img_bgr = crop_img
        H_img, W_img = metal_mask.shape

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found for patch embedding extraction.")
            return None, None

        # bounding box over metal region
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        embeddings = []
        centers = []

        print("Extracting patch embeddings with ResNet (METAL ONLY)...")
        with torch.no_grad():
            for y in range(ymin, ymax + 1, stride):
                for x in range(xmin, xmax + 1, stride):
                    y2 = min(y + patch_size, H_img)
                    x2 = min(x + patch_size, W_img)

                    patch_mask = metal_mask[y:y2, x:x2]
                    if patch_mask.size == 0:
                        continue

                    # ensure the patch is mostly metal
                    if patch_mask.sum() < min_metal_ratio * patch_mask.size:
                        continue

                    patch_bgr = img_bgr[y:y2, x:x2]
                    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(patch_rgb)

                    img_input = self.feat_preprocess(pil_img).unsqueeze(0).to(self.device)
                    feat = self.feat_model(img_input)          # (1, 2048, 1, 1)
                    feat = torch.flatten(feat, 1)              # (1, 2048)
                    feat = F.normalize(feat, dim=-1)           # [1, D]

                    embeddings.append(feat.squeeze(0).cpu().numpy())
                    # store patch center for visualization
                    cx = (x + x2) // 2
                    cy = (y + y2) // 2
                    centers.append((cx + crop_coords[0], cy + crop_coords[1]))

        if len(embeddings) == 0:
            print("No valid patches found for embedding extraction.")
            return None, None

        embeddings = np.array(embeddings)  # (N, D)
        centers = np.array(centers)        # (N, 2)

        # Optional: random subsample if too many patches for t-SNE
        if max_patches is not None and embeddings.shape[0] > max_patches:
            print(f"Subsampling patches from {embeddings.shape[0]} to {max_patches} for t-SNE...")
            idx = np.random.choice(embeddings.shape[0], size=max_patches, replace=False)
            embeddings = embeddings[idx]
            centers = centers[idx]

        print(f"Total patches used for analysis (METAL ONLY): {embeddings.shape[0]}")
        return embeddings, centers

    # ------------------------------------------------------------------
    # T-SNE + CLUSTERING WITH IMAGE-ONLY LABELS (2D ONLY)
    # ------------------------------------------------------------------
    def visualize_tsne_clusters(self, embeddings, centers, metal_mask=None, n_clusters=2):
        """
        Run t-SNE on METAL-ONLY patch embeddings and visualize:
          - 2D t-SNE scatter plot (colored by KMeans clusters)
          - cluster positions on the metal region of the original image

        Uses ONLY image embeddings (ResNet), no text.
        """
        if embeddings is None or centers is None:
            print("No embeddings or centers provided for t-SNE visualization.")
            return

        n_samples = embeddings.shape[0]
        if n_samples < 3:
            print(f"Not enough samples for t-SNE (got {n_samples}, need >= 3).")
            return

        # ---- ADAPTIVE PERPLEXITY ----
        base_perplexity = 30.0
        perplexity = min(base_perplexity, (n_samples - 1) / 3.0)
        perplexity = max(5.0, perplexity)
        if perplexity >= n_samples:
            perplexity = n_samples - 1e-4

        print(f"Running KMeans clustering on {n_samples} embeddings...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # ---- CONSISTENT COLORS FOR ALL PLOTS ----
        tab10_colors = np.array(plt.cm.tab10.colors)
        cluster_colors = tab10_colors[:n_clusters]       # (n_clusters, 4)
        point_colors = cluster_colors[cluster_labels]    # (N, 4)

        # ---- t-SNE 2D ----
        print(f"Running t-SNE projection (2D) with perplexity={perplexity:.2f} ...")
        tsne_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='random',
            random_state=42
        )
        embeddings_2d = tsne_2d.fit_transform(embeddings)  # (N, 2)
        self.embeddings_2d = embeddings_2d

        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # If a metal_mask is provided, zero out everything else for visualization
        if metal_mask is None:
            metal_mask = self.processed_mask if self.processed_mask is not None else self.mask

        if metal_mask is not None:
            metal_mask_vis = (metal_mask > 0).astype(np.uint8)
            img_rgb_metal = img_rgb.copy()
            img_rgb_metal[metal_mask_vis == 0] = 255
        else:
            # Fallback: no mask, show full image
            img_rgb_metal = img_rgb

        # Plot: t-SNE 2D + patch centers on METAL-ONLY image
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Patch Embedding Structure on METAL REGION (ResNet + t-SNE)",
                     fontsize=16, fontweight="bold")

        def cluster_name(k: int) -> str:
            return f"cluster {k}"

        # 1) t-SNE 2D scatter
        ax0 = axes[0]
        ax0.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            color=point_colors,
            s=20,
            alpha=0.8
        )
        ax0.set_title("t-SNE (2D) of METAL Patch Embeddings\nColor = KMeans clusters")
        ax0.set_xlabel("t-SNE 1")
        ax0.set_ylabel("t-SNE 2")

        # Legend
        for k in range(n_clusters):
            ax0.scatter([], [], color=cluster_colors[k], label=cluster_name(k))
        ax0.legend(title="Cluster label", loc="best")

        # 2) Metal-only image with patch centers colored by cluster
        ax1 = axes[1]
        ax1.imshow(img_rgb_metal)
        ax1.set_title("METAL Region: Patch Locations Colored by Cluster")
        ax1.axis("off")

        for (cx, cy), label in zip(centers, cluster_labels):
            ax1.scatter(cx, cy,
                        color=[cluster_colors[label]],
                        s=35,
                        edgecolors="black",
                        linewidths=0.5)

        plt.tight_layout()
        plt.show()

        print("t-SNE (2D) + clustering visualization on METAL ONLY completed.")
        print("Interpretation ideas:")
        print("  - KMeans clusters patches based on ResNet embeddings.")
        print("  - You can inspect which cluster visually looks like 'rust' vs 'clean metal'.")

    def analyze_embedding_structure(self):
        """
        Full pipeline for embedding structure on METAL ONLY:

        - Segment metal in the image
        - Extract patches from metal region
        - Compute ResNet embeddings (image only)
        - Visualize with t-SNE (2D) and clustering
        """
        if self.original_image is None:
            print("No image loaded; cannot analyze embedding structure.")
            return

        if self.processed_mask is None and self.mask is not None:
            self.processed_mask = self.apply_morphological_operations(self.mask)

        if self.processed_mask is None:
            print("No metal mask available; cannot restrict patches to metal.")
            return

        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask

        embeddings, centers = self.extract_patch_embeddings(
            patch_size=224,
            stride=112,
            min_metal_ratio=0.3,
            max_patches=500
        )

        if embeddings is not None and centers is not None:
            # Pass metal_mask so that visualization also uses METAL ONLY view
            self.visualize_tsne_clusters(embeddings, centers, metal_mask=metal_mask)
        else:
            print("Embedding structure analysis skipped (no valid embeddings).")

    # ------------------------------------------------------------------
    # EMBEDDING-BASED RUST DETECTOR (ResNet + KMEANS)
    # ------------------------------------------------------------------
    def rust_resnet_embedding_based(self, patch_size=16, stride=4):
        """
        Embedding-based rust detector using ResNet embeddings + KMeans.

        - Extract patch embeddings only from metal region.
        - Cluster embeddings into 2 clusters.
        - Decide which cluster is 'rust' using a simple color heuristic
          (rust patches tend to be more reddish / darker).
        """
        if self.feat_model is None or self.feat_preprocess is None:
            print("ResNet feature model not available.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        if self.original_image is None:
            print("No image loaded for rust detection.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        metal_mask = self._get_active_metal_mask()
        if metal_mask is None:
            print("No metal mask available for rust detection.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        crop_img, crop_mask, crop_coords = self._compute_crop_from_mask(metal_mask, pad=20)
        if crop_img is None or crop_mask is None:
            print("No valid crop available for rust detection.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        self.crop_image = crop_img
        self.crop_mask = crop_mask
        self.crop_coords = crop_coords

        metal_mask_full = (metal_mask > 0).astype(np.uint8)
        metal_mask = (crop_mask > 0).astype(np.uint8)
        img_bgr = crop_img
        H_img, W_img = metal_mask.shape

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        print("Running ResNet-based rust detection (unsupervised clustering)...")

        patch_feats = []
        patch_coords = []   # (y, y2, x, x2) in FULL image coords
        patch_colors = []   # mean RGB per patch

        with torch.no_grad():
            for y in range(ymin, ymax + 1, stride):
                for x in range(xmin, xmax + 1, stride):
                    y2 = min(y + patch_size, H_img)
                    x2 = min(x + patch_size, W_img)

                    patch_mask = metal_mask[y:y2, x:x2]
                    # Skip patch if almost no metal
                    if patch_mask.sum() < 0.3 * patch_mask.size:
                        continue

                    patch_bgr = img_bgr[y:y2, x:x2]
                    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(patch_rgb)

                    img_input = self.feat_preprocess(pil_img).unsqueeze(0).to(self.device)
                    feat = self.feat_model(img_input)          # (1, 2048, 1, 1)
                    feat = torch.flatten(feat, 1)              # (1, 2048)
                    feat = F.normalize(feat, dim=-1)           # [1, D]

                    patch_feats.append(feat.squeeze(0).cpu().numpy())
                    patch_coords.append(
                        (y + crop_coords[1], y2 + crop_coords[1], x + crop_coords[0], x2 + crop_coords[0])
                    )

                    # mean RGB for color-based rust heuristic
                    mean_rgb = patch_rgb.mean(axis=(0, 1))     # (3,)
                    patch_colors.append(mean_rgb)

        if len(patch_feats) == 0:
            print("No valid patches for rust detection.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        patch_feats = np.array(patch_feats)   # (N, D)
        patch_colors = np.array(patch_colors) # (N, 3)

        # Cluster patch embeddings into 2 groups
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patch_feats)

        # Decide which cluster is 'rust' using a simple color heuristic:
        # rust is typically more reddish and a bit darker than clean metal.
        rust_scores = []
        for k in range(2):
            idx = (cluster_labels == k)
            if not np.any(idx):
                rust_scores.append(-1e9)
                continue
            colors_k = patch_colors[idx]  # (Nk, 3) in RGB
            R = colors_k[:, 0]
            G = colors_k[:, 1]
            B = colors_k[:, 2]
            # heuristic: higher (R-G) and (R-B), and slightly lower brightness
            redness = (R - G) + (R - B)
            brightness = (R + G + B) / 3.0
            score = redness.mean() - 0.3 * brightness.mean()
            rust_scores.append(score)
            print(f"Cluster {k}: redness_score = {score:.2f}")

        rust_cluster = int(np.argmax(rust_scores))
        clean_cluster = 1 - rust_cluster
        print(f"Heuristic mapping: rust_cluster = {rust_cluster}, clean_cluster = {clean_cluster}")

        # Build rust mask
        rust_mask = np.zeros_like(metal_mask_full, dtype=np.uint8)
        for (y, y2, x, x2), label in zip(patch_coords, cluster_labels):
            if label == rust_cluster:
                rust_mask[y:y2, x:x2] = 1

        # Only count rust inside metal region
        rust_mask = rust_mask * metal_mask_full

        rust_pixels = rust_mask.sum()
        metal_pixels = metal_mask_full.sum()
        rust_percentage = (rust_pixels / metal_pixels * 100.0) if metal_pixels > 0 else 0.0

        return rust_mask.astype(np.uint8), rust_percentage

    def run_rust_detection(self):
        """Run ResNet-based unsupervised rust detector on metal region."""
        if self.original_image is None:
            print("No image loaded; cannot run rust detection.")
            return
        if self.processed_mask is None and self.mask is None:
            print("No mask available; run segmentation first.")
            return

        self.rust_mask, self.rust_percentage = self.rust_resnet_embedding_based()

        if self.rust_mask is not None:
            self.show_rust_results()
        else:
            print("Rust detection failed or skipped.")

    def show_rust_results(self):
        """Show ONLY the final 3-panel rust figure."""
        if self.rust_mask is None:
            print("No rust mask to display.")
            return

        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask
        metal_mask = (metal_mask > 0).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Embedding-Based Rust Detection (ResNet + KMeans)",
                     fontsize=16, fontweight="bold")

        # 1) Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontweight="bold")
        axes[0].axis("off")

        # 2) Metal-only region
        metal_only = img_rgb.copy()
        metal_only[metal_mask == 0] = 255
        axes[1].imshow(metal_only)
        axes[1].set_title("Metal Region", fontweight="bold")
        axes[1].axis("off")

        # 3) Rust overlay ON METAL REGION ONLY
        base = metal_only.copy()
        overlay = base.copy()
        overlay[self.rust_mask == 1] = [255, 0, 0]   # red rust on metal
        blended = cv2.addWeighted(base, 0.7, overlay, 0.3, 0)

        title = "Rust Areas (Red)"
        if self.rust_percentage is not None:
            title += f"\nEstimated Corrosion: {self.rust_percentage:.2f}% of metal"
        axes[2].imshow(blended)
        axes[2].set_title(title, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        if self.rust_percentage is not None:
            print(f"Estimated rust / corrosion: {self.rust_percentage:.2f}% of metal region.")

    # ------------------------------------------------------------------
    # SEGMENTATION PIPELINE
    # ------------------------------------------------------------------
    def detect_metal_regions(self):
        """Automatically detect metal-like regions based on visual properties."""
        if self.original_image is None:
            return []
        try:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            _, A, _ = cv2.split(lab)
            metal_mask1 = cv2.inRange(A, 120, 135)

            _, S, V = cv2.split(hsv)
            metal_mask2 = cv2.inRange(S, 30, 100) & cv2.inRange(V, 100, 255)

            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)

            combined_metal = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined_metal = cv2.bitwise_or(combined_metal, edges_dilated)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_CLOSE, kernel)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_metal, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            min_area = 500
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    regions.append(
                        {
                            "id": len(regions),
                            "bbox": [x, y, x + w, y + h],
                            "center": [center_x, center_y],
                            "area": area,
                        }
                    )
            return regions
        except Exception as e:
            print(f"Metal detection error: {e}")
            return []

    def sam2_predict(self, points, labels):
        """Predict segmentation mask using SAM 2, then apply morphology."""
        if self.model is None:
            messagebox.showerror("Error", "SAM 2 model not loaded!")
            return None
        try:
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            results = self.model.predict(image_rgb, points=points, labels=labels)
            if (
                results
                and len(results) > 0
                and results[0].masks is not None
                and len(results[0].masks.data) > 0
            ):
                mask = results[0].masks.data[0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                self.processed_mask = self.apply_morphological_operations(binary_mask)
                return binary_mask
            return None
        except Exception as e:
            print(f"SAM 2 prediction error: {e}")
            return self.fallback_segmentation(points, labels)

    def fallback_segmentation(self, points, labels):
        """Fallback segmentation using GrabCut if SAM fails."""
        if len(points) == 0:
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.processed_mask = mask.copy()
            return mask

        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        for point, label in zip(points, labels):
            x, y = point
            if label == 1:  # foreground
                cv2.circle(mask, (x, y), 15, 3, -1)
            else:  # background
                cv2.circle(mask, (x, y), 15, 0, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if any(l == 1 for l in labels):
            fg_points = [points[i] for i in range(len(points)) if labels[i] == 1]
            x_coords = [p[0] for p in fg_points]
            y_coords = [p[1] for p in fg_points]
            x1 = max(0, min(x_coords) - 30)
            x2 = min(self.image.shape[1], max(x_coords) + 30)
            y1 = max(0, min(y_coords) - 30)
            y2 = min(self.image.shape[0], max(y_coords) + 30)
            rect = (x1, y1, x2 - x1, y2 - y1)

            cv2.grabCut(
                self.original_image,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT,
            )

        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        self.processed_mask = self.apply_morphological_operations(final_mask)
        return final_mask

    def interactive_metal_segmentation(self):
        """Main interactive segmentation for metal objects."""
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return False

        print("Detecting metal regions...")
        regions = self.detect_metal_regions()
        if regions:
            print(
                f"Found {len(regions)} potential metal regions - using auto-detection mode"
            )
            return self.auto_detection_mode(regions)
        else:
            print("No metal regions auto-detected - using manual point mode")
            return self.manual_point_mode()

    def auto_detection_mode(self, regions):
        """Auto-detection mode where user clicks on detected metal regions."""
        cv2.namedWindow("Click on METAL Parts - Press 's' when done")

        def click_callback(event, x, y, flags, param):
            regions_param = param["regions"]
            if event == cv2.EVENT_LBUTTONDOWN:
                for region in regions_param:
                    x1, y1, x2, y2 = map(int, region["bbox"])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        print(f"Selected metal region {region['id']}")
                        center_x, center_y = region["center"]
                        self.mask = self.sam2_predict(
                            points=[[center_x, center_y]], labels=[1]
                        )
                        self.update_auto_display(regions_param, region["id"])
                        break

        self.update_auto_display(regions, -1)
        param = {"regions": regions}
        cv2.setMouseCallback(
            "Click on METAL Parts - Press 's' when done", click_callback, param
        )

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                cv2.destroyAllWindows()
                return self.manual_point_mode()
            elif key == ord("s"):
                if self.mask is not None and np.any(self.mask):
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning(
                        "No Selection", "Please select a metal region first!"
                    )
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return False

    def update_auto_display(self, regions, selected_id):
        """Update display for auto-detection mode (OpenCV only)."""
        display_image = self.original_image.copy()
        for region in regions:
            x1, y1, x2, y2 = map(int, region["bbox"])
            center_x, center_y = map(int, region["center"])
            if region["id"] == selected_id:
                color = (0, 255, 0)
                thickness = 3
                if self.mask is not None and np.any(self.mask):
                    overlay = display_image.copy()
                    overlay[self.mask == 1] = [0, 255, 0]
                    display_image = cv2.addWeighted(
                        display_image, 0.7, overlay, 0.3, 0
                    )
            else:
                color = (255, 0, 0)
                thickness = 2

            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                display_image,
                f"M{region['id']}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                display_image,
                f"M{region['id']}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        instructions = [
            "METAL DETECTION MODE",
            f"Found {len(regions)} potential metal regions",
            "Click on any METAL REGION (M0, M1, etc.) to select it",
            "Green = Selected metal, Blue = Potential metal",
            "Press 'm' for manual mode, 's' to save selection, 'q' to quit",
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(
                display_image,
                instruction,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                display_image,
                instruction,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Click on METAL Parts - Press 's' when done", display_image)

    def manual_point_mode(self):
        """Manual point-based segmentation for metal objects."""
        print("Manual metal segmentation mode")
        cv2.namedWindow("Click on METAL Objects - Press 's' when done")
        points, labels = [], []

        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                labels.append(1)
                print(f"Added metal point at ({x}, {y})")
                self.mask = self.sam2_predict(points, labels)
                self.update_manual_display(points)

        cv2.setMouseCallback(
            "Click on METAL Objects - Press 's' when done", click_callback
        )
        self.update_manual_display(points)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                points.clear()
                labels.clear()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.processed_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.update_manual_display(points)
                print("Cleared all points")
            elif key == ord("s"):
                if points:
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning(
                        "No Points", "Please click on metal areas first!"
                    )
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return False

    def update_manual_display(self, points):
        """Update display for manual point mode (OpenCV only)."""
        display_image = self.original_image.copy()
        for x, y in points:
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)

        if self.mask is not None and np.any(self.mask):
            overlay = display_image.copy()
            overlay[self.mask == 1] = [0, 255, 0]
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)

        instructions = [
            "MANUAL METAL SEGMENTATION",
            "Click on METAL objects to segment them",
            "Green circles = Your clicks, Green overlay = Detected metal",
            "Press 'c' to clear points, 's' to save selection, 'q' to quit",
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(
                display_image,
                instruction,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                display_image,
                instruction,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Click on METAL Objects - Press 's' when done", display_image)

    # ------------------------------------------------------------------
    # FINAL STEP WRAPPERS
    # ------------------------------------------------------------------
    def show_results(self):
        """Run embedding-based rust detection and show the final figure."""
        if self.image is None or self.mask is None:
            messagebox.showwarning("No Results", "No segmentation results to display.")
            return

        if self.processed_mask is None:
            self.processed_mask = self.apply_morphological_operations(self.mask)

        self.run_rust_detection()

    def run(self):
        """
        Main workflow.
        """
        print("METAL SEGMENTATION + ResNet EMBEDDINGS + METAL-ONLY t-SNE STRUCTURE ANALYSIS")
        print("=" * 80)
        if not self.load_image():
            return
        if not self.interactive_metal_segmentation():
            return

        # 1) Analyze embedding structure (t-SNE & clusters) - METAL-ONLY
        self.analyze_embedding_structure()

        # 2) Run embedding-based rust detection pipeline
        self.show_results()
        print("Segmentation, METAL-ONLY embedding-structure analysis, and rust estimation completed.")


if __name__ == "__main__":
    segmenter = MetalSegmenter()
    segmenter.run()
