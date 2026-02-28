import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import SAM
import torch

# For t-SNE and clustering
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class MetalSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None               # raw segmentation mask (metal = 1)
        self.processed_mask = None     # mask after morphology
        self.model = None              # SAM model
        self.image_path = None

        # morphology parameters
        self.kernel_size = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 1

        # rust output
        self.rust_mask = None
        self.rust_percentage = None

        # t-SNE outputs (2D only)
        self.embeddings_2d = None

        print("Initializing MetalSegmenter (no CLIP, using color/texture features)...")
        self.load_sam2_model()

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

    # ------------------------------------------------------------------
    # PATCH FEATURE EXTRACTION (NO CLIP)
    # ------------------------------------------------------------------
    def compute_patch_features(self, patch_bgr):
        """
        Compute a hand-crafted feature vector for a patch.

        Features (fixed-length):
          - Mean B, G, R
          - Mean H, S, V
          - Mean L, A, B (Lab)
          - Std of grayscale
          - Edge density (Canny)
        """
        if patch_bgr.size == 0:
            return None

        # Mean color in BGR
        bgr_mean = patch_bgr.reshape(-1, 3).mean(axis=0)  # [B, G, R]

        # HSV
        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        hsv_mean = patch_hsv.reshape(-1, 3).mean(axis=0)  # [H, S, V]

        # Lab
        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        lab_mean = patch_lab.reshape(-1, 3).mean(axis=0)  # [L, A, B]

        # Texture: grayscale std + edge density
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        gray_std = gray.std()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean()

        # Concatenate into one feature vector
        features = np.concatenate(
            [
                bgr_mean,        # 0-2
                hsv_mean,        # 3-5
                lab_mean,        # 6-8
                np.array([gray_std, edge_density], dtype=np.float32),  # 9-10
            ]
        ).astype(np.float32)

        return features

    def extract_patch_embeddings(
        self,
        patch_size=224,
        stride=112,
        min_metal_ratio=0.3,
        max_patches=500
    ):
        """
        Extract COLOR/TEXTURE feature vectors of patches from the METAL REGION ONLY.

        Returns:
            embeddings_np: (N, D) numpy array
            centers:       (N, 2) array of (x_center, y_center) for each patch
        """
        if self.original_image is None:
            print("No image loaded for patch feature extraction.")
            return None, None

        # Metal-only mask
        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask
        if metal_mask is None:
            print("No metal mask available for patch feature extraction.")
            return None, None

        metal_mask = (metal_mask > 0).astype(np.uint8)
        img_bgr = self.original_image
        H_img, W_img = metal_mask.shape

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found for patch feature extraction.")
            return None, None

        # bounding box over metal region
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        embeddings = []
        centers = []

        print("Extracting patch features (purely image-based, METAL ONLY)...")
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

                feat = self.compute_patch_features(patch_bgr)
                if feat is None:
                    continue

                embeddings.append(feat)
                # store patch center for visualization
                cx = (x + x2) // 2
                cy = (y + y2) // 2
                centers.append((cx, cy))

        if len(embeddings) == 0:
            print("No valid patches found for feature extraction.")
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
    # T-SNE + CLUSTERING WITH DATA-DRIVEN rust/metal LABELS (2D ONLY)
    # ------------------------------------------------------------------
    def visualize_tsne_clusters(self, embeddings, centers, metal_mask=None, n_clusters=2):
        """
        Run t-SNE on METAL-ONLY patch feature vectors and visualize:
          - 2D t-SNE scatter plot (colored by KMeans clusters)
          - cluster positions on the metal region of the original image

        Uses ONLY color/texture features for t-SNE / KMeans.
        Rust vs metal cluster mapping is inferred from color statistics:
        rust patches are typically more reddish/brown (R >> G,B, higher S, lower V).
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

        print(f"Running KMeans clustering on {n_samples} feature vectors...")
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

        # -------- DATA-DRIVEN rust vs metal MAPPING (color-based) --------
        # embeddings columns: [B, G, R, H, S, V, L, A, B_lab, gray_std, edge_density]
        B_vals = embeddings[:, 0]
        G_vals = embeddings[:, 1]
        R_vals = embeddings[:, 2]
        H_vals = embeddings[:, 3]
        S_vals = embeddings[:, 4]
        V_vals = embeddings[:, 5]

        def compute_rust_score(mask_k):
            """Higher score => more likely rust: reddish, saturated, slightly darker."""
            if not np.any(mask_k):
                return -1e9
            rg = (R_vals[mask_k] - G_vals[mask_k]).mean()
            rb = (R_vals[mask_k] - B_vals[mask_k]).mean()
            sat = S_vals[mask_k].mean() / 255.0  # 0-1
            val = V_vals[mask_k].mean() / 255.0  # 0-1

            # emphasize red dominance and saturation, penalize brightness a bit
            score = 0.6 * ((rg + rb) / 2.0) + 40.0 * sat - 20.0 * val
            return score

        rust_cluster = None
        metal_cluster = None
        cluster_scores = []
        for k in range(n_clusters):
            mask_k = (cluster_labels == k)
            score_k = compute_rust_score(mask_k)
            cluster_scores.append(score_k)
            print(f"Cluster {k}: rust-like score = {score_k:.4f}")

        rust_cluster = int(np.argmax(cluster_scores))
        metal_cluster = 1 - rust_cluster if n_clusters == 2 else None

        print(f"Color-based mapping: rust_cluster = {rust_cluster}, metal_cluster = {metal_cluster}")

        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # If a metal_mask is provided, zero out everything else for visualization
        if metal_mask is None:
            metal_mask = self.processed_mask if self.processed_mask is not None else self.mask

        if metal_mask is not None:
            metal_mask_vis = (metal_mask > 0).astype(np.uint8)
            img_rgb_metal = img_rgb.copy()
            img_rgb_metal[metal_mask_vis == 0] = 0
        else:
            # Fallback: no mask, show full image
            img_rgb_metal = img_rgb

        # Plot: t-SNE 2D + patch centers on METAL-ONLY image
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Patch Feature Structure on METAL REGION (Color/Texture + t-SNE)",
            fontsize=16,
            fontweight="bold"
        )

        # Helper to map cluster index -> human-friendly name
        def cluster_name(k: int) -> str:
            if rust_cluster is not None and metal_cluster is not None:
                if k == rust_cluster:
                    return "rust"
                elif k == metal_cluster:
                    return "metal"
                else:
                    return f"cluster {k}"
            # fallback
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
        ax0.set_title("t-SNE (2D) of METAL Patch Features\nColor = rust / metal clusters")
        ax0.set_xlabel("t-SNE 1")
        ax0.set_ylabel("t-SNE 2")

        # Legend
        for k in range(n_clusters):
            ax0.scatter([], [], color=cluster_colors[k], label=cluster_name(k))
        ax0.legend(title="Cluster label", loc="best")

        # 2) Metal-only image with patch centers colored by cluster
        ax1 = axes[1]
        ax1.imshow(img_rgb_metal)
        ax1.set_title("METAL Region: Patch Locations Colored by Cluster (rust vs metal)")
        ax1.axis("off")

        for (cx, cy), label in zip(centers, cluster_labels):
            ax1.scatter(
                cx,
                cy,
                color=[cluster_colors[label]],
                s=35,
                edgecolors="black",
                linewidths=0.5
            )

        plt.tight_layout()
        plt.show()

        print("t-SNE (2D) + clustering visualization on METAL ONLY completed.")
        print("Interpretation ideas:")
        print("  - Color statistics decide which cluster is 'rust' vs 'metal'.")
        print("  - The 2D plot shows the structure of patch features restricted to the metal region.")

    def analyze_embedding_structure(self):
        """
        Full pipeline for feature structure on METAL ONLY:

        - Segment metal in the image
        - Extract patches from metal region
        - Compute color/texture features (image only)
        - Visualize with t-SNE (2D) and clustering
        """
        if self.original_image is None:
            print("No image loaded; cannot analyze feature structure.")
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
            print("Feature structure analysis skipped (no valid features).")

    # ------------------------------------------------------------------
    # COLOR/FEATURE-BASED RUST DETECTOR (NO CLIP)
    # ------------------------------------------------------------------
    def rust_color_based_detection(self, patch_size=64, stride=32):
        """
        Rust detector based on color clustering inside the METAL region.
        No CLIP, no text prompts. Uses the same color/texture features.
        """
        if self.original_image is None:
            print("No image loaded for rust detection.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask
        if metal_mask is None:
            print("No metal mask available for rust detection.")
            return np.zeros_like(self.original_image[:, :, 0], dtype=np.uint8), 0.0

        metal_mask = (metal_mask > 0).astype(np.uint8)
        H_img, W_img = metal_mask.shape

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        # Extract patch features on metal region (denser sampling for rust detection)
        embeddings, centers = self.extract_patch_embeddings(
            patch_size=patch_size,
            stride=stride,
            min_metal_ratio=0.3,
            max_patches=None
        )

        if embeddings is None or centers is None:
            print("No features extracted for rust detection.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        # KMeans clustering into 2 clusters (rust vs metal)
        print("Running color/feature-based KMeans for rust detection...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # embeddings columns: [B, G, R, H, S, V, L, A, B_lab, gray_std, edge_density]
        B_vals = embeddings[:, 0]
        G_vals = embeddings[:, 1]
        R_vals = embeddings[:, 2]
        H_vals = embeddings[:, 3]
        S_vals = embeddings[:, 4]
        V_vals = embeddings[:, 5]

        def compute_rust_score(mask_k):
            if not np.any(mask_k):
                return -1e9
            rg = (R_vals[mask_k] - G_vals[mask_k]).mean()
            rb = (R_vals[mask_k] - B_vals[mask_k]).mean()
            sat = S_vals[mask_k].mean() / 255.0
            val = V_vals[mask_k].mean() / 255.0
            score = 0.6 * ((rg + rb) / 2.0) + 40.0 * sat - 20.0 * val
            return score

        cluster_scores = []
        for k in range(2):
            mask_k = (cluster_labels == k)
            score_k = compute_rust_score(mask_k)
            cluster_scores.append(score_k)
            print(f"Rust detection - Cluster {k}: rust-like score = {score_k:.4f}")

        rust_cluster = int(np.argmax(cluster_scores))
        print(f"Rust detection - chosen rust_cluster = {rust_cluster}")

        # Build pixel-level rust mask from patch-level cluster assignments
        rust_mask = np.zeros_like(metal_mask, dtype=np.uint8)

        ys, xs = np.where(metal_mask == 1)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        print("Filling rust mask from patch clusters...")
        idx = 0
        for y in range(ymin, ymax + 1, stride):
            for x in range(xmin, xmax + 1, stride):
                y2 = min(y + patch_size, H_img)
                x2 = min(x + patch_size, W_img)

                patch_mask = metal_mask[y:y2, x:x2]
                if patch_mask.size == 0:
                    continue
                if patch_mask.sum() < 0.3 * patch_mask.size:
                    continue

                if idx >= len(cluster_labels):
                    continue

                if cluster_labels[idx] == rust_cluster:
                    rust_mask[y:y2, x:x2] = 1

                idx += 1

        # Only count rust inside metal region
        rust_mask = rust_mask * metal_mask

        rust_pixels = rust_mask.sum()
        metal_pixels = metal_mask.sum()
        rust_percentage = (rust_pixels / metal_pixels * 100.0) if metal_pixels > 0 else 0.0

        return rust_mask.astype(np.uint8), rust_percentage

    def run_rust_detection(self):
        """Run color/texture-based rust detector on metal region."""
        if self.original_image is None:
            print("No image loaded; cannot run rust detection.")
            return
        if self.processed_mask is None and self.mask is None:
            print("No mask available; run segmentation first.")
            return

        self.rust_mask, self.rust_percentage = self.rust_color_based_detection()

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
        fig.suptitle(
            "Color/Texture-Based Rust Detection",
            fontsize=16,
            fontweight="bold"
        )

        # 1) Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontweight="bold")
        axes[0].axis("off")

        # 2) Metal-only region
        metal_only = img_rgb.copy()
        metal_only[metal_mask == 0] = 0
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
        if self.image is None:
            return []
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

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

            contours, _ = cv2.findContours(
                combined_metal,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

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
        """Run rust detection and show the final figure."""
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
        print("METAL SEGMENTATION + COLOR/TEXTURE FEATURES + METAL-ONLY t-SNE STRUCTURE ANALYSIS")
        print("=" * 80)
        if not self.load_image():
            return
        if not self.interactive_metal_segmentation():
            return

        # 1) Analyze feature structure (t-SNE & clusters) - METAL-ONLY
        self.analyze_embedding_structure()

        # 2) Run color/texture-based rust detection pipeline
        self.show_results()
        print("Segmentation, METAL-ONLY feature-structure analysis, and rust estimation completed.")


if __name__ == "__main__":
    segmenter = MetalSegmenter()
    segmenter.run()
