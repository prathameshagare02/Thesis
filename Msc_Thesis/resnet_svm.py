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

# SVM for rust / non-rust classification on patch embeddings
from sklearn.svm import LinearSVC


class MetalSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None               # raw segmentation mask (metal = 1)
        self.processed_mask = None     # mask after morphology
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

        # SVM model for rust / non-rust patches
        self.svm_model = None

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

    # ------------------------------------------------------------------
    # ResNet + SVM: training and inference on patches (replaces CLIP+t-SNE)
    # ------------------------------------------------------------------
    def train_rust_svm(
        self,
        rust_gt_mask,
        patch_size=64,
        stride=32,
        rust_ratio_thresh=0.5,
        nonrust_ratio_thresh=0.1,
    ):
        """
        Train an SVM to classify patches as rust / non-rust based on ResNet embeddings.

        rust_gt_mask: 2D numpy array, same HxW as image, values:
                      0 = non-rust, 1 (or 255) = rust
        """
        if self.feat_model is None or self.feat_preprocess is None:
            print("ResNet feature model not available, cannot train SVM.")
            return

        if self.original_image is None:
            print("No image loaded, cannot train SVM.")
            return

        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask
        if metal_mask is None:
            print("No metal mask available, cannot train SVM.")
            return

        # Normalize rust mask to {0,1}
        if rust_gt_mask.dtype != np.uint8:
            rust_gt_mask = rust_gt_mask.astype(np.uint8)
        rust_mask_bin = (rust_gt_mask > 0).astype(np.uint8)

        # sanity check size
        H_img, W_img = metal_mask.shape
        if rust_mask_bin.shape != (H_img, W_img):
            raise ValueError(f"rust_gt_mask shape {rust_mask_bin.shape} "
                             f"does not match image/mask shape {(H_img, W_img)}")

        metal_mask = (metal_mask > 0).astype(np.uint8)
        img_bgr = self.original_image

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found, cannot train SVM.")
            return

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        X_feats = []
        y_labels = []

        print("Collecting patch features + labels for SVM training...")
        with torch.no_grad():
            for y in range(ymin, ymax + 1, stride):
                for x in range(xmin, xmax + 1, stride):
                    y2 = min(y + patch_size, H_img)
                    x2 = min(x + patch_size, W_img)

                    patch_metal = metal_mask[y:y2, x:x2]
                    if patch_metal.sum() < 0.3 * patch_metal.size:
                        continue

                    patch_rust = rust_mask_bin[y:y2, x:x2]
                    rust_ratio = patch_rust.sum() / patch_rust.size

                    # Decide label
                    if rust_ratio >= rust_ratio_thresh:
                        label = 1  # rust
                    elif rust_ratio <= nonrust_ratio_thresh:
                        label = 0  # clean metal
                    else:
                        # ambiguous patch, skip
                        continue

                    patch_bgr = img_bgr[y:y2, x:x2]
                    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(patch_rgb)

                    img_input = self.feat_preprocess(pil_img).unsqueeze(0).to(self.device)
                    feat = self.feat_model(img_input)          # (1, 2048, 1, 1)
                    feat = torch.flatten(feat, 1)              # (1, 2048)
                    feat = F.normalize(feat, dim=-1)           # [1, D]

                    X_feats.append(feat.squeeze(0).cpu().numpy())
                    y_labels.append(label)

        if len(X_feats) == 0:
            print("No training patches collected; SVM not trained.")
            return

        X_feats = np.array(X_feats)
        y_labels = np.array(y_labels)
        print(f"Training SVM on {X_feats.shape[0]} patches with {X_feats.shape[1]}-D features.")
        print(f"Class balance: rust={np.sum(y_labels==1)}, non-rust={np.sum(y_labels==0)}")

        svm = LinearSVC(class_weight="balanced", max_iter=5000)
        svm.fit(X_feats, y_labels)
        self.svm_model = svm
        print("SVM training complete. You can now run rust_resnet_svm_based().")

    def rust_resnet_svm_based(self, patch_size=64, stride=32):
        """
        Use trained SVM + ResNet embeddings to detect rust patches
        and build a pixel-level rust mask.
        Replaces CLIP-based zero-shot rust detector.
        """
        if self.svm_model is None:
            print("SVM model not trained; call train_rust_svm(...) first.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        if self.feat_model is None or self.feat_preprocess is None:
            print("ResNet feature model not available.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        if self.original_image is None:
            print("No image loaded for rust detection.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        metal_mask = self.processed_mask if self.processed_mask is not None else self.mask
        if metal_mask is None:
            print("No metal mask available for rust detection.")
            return np.zeros_like(self.processed_mask, dtype=np.uint8), 0.0

        metal_mask = (metal_mask > 0).astype(np.uint8)
        img_bgr = self.original_image
        H_img, W_img = metal_mask.shape

        ys, xs = np.where(metal_mask == 1)
        if len(xs) == 0:
            print("No metal pixels found.")
            return np.zeros_like(metal_mask, dtype=np.uint8), 0.0

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        rust_mask = np.zeros_like(metal_mask, dtype=np.uint8)

        print("Running ResNet+SVM rust detection on metal region...")
        with torch.no_grad():
            for y in range(ymin, ymax + 1, stride):
                for x in range(xmin, xmax + 1, stride):
                    y2 = min(y + patch_size, H_img)
                    x2 = min(x + patch_size, W_img)

                    patch_metal = metal_mask[y:y2, x:x2]
                    if patch_metal.sum() < 0.3 * patch_metal.size:
                        continue

                    patch_bgr = img_bgr[y:y2, x:x2]
                    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(patch_rgb)

                    img_input = self.feat_preprocess(pil_img).unsqueeze(0).to(self.device)
                    feat = self.feat_model(img_input)          # (1, 2048, 1, 1)
                    feat = torch.flatten(feat, 1)              # (1, 2048)
                    feat = F.normalize(feat, dim=-1)           # [1, D]

                    feat_np = feat.squeeze(0).cpu().numpy().reshape(1, -1)
                    pred = self.svm_model.predict(feat_np)[0]  # 0 or 1

                    if pred == 1:
                        rust_mask[y:y2, x:x2] = 1

        rust_mask = rust_mask * metal_mask
        rust_pixels = rust_mask.sum()
        metal_pixels = metal_mask.sum()
        rust_percentage = (rust_pixels / metal_pixels * 100.0) if metal_pixels > 0 else 0.0

        return rust_mask.astype(np.uint8), rust_percentage

    def run_rust_detection(self):
        """Run ResNet+SVM rust detector on metal region (replaces CLIP-based)."""
        if self.original_image is None:
            print("No image loaded; cannot run rust detection.")
            return
        if self.processed_mask is None and self.mask is None:
            print("No mask available; run segmentation first.")
            return
        if self.svm_model is None:
            print("SVM not trained yet; rust detection skipped. "
                  "Call train_rust_svm(...) with a ground-truth rust mask.")
            return

        self.rust_mask, self.rust_percentage = self.rust_resnet_svm_based()

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
        fig.suptitle("ResNet + SVM Rust Detection",
                     fontsize=16, fontweight="bold")

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
    # SEGMENTATION PIPELINE (kept the same)
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
        """Run SVM-based rust detection and show the final figure."""
        if self.image is None or self.mask is None:
            messagebox.showwarning("No Results", "No segmentation results to display.")
            return

        if self.processed_mask is None:
            self.processed_mask = self.apply_morphological_operations(self.mask)

        self.run_rust_detection()

    def run(self):
        """
        Main workflow.
        NOTE: This will NOT automatically train the SVM.
              Call train_rust_svm(...) beforehand with a ground-truth rust mask.
        """
        print("METAL SEGMENTATION + ResNet EMBEDDINGS + SVM RUST DETECTION")
        print("=" * 80)
        if not self.load_image():
            return
        if not self.interactive_metal_segmentation():
            return

        # Run SVM-based rust detection (only if SVM is trained)
        self.show_results()
        print("Segmentation and rust estimation completed (if SVM was trained).")


if __name__ == "__main__":
    segmenter = MetalSegmenter()
    segmenter.run()
