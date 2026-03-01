import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Toplevel
from ultralytics import SAM
import torch
from matplotlib.colors import ListedColormap

# --- NEW: classifier imports ---
import timm
from torchvision import transforms
from PIL import Image
import json
import argparse  # <--- NEW

# -------------------------------
# Rust / Non-Rust Classifier
# -------------------------------
class RustClassifier:
    """
    Loads a timm-based classifier checkpoint produced by train_rust.py.
    Checkpoint must contain: model_name, class_names, img_size, model (state_dict).
    """
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.model = None
        self.class_names = None
        self.img_size = 224
        self.val_tfms = None
        self._load()

    def _load(self):
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint not found: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)

        model_name = ckpt.get("model_name", "resnet50")
        self.class_names = ckpt.get("class_names", ["clean", "rust"])
        self.img_size = int(ckpt.get("img_size", 224))

        self.model = timm.create_model(model_name, pretrained=False, num_classes=len(self.class_names))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.val_tfms = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f"[CKPT] Loaded {model_name} | classes={self.class_names} | img_size={self.img_size}")

    @torch.no_grad()
    def predict_pil(self, pil_img: Image.Image):
        x = self.val_tfms(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "pred_idx": pred_idx,
            "pred_label": self.class_names[pred_idx],
            "probs": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        }


class MetalSegmenter:
    def __init__(self, classifier_ckpt: str = None, save_pred_json: bool = True):
        self.image = None
        self.original_image = None
        self.mask = None
        self.processed_mask = None  # mask after morphology
        self.model = None
        self.image_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel_size = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 1
        print(f"Using device: {self.device}")
        self.load_sam2_model()  # load once

        # --- classifier bits ---
        self.classifier = None
        self.classifier_ckpt = classifier_ckpt
        self.save_pred_json = save_pred_json
        if self.classifier_ckpt:
            try:
                self.classifier = RustClassifier(self.classifier_ckpt, device=self.device)
            except Exception as e:
                print(f"[Classifier] Failed to initialize: {e}")
                self.classifier = None

    # ---------------------------
    # Loading & IO
    # ---------------------------
    def load_sam2_model(self):
        try:
            print("Loading SAM 2 model...")
            self.model = SAM('sam2.1_b.pt')
            print("SAM 2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading SAM 2 model: {e}")
            messagebox.showerror("Error", f"Failed to load SAM 2 model: {e}")
            return False

    def load_image(self, image_path=None):
        if image_path is None:
            root = tk.Tk(); root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select metal object image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
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

    # ---------------------------
    # Morphology
    # ---------------------------
    def apply_morphological_operations(self, mask):
        if mask is None:
            return None
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations) if self.erosion_iterations > 0 else mask_uint8.copy()
        mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=self.dilation_iterations) if self.dilation_iterations > 0 else mask_eroded.copy()
        processed_mask = (mask_dilated > 127).astype(np.uint8)
        return processed_mask

    # ---------------------------
    # Classification helpers
    # ---------------------------
    def _largest_component_bbox(self, binary_mask: np.ndarray, min_area: int = 500):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = int(np.argmax(areas)) + 1
        if areas[max_idx - 1] < min_area:
            return None
        x = int(stats[max_idx, cv2.CC_STAT_LEFT])
        y = int(stats[max_idx, cv2.CC_STAT_TOP])
        w = int(stats[max_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[max_idx, cv2.CC_STAT_HEIGHT])
        return (x, y, x + w, y + h)

    def _crop_for_classifier(self, bgr_img: np.ndarray, binary_mask: np.ndarray, bbox):
        """Crop bbox and set background to white within the crop; return PIL RGB."""
        x1, y1, x2, y2 = bbox
        crop = bgr_img[y1:y2, x1:x2].copy()
        crop_mask = binary_mask[y1:y2, x1:x2]
        white = np.ones_like(crop, dtype=np.uint8) * 255
        crop_fg = np.where(crop_mask[..., None] == 1, crop, white)
        rgb = cv2.cvtColor(crop_fg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def classify_segment(self):
        """Run classifier on the largest connected component of processed_mask."""
        if self.classifier is None:
            print("[Classifier] Not initialized. Skipping classification.")
            return None
        if self.processed_mask is None or not np.any(self.processed_mask):
            print("[Classifier] No processed mask to classify. Skipping.")
            return None

        bbox = self._largest_component_bbox(self.processed_mask, min_area=500)
        if bbox is None:
            print("[Classifier] No sufficiently large component found. Skipping.")
            return None

        pil_crop = self._crop_for_classifier(self.original_image, self.processed_mask, bbox)
        result = self.classifier.predict_pil(pil_crop)
        pred_label = result["pred_label"]
        prob = result["probs"][pred_label]
        print(f"[Classifier] Prediction: {pred_label} ({prob:.2%}) | probs={result['probs']}")
        return {"bbox": bbox, **result}

    # ---------------------------
    # Save ONLY the processed metal parts image
    # ---------------------------
    def save_final_image(self, cls_result=None):
        """
        Save ONLY the processed metal parts (masked original image) to:
            ./results/<original_name>_processed.png
        Also saves classification JSON if provided.
        """
        if self.original_image is None or self.processed_mask is None or not np.any(self.processed_mask):
            messagebox.showwarning("No Result", "No processed metal parts to save.")
            return None

        metal_only = self.original_image * self.processed_mask[:, :, np.newaxis]

        save_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(save_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(self.image_path or "image"))[0]
        save_path = os.path.join(save_dir, f"{base}_processed.png")

        ok = cv2.imwrite(save_path, metal_only)
        if ok:
            print(f"[SAVED] Processed metal parts image saved at: {save_path}")
            messagebox.showinfo("Saved", f"Processed metal parts saved at:\n{save_path}")
            if cls_result is not None:
                json_path = os.path.join(save_dir, f"{base}_processed_pred.json")
                with open(json_path, "w") as f:
                    json.dump({
                        "image_path": self.image_path,
                        "output_image": save_path,
                        **cls_result
                    }, f, indent=2)
                print(f"[SAVED] Classification JSON: {json_path}")
            return save_path
        else:
            messagebox.showerror("Error", "Failed to save the processed image.")
            return None

    # ---------------------------
    # UI - Morphology window
    # ---------------------------
    def show_morphological_controls(self):
        control_window = Toplevel()
        control_window.title("Morphological Operations Controls")
        control_window.geometry("520x380")

        tk.Label(control_window, text="Kernel Size:").pack(pady=5)
        kernel_scale = Scale(control_window, from_=1, to=15, orient=tk.HORIZONTAL, length=380, showvalue=True)
        kernel_scale.set(self.kernel_size); kernel_scale.pack(pady=5)

        tk.Label(control_window, text="Erosion Iterations:").pack(pady=5)
        erosion_scale = Scale(control_window, from_=0, to=10, orient=tk.HORIZONTAL, length=380, showvalue=True)
        erosion_scale.set(self.erosion_iterations); erosion_scale.pack(pady=5)

        tk.Label(control_window, text="Dilation Iterations:").pack(pady=5)
        dilation_scale = Scale(control_window, from_=0, to=10, orient=tk.HORIZONTAL, length=380, showvalue=True)
        dilation_scale.set(self.dilation_iterations); dilation_scale.pack(pady=5)

        def apply_ops():
            self.kernel_size = kernel_scale.get()
            self.erosion_iterations = erosion_scale.get()
            self.dilation_iterations = dilation_scale.get()
            if self.mask is not None:
                self.processed_mask = self.apply_morphological_operations(self.mask)
                self.update_results_display()
            else:
                messagebox.showwarning("Warning", "No mask available to process")

        def reset_ops():
            kernel_scale.set(3); erosion_scale.set(1); dilation_scale.set(1)
            self.kernel_size = 3; self.erosion_iterations = 1; self.dilation_iterations = 1
            if self.mask is not None:
                self.processed_mask = self.mask.copy()
                self.update_results_display()

        btns = tk.Frame(control_window); btns.pack(pady=10)
        tk.Button(btns, text="Apply Operations", command=apply_ops).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Reset", command=reset_ops).pack(side=tk.LEFT, padx=6)

        # Save only the final processed metal parts (and classify)
        tk.Button(control_window, text="Save Processed Metal Parts",
                  command=lambda: self.save_final_image(self.classify_segment()),
                  bg="#28a745", fg="white", padx=12, pady=5).pack(pady=8)

        tk.Button(control_window, text="Close", command=control_window.destroy).pack(pady=6)

        if self.mask is not None and self.processed_mask is None:
            self.processed_mask = self.apply_morphological_operations(self.mask)

    # ---------------------------
    # Plots (for visualization only; not saved)
    # ---------------------------
    def update_results_display(self):
        if self.mask is None or self.processed_mask is None:
            return
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metal Segmentation with Morphological Operations', fontsize=16, fontweight='bold')

        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image"); axes[0, 0].axis('off')

        axes[0, 1].imshow(self.mask, cmap='viridis')
        axes[0, 1].set_title("Original Mask"); axes[0, 1].axis('off')

        metal_only_original = self.original_image * self.mask[:, :, np.newaxis]
        axes[0, 2].imshow(cv2.cvtColor(metal_only_original, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Original Metal Parts"); axes[0, 2].axis('off')

        axes[1, 0].imshow(self.processed_mask, cmap='viridis')
        axes[1, 0].set_title(f"Processed Mask\nKernel:{self.kernel_size}, Erosion:{self.erosion_iterations}, Dilation:{self.dilation_iterations}")
        axes[1, 0].axis('off')

        comparison = np.zeros_like(self.mask, dtype=np.uint8)
        comparison[(self.mask == 1) & (self.processed_mask == 0)] = 1
        comparison[(self.mask == 0) & (self.processed_mask == 1)] = 2
        comparison[(self.mask == 1) & (self.processed_mask == 1)] = 3
        custom_cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
        im = axes[1, 1].imshow(comparison, cmap=custom_cmap, vmin=0, vmax=3)
        axes[1, 1].set_title("Comparison"); axes[1, 1].axis('off')
        cbar = fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5]); cbar.set_ticklabels(['Background', 'Original Only', 'Processed Only', 'Both'])

        metal_only_processed = self.original_image * self.processed_mask[:, :, np.newaxis]
        axes[1, 2].imshow(cv2.cvtColor(metal_only_processed, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Processed Metal Parts"); axes[1, 2].axis('off')

        plt.tight_layout(); plt.show()

    # ---------------------------
    # Metal region proposals
    # ---------------------------
    def detect_metal_regions(self):
        if self.image is None:
            return []
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            _, A, _ = cv2.split(lab)
            metal_mask1 = cv2.inRange(A, 120, 135)  # grayish tones

            _, S, V = cv2.split(hsv)
            metal_mask2 = cv2.inRange(S, 30, 100) & cv2.inRange(V, 100, 255)

            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)

            combined_metal = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined_metal = cv2.bitwise_or(combined_metal, edges_dilated)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_CLOSE, kernel)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            min_area = 500
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    regions.append({
                        'id': len(regions),
                        'bbox': [x, y, x + w, y + h],
                        'center': [center_x, center_y],
                        'area': area
                    })
            return regions
        except Exception as e:
            print(f"Metal detection error: {e}")
            return []

    # ---------------------------
    # SAM + Fallback
    # ---------------------------
    def sam2_predict(self, points, labels):
        if self.model is None:
            messagebox.showerror("Error", "SAM 2 model not loaded!")
            return None
        try:
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            results = self.model.predict(image_rgb, points=points, labels=labels)
            if results and len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                mask = results[0].masks.data[0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                self.processed_mask = self.apply_morphological_operations(binary_mask)
                return binary_mask
            return None
        except Exception as e:
            print(f"SAM 2 prediction error: {e}")
            return self.fallback_segmentation(points, labels)

    def fallback_segmentation(self, points, labels):
        if len(points) == 0:
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.processed_mask = mask.copy()
            return mask

        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        for point, label in zip(points, labels):
            x, y = point
            if label == 1:
                cv2.circle(mask, (x, y), 15, 3, -1)  # sure foreground
            else:
                cv2.circle(mask, (x, y), 15, 0, -1)  # sure background
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        if any(l == 1 for l in labels):
            xs = [p[0] for p in points]; ys = [p[1] for p in points]
            x1 = max(0, min(xs) - 30); x2 = min(self.image.shape[1], max(xs) + 30)
            y1 = max(0, min(ys) - 30); y2 = min(self.image.shape[0], max(ys) + 30)
            rect = (x1, y1, x2 - x1, y2 - y1)
            cv2.grabCut(self.original_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        self.processed_mask = self.apply_morphological_operations(final_mask)
        return final_mask

    # ---------------------------
    # Interactive modes
    # ---------------------------
    def interactive_metal_segmentation(self):
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return False
        print("Detecting metal regions...")
        regions = self.detect_metal_regions()
        if regions:
            print(f"Found {len(regions)} potential metal regions - using auto-detection mode")
            return self.auto_detection_mode(regions)
        else:
            print("No metal regions auto-detected - using manual point mode")
            return self.manual_point_mode()

    def auto_detection_mode(self, regions):
        cv2.namedWindow("Click on METAL Parts - Press 's' when done")

        def click_callback(event, x, y, flags, param):
            regions_param = param['regions']
            if event == cv2.EVENT_LBUTTONDOWN:
                for region in regions_param:
                    x1, y1, x2, y2 = map(int, region['bbox'])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        print(f"Selected metal region {region['id']}")
                        center_x, center_y = region['center']
                        self.mask = self.sam2_predict(points=[[center_x, center_y]], labels=[1])
                        self.update_auto_display(regions_param, region['id'])
                        break

        self.update_auto_display(regions, -1)
        param = {'regions': regions}
        cv2.setMouseCallback("Click on METAL Parts - Press 's' when done", click_callback, param)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                cv2.destroyAllWindows()
                return self.manual_point_mode()
            elif key == ord('s'):
                if self.mask is not None and np.any(self.mask):
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Selection", "Please select a metal region first!")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def update_auto_display(self, regions, selected_id):
        display_image = self.original_image.copy()
        for region in regions:
            x1, y1, x2, y2 = map(int, region['bbox'])
            center_x, center_y = map(int, region['center'])
            if region['id'] == selected_id:
                color = (0, 255, 0); thickness = 3
                if self.mask is not None and np.any(self.mask):
                    overlay = display_image.copy()
                    overlay[self.mask == 1] = [0, 255, 0]
                    display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
            else:
                color = (255, 0, 0); thickness = 2
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display_image, f"M{region['id']}", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(display_image, f"M{region['id']}", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Click on METAL Parts - Press 's' when done", display_image)

    def manual_point_mode(self):
        print("Manual metal segmentation mode")
        cv2.namedWindow("Click on METAL Objects - Press 's' when done")
        points, labels = [], []

        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y]); labels.append(1)
                print(f"Added metal point at ({x}, {y})")
                self.mask = self.sam2_predict(points, labels)
                self.update_manual_display(points)

        cv2.setMouseCallback("Click on METAL Objects - Press 's' when done", click_callback)
        self.update_manual_display(points)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                points.clear(); labels.clear()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.processed_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.update_manual_display(points)
                print("Cleared all points")
            elif key == ord('s'):
                if points:
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Points", "Please click on metal areas first!")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def update_manual_display(self, points):
        display_image = self.original_image.copy()
        for x, y in points:
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
        if self.mask is not None and np.any(self.mask):
            overlay = display_image.copy()
            overlay[self.mask == 1] = [0, 255, 0]
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
        cv2.imshow("Click on METAL Objects - Press 's' when done", display_image)

    # ---------------------------
    # Orchestration
    # ---------------------------
    def show_results(self):
        if self.image is None or self.mask is None:
            messagebox.showwarning("No Results", "No segmentation results to display.")
            return
        self.show_morphological_controls()
        self.update_results_display()

        # Run classification (if available) and save final image + JSON
        cls_result = self.classify_segment()
        self.save_final_image(cls_result)

    def run(self, image_path=None):
        print("METAL OBJECT SEGMENTATION TOOL")
        print("=" * 50)
        if not self.load_image(image_path=image_path):
            return
        if not self.interactive_metal_segmentation():
            return
        self.show_results()
        print("Metal object segmentation completed.")

if __name__ == "__main__":
    # CLI: allow passing --ckpt and --image
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r"runs\rust_cls\resnet50_best.pth",
                        help="Path to trained checkpoint from train_rust.py")
    parser.add_argument("--image", type=str, default=None, help="Optional path to an image; if unset, a file dialog opens")
    args = parser.parse_args()

    segmenter = MetalSegmenter(classifier_ckpt=args.ckpt)
    segmenter.run(image_path=args.image)
