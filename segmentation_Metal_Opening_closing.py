import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Toplevel
from ultralytics import SAM
import torch

class MetalSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None
        self.processed_mask = None  # For storing mask after morphological operations
        self.model = None
        self.image_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel_size = 5
        self.erosion_iterations = 1
        self.dilation_iterations = 1
        print(f"Using device: {self.device}")
        self.load_sam2_model()
    
    def load_sam2_model(self):
        """Load SAM 2 model from Ultralytics"""
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
        """Load an image from file"""
        if image_path is None:
            root = tk.Tk()
            root.withdraw()
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
    
    def apply_morphological_operations(self, mask):
        """Apply erosion and dilation operations to the mask"""
        if mask is None:
            return None
            
        # Convert to uint8 if needed
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Create kernel for morphological operations
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Apply erosion
        if self.erosion_iterations > 0:
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=self.erosion_iterations)
        else:
            mask_eroded = mask_uint8.copy()
        
        # Apply dilation
        if self.dilation_iterations > 0:
            mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=self.dilation_iterations)
        else:
            mask_dilated = mask_eroded.copy()
        
        # Convert back to binary mask
        processed_mask = (mask_dilated > 127).astype(np.uint8)
        
        return processed_mask
    
    def show_morphological_controls(self):
        """Show controls for morphological operations"""
        control_window = Toplevel()
        control_window.title("Morphological Operations Controls")
        control_window.geometry("400x300")
        
        # Kernel size control
        tk.Label(control_window, text="Kernel Size:").pack(pady=5)
        kernel_scale = Scale(control_window, from_=1, to=15, orient=tk.HORIZONTAL, 
                           length=300, showvalue=True)
        kernel_scale.set(self.kernel_size)
        kernel_scale.pack(pady=5)
        
        # Erosion iterations control
        tk.Label(control_window, text="Erosion Iterations:").pack(pady=5)
        erosion_scale = Scale(control_window, from_=0, to=10, orient=tk.HORIZONTAL, 
                            length=300, showvalue=True)
        erosion_scale.set(self.erosion_iterations)
        erosion_scale.pack(pady=5)
        
        # Dilation iterations control
        tk.Label(control_window, text="Dilation Iterations:").pack(pady=5)
        dilation_scale = Scale(control_window, from_=0, to=10, orient=tk.HORIZONTAL, 
                             length=300, showvalue=True)
        dilation_scale.set(self.dilation_iterations)
        dilation_scale.pack(pady=5)
        
        def apply_operations():
            self.kernel_size = kernel_scale.get()
            self.erosion_iterations = erosion_scale.get()
            self.dilation_iterations = dilation_scale.get()
            
            if self.mask is not None:
                self.processed_mask = self.apply_morphological_operations(self.mask)
                self.update_results_display()
            else:
                messagebox.showwarning("Warning", "No mask available to process")
        
        def reset_operations():
            kernel_scale.set(3)
            erosion_scale.set(1)
            dilation_scale.set(1)
            self.kernel_size = 3
            self.erosion_iterations = 1
            self.dilation_iterations = 1
            
            if self.mask is not None:
                self.processed_mask = self.mask.copy()
                self.update_results_display()
        
        # Buttons
        button_frame = tk.Frame(control_window)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Apply Operations", command=apply_operations).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset", command=reset_operations).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=control_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Initialize processed mask
        if self.mask is not None and self.processed_mask is None:
            self.processed_mask = self.apply_morphological_operations(self.mask)
    
    def update_results_display(self):
        """Update the results display with current morphological operations"""
        if self.mask is None or self.processed_mask is None:
            return
            
        # Create comparison display
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metal Segmentation with Morphological Operations', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        # Original mask
        axes[0, 1].imshow(self.mask, cmap='viridis')
        axes[0, 1].set_title("Original Mask", fontweight='bold')
        axes[0, 1].axis('off')
        
        # Original metal parts
        metal_only_original = self.original_image * self.mask[:, :, np.newaxis]
        axes[0, 2].imshow(cv2.cvtColor(metal_only_original, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Original Metal Parts", fontweight='bold')
        axes[0, 2].axis('off')
        
        # Processed mask
        axes[1, 0].imshow(self.processed_mask, cmap='viridis')
        axes[1, 0].set_title(f"Processed Mask\nKernel: {self.kernel_size}, Erosion: {self.erosion_iterations}, Dilation: {self.dilation_iterations}", 
                           fontweight='bold')
        axes[1, 0].axis('off')
        
        # Comparison: Original vs Processed
        comparison = np.zeros_like(self.mask, dtype=np.uint8)
        comparison[self.mask == 1] = 1  # Original only
        comparison[self.processed_mask == 1] = 2  # Processed only
        comparison[(self.mask == 1) & (self.processed_mask == 1)] = 3  # Both
        
        axes[1, 1].imshow(comparison, cmap='tab10')
        axes[1, 1].set_title("Comparison\n(Red=Original, Green=Processed, Yellow=Both)", fontweight='bold')
        axes[1, 1].axis('off')
        
        # Processed metal parts
        metal_only_processed = self.original_image * self.processed_mask[:, :, np.newaxis]
        axes[1, 2].imshow(cv2.cvtColor(metal_only_processed, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Processed Metal Parts", fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def detect_metal_regions(self):
        """Automatically detect metal-like regions based on visual properties"""
        if self.image is None:
            return []
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            
            # Calculate image statistics for adaptive thresholding
            gray_mean = np.mean(gray)
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)

            # Enhanced metal detection in LAB space - focus on metallic reflections
            L, A, B = cv2.split(lab)
            
            # Metal typically has specific characteristics in LAB space
            # Higher A channel values for metallic gray/blue tones
            metal_mask1 = cv2.inRange(A, 125, 140)
            
            # Look for high contrast regions (typical of metal reflections)
            contrast = cv2.Laplacian(gray, cv2.CV_64F)
            contrast = np.uint8(np.absolute(contrast))
            _, high_contrast = cv2.threshold(contrast, 30, 255, cv2.THRESH_BINARY)
            
            # HSV-based metal detection - exclude rope-like colors
            H, S, V = cv2.split(hsv)
            # Rope typically has brown/yellow tones (H: 10-30, S: 40-100, V: 50-150)
            rope_mask = cv2.inRange(hsv, (10, 40, 50), (30, 100, 150))
            rope_mask = cv2.bitwise_not(rope_mask)  # Invert to exclude ropes
            
            # Metal in HSV: high saturation, medium-high value
            metal_mask2 = cv2.inRange(S, 40, 120) & cv2.inRange(V, 100, 220) & rope_mask
            
            # Edge detection for metal boundaries
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Combine metal detection methods
            combined_metal = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined_metal = cv2.bitwise_or(combined_metal, high_contrast)
            
            # Use edges to refine metal regions
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            combined_metal = cv2.bitwise_and(combined_metal, edges_dilated)
            
            # Morphological operations to clean up
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_CLOSE, kernel)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter regions based on shape and size
            regions = []
            min_area = 800  # Increased minimum area to exclude small noise
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Calculate solidity to exclude irregular shapes (like ropes)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    # Metal parts tend to be more solid/compact
                    if solidity > 0.3:  # Exclude very irregular shapes
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        regions.append({
                            'id': len(regions),
                            'bbox': [x, y, x + w, y + h],
                            'center': [center_x, center_y],
                            'area': area,
                            'solidity': solidity
                        })
            
            # Sort regions by area (largest first) - likely to be main metal parts
            regions.sort(key=lambda x: x['area'], reverse=True)
            
            return regions
        except Exception as e:
            print(f"Metal detection error: {e}")
            return []
    
    def sam2_predict(self, points, labels):
        """Predict segmentation mask using SAM 2 with enhanced metal detection"""
        if self.model is None:
            messagebox.showerror("Error", "SAM 2 model not loaded!")
            return None
        try:
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            results = self.model.predict(image_rgb, points=points, labels=labels)
            if results and len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                mask = results[0].masks.data[0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # Post-process mask to remove rope-like regions
                binary_mask = self.remove_rope_regions(binary_mask)
                
                # Apply initial morphological operations
                self.processed_mask = self.apply_morphological_operations(binary_mask)
                return binary_mask
            return None
        except Exception as e:
            print(f"SAM 2 prediction error: {e}")
            return self.fallback_segmentation(points, labels)
    
    def remove_rope_regions(self, mask):
        """Remove rope-like regions from the mask based on color and shape"""
        if mask is None or not np.any(mask):
            return mask
            
        # Create masked image
        masked_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # Define rope color range (brown/yellow tones)
        rope_lower = np.array([10, 40, 50])
        rope_upper = np.array([30, 100, 150])
        rope_mask = cv2.inRange(hsv, rope_lower, rope_upper)
        
        # Remove rope-colored regions from the mask
        cleaned_mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(rope_mask))
        
        # Also remove very elongated regions (characteristic of ropes)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros_like(cleaned_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area
                # Calculate aspect ratio to detect elongated shapes
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                # Ropes are typically very elongated (high aspect ratio)
                if aspect_ratio < 8:  # Keep only moderately elongated shapes
                    cv2.fillPoly(final_mask, [contour], 1)
        
        return final_mask if np.any(final_mask) else cleaned_mask
    
    def fallback_segmentation(self, points, labels):
        """Fallback segmentation using GrabCut with rope exclusion"""
        if len(points) == 0:
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            self.processed_mask = mask.copy()
            return mask
            
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        for point, label in zip(points, labels):
            x, y = point
            if label == 1:  # Foreground (metal)
                cv2.circle(mask, (x, y), 15, 3, -1)  # Sure foreground
            else:
                cv2.circle(mask, (x, y), 15, 0, -1)  # Sure background
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        if any(l == 1 for l in labels):
            fg_points = [points[i] for i in range(len(points)) if labels[i] == 1]
            if fg_points:
                x_coords = [p[0] for p in fg_points]
                y_coords = [p[1] for p in fg_points]
                x1 = max(0, min(x_coords) - 30)
                x2 = min(self.image.shape[1], max(x_coords) + 30)
                y1 = max(0, min(y_coords) - 30)
                y2 = min(self.image.shape[0], max(y_coords) + 30)
                rect = (x1, y1, x2 - x1, y2 - y1)
                cv2.grabCut(self.original_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        
        # Remove rope regions from GrabCut result
        final_mask = self.remove_rope_regions(final_mask)
        
        self.processed_mask = self.apply_morphological_operations(final_mask)
        return final_mask
    
    def interactive_metal_segmentation(self):
        """Main interactive segmentation for metal objects"""
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
        """Auto-detection mode where user clicks on detected metal regions"""
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
        """Update display for auto-detection mode"""
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
        
        instructions = [
            "METAL DETECTION MODE",
            f"Found {len(regions)} potential metal regions",
            "Click on any METAL REGION (M0, M1, etc.) to select it",
            "Green = Selected metal, Blue = Potential metal",
            "Press 'm' for manual mode, 's' to save selection, 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_image, instruction, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Click on METAL Parts - Press 's' when done", display_image)
    
    def manual_point_mode(self):
        """Manual point-based segmentation for metal objects with rope exclusion"""
        print("Manual metal segmentation mode")
        cv2.namedWindow("Click on METAL Objects - Press 's' when done")
        points, labels = [], []
        
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click for metal points
                points.append([x, y])
                labels.append(1)
                print(f"Added metal point at ({x}, {y})")
                self.mask = self.sam2_predict(points, labels)
                self.update_manual_display(points, [])
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click for non-metal points (to exclude ropes)
                points.append([x, y])
                labels.append(0)
                print(f"Added exclusion point at ({x}, {y}) - to remove rope")
                self.mask = self.sam2_predict(points, labels)
                self.update_manual_display([p for i, p in enumerate(points) if labels[i] == 1], 
                                         [p for i, p in enumerate(points) if labels[i] == 0])
        
        cv2.setMouseCallback("Click on METAL Objects - Press 's' when done", click_callback)
        self.update_manual_display(points, [])
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                points.clear()
                labels.clear()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.processed_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.update_manual_display(points, [])
                print("Cleared all points")
            elif key == ord('s'):
                if any(l == 1 for l in labels):  # At least one metal point
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Points", "Please click on metal areas first!")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
    
    def update_manual_display(self, metal_points, exclude_points):
        """Update display for manual point mode"""
        display_image = self.original_image.copy()
        
        # Draw metal points (green)
        for x, y in metal_points:
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
        
        # Draw exclusion points (red)
        for x, y in exclude_points:
            cv2.circle(display_image, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(display_image, "EXCLUDE", (x+12, y-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if self.mask is not None and np.any(self.mask):
            overlay = display_image.copy()
            overlay[self.mask == 1] = [0, 255, 0]
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
        
        instructions = [
            "MANUAL METAL SEGMENTATION",
            "LEFT CLICK on METAL objects to select them",
            "RIGHT CLICK on ROPES/NON-METAL to exclude them",
            "Green circles = Metal points, Red circles = Exclusion points",
            "Green overlay = Detected metal",
            "Press 'c' to clear points, 's' to save selection, 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_image, instruction, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Click on METAL Objects - Press 's' when done", display_image)
    
    def show_results(self):
        """Display metal segmentation results with morphological operations controls"""
        if self.image is None or self.mask is None:
            messagebox.showwarning("No Results", "No segmentation results to display.")
            return
        
        # Show morphological controls
        self.show_morphological_controls()
        
        # Show initial results
        self.update_results_display()
    
    def run(self):
        """Main workflow"""
        print("METAL OBJECT SEGMENTATION TOOL")
        print("=" * 50)
        if not self.load_sam2_model():
            return
        if not self.load_image():
            return
        if not self.interactive_metal_segmentation():
            return
        self.show_results()
        print("Metal object segmentation completed.")

if __name__ == "__main__":
    segmenter = MetalSegmenter()
    segmenter.run()