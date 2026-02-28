import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from ultralytics import SAM
import torch
from datetime import datetime

class MetalSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None
        self.model = None
        self.image_path = None
        self.results_dir = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        
        # Create results directory in the same location as the image
        self.create_results_directory()
        
        print(f"Image loaded: {self.image.shape}")
        return True
    
    def create_results_directory(self):
        """Create results directory in the same location as the input image"""
        if self.image_path:
            image_dir = os.path.dirname(self.image_path)
            image_filename = os.path.splitext(os.path.basename(self.image_path))[0]
            
            # Create results directory name based on image filename
            self.results_dir = os.path.join(image_dir, f"{image_filename}_results")
            
            # Create the directory if it doesn't exist
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"Results will be saved to: {self.results_dir}")
    
    def detect_metal_regions(self):
        """Automatically detect metal-like regions based on visual properties"""
        if self.image is None:
            return []
        
        try:
            # Convert to different color spaces for metal detection
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            
            # Metal detection strategies:
            
            # 1. Detect shiny/reflective areas (high contrast in grayscale)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # 2. Detect metallic colors (grayish, silver tones in LAB space)
            L, A, B = cv2.split(lab)
            # Metal often has specific A and B channel characteristics
            metal_mask1 = cv2.inRange(A, 120, 135)  # Grayish tones
            
            # 3. Detect in HSV for metallic reflections
            H, S, V = cv2.split(hsv)
            # Metal often has medium saturation and high value
            metal_mask2 = cv2.inRange(S, 30, 100) & cv2.inRange(V, 100, 255)
            
            # 4. Edge density - metal objects often have clear edges
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Combine detection methods
            combined_metal = cv2.bitwise_or(metal_mask1, metal_mask2)
            combined_metal = cv2.bitwise_or(combined_metal, edges_dilated)
            
            # Clean up the mask
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_CLOSE, kernel)
            combined_metal = cv2.morphologyEx(combined_metal, cv2.MORPH_OPEN, kernel)
            
            # Find contours of potential metal regions
            contours, _ = cv2.findContours(combined_metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and create regions
            regions = []
            min_area = 500  # Minimum area for metal regions
            
            for i, contour in enumerate(contours):
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
    
    def sam2_predict(self, points, labels):
        """Predict segmentation mask using SAM 2"""
        if self.model is None:
            messagebox.showerror("Error", "SAM 2 model not loaded!")
            return None
        
        try:
            # Convert image to RGB for SAM 2
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Run SAM 2 prediction with points
            results = self.model.predict(image_rgb, points=points, labels=labels)
            
            if results and len(results) > 0:
                # Get masks from results
                masks = results[0].masks
                if masks is not None and len(masks.data) > 0:
                    # Get the first mask
                    mask = masks.data[0].cpu().numpy()
                    return mask
                
            return None
            
        except Exception as e:
            print(f"SAM 2 prediction error: {e}")
            return self.fallback_segmentation(points, labels)
    
    def fallback_segmentation(self, points, labels):
        """Fallback segmentation using GrabCut"""
        if len(points) == 0:
            return np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        # Create mask based on points
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        # Mark points on mask
        for point, label in zip(points, labels):
            x, y = point
            if label == 1:  # Foreground (metal)
                cv2.circle(mask, (x, y), 15, 3, -1)  # Sure foreground
            else:  # Background
                cv2.circle(mask, (x, y), 15, 0, -1)  # Sure background
        
        # Use GrabCut for segmentation
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Create bounding box from points
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
                        # Use the center point for SAM 2 segmentation
                        center_x, center_y = region['center']
                        self.mask = self.sam2_predict(points=[[center_x, center_y]], labels=[1])
                        self.update_auto_display(regions_param, region['id'])
                        break
        
        # Initial display
        self.update_auto_display(regions, -1)
        
        # Set mouse callback
        param = {'regions': regions}
        cv2.setMouseCallback("Click on METAL Parts - Press 's' when done", click_callback, param)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('m'):  # Switch to manual mode
                cv2.destroyAllWindows()
                return self.manual_point_mode()
                
            elif key == ord('s'):  # Save selection
                if self.mask is not None and np.any(self.mask):
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Selection", "Please select a metal region first!")
                    
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def update_auto_display(self, regions, selected_id):
        """Update display for auto-detection mode"""
        display_image = self.original_image.copy()
        
        for region in regions:
            x1, y1, x2, y2 = map(int, region['bbox'])
            center_x, center_y = map(int, region['center'])
            
            if region['id'] == selected_id:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
                
                # Show segmentation overlay
                if self.mask is not None and np.any(self.mask):
                    overlay = display_image.copy()
                    overlay[self.mask == 1] = [0, 255, 0]  # Green for metal
                    display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
            else:
                color = (255, 0, 0)  # Blue for potential metal regions
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw region ID
            cv2.putText(display_image, f"M{region['id']}", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(display_image, f"M{region['id']}", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add instructions
        instructions = [
            "METAL DETECTION MODE",
            f"Found {len(regions)} potential metal regions",
            "Click on any METAL REGION (M0, M1, etc.) to select it",
            "Green = Selected metal, Blue = Potential metal",
            "Press 'm' for manual mode, 's' to save, 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_image, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Click on METAL Parts - Press 's' when done", display_image)
    
    def manual_point_mode(self):
        """Manual point-based segmentation for metal objects"""
        print("Manual metal segmentation mode")
        
        cv2.namedWindow("Click on METAL Objects - Press 's' when done")
        
        points = []
        labels = []
        
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                labels.append(1)  # Metal foreground
                print(f"Added metal point at ({x}, {y})")
                
                # Update segmentation
                self.mask = self.sam2_predict(points, labels)
                self.update_manual_display(points)
        
        cv2.setMouseCallback("Click on METAL Objects - Press 's' when done", click_callback)
        self.update_manual_display(points)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Clear points
                points = []
                labels = []
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.update_manual_display(points)
                print("Cleared all points")
                
            elif key == ord('s'):  # Save
                if points:
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Points", "Please click on metal areas first!")
                    
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def update_manual_display(self, points):
        """Update display for manual point mode"""
        display_image = self.original_image.copy()
        
        # Draw points
        for point in points:
            x, y = point
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)  # Green for metal
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
        
        # Show current mask if available
        if self.mask is not None and np.any(self.mask):
            overlay = display_image.copy()
            overlay[self.mask == 1] = [0, 255, 0]  # Green overlay for metal
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
        
        # Add instructions
        instructions = [
            "MANUAL METAL SEGMENTATION",
            "Click on METAL objects to segment them",
            "Green circles = Your clicks, Green overlay = Detected metal",
            "Press 'c' to clear points, 's' to save, 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_image, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Click on METAL Objects - Press 's' when done", display_image)
    
    def show_results(self):
        """Display metal segmentation results"""
        if self.image is None or self.mask is None:
            return
        
        # Create output images
        metal_only = self.original_image * self.mask[:, :, np.newaxis]
        
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]
        
        transparent_bg = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2BGRA)
        transparent_bg[:, :, 3] = self.mask * 255
        
        black_bg = self.original_image.copy()
        black_bg[self.mask == 0] = [0, 0, 0]
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metal Object Segmentation Results', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        axes[0, 1].imshow(self.mask, cmap='viridis')
        axes[0, 1].set_title("Metal Segmentation Mask", fontweight='bold')
        axes[0, 1].axis('off')
        
        # Metal only (black background)
        axes[0, 2].imshow(cv2.cvtColor(metal_only, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Metal Parts Only", fontweight='bold')
        axes[0, 2].axis('off')
        
        # White background
        axes[1, 0].imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("White Background", fontweight='bold')
        axes[1, 0].axis('off')
        
        # Black background
        axes[1, 1].imshow(cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Black Background", fontweight='bold')
        axes[1, 1].axis('off')
        
        # Area analysis
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        metal_pixels = np.sum(self.mask > 0)
        metal_percentage = (metal_pixels / total_pixels) * 100
        
        labels = ['Metal', 'Background']
        sizes = [metal_percentage, 100 - metal_percentage]
        colors = ['silver', 'lightgray']
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title("Metal Area Distribution", fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Metal Area: {metal_pixels} pixels ({metal_percentage:.2f}%)")
        print(f"Background Area: {total_pixels - metal_pixels} pixels ({100-metal_percentage:.2f}%)")
    
    def save_results(self):
        """Automatically save metal segmentation results to results directory"""
        if self.image is None or self.mask is None or self.results_dir is None:
            messagebox.showerror("Error", "No results to save or results directory not created!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"metal_segmentation_{timestamp}"
        
        # Calculate metrics
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        metal_pixels = np.sum(self.mask > 0)
        metal_percentage = (metal_pixels / total_pixels) * 100
        
        # Save images
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_original.jpg"), self.original_image)
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_metal_mask.png"), self.mask * 255)
        
        # Metal only (black background)
        metal_only = self.original_image * self.mask[:, :, np.newaxis]
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_metal_only.png"), metal_only)
        
        # White background
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_white_bg.jpg"), white_bg)
        
        # Black background
        black_bg = self.original_image.copy()
        black_bg[self.mask == 0] = [0, 0, 0]
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_black_bg.jpg"), black_bg)
        
        # Transparent background
        transparent_bg = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2BGRA)
        transparent_bg[:, :, 3] = self.mask * 255
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_transparent.png"), transparent_bg)
        
        # Save analysis report
        with open(os.path.join(self.results_dir, f"{base_name}_report.txt"), "w") as f:
            f.write("METAL OBJECT SEGMENTATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Image: {os.path.basename(self.image_path)}\n")
            f.write(f"Image Dimensions: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n")
            f.write(f"Total Image Area: {total_pixels} pixels\n")
            f.write(f"Metal Area: {metal_pixels} pixels\n")
            f.write(f"Background Area: {total_pixels - metal_pixels} pixels\n")
            f.write(f"Metal Percentage: {metal_percentage:.2f}%\n")
            f.write(f"Background Percentage: {100 - metal_percentage:.2f}%\n")
        
        # Save visualization plot
        self.save_visualization_plot(metal_percentage, metal_pixels, total_pixels)
        
        print(f"Results automatically saved to: {self.results_dir}")
        messagebox.showinfo("Success", f"Metal segmentation results automatically saved to:\n{self.results_dir}")
    
    def save_visualization_plot(self, metal_percentage, metal_pixels, total_pixels):
        """Save the visualization plot as an image"""
        # Create output images for the plot
        metal_only = self.original_image * self.mask[:, :, np.newaxis]
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]
        black_bg = self.original_image.copy()
        black_bg[self.mask == 0] = [0, 0, 0]
        
        # Create and save the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metal Object Segmentation Results', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        axes[0, 1].imshow(self.mask, cmap='viridis')
        axes[0, 1].set_title("Metal Segmentation Mask", fontweight='bold')
        axes[0, 1].axis('off')
        
        # Metal only
        axes[0, 2].imshow(cv2.cvtColor(metal_only, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Metal Parts Only", fontweight='bold')
        axes[0, 2].axis('off')
        
        # White background
        axes[1, 0].imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("White Background", fontweight='bold')
        axes[1, 0].axis('off')
        
        # Black background
        axes[1, 1].imshow(cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Black Background", fontweight='bold')
        axes[1, 1].axis('off')
        
        # Area distribution pie chart
        labels = ['Metal', 'Background']
        sizes = [metal_percentage, 100 - metal_percentage]
        colors = ['silver', 'lightgray']
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title("Metal Area Distribution", fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.results_dir, f"metal_segmentation_{timestamp}_visualization.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        # Automatically save results without asking
        self.save_results()
        
        print("Metal object segmentation completed and automatically saved!")

if __name__ == "__main__":
    segmenter = MetalSegmenter()
    segmenter.run()