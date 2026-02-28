import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from ultralytics import SAM
import torch
from datetime import datetime

class CorrosionSegmenter:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.mask = None
        self.corrosion_mask = None
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
                title="Select corrosion image",
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
    
    def auto_detect_corrosion(self):
        """Automatically detect corrosion/dark regions using image processing"""
        if self.image is None:
            return None
        
        # Convert to different color spaces for better corrosion detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Method 1: Detect dark regions in grayscale
        _, dark_mask1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Method 2: Detect dark regions in LAB color space (L channel)
        L, A, B = cv2.split(lab)
        _, dark_mask2 = cv2.threshold(L, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Method 3: Detect in HSV for dark/black regions
        _, S, V = cv2.split(hsv)
        _, dark_mask3 = cv2.threshold(V, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask1, dark_mask2)
        combined_mask = cv2.bitwise_or(combined_mask, dark_mask3)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of corrosion regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small noise
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not filtered_contours:
            return None
        
        # Create final corrosion mask
        self.corrosion_mask = np.zeros_like(gray)
        cv2.drawContours(self.corrosion_mask, filtered_contours, -1, 255, -1)
        
        return filtered_contours
    
    def refine_with_sam2(self, contours):
        """Use SAM 2 to refine the corrosion detection"""
        if self.model is None or not contours:
            return self.corrosion_mask
        
        try:
            # Get the largest corrosion region
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of the largest corrosion area
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate center point for SAM 2 prompt
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Convert image to RGB for SAM 2
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Use the center point as foreground prompt for SAM 2
            points = np.array([[center_x, center_y]])
            labels = np.array([1])  # Foreground
            
            # Run SAM 2 prediction
            results = self.model.predict(image_rgb, points=points, labels=labels)
            
            if results and len(results) > 0:
                masks = results[0].masks
                if masks is not None and len(masks.data) > 0:
                    sam_mask = masks.data[0].cpu().numpy()
                    # Combine SAM mask with our corrosion mask
                    refined_mask = np.logical_or(self.corrosion_mask > 0, sam_mask > 0.5)
                    self.mask = refined_mask.astype(np.uint8) * 255
                    return self.mask
            
        except Exception as e:
            print(f"SAM 2 refinement error: {e}")
        
        # Fallback: use the corrosion mask directly
        self.mask = self.corrosion_mask
        return self.mask
    
    def interactive_corrosion_selection(self):
        """Allow user to select which corrosion regions to keep"""
        if self.image is None:
            return False
        
        # Auto-detect corrosion
        print("Detecting corrosion regions...")
        contours = self.auto_detect_corrosion()
        
        if not contours:
            messagebox.showinfo("No Corrosion", "No corrosion regions detected automatically.")
            return self.manual_corrosion_selection()
        
        print(f"Found {len(contours)} corrosion regions")
        
        # Refine with SAM 2
        self.refine_with_sam2(contours)
        
        # Show initial detection
        self.show_corrosion_detection(contours)
        
        cv2.namedWindow("Corrosion Detection - Press 's' to Save or 'q' to Quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save detection
                cv2.destroyAllWindows()
                return True
                
            elif key == ord('m'):  # Manual mode
                cv2.destroyAllWindows()
                return self.manual_corrosion_selection()
                
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def manual_corrosion_selection(self):
        """Manual selection of corrosion regions using SAM 2"""
        print("Manual corrosion selection mode")
        
        cv2.namedWindow("Click on Corrosion Areas - Press 's' when done")
        
        points = []
        labels = []
        
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                labels.append(1)  # Foreground (corrosion)
                print(f"Marked corrosion at ({x}, {y})")
                
                # Update segmentation
                self.update_manual_segmentation(points, labels)
        
        cv2.setMouseCallback("Click on Corrosion Areas - Press 's' when done", click_callback)
        
        # Initial display
        display_image = self.original_image.copy()
        cv2.putText(display_image, "Click on corrosion areas", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, "Press 's' to save, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Click on Corrosion Areas - Press 's' when done", display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                if points:
                    cv2.destroyAllWindows()
                    return True
                else:
                    messagebox.showwarning("No Points", "Click on corrosion areas first!")
                    
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def update_manual_segmentation(self, points, labels):
        """Update segmentation display for manual mode"""
        if not points:
            return
        
        try:
            # Use SAM 2 for segmentation
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            points_array = np.array(points)
            labels_array = np.array(labels)
            
            results = self.model.predict(image_rgb, points=points_array, labels=labels_array)
            
            if results and len(results) > 0:
                masks = results[0].masks
                if masks is not None and len(masks.data) > 0:
                    sam_mask = masks.data[0].cpu().numpy()
                    self.mask = (sam_mask > 0.5).astype(np.uint8) * 255
        
        except Exception as e:
            print(f"SAM 2 manual segmentation error: {e}")
            # Fallback: create mask from points
            self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            for point in points:
                x, y = point
                cv2.circle(self.mask, (x, y), 15, 255, -1)
        
        # Show result
        display_image = self.original_image.copy()
        
        # Draw points
        for point in points:
            x, y = point
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
        
        # Show segmentation overlay
        if self.mask is not None and np.any(self.mask):
            overlay = display_image.copy()
            overlay[self.mask > 0] = [0, 0, 255]  # Red for corrosion
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
        
        cv2.imshow("Click on Corrosion Areas - Press 's' when done", display_image)
    
    def show_corrosion_detection(self, contours):
        """Show the detected corrosion regions"""
        display_image = self.original_image.copy()
        
        # Draw contours
        cv2.drawContours(display_image, contours, -1, (0, 0, 255), 2)
        
        # Add info
        cv2.putText(display_image, f"Detected {len(contours)} corrosion regions", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, "Press 's' to save, 'm' for manual, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Corrosion Detection - Press 's' to Save or 'q' to Quit", display_image)
    
    def show_results(self):
        """Display corrosion segmentation results"""
        if self.image is None or self.mask is None:
            return
        
        # Create output images
        corrosion_only = self.original_image.copy()
        corrosion_only[self.mask == 0] = 0  # Black out non-corrosion areas
        
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]  # White background for non-corrosion
        
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = [0, 0, 255]  # Red overlay for corrosion
        
        # Calculate corrosion percentage
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        corrosion_pixels = np.sum(self.mask > 0)
        corrosion_percentage = (corrosion_pixels / total_pixels) * 100
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Corrosion Analysis - {corrosion_percentage:.1f}% Corrosion Detected', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        # Corrosion mask
        axes[0, 1].imshow(self.mask, cmap='hot')
        axes[0, 1].set_title("Corrosion Mask", fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Corrosion Overlay (Red)", fontweight='bold')
        axes[0, 2].axis('off')
        
        # Corrosion only
        axes[1, 0].imshow(cv2.cvtColor(corrosion_only, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Corrosion Areas Only", fontweight='bold')
        axes[1, 0].axis('off')
        
        # White background
        axes[1, 1].imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("White Background", fontweight='bold')
        axes[1, 1].axis('off')
        
        # Corrosion percentage pie chart
        labels = ['Corrosion', 'Intact']
        sizes = [corrosion_percentage, 100 - corrosion_percentage]
        colors = ['red', 'green']
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title("Surface Condition", fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Corrosion Analysis: {corrosion_percentage:.2f}% of surface is corroded")
        print(f"Corrosion Area: {corrosion_pixels} pixels")
        print(f"Total Area: {total_pixels} pixels")
    
    def save_results(self):
        """Automatically save corrosion analysis results to results directory"""
        if self.image is None or self.mask is None or self.results_dir is None:
            messagebox.showerror("Error", "No results to save or results directory not created!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"corrosion_analysis_{timestamp}"
        
        # Calculate metrics
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        corrosion_pixels = np.sum(self.mask > 0)
        corrosion_percentage = (corrosion_pixels / total_pixels) * 100
        
        # Save images
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_original.jpg"), self.original_image)
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_corrosion_mask.png"), self.mask)
        
        corrosion_only = self.original_image.copy()
        corrosion_only[self.mask == 0] = 0
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_corrosion_only.png"), corrosion_only)
        
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_white_bg.jpg"), white_bg)
        
        # Save overlay image
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = [0, 0, 255]
        cv2.imwrite(os.path.join(self.results_dir, f"{base_name}_overlay.jpg"), overlay)
        
        # Save analysis report
        with open(os.path.join(self.results_dir, f"{base_name}_report.txt"), "w") as f:
            f.write("CORROSION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Image: {os.path.basename(self.image_path)}\n")
            f.write(f"Image Dimensions: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n")
            f.write(f"Total Surface Area: {total_pixels} pixels\n")
            f.write(f"Corroded Area: {corrosion_pixels} pixels\n")
            f.write(f"Corrosion Percentage: {corrosion_percentage:.2f}%\n\n")
            
            if corrosion_percentage < 5:
                f.write("SEVERITY: LOW\n")
                f.write("RECOMMENDATION: Regular monitoring recommended\n")
            elif corrosion_percentage < 20:
                f.write("SEVERITY: MEDIUM\n")
                f.write("RECOMMENDATION: Schedule inspection and consider protective coatings\n")
            elif corrosion_percentage < 50:
                f.write("SEVERITY: HIGH\n")
                f.write("RECOMMENDATION: Immediate inspection and maintenance required\n")
            else:
                f.write("SEVERITY: CRITICAL\n")
                f.write("RECOMMENDATION: Urgent action required - structural integrity may be compromised\n")
        
        # Save visualization plot
        self.save_visualization_plot(corrosion_percentage, corrosion_pixels, total_pixels)
        
        print(f"Results automatically saved to: {self.results_dir}")
        messagebox.showinfo("Success", f"Corrosion analysis automatically saved to:\n{self.results_dir}")
    
    def save_visualization_plot(self, corrosion_percentage, corrosion_pixels, total_pixels):
        """Save the visualization plot as an image"""
        # Create output images for the plot
        corrosion_only = self.original_image.copy()
        corrosion_only[self.mask == 0] = 0
        
        white_bg = self.original_image.copy()
        white_bg[self.mask == 0] = [255, 255, 255]
        
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = [0, 0, 255]
        
        # Create and save the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Corrosion Analysis - {corrosion_percentage:.1f}% Corrosion Detected', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        # Corrosion mask
        axes[0, 1].imshow(self.mask, cmap='hot')
        axes[0, 1].set_title("Corrosion Mask", fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Corrosion Overlay (Red)", fontweight='bold')
        axes[0, 2].axis('off')
        
        # Corrosion only
        axes[1, 0].imshow(cv2.cvtColor(corrosion_only, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Corrosion Areas Only", fontweight='bold')
        axes[1, 0].axis('off')
        
        # White background
        axes[1, 1].imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("White Background", fontweight='bold')
        axes[1, 1].axis('off')
        
        # Corrosion percentage pie chart
        labels = ['Corrosion', 'Intact']
        sizes = [corrosion_percentage, 100 - corrosion_percentage]
        colors = ['red', 'green']
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title("Surface Condition", fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.results_dir, f"corrosion_analysis_{timestamp}_visualization.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
    
    def run(self):
        """Main workflow"""
        print("CORROSION SEGMENTATION AND ANALYSIS TOOL")
        print("=" * 50)
        
        if not self.load_sam2_model():
            return
        
        if not self.load_image():
            return
        
        if not self.interactive_corrosion_selection():
            return
        
        self.show_results()
        
        # Automatically save results without asking
        self.save_results()
        
        print("Corrosion analysis completed and automatically saved!")

if __name__ == "__main__":
    segmenter = CorrosionSegmenter()
    segmenter.run()