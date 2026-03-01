import os
import numpy as np
from PIL import Image
import torch

# File picker (native file explorer)
import tkinter as tk
from tkinter import filedialog

from transformers import Sam3Processor, Sam3Model


def pick_image_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def main():
    # ---- Settings you may want to tweak ----
    TEXT_PROMPT = "metal"          # try: "metal part", "steel", "aluminum", "metallic"
    THRESHOLD = 0.5                # object confidence threshold
    MASK_THRESHOLD = 0.5           # mask binarization threshold
    OUTPUT_DIR = "sam3_outputs"
    # ---------------------------------------

    image_path = pick_image_file()
    if not image_path:
        print("No image selected. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SAM3
    # Requires HF auth if the model is gated:
    #   pip install -U huggingface_hub
    #   hf auth login
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run text-prompted concept segmentation
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks", None)  # expected shape: [N, H, W]
    if masks is None or len(masks) == 0:
        print(f"No masks found for prompt: '{TEXT_PROMPT}'. Try a different prompt.")
        return

    # Combine all detected "metal" instance masks into one mask
    # masks is a torch tensor; convert to numpy uint8
    combined = torch.any(masks.bool(), dim=0).cpu().numpy().astype(np.uint8) * 255  # [H, W], {0,255}

    # Save mask
    mask_img = Image.fromarray(combined, mode="L")
    mask_path = os.path.join(OUTPUT_DIR, "metal_mask.png")
    mask_img.save(mask_path)

    # Create transparent cutout (RGBA): keep only metal pixels
    rgb = np.array(image)  # [H, W, 3]
    alpha = combined  # [H, W]
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    cutout = Image.fromarray(rgba, mode="RGBA")

    cutout_path = os.path.join(OUTPUT_DIR, "metal_cutout.png")
    cutout.save(cutout_path)

    print(f"Saved:\n  {mask_path}\n  {cutout_path}")
    print(f"Detected {masks.shape[0]} instance(s) for prompt: '{TEXT_PROMPT}'")


if __name__ == "__main__":
    main()
