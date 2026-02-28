import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# folder with the images to analyse
input_dir = Path(r"data/raw/Stahl - Cropped/APZ10/(2013) 5082/336 Hrs")
output_dir = Path("result")
output_dir.mkdir(exist_ok=True, parents=True)

# HSV ranges roughly matching rust colours
low1, high1 = np.array([0, 50, 50]), np.array([20, 255, 255])
low2, high2 = np.array([10, 100, 100]), np.array([30, 255, 255])
kernel = np.ones((5, 5), np.uint8)

def rust_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, low1, high1)
    m2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def shrink(img, scale=0.55):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def caption(img, text):
    out = img.copy()
    cv2.rectangle(out, (5, 5), (260, 45), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def process_image(img_path):
    name = Path(img_path).stem
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping unreadable file: {img_path}")
        return

    original = img.copy()
    mask = rust_mask(img)
    corrosion = cv2.bitwise_and(img, img, mask=mask)
    highlight = original.copy()
    highlight[mask > 0] = [0, 0, 255]

    # prepare smaller views for summary grid
    hsv_vis = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR)
    mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    img1 = caption(shrink(original), "Original")
    img2 = caption(shrink(hsv_vis), "HSV View")
    img3 = caption(shrink(mask_col), "Rust Mask")
    img4 = caption(shrink(corrosion), "Detected")
    img5 = caption(shrink(highlight), "Highlighted")

    total = mask.size
    rust_pixels = cv2.countNonZero(mask)
    ratio = (rust_pixels / total) * 100

    blank = np.zeros_like(img1)
    lines = [
        "Corrosion Summary",
        f"Total Pixels: {total:,}",
        f"Rust Pixels: {rust_pixels:,}",
        f"Rust %: {ratio:.2f}%"
    ]
    for i, line in enumerate(lines):
        y = 70 + i * 45
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(blank, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)

    # combine into grid
    row1 = cv2.hconcat([img1, img2, img3])
    row2 = cv2.hconcat([img4, img5, blank])
    combined = cv2.vconcat([row1, row2])

    # save results for this image
    cv2.imwrite(str(output_dir / f"{name}_mask.jpg"), mask)
    cv2.imwrite(str(output_dir / f"{name}_highlighted.jpg"), highlight)
    cv2.imwrite(str(output_dir / f"{name}_summary.jpg"), combined)
    print(f"{name}: {ratio:.2f}% rust area")

# run on all image files in the folder
for file in sorted(input_dir.glob("*.jpg")):
    process_image(file)

print("\nAll images processed. Results saved in:", output_dir)
