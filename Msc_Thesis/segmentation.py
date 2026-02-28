import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# make sure the output folder exists
os.makedirs("result", exist_ok=True)

# choose the image to check
image_path = r"data/raw/Stahl - Cropped/APZ10/(2013) 5082/336 Hrs/336h (2)_20130211 3rd Back.JPG"
#image_path = r"data/raw/Cronidur 30 - Cropped/(2013) 5225/480/480h (1)_20130408 1st Back.JPG"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")

img = cv2.imread(image_path)
if img is None:
    raise ValueError("Unable to open the image file.")

orig = img.copy()

# --- exposure balance and glare cleanup ---
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_eq = clahe.apply(l)
lab_eq = cv2.merge((l_eq, a, b))
img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
img[bright == 255] = [0, 0, 0]

# --- noise filtering ---
img = cv2.bilateralFilter(img, 9, 75, 75)
img = cv2.medianBlur(img, 3)

# --- rust segmentation in HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_med = np.median(hsv[:, :, 2])
v_low = max(30, int(v_med * 0.4))
v_high = min(255, int(v_med * 1.4))

low1 = np.array([0, 40, v_low])
high1 = np.array([25, 255, v_high])
low2 = np.array([10, 60, v_low])
high2 = np.array([35, 255, v_high])

m1 = cv2.inRange(hsv, low1, high1)
m2 = cv2.inRange(hsv, low2, high2)
mask = cv2.bitwise_or(m1, m2)

# --- morphological cleaning ---
noise = np.std(gray)
k = int(np.clip(noise / 20, 3, 9))
if k % 2 == 0:
    k += 1
kernel = np.ones((k, k), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean = np.zeros_like(mask)
for c in contours:
    if cv2.contourArea(c) > 200:
        cv2.drawContours(clean, [c], -1, 255, -1)
mask = clean

# --- build result images ---
detected = cv2.bitwise_and(orig, orig, mask=mask)
highlight = orig.copy()
highlight[mask > 0] = [0, 0, 255]

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def resize(img, scale=0.55):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def tag(img, text):
    out = img.copy()
    cv2.rectangle(out, (5, 5), (260, 50), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out

a = tag(resize(orig), "Original")
b = tag(resize(hsv_rgb), "HSV")
c = tag(resize(mask_rgb), "Mask")
d = tag(resize(detected), "Detected")
e = tag(resize(highlight), "Highlighted")

# --- simple stats display ---
total = mask.size
rust_pixels = cv2.countNonZero(mask)
rust_pct = (rust_pixels / total) * 100

blank = np.zeros_like(a)
lines = [
    "Corrosion Summary",
    f"Total Pixels: {total:,}",
    f"Rust Pixels: {rust_pixels:,}",
    f"Rust %: {rust_pct:.2f}%"
]
for i, line in enumerate(lines):
    y = 70 + i * 45
    col = (0, 255, 255) if i == 0 else (255, 255, 255)
    cv2.putText(blank, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, col, 2, cv2.LINE_AA)

# --- combine for overview ---
r1 = cv2.hconcat([a, b, c])
r2 = cv2.hconcat([d, e, blank])
grid = cv2.vconcat([r1, r2])

# --- save results ---
cv2.imwrite("result/original.jpg", orig)
cv2.imwrite("result/denoised.jpg", img)
cv2.imwrite("result/mask.jpg", mask)
cv2.imwrite("result/highlighted.jpg", highlight)
cv2.imwrite("result/overview.jpg", grid)

print(f"Saved to result/overview.jpg\nRust area: {rust_pct:.2f}%")

# --- show result ---
h, w = grid.shape[:2]
plt.figure(figsize=(w / 100, h / 100))
plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout(pad=0)
plt.title("Corrosion Detection", fontsize=18)
plt.show()
