import numpy as np
import cv2

# Load the saved feature tensor for 12.JPG
data = np.load('results/exp-2-stage/12_pixel_features.npz', allow_pickle=True)

feature_tensor = data['pixel_feature_tensor']
feature_names = list(data['pixel_feature_names'])
rust_mask_roi = data['rust_mask_roi']

h, w, f = feature_tensor.shape
X_raw = feature_tensor.reshape(-1, f)
rust_flat = rust_mask_roi.reshape(-1).astype(bool)

idx = {str(name): i for i, name in enumerate(feature_names)}

print(f'ROI total pixels: {h*w}')
print(f'Rust pixels: {rust_flat.sum()}')
print(f'Non-rust pixels: {(~rust_flat).sum()}')

print(f'\\n--- RUST PIXELS ---')
print(f'  L_norm: mean={X_raw[rust_flat, idx["L_norm"]].mean():.3f}')
print(f'  a_norm: mean={X_raw[rust_flat, idx["a_norm"]].mean():.3f}')
print(f'  b_norm: mean={X_raw[rust_flat, idx["b_norm"]].mean():.3f}')
print(f'  H_norm: mean={X_raw[rust_flat, idx["H_norm"]].mean():.3f}')
print(f'  S_norm: mean={X_raw[rust_flat, idx["S_norm"]].mean():.3f}')
print(f'  V_norm: mean={X_raw[rust_flat, idx["V_norm"]].mean():.3f}')
print(f'  redness: mean={X_raw[rust_flat, idx["redness"]].mean():.3f}, min={X_raw[rust_flat, idx["redness"]].min():.3f}')
print(f'  brownness: mean={X_raw[rust_flat, idx["brownness"]].mean():.3f}, min={X_raw[rust_flat, idx["brownness"]].min():.3f}')

print(f'\\n--- NON-RUST PIXELS ---')
print(f'  L_norm: mean={X_raw[~rust_flat, idx["L_norm"]].mean():.3f}')
print(f'  a_norm: mean={X_raw[~rust_flat, idx["a_norm"]].mean():.3f}')
print(f'  b_norm: mean={X_raw[~rust_flat, idx["b_norm"]].mean():.3f}')
print(f'  H_norm: mean={X_raw[~rust_flat, idx["H_norm"]].mean():.3f}')
print(f'  S_norm: mean={X_raw[~rust_flat, idx["S_norm"]].mean():.3f}')
print(f'  V_norm: mean={X_raw[~rust_flat, idx["V_norm"]].mean():.3f}')
print(f'  redness: mean={X_raw[~rust_flat, idx["redness"]].mean():.3f}')
print(f'  brownness: mean={X_raw[~rust_flat, idx["brownness"]].mean():.3f}')

# Check distribution of redness in rust pixels
rust_redness = X_raw[rust_flat, idx["redness"]]
print(f'\\n--- RUST REDNESS Distribution ---')
print(f'  Mean: {rust_redness.mean():.3f}')
print(f'  10th percentile: {np.percentile(rust_redness, 10):.3f}')
print(f'  25th percentile: {np.percentile(rust_redness, 25):.3f}')
print(f'  50th percentile: {np.percentile(rust_redness, 50):.3f}')
print(f'  Min: {rust_redness.min():.3f}')
