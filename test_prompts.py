"""Test different prompts for rust detection."""
import cv2
import numpy as np
import importlib.util

# Load module
spec = importlib.util.spec_from_file_location('sam3_exp', 'sam3-exp-2-stage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

det = module.FastRustDetector(verbose=False)

# Load and crop image
img = cv2.imread('data/New/20.JPG')
results = det._run_sam3_text_prompt(img, 'metal')
r0 = results[0]
box = r0.boxes.xyxy[0].cpu().numpy().astype(int)
x1,y1,x2,y2 = box
crop = img[y1:y2, x1:x2]

test_prompts = [
    'rust', 'corrosion', 'rust stains', 'brown rust', 'oxidation',
    'brown spots', 'rusty surface', 'rust damage', 'corroded metal',
    'rusted area', 'orange rust', 'iron rust', 'stain', 'discoloration',
    'brown', 'orange', 'damaged surface', 'dirty metal'
]

print('Testing prompts on cropped ROI:')
print('='*60)
for prompt in test_prompts:
    try:
        r = det._run_sam3_text_prompt(crop, prompt)
        if r and len(r) > 0 and r[0].masks is not None:
            n = len(r[0].masks.data)
            confs = r[0].boxes.conf.cpu().numpy() if r[0].boxes is not None else []
            print(f'{prompt:25s}: {n} det, conf={[f"{c:.3f}" for c in confs]}')
        else:
            print(f'{prompt:25s}: 0 det')
    except Exception as e:
        print(f'{prompt:25s}: error - {e}')
