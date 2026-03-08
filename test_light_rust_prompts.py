"""Test prompts for detecting light/mild rust."""
import cv2
import numpy as np
import importlib.util

# Load module
spec = importlib.util.spec_from_file_location('sam3_exp', 'sam3-exp-2-stage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

det = module.FastRustDetector(verbose=False)

# Load and crop image
img = cv2.imread('data/New/2.JPG')
results = det._run_sam3_text_prompt(img, 'metal')
r0 = results[0]
box = r0.boxes.xyxy[0].cpu().numpy().astype(int)
x1,y1,x2,y2 = box
crop = img[y1:y2, x1:x2]

# Prompts targeting lighter rust
test_prompts = [
    # Current
    'rusted area',
    # Light rust variants
    'light rust',
    'mild rust', 
    'rust stain',
    'rust spots',
    'surface rust',
    'rust discoloration',
    'oxidized metal',
    'tarnished metal',
    'brown stain',
    'rust patch',
    'rust streak',
    'early rust',
    'beginning rust',
]

print('Testing prompts for light rust detection:')
print('='*60)
for prompt in test_prompts:
    try:
        r = det._run_sam3_text_prompt(crop, prompt)
        if r and len(r) > 0 and r[0].masks is not None:
            n = len(r[0].masks.data)
            confs = r[0].boxes.conf.cpu().numpy() if r[0].boxes is not None else []
            # Calculate total mask area
            masks = r[0].masks.data.cpu().numpy()
            total_area = sum((m > 0.5).sum() for m in masks)
            print(f'{prompt:25s}: {n} det, conf={[f"{c:.3f}" for c in confs[:3]]}, area={total_area}px')
        else:
            print(f'{prompt:25s}: 0 det')
    except Exception as e:
        print(f'{prompt:25s}: error - {e}')
