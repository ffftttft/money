import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

print("Loading lightweight VLM (BLIP-base) for edge-side inference...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

frames = [
    ("outputs/deliverables/PPT_Assets/Risk_Frames_Images/rank2_idx233_score33.452_t13:56.jpg", 33.452),
    ("outputs/deliverables/PPT_Assets/Risk_Frames_Images/rank4_idx254_score32.377_t14:07.jpg", 32.377)
]

for img_path, original_score in frames:
    print("-" * 60)
    print(f"Processing: {img_path}")
    raw_image = Image.open(img_path).convert('RGB')
    
    start_time = time.time()
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    infer_time = time.time() - start_time
    
    print(f"Original Score : {original_score:.3f}")
    print(f"AI Caption     : {caption.capitalize()}")
    print(f"Infer Time     : {infer_time:.3f} sec")

print("-" * 60)
