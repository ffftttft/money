import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def generate_explanations(risk_frames_dir, top_k=5):
    risk_dir = resolve_repo_path(risk_frames_dir)
    meta_csv_path = risk_dir / "meta.csv"
    
    if not meta_csv_path.exists():
        print(f"Error: {meta_csv_path} not found.")
        return
        
    df = pd.read_csv(meta_csv_path).head(top_k)
    
    print("Loading lightweight VLM (BLIP-base) for edge-side inference...")
    # Using BLIP as it is very lightweight and runs well on CPU for prototyping
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps" # Use Apple Silicon if available
        
    model.to(device)
    print(f"Model loaded onto {device}.")
    
    results = []
    
    print("\nGenerating Semantic Explanations for Top Risk Frames:")
    print("-" * 60)
    for idx, row in df.iterrows():
        img_path = resolve_repo_path(row["dst"])
        if not img_path.exists():
            continue
            
        raw_image = Image.open(img_path).convert('RGB')
        
        # Unconditional image captioning
        start_time = time.time()
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        infer_time = time.time() - start_time
        
        print(f"Frame Index : {row['frame_index']}")
        print(f"Risk Score  : {row['risk_score']:.2f}")
        print(f"AI Caption  : {caption.capitalize()}")
        print(f"Infer Time  : {infer_time:.3f} sec")
        print("-" * 60)
        
        results.append({
            "rank": row['rank'],
            "frame_index": row['frame_index'],
            "score": row['risk_score'],
            "timestamp": row['video_time'],
            "ai_explanation": caption.capitalize(),
            "inference_time_sec": round(infer_time, 3)
        })
        
    out_df = pd.DataFrame(results)
    out_csv = risk_dir.parent / f"{risk_dir.name}_semantic_report.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved semantic report to: {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk_dir",
        type=str,
        default=str(ROOT / "results" / "final_for_ppt_2d" / "gaussian_risk_frames"),
        help="Directory containing risk frames and meta.csv",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of frames to process")
    args = parser.parse_args()
    
    generate_explanations(args.risk_dir, args.top_k)
