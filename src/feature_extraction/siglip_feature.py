import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, processor):
        self.root_dir = Path(root_dir)
        self.paths = sorted([p for p in self.root_dir.glob("*.jpg")])
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.processor:
            img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return img, path.name


def parse_args():
    p = argparse.ArgumentParser(description="Extract SigLIP features.")
    p.add_argument("--data", required=True, help="Input image folder")
    p.add_argument("--out", required=True, help="Output .npy file")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--model",
        default="google/siglip2-base-patch16-224",
        help="HuggingFace model id",
    )
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = pick_device()

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).vision_model
    model.eval()
    model.to(device)

    dataset = ImageFolderDataset(Path(args.data), processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    features = []

    with torch.no_grad():
        for batch, _ in tqdm(loader, desc=f"Extracting ({device.type})"):
            batch = batch.to(device)
            outputs = model(pixel_values=batch)
            feats = outputs.pooler_output
            feats = feats.detach().float().cpu().numpy()
            features.append(feats)

    if len(features) == 0:
        raise RuntimeError("No features extracted. Check input folder.")

    features = np.concatenate(features, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, features)

    print("Device:", device)
    print("Model:", args.model)
    print("Samples:", features.shape[0])
    print("Feature dim:", features.shape[1])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
