import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from timm import create_model
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform):
        self.root_dir = Path(root_dir)
        self.paths = sorted([p for p in self.root_dir.glob("*.jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path.name


def parse_args():
    p = argparse.ArgumentParser(description="Extract features using timm models.")
    p.add_argument("--data", required=True, help="Input image folder")
    p.add_argument("--out", required=True, help="Output .npy file")
    p.add_argument("--model", required=True, help="Timm model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
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

    model = create_model(args.model, pretrained=True, num_classes=0, global_pool="avg")
    model.eval()
    model.to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = ImageFolderDataset(Path(args.data), transform)
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
            feats = model(batch)
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
