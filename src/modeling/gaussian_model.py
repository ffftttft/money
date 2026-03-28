import argparse
from pathlib import Path

import numpy as np


def fit_gaussian(features: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    centered = features - mean
    cov = np.cov(centered, rowvar=False)
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    return mean, cov


def mahalanobis_scores(features: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    diff = features - mean
    # score_i = sqrt((x_i - mu)^T Sigma^-1 (x_i - mu))
    scores = np.sqrt(np.einsum("bi,ij,bj->b", diff, inv_cov, diff))
    return scores


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian modeling + Mahalanobis scoring")
    p.add_argument("--train", required=True, help="Train feature .npy path")
    p.add_argument("--test", required=True, help="Test feature .npy path")
    p.add_argument("--out-model", required=True, help="Output model .npz path")
    p.add_argument("--out-score", required=True, help="Output score .npy path")
    p.add_argument("--reg", type=float, default=1e-4, help="Cov regularization")
    return p.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    out_model = Path(args.out_model)
    out_score = Path(args.out_score)

    train_feat = np.load(train_path)
    test_feat = np.load(test_path)
    if train_feat.ndim != 2 or test_feat.ndim != 2:
        raise ValueError("Train/Test features must be 2D arrays [N, D]")
    if train_feat.shape[1] != test_feat.shape[1]:
        raise ValueError("Train/Test feature dims mismatch")

    mean, cov = fit_gaussian(train_feat, args.reg)
    test_scores = mahalanobis_scores(test_feat, mean, cov)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_score.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_model, mean=mean, cov=cov, reg=np.array(args.reg, dtype=np.float32))
    np.save(out_score, test_scores)

    print("Train:", train_path)
    print("Test:", test_path)
    print("Train samples:", train_feat.shape[0])
    print("Test samples:", test_feat.shape[0])
    print("Feature dim:", train_feat.shape[1])
    print("Reg:", args.reg)
    print("Score mean/std:", float(test_scores.mean()), float(test_scores.std()))
    print("Model saved:", out_model)
    print("Score saved:", out_score)


if __name__ == "__main__":
    main()
