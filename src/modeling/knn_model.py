import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args():
    p = argparse.ArgumentParser(description="KNN anomaly scoring")
    p.add_argument("--train", required=True, help="Train feature .npy")
    p.add_argument("--test", required=True, help="Test feature .npy")
    p.add_argument("--out-model", required=True, help="Output joblib model path")
    p.add_argument("--out-score", required=True, help="Output score .npy path")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors")
    p.add_argument("--metric", default="euclidean", help="KNN distance metric")
    p.add_argument("--reg", type=float, default=1e-4, help="Cov regularization for mahalanobis")
    return p.parse_args()


def main():
    args = parse_args()
    train = np.load(args.train)
    test = np.load(args.test)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Train/Test features must be 2D arrays")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Feature dimensions do not match")

    metric = args.metric.lower()
    train_used = train
    test_used = test
    model_bundle = {}

    if metric == "mahalanobis":
        # Fast equivalent: whiten features so Euclidean distance equals Mahalanobis distance.
        cov = np.cov(train, rowvar=False)
        cov = cov + args.reg * np.eye(cov.shape[0], dtype=cov.dtype)
        inv_cov = np.linalg.inv(cov)
        chol = np.linalg.cholesky(inv_cov)  # inv_cov = L @ L.T
        train_used = train @ chol
        test_used = test @ chol
        nbr_metric = "euclidean"
        model_bundle["mahalanobis_chol"] = chol
    else:
        nbr_metric = metric

    model = NearestNeighbors(n_neighbors=args.k, metric=nbr_metric)
    model.fit(train_used)
    dists, _ = model.kneighbors(test_used, return_distance=True)
    scores = dists.mean(axis=1)

    out_model = Path(args.out_model)
    out_score = Path(args.out_score)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_score.parent.mkdir(parents=True, exist_ok=True)
    model_bundle.update({"neighbors_model": model, "metric": metric, "k": args.k})
    joblib.dump(model_bundle, out_model)
    np.save(out_score, scores)

    print("Train samples:", train.shape[0])
    print("Test samples:", test.shape[0])
    print("Feature dim:", train.shape[1])
    print("k:", args.k)
    print("metric:", metric)
    if metric == "mahalanobis":
        print("reg:", args.reg)
    print("Score mean/std:", float(scores.mean()), float(scores.std()))
    print("Model saved:", out_model)
    print("Scores saved:", out_score)


if __name__ == "__main__":
    main()
