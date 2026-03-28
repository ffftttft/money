from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import umap


ROOT = Path(__file__).resolve().parents[2]
FEATURES_SRC = ROOT / "data" / "features"
FEATURES_2D = ROOT / "outputs" / "features_2d"
SCORES_2D = ROOT / "outputs" / "scores_2d"
MODELS_2D = ROOT / "outputs" / "models_2d"
RESULT_2D = ROOT / "results" / "final_for_ppt_2d"

MODEL_FILES = {
    "siglip": (
        FEATURES_SRC / "siglip_train.npy",
        FEATURES_SRC / "siglip_test.npy",
    ),
    "resnet101.a1h_in1k": (
        FEATURES_SRC / "timm" / "resnet101.a1h_in1k_train.npy",
        FEATURES_SRC / "timm" / "resnet101.a1h_in1k_test.npy",
    ),
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": (
        FEATURES_SRC / "timm" / "vit_base_patch16_224.augreg2_in21k_ft_in1k_train.npy",
        FEATURES_SRC / "timm" / "vit_base_patch16_224.augreg2_in21k_ft_in1k_test.npy",
    ),
    "convnext_base.fb_in22k_ft_in1k": (
        FEATURES_SRC / "timm" / "convnext_base.fb_in22k_ft_in1k_train.npy",
        FEATURES_SRC / "timm" / "convnext_base.fb_in22k_ft_in1k_test.npy",
    ),
    "efficientnet_b0.ra_in1k": (
        FEATURES_SRC / "timm" / "efficientnet_b0.ra_in1k_train.npy",
        FEATURES_SRC / "timm" / "efficientnet_b0.ra_in1k_test.npy",
    ),
}

SHORT = {
    "siglip": "SigLIP",
    "resnet101.a1h_in1k": "ResNet101",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
    "convnext_base.fb_in22k_ft_in1k": "ConvNeXt-B",
    "efficientnet_b0.ra_in1k": "EffNet-B0",
}


def ensure_dirs():
    for p in [
        FEATURES_2D,
        SCORES_2D / "gaussian",
        SCORES_2D / "knn",
        MODELS_2D / "gaussian",
        MODELS_2D / "knn",
        RESULT_2D / "train_2d_vis",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def reduce_features_to_2d():
    lines = ["model,orig_dim,reducer,n_neighbors,min_dist,metric,random_state"]
    reduced = {}
    for name, (train_p, test_p) in MODEL_FILES.items():
        train = np.load(train_p)
        test = np.load(test_p)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        train2 = reducer.fit_transform(train)
        test2 = reducer.transform(test)
        np.save(FEATURES_2D / f"{name}_train.npy", train2)
        np.save(FEATURES_2D / f"{name}_test.npy", test2)
        lines.append(f"{name},{train.shape[1]},UMAP,30,0.1,euclidean,42")
        reduced[name] = (train2, test2)
    return reduced, lines


def plot_train_2d(reduced):
    # per model
    for name, (train2, _) in reduced.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(train2[:, 0], train2[:, 1], s=8, alpha=0.5, color="#2A9D8F")
        ax.set_title(f"Train UMAP-2D: {SHORT.get(name, name)}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(RESULT_2D / "train_2d_vis" / f"{name}_train_umap2d.png", dpi=250)
        plt.close(fig)

    # all models in one figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.reshape(-1)
    names = list(MODEL_FILES.keys())
    for i, name in enumerate(names):
        ax = axes[i]
        train2 = reduced[name][0]
        ax.scatter(train2[:, 0], train2[:, 1], s=6, alpha=0.45, color="#4C78A8")
        ax.set_title(SHORT.get(name, name), fontsize=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Train Feature UMAP-2D Visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULT_2D / "train_2d_vis" / "00_all_models_train_umap2d.png", dpi=250)
    plt.close(fig)


def run_gaussian_and_knn_on_2d():
    for name in MODEL_FILES.keys():
        tr = np.load(FEATURES_2D / f"{name}_train.npy")
        te = np.load(FEATURES_2D / f"{name}_test.npy")

        # Gaussian (Mahalanobis)
        mean = tr.mean(axis=0)
        cov = np.cov(tr, rowvar=False) + 1e-4 * np.eye(2)
        inv_cov = np.linalg.inv(cov)
        diff = te - mean
        g_score = np.sqrt(np.einsum("bi,ij,bj->b", diff, inv_cov, diff))
        np.save(SCORES_2D / "gaussian" / f"{name}_test_scores.npy", g_score)
        np.savez(MODELS_2D / "gaussian" / f"{name}_model.npz", mean=mean, cov=cov)

        # KNN with Mahalanobis equivalent (whiten + euclidean)
        chol = np.linalg.cholesky(inv_cov)
        tr_w = tr @ chol
        te_w = te @ chol
        knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        knn.fit(tr_w)
        dists, _ = knn.kneighbors(te_w, return_distance=True)
        k_score = dists.mean(axis=1)
        np.save(SCORES_2D / "knn" / f"{name}_test_scores.npy", k_score)
        np.savez(MODELS_2D / "knn" / f"{name}_model.npz", chol=chol)


def export_final_pack():
    cmd = [
        sys.executable,
        str(ROOT / "src" / "pipeline" / "export_final_results.py"),
        "--score-base",
        str(SCORES_2D),
        "--out-dir",
        str(RESULT_2D),
    ]
    subprocess.run(cmd, check=True)


def main():
    ensure_dirs()
    reduced, cfg_lines = reduce_features_to_2d()
    run_gaussian_and_knn_on_2d()
    export_final_pack()
    # Export step cleans output dir, so draw train-vis after export.
    (RESULT_2D / "train_2d_vis").mkdir(parents=True, exist_ok=True)
    plot_train_2d(reduced)
    (RESULT_2D / "00_umap2d_config.csv").write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")
    print("Saved 2D pipeline results:", RESULT_2D)


if __name__ == "__main__":
    main()
