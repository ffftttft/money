import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SHORT_NAME = {
    "siglip": "SigLIP",
    "resnet101.a1h_in1k": "ResNet101",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
    "convnext_base.fb_in22k_ft_in1k": "ConvNeXt-B",
    "efficientnet_b0.ra_in1k": "EffNet-B0",
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate PPT-ready visuals from score files.")
    p.add_argument("--score-dir", default=str(ROOT / "outputs" / "scores" / "gaussian"))
    p.add_argument("--out-root", default=str(ROOT / "results"))
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    p.add_argument("--segment-min-len", type=int, default=5)
    return p.parse_args()


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x_pad, kernel, mode="valid")


def load_scores(score_dir: Path):
    names = []
    scores = []
    for p in sorted(score_dir.glob("*_test_scores.npy")):
        names.append(p.name.replace("_test_scores.npy", ""))
        scores.append(np.load(p))
    if not scores:
        raise RuntimeError(f"No *_test_scores.npy found in {score_dir}")
    return names, scores


def short_name(name: str) -> str:
    return SHORT_NAME.get(name, name)


def safe_name(name: str) -> str:
    return short_name(name).replace("/", "").replace(" ", "_")


def has_anomaly_segment(score: np.ndarray, smooth_window: int, q: float, min_len: int) -> tuple[bool, float]:
    sm = moving_average(score, smooth_window)
    th = float(np.quantile(sm, q))
    b = (sm >= th).astype(np.int32)
    ratio = float(b.mean())
    run = 0
    for v in b:
        run = run + 1 if v == 1 else 0
        if run >= min_len:
            return True, ratio
    return False, ratio


def save_stats_table(names, scores, out_dir: Path, smooth_window: int, q: float, min_len: int):
    lines = ["model_short,model_full,samples,mean,std,p95,p99,is_anomaly,anomaly_ratio"]
    for n, s in zip(names, scores):
        is_abn, ratio = has_anomaly_segment(s, smooth_window, q, min_len)
        lines.append(
            ",".join(
                [
                    short_name(n),
                    n,
                    str(len(s)),
                    f"{float(np.mean(s)):.6f}",
                    f"{float(np.std(s)):.6f}",
                    f"{float(np.percentile(s, 95)):.6f}",
                    f"{float(np.percentile(s, 99)):.6f}",
                    "yes" if is_abn else "no",
                    f"{ratio:.6f}",
                ]
            )
        )
    (out_dir / "00_visual_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mean_std_bar(names, scores, out_dir: Path, smooth_window: int, q: float, min_len: int):
    means = np.array([np.mean(s) for s in scores], dtype=np.float64)
    stds = np.array([np.std(s) for s in scores], dtype=np.float64)
    status = [has_anomaly_segment(s, smooth_window, q, min_len)[0] for s in scores]
    order = np.argsort(means)
    means, stds = means[order], stds[order]
    names_order = [short_name(names[i]) for i in order]
    status_order = [status[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names_order))
    colors = ["#D1495B" if s else "#3A7CA5" for s in status_order]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names_order, rotation=25, ha="right")
    ax.set_title("Model Risk Mean (+/- std), red=anomaly")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    for i, v in enumerate(means):
        tag = "Anomaly" if status_order[i] else "Normal"
        ax.text(i, v + stds[i] + 0.6, tag, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "01_model_mean_std_bar.png", dpi=300)
    plt.close(fig)


def plot_tail_bar(names, scores, out_dir: Path):
    p95 = np.array([np.percentile(s, 95) for s in scores], dtype=np.float64)
    p99 = np.array([np.percentile(s, 99) for s in scores], dtype=np.float64)
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, p95, width=w, label="P95", color="#2A9D8F")
    ax.bar(x + w / 2, p99, width=w, label="P99", color="#E76F51")
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(n) for n in names], rotation=25, ha="right")
    ax.set_title("Tail Risk Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_tail_risk_p95_p99.png", dpi=300)
    plt.close(fig)


def plot_distributions(names, scores, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(scores, tick_labels=[short_name(n) for n in names], showfliers=False)
    ax.set_title("Score Distribution by Model (Boxplot)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "03_score_distribution_boxplot.png", dpi=300)
    plt.close(fig)


def plot_timeline_heatmap(names, scores, out_dir: Path, smooth_window: int, q: float):
    max_len = max(len(s) for s in scores)
    mat = np.zeros((len(names), max_len), dtype=np.float32)
    for i, s in enumerate(scores):
        sm = moving_average(s, smooth_window)
        th = float(np.quantile(sm, q))
        b = (sm >= th).astype(np.float32)
        mat[i, : len(b)] = b

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title("Anomaly Segment Timeline")
    ax.set_xlabel("frame index")
    ax.set_ylabel("model")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([short_name(n) for n in names])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "05_anomaly_timeline_heatmap.png", dpi=300)
    plt.close(fig)


def save_topk(names, scores, out_dir: Path, k: int = 10):
    lines = ["model_short,model_full,rank,frame_index,score"]
    for n, s in zip(names, scores):
        idx = np.argsort(-s)[:k]
        for r, i in enumerate(idx, 1):
            lines.append(f"{short_name(n)},{n},{r},{int(i)},{float(s[i]):.6f}")
    (out_dir / "06_topk_anomaly_frames.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_single_model_detail(name: str, score: np.ndarray, out_dir: Path, smooth_window: int, q: float, min_len: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = moving_average(score, smooth_window)
    th = float(np.quantile(sm, q))
    b = (sm >= th).astype(np.int32)

    # curve
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(sm))
    ax.plot(x, sm, color="#2A9D8F", linewidth=1.6, label="smoothed risk")
    ax.axhline(th, color="#D1495B", linestyle="--", linewidth=1.3, label=f"q{int(q*100)} threshold")
    ax.fill_between(x, sm, th, where=(sm >= th), color="#D1495B", alpha=0.22)
    ax.set_title(f"{short_name(name)} Risk Curve")
    ax.set_xlabel("frame index")
    ax.set_ylabel("risk")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "01_risk_curve.png", dpi=300)
    plt.close(fig)

    # anomaly timeline
    fig, ax = plt.subplots(figsize=(12, 1.8))
    ax.imshow(b.reshape(1, -1), aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title(f"{short_name(name)} Timeline (0=normal, 1=anomaly)")
    ax.set_xlabel("frame index")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "02_timeline.png", dpi=300)
    plt.close(fig)

    # topk
    idx = np.argsort(-score)[:10]
    lines = ["model_short,model_full,rank,frame_index,score"]
    for r, i in enumerate(idx, 1):
        lines.append(f"{short_name(name)},{name},{r},{int(i)},{float(score[i]):.6f}")
    (out_dir / "03_topk_frames.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # status summary
    is_abn, ratio = has_anomaly_segment(score, smooth_window, q, min_len)
    msg = [
        f"model_short={short_name(name)}",
        f"model_full={name}",
        f"samples={len(score)}",
        f"threshold_quantile={q}",
        f"smooth_window={smooth_window}",
        f"segment_min_len={min_len}",
        f"is_anomaly={'yes' if is_abn else 'no'}",
        f"anomaly_ratio={ratio:.6f}",
    ]
    (out_dir / "00_status.txt").write_text("\n".join(msg) + "\n", encoding="utf-8")


def write_explain_md(out_root: Path):
    text = """# 图表解读（详细版）

这份说明对应两个层级：
- `figures_compare/`：必须放在一起看的“多模型对比图”。
- `figures_models/`：必须分开看的“单模型细节图”。

---
## 第一部分：先让你看懂项目（给你自己看）

### 1) 项目到底在做什么
- 目标：从交通视频中自动找“异常风险时段”。
- 方法：先用训练视频提取“正常模式特征”，再对测试视频每一帧打风险分数。
- 分数解释：分数越高，说明这一帧越不像训练阶段的“正常状态”。

### 2) 什么叫“异常”
- 不是人工标注类别异常（你现在没做类别分类）。
- 是统计意义异常：分数持续超过阈值，就当作异常片段。

### 3) 判定规则（你要背）
- 平滑窗口：9 帧。
- 阈值：平滑后分数的 95% 分位数。
- 连续超过阈值 >=5 帧 => 判定有异常片段。

---
## 第二部分：答辩时怎么讲（给老师讲）

### A. 对比图（放一起讲）
- 文件夹：`figures_compare/`
- 用法：先讲全局结论，再讲模型差异。

1) `01_model_mean_std_bar.png`
- 含义：展示每个模型在测试集上的平均风险分数（柱高）和波动（误差棒）。
- 颜色和标签：`red + Anomaly` 表示该模型触发了异常片段规则。
- 讲法：
  1. 哪个均值低，说明整体更稳定。
  2. 哪个误差棒大，说明波动更强。
  3. 结合标签说明是否触发片段级预警。

2) `02_tail_risk_p95_p99.png`
- 含义：展示高风险尾部（P95、P99）。
- 讲法：
  1. P99 越高，代表极端风险帧越高。
  2. P95 和 P99 差距越大，尖峰风险越明显。

3) `03_score_distribution_boxplot.png`
- 含义：展示分数分布（中位数、四分位范围、上下须）。
- 讲法：
  1. 中位线越高，整体风险基线越高。
  2. 箱体越宽，波动越大。

4) `05_anomaly_timeline_heatmap.png`
- 含义：把每个模型每一帧是否超阈值压成 0/1 时间图。
- 讲法：
  1. 横轴看时间，纵轴看模型。
  2. 同时出现异常的时段，说明跨模型一致性更高。

### B. 细节图（分开讲）
- 文件夹：`figures_models/<模型简称>/`
- 每个模型都有：
  1. `01_risk_curve.png`：这一模型自己的风险曲线和阈值。
  2. `02_timeline.png`：这一模型自己的异常时间条。
  3. `03_topk_frames.csv`：该模型最高风险 Top10 帧。
  4. `00_status.txt`：该模型是否异常、异常比例。

答辩讲法模板（每个模型 20 秒）：
1. 先说这模型是否触发异常（`00_status.txt`）。
2. 再指曲线里最明显峰值在哪段。
3. 最后补一句 Top 帧可回放定位，系统可落地。

---
## 你可以直接照读的结论模板
- “我们采用无监督风险评分，不依赖异常标签。”
- “通过时间平滑与分位阈值，将逐帧分数转为片段级预警。”
- “对比实验显示不同骨干模型在风险敏感度与波动性上存在差异。”
- “系统输出包含对比图、单模型细节图和关键帧索引，便于工程部署与人工复核。”
"""
    (out_root / "FIGURE_EXPLANATION.md").write_text(text, encoding="utf-8")


def build_compare_pack(names, scores, out_dir: Path, smooth_window: int, q: float, min_len: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_stats_table(names, scores, out_dir, smooth_window, q, min_len)
    plot_mean_std_bar(names, scores, out_dir, smooth_window, q, min_len)
    plot_tail_bar(names, scores, out_dir)
    plot_distributions(names, scores, out_dir)
    plot_timeline_heatmap(names, scores, out_dir, smooth_window, q)
    save_topk(names, scores, out_dir, 10)


def main():
    args = parse_args()
    score_dir = Path(args.score_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names, scores = load_scores(score_dir)
    compare_dir = out_root / "figures_compare"
    model_dir_root = out_root / "figures_models"
    build_compare_pack(names, scores, compare_dir, args.smooth_window, args.threshold_quantile, args.segment_min_len)
    for n, s in zip(names, scores):
        plot_single_model_detail(
            n,
            s,
            model_dir_root / safe_name(n),
            args.smooth_window,
            args.threshold_quantile,
            args.segment_min_len,
        )
    write_explain_md(out_root)

    print("Saved visual packs:")
    print("-", compare_dir)
    print("-", model_dir_root)
    print("-", out_root / "FIGURE_EXPLANATION.md")


if __name__ == "__main__":
    main()
