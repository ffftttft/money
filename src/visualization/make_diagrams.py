from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "figures"


def setup_chinese_font():
    candidates = ["PingFang HK", "Songti SC", "STHeiti", "Heiti TC", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, w, h, text, fc="#E9F1F7", ec="#2F4F6F", fontsize=11):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, x1, y1, x2, y2, color="#3B556E"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=color),
    )


def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title("城市道路异常风险预警系统流程图", fontsize=18, weight="bold", pad=14)

    w, h = 0.14, 0.12
    y = 0.54
    xs = [0.03, 0.20, 0.37, 0.54, 0.71, 0.84]
    labels = [
        "视频输入\n(MP4)",
        "抽帧预处理\n(2 FPS, 224x224)",
        "特征提取\n(SigLIP / 对比模型)",
        "异常建模\n(高斯/KNN)",
        "风险评分\n+ 时间平滑",
        "异常报警输出\n+ 可视化",
    ]

    for x, t in zip(xs, labels):
        add_box(ax, (x, y), w, h, t)

    for i in range(len(xs) - 1):
        add_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    add_box(ax, (0.22, 0.30), 0.24, 0.10, "训练集: 前12分钟", fc="#F4F8EC", ec="#607D3B")
    add_box(ax, (0.52, 0.30), 0.24, 0.10, "测试集: 后3分钟", fc="#F9EFEF", ec="#8C4A4A")
    add_arrow(ax, 0.31, 0.42, 0.31, 0.40)
    add_arrow(ax, 0.61, 0.42, 0.61, 0.40)

    add_box(ax, (0.34, 0.10), 0.32, 0.12, "评估输出: AUC / F1 / 风险曲线 / 对比图", fc="#FFF6E8", ec="#B07A2A")
    add_arrow(ax, 0.84 + w / 2, y, 0.50, 0.22)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_system_pipeline_diagram.png", dpi=300)
    plt.close(fig)


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title("系统模块架构图", fontsize=18, weight="bold", pad=14)

    add_box(ax, (0.05, 0.70), 0.20, 0.15, "数据层\nvideo / frames / features", fc="#E8F4FA")
    add_box(ax, (0.30, 0.70), 0.20, 0.15, "感知层\nSigLIP + TIMM", fc="#EAF7EE", ec="#3D7A57")
    add_box(ax, (0.55, 0.70), 0.20, 0.15, "建模层\nGaussian + KNN", fc="#FFF4E8", ec="#9C6B2E")
    add_box(ax, (0.80, 0.70), 0.15, 0.15, "决策层\n阈值/片段检测", fc="#FCEDEE", ec="#A34E5A")

    add_box(ax, (0.12, 0.40), 0.22, 0.14, "工程脚本层\n01~06 codes", fc="#F3F4F8", ec="#5A6273")
    add_box(ax, (0.40, 0.40), 0.22, 0.14, "结果层\nscores / models", fc="#F3F4F8", ec="#5A6273")
    add_box(ax, (0.68, 0.40), 0.22, 0.14, "展示层\nPPT图表/结论", fc="#F3F4F8", ec="#5A6273")

    add_arrow(ax, 0.25, 0.775, 0.30, 0.775)
    add_arrow(ax, 0.50, 0.775, 0.55, 0.775)
    add_arrow(ax, 0.75, 0.775, 0.80, 0.775)

    add_arrow(ax, 0.15, 0.70, 0.20, 0.54)
    add_arrow(ax, 0.40, 0.70, 0.46, 0.54)
    add_arrow(ax, 0.65, 0.70, 0.51, 0.54)
    add_arrow(ax, 0.88, 0.70, 0.79, 0.54)

    add_arrow(ax, 0.34, 0.47, 0.40, 0.47)
    add_arrow(ax, 0.62, 0.47, 0.68, 0.47)

    add_box(ax, (0.28, 0.15), 0.44, 0.14, "可部署闭环:\n摄像头接入 -> 边缘推理 -> 风险评分 -> 告警处置", fc="#EEF0FB", ec="#495D9A")
    add_arrow(ax, 0.79, 0.40, 0.58, 0.29)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_system_architecture_diagram.png", dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    make_pipeline_diagram()
    make_architecture_diagram()
    print("Saved:")
    print("-", OUT_DIR / "07_system_pipeline_diagram.png")
    print("-", OUT_DIR / "08_system_architecture_diagram.png")


if __name__ == "__main__":
    main()
