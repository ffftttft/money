import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def plot_trajectory():
    # 指定 SigLIP 2D 特征路径
    train_feat_path = ROOT / "outputs" / "features_2d" / "siglip_train.npy"
    test_feat_path = ROOT / "outputs" / "features_2d" / "siglip_test.npy"
    out_path = ROOT / "outputs" / "deliverables" / "PPT_Assets" / "05_state_trajectory.png"
    
    print("Loading 2D features for SigLIP model...")
    # 载入 2D 特征
    train_feat = np.load(train_feat_path)
    test_feat = np.load(test_feat_path)
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 1. 绘制正常状态空间（训练集背景）
    # 使用浅蓝色，表示“安全舒适区”
    plt.scatter(train_feat[:, 0], train_feat[:, 1], 
                c='#87CEEB', alpha=0.3, s=30, edgecolors='none', 
                label='Normal State Space (Safe Zone)')
    
    # 2. 绘制测试集状态演化轨迹 (只画异常点前后的部分，避免画面太乱)
    # 找到最高风险点 (已知 index 232 附近是高风险区)
    center_idx = 232
    window = 30  # 前后各画 30 帧
    start_idx = max(0, center_idx - window)
    end_idx = min(len(test_feat), center_idx + window)
    
    x = test_feat[start_idx:end_idx, 0]
    y = test_feat[start_idx:end_idx, 1]
    
    # 提取完整测试集的背景点作为灰色散点，表示测试集的其他时间
    other_x = np.concatenate([test_feat[:start_idx, 0], test_feat[end_idx:, 0]])
    other_y = np.concatenate([test_feat[:start_idx, 1], test_feat[end_idx:, 1]])
    plt.scatter(other_x, other_y, c='gray', alpha=0.15, s=15, 
                label='Other Test Frames')
    
    # 将离散的点转化为线段，用于映射颜色渐变
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 生成时间序列用于着色 (颜色代表时间的推进)
    time_steps = np.arange(start_idx, end_idx)
    norm = plt.Normalize(time_steps.min(), time_steps.max())
    
    # 选用一个从黄到红的渐变色 (YlOrRd)，暗示随着时间推移，系统向“危险状态”演化
    cmap = plt.get_cmap('YlOrRd')
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, alpha=0.8)
    lc.set_array(time_steps)
    
    ax = plt.gca()
    line = ax.add_collection(lc)
    
    # 添加时间颜色条
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Time Evolution (Test Frames)', fontsize=14, fontweight='bold')
    
    # 3. 每隔一段距离，加上小箭头，强调“漂移方向”
    arrow_interval = max(len(x) // 15, 1)  # 大概画15个箭头
    for i in range(0, len(x)-1, arrow_interval):
        # 计算箭头颜色，和线段颜色保持一致
        arrow_color = cmap(norm(i))
        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", 
                                    color=arrow_color, lw=2))
        
    # 4. 重点标记起点和终点
    plt.scatter(x[0], y[0], c='#00FF00', marker='o', s=200, edgecolor='black', 
                linewidth=1.5, zorder=5, label='Trajectory Start (t=0)')
    plt.scatter(x[-1], y[-1], c='#FF0000', marker='X', s=200, edgecolor='black', 
                linewidth=1.5, zorder=5, label='Trajectory End (Anomaly)')
    
    # 5. 美化图表
    plt.title('Spatiotemporal State Trajectory of System Risk\n(Early Warning of State Drift)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Latent State Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('Latent State Dimension 2', fontsize=14, fontweight='bold')
    
    # 定制图例
    legend = plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSuccess! Highly aesthetic trajectory plot saved to:")
    print(f" -> {out_path}")

if __name__ == "__main__":
    plot_trajectory()
