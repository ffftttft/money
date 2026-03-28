import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

print("Loading SigLIP 2D features...")
train_feat = np.load("outputs/features_2d/siglip_train.npy")

fig, ax = plt.subplots(figsize=(8, 6))

# Set background to a soft, academic color or white
ax.set_facecolor('#F8F9FA') 
fig.patch.set_facecolor('white')

# Plot the scatter points of the normal training data (the "Safe Zone Manifold")
scatter = ax.scatter(
    train_feat[:, 0], train_feat[:, 1],
    s=15, alpha=0.6, 
    c='#3A7CA5',  # A nice technical blue
    edgecolors='white', linewidths=0.5,
    label='Normal State Manifold (Training Data)'
)

ax.set_title("Normal Traffic State Manifold (SigLIP + UMAP)", fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
ax.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight='bold')
ax.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight='bold')

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
ax.grid(True, linestyle='--', alpha=0.5, color='gray')

# Add legend
ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()

out_path = "outputs/deliverables/PPT_Assets/07_siglip_static_manifold.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated static UMAP plot at: {out_path}")

