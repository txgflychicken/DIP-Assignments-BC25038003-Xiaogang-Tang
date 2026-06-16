"""Plot training results for Assignment 4 report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. Loss Curve ==========
epochs = list(range(60))
losses = [0.0958, 0.0726, 0.0627, 0.0579, 0.0552, 0.0535, 0.0525, 0.0517, 0.0511, 0.0507,
          0.0503, 0.0500, 0.0496, 0.0494, 0.0491, 0.0489, 0.0487, 0.0485, 0.0483, 0.0482,
          0.0480, 0.0479, 0.0478, 0.0477, 0.0476, 0.0474, 0.0473, 0.0472, 0.0471, 0.0470,
          0.0469, 0.0468, 0.0468, 0.0467, 0.0466, 0.0465, 0.0464, 0.0464, 0.0463, 0.0462,
          0.0462, 0.0461, 0.0460, 0.0460, 0.0459, 0.0459, 0.0458, 0.0458, 0.0457, 0.0457,
          0.0456, 0.0456, 0.0455, 0.0455, 0.0454, 0.0454, 0.0453, 0.0453, 0.0452, 0.0452]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, losses, 'b-', linewidth=1.5, markersize=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('L1 Loss', fontsize=12)
ax.set_title('3DGS Training Loss Curve (Chair, 3000 Gaussians, 100x100)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 59)
fig.tight_layout()
fig.savefig('loss_curve.png', dpi=150)
plt.close(fig)
print('Loss curve saved to loss_curve.png')

# ========== 2. Rendered vs GT comparison at different epochs ==========
debug_dir = 'data/chair/checkpoints/debug_images'
epochs_to_show = [0, 10, 20, 40, 59]

fig, axes = plt.subplots(len(epochs_to_show), 1, figsize=(12, 4 * len(epochs_to_show)))
if len(epochs_to_show) == 1:
    axes = [axes]

for i, ep in enumerate(epochs_to_show):
    img_path = os.path.join(debug_dir, f'epoch_{ep:04d}.png')
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f'Epoch {ep}', fontsize=12)
        axes[i].axis('off')
    else:
        axes[i].text(0.5, 0.5, f'Image not found: epoch_{ep:04d}.png',
                     ha='center', va='center')

fig.suptitle('3DGS Training Progress: GT (top row) vs Rendered (bottom row)', fontsize=14)
fig.tight_layout()
fig.savefig('training_progress.png', dpi=150)
plt.close(fig)
print('Training progress saved to training_progress.png')

print('All plots generated successfully!')
