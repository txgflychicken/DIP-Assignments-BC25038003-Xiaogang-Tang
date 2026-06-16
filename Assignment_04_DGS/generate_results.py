"""Generate 3D rendering results for the chair scene."""
import torch
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer

device = torch.device("cuda")
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# Load dataset and trained model
print("Loading dataset and model...")
dataset = ColmapDataset("data/chair")
sample = dataset[0]
H, W = sample['image'].shape[:2]
K_ref = sample['K'].to(device)

model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
ckpt = torch.load("data/chair/checkpoints/checkpoint_000040.pt", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

renderer = GaussianRenderer(H, W).to(device)

# Get all gaussian params once
with torch.no_grad():
    gp = model()

# ==========================================
# 1. Render specific test views
# ==========================================
print("\nRendering test views...")
test_indices = [0, 15, 30, 50, 75, 99]  # Different angles
results = []
for idx in test_indices:
    sample = dataset[idx]
    gt = (sample['image'].numpy() * 255).astype(np.uint8)
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)

    with torch.no_grad():
        rendered = renderer(
            means3D=gp['positions'], covs3d=gp['covariance'],
            colors=gp['colors'], opacities=gp['opacities'],
            K=K, R=R, t=t
        )
    rend = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Side by side: GT | Rendered
    side = np.concatenate([gt, rend], axis=1)
    # Add labels
    cv2.putText(side, "GT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(side, "Ours", (W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{out_dir}/view_{idx:03d}.png", side_bgr)
    results.append(side)
    print(f"  View {idx}: saved")

# ==========================================
# 2. Multi-view grid (4x4 = 16 views)
# ==========================================
print("\nCreating multi-view grid...")
grid_indices = list(range(0, 100, 6))[:16]  # 16 evenly spaced views
grid_cells = []
for idx in grid_indices:
    sample = dataset[idx]
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)

    with torch.no_grad():
        rendered = renderer(
            means3D=gp['positions'], covs3d=gp['covariance'],
            colors=gp['colors'], opacities=gp['opacities'],
            K=K, R=R, t=t
        )
    rend = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    grid_cells.append(rend)

# Build 4x4 grid
rows = []
for r in range(4):
    rows.append(np.concatenate(grid_cells[r*4:(r+1)*4], axis=1))
grid = np.concatenate(rows, axis=0)
grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"{out_dir}/multi_view_grid.png", grid_bgr)
print(f"  Grid saved: {out_dir}/multi_view_grid.png ({grid.shape[1]}x{grid.shape[0]})")

# ==========================================
# 3. High-quality multi-view video (original camera path)
# ==========================================
print("\nRendering camera path video (100 frames)...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f"{out_dir}/chair_flythrough.mp4", fourcc, 10, (W * 2, H))

for idx in tqdm(range(len(dataset)), desc="Video"):
    sample = dataset[idx]
    gt = (sample['image'].numpy() * 255).astype(np.uint8)
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)

    with torch.no_grad():
        rendered = renderer(
            means3D=gp['positions'], covs3d=gp['covariance'],
            colors=gp['colors'], opacities=gp['opacities'],
            K=K, R=R, t=t
        )
    rend = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    frame = np.concatenate([gt, rend], axis=1)
    cv2.putText(frame, "GT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Rendered", (W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
print(f"  Video saved: {out_dir}/chair_flythrough.mp4")

# ==========================================
# 4. Horizontal orbit video with gsplat
# ==========================================
print("\nRendering horizontal orbit video (240 frames)...")
from render_3dgs_mv import build_horizontal_orbit, look_at_colmap

R_path, t_path = build_horizontal_orbit(dataset, 240)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f"{out_dir}/chair_orbit_video.mp4", fourcc, 30, (W, H))

for i in tqdm(range(len(R_path)), desc="Orbit"):
    R = torch.as_tensor(R_path[i], device=device)
    t = torch.as_tensor(t_path[i], device=device)

    with torch.no_grad():
        rendered = renderer(
            means3D=gp['positions'], covs3d=gp['covariance'],
            colors=gp['colors'], opacities=gp['opacities'],
            K=K_ref, R=R, t=t
        )
    rend = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    writer.write(cv2.cvtColor(rend, cv2.COLOR_RGB2BGR))

writer.release()
print(f"  Video saved: {out_dir}/chair_orbit_video.mp4")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("All results generated in 'results/' directory:")
print("=" * 60)
for f in sorted(os.listdir(out_dir)):
    size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
    print(f"  {f:30s}  {size_kb:8.1f} KB")
