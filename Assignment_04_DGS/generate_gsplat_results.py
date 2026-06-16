"""Generate renderings using gsplat CUDA rasterizer (official 3DGS-equivalent)."""
import os, sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Setup MSVC env for CUDA JIT compilation
os.environ["PATH"] = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64;C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64;" + os.environ.get("PATH", "")
os.environ["INCLUDE"] = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/include;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared"
os.environ["LIB"] = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/lib/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64;D:/Anaconda/conda/Library/lib;D:/Anaconda/conda/Library/lib/x64"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from render_3dgs_mv import build_horizontal_orbit
import gsplat

device = torch.device("cuda")
out_dir = "results_gsplat"
os.makedirs(out_dir, exist_ok=True)

# Load dataset and trained model
print("Loading dataset and model...")
dataset = ColmapDataset("data/chair")
sample = dataset[0]
H, W = sample['image'].shape[:2]
K_ref = sample['K'].to(device).unsqueeze(0)  # (1, 3, 3)

model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
ckpt = torch.load("data/chair/checkpoints/checkpoint_000040.pt", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Also load our renderer for side-by-side comparison
renderer_ours = GaussianRenderer(H, W).to(device)

print("Model loaded. GPU VRAM used:", torch.cuda.memory_allocated() / 1024**3, "GB")

# Helper: render with gsplat
def render_gsplat(gp, R, t, K):
    """Render one view with gsplat CUDA rasterizer."""
    R_np = R.cpu().numpy() if isinstance(R, torch.Tensor) else R
    t_np = t.cpu().numpy().reshape(3) if isinstance(t, torch.Tensor) else t.reshape(3)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_np
    viewmat[:3, 3] = t_np
    viewmat_t = torch.from_numpy(viewmat).to(device).unsqueeze(0)  # (1, 4, 4)
    K_t = K.to(device)
    if K_t.dim() == 2:
        K_t = K_t.unsqueeze(0)

    rendered, _, _ = gsplat.rasterization(
        means=gp.positions, quats=gp.rotations,
        scales=gp.scales, opacities=gp.opacities.squeeze(-1),
        colors=gp.colors, viewmats=viewmat_t, Ks=K_t,
        width=W, height=H, packed=True, tile_size=16
    )
    return rendered.squeeze(0)  # (H, W, 3)

# Get params
with torch.no_grad():
    gp = model.get_gaussian_params()

# ==========================================
# 1. Side-by-side test views: GT | Ours | gsplat
# ==========================================
print("\nRendering comparison views...")
test_indices = [0, 15, 30, 50, 75, 99]
for idx in test_indices:
    sample = dataset[idx]
    gt = (sample['image'].numpy() * 255).astype(np.uint8)
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)
    K_gsplat = sample['K'].to(device)

    with torch.no_grad():
        # Our renderer
        our_rend = renderer_ours(
            means3D=gp.positions, covs3d=gp.covariance,
            colors=gp.colors, opacities=gp.opacities,
            K=K, R=R, t=t
        )
        our_img = (our_rend.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # gsplat renderer
        gsplat_img = (render_gsplat(gp, R, t, K_gsplat).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # GT | Ours | gsplat
    side = np.concatenate([gt, our_img, gsplat_img], axis=1)
    cv2.putText(side, "GT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(side, "Ours", (W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(side, "gsplat", (W*2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(f"{out_dir}/compare_{idx:03d}.png", cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"  View {idx}: saved (GT | Ours | gsplat)")

# ==========================================
# 2. gsplat multi-view grid (4x4)
# ==========================================
print("\nCreating gsplat multi-view grid...")
grid_indices = list(range(0, 100, 6))[:16]
grid_cells = []
for idx in grid_indices:
    sample = dataset[idx]
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)
    with torch.no_grad():
        img = render_gsplat(gp, R, t, K)
        grid_cells.append((img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

rows = [np.concatenate(grid_cells[r*4:(r+1)*4], axis=1) for r in range(4)]
grid = np.concatenate(rows, axis=0)
cv2.imwrite(f"{out_dir}/gsplat_multi_view.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print(f"  Grid saved ({grid.shape[1]}x{grid.shape[0]})")

# ==========================================
# 3. gsplat flythrough video (GT | Ours | gsplat)
# ==========================================
print("\nRendering flythrough comparison video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f"{out_dir}/gsplat_flythrough_compare.mp4", fourcc, 5, (W * 3, H))

for idx in tqdm(range(0, len(dataset), 2), desc="Video"):  # every 2nd for speed
    sample = dataset[idx]
    gt = (sample['image'].numpy() * 255).astype(np.uint8)
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1)

    with torch.no_grad():
        our_img = (renderer_ours(means3D=gp.positions, covs3d=gp.covariance,
                    colors=gp.colors, opacities=gp.opacities, K=K, R=R, t=t)
                    .cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gs_img = (render_gsplat(gp, R, t, K).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    frame = np.concatenate([gt, our_img, gs_img], axis=1)
    cv2.putText(frame, "GT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Ours (PyTorch)", (W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "gsplat (CUDA)", (W*2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
print(f"  Video saved: {out_dir}/gsplat_flythrough_compare.mp4")

# ==========================================
# 4. gsplat orbit video
# ==========================================
print("\nRendering gsplat orbit video...")
R_path, t_path = build_horizontal_orbit(dataset, 240)
writer = cv2.VideoWriter(f"{out_dir}/gsplat_orbit.mp4", fourcc, 30, (W, H))

for i in tqdm(range(len(R_path)), desc="Orbit"):
    R = torch.as_tensor(R_path[i], device=device)
    t = torch.as_tensor(t_path[i], device=device)
    with torch.no_grad():
        img = render_gsplat(gp, R, t, K_ref)
        frame = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
print(f"  Video saved: {out_dir}/gsplat_orbit.mp4")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print(f"gsplat results in '{out_dir}/':")
print("=" * 60)
for f in sorted(os.listdir(out_dir)):
    size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
    print(f"  {f:35s}  {size_kb:8.1f} KB")
