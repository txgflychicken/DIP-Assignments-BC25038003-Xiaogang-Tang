"""Compare simplified PyTorch 3DGS vs gsplat (CUDA-accelerated) on chair dataset.

Measures: training speed (it/s), VRAM usage, rendering quality (PSNR).
"""
import os
import sys

# Set up MSVC environment before importing torch/gsplat
VC_BIN = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
SDK_BIN = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
if os.path.isdir(VC_BIN):
    os.environ["PATH"] = VC_BIN + os.pathsep + SDK_BIN + os.pathsep + os.environ.get("PATH", "")
    # Also set LIB and INCLUDE for MSVC
    VC = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
    SDK = "C:/Program Files (x86)/Windows Kits/10"
    os.environ["INCLUDE"] = f"{VC}/include;{SDK}/Include/10.0.26100.0/ucrt;{SDK}/Include/10.0.26100.0/um;{SDK}/Include/10.0.26100.0/shared"
    os.environ["LIB"] = f"{VC}/lib/x64;{SDK}/Lib/10.0.26100.0/ucrt/x64;{SDK}/Lib/10.0.26100.0/um/x64"
    print(f"MSVC env set. cl.exe at: {VC_BIN}")

import torch
import time
import numpy as np
from tqdm import tqdm

# Add our code path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import ColmapDataset
from gaussian_model import GaussianModel

import gsplat

device = torch.device("cuda")

# Load data
print("Loading chair dataset...")
dataset = ColmapDataset("data/chair")
print(f"  Images: {len(dataset)}, Points: {dataset.points3D_xyz.shape[0]}")

# Get image dims
sample = dataset[0]
H, W = sample['image'].shape[:2]
print(f"  Resolution: {W}x{H}")

# ========================
# Part 1: Our simplified PyTorch renderer speed
# ========================
print("\n" + "="*60)
print("Part 1: Simplified PyTorch Renderer (our implementation)")
print("="*60)

from gaussian_renderer import GaussianRenderer
model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
renderer = GaussianRenderer(H, W).to(device)

# Warmup
with torch.no_grad():
    gp = model()
    K = sample['K'].to(device)
    R = sample['R'].to(device)
    t = sample['t'].to(device).reshape(-1, 3)
    _ = renderer(gp['positions'], gp['covariance'], gp['colors'], gp['opacities'],
                 K=K, R=R, t=t.squeeze(0))

torch.cuda.synchronize()

# Measure rendering speed (pure forward, no backward)
N_warmup = 3
N_measure = 20
times = []
with torch.no_grad():
    for i in range(N_warmup + N_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gp = model()
        _ = renderer(gp['positions'], gp['covariance'], gp['colors'], gp['opacities'],
                     K=K, R=R, t=t.squeeze(0))
        torch.cuda.synchronize()
        if i >= N_warmup:
            times.append(time.perf_counter() - t0)

our_fwd_time = np.mean(times)
print(f"  Forward render time: {our_fwd_time*1000:.1f} ms")
print(f"  Render FPS: {1.0/our_fwd_time:.1f}")

# Measure training step time (forward + backward)
times = []
for i in range(N_warmup + N_measure):
    gp = model()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rendered = renderer(gp['positions'], gp['covariance'], gp['colors'], gp['opacities'],
                        K=K, R=R, t=t.squeeze(0))
    loss = torch.abs(rendered - sample['image'].to(device)).mean()
    loss.backward()
    torch.cuda.synchronize()
    if i >= N_warmup:
        times.append(time.perf_counter() - t0)
    model.zero_grad()

our_train_time = np.mean(times)
print(f"  Training step time (fwd+bwd): {our_train_time*1000:.1f} ms")
print(f"  Training it/s: {1.0/our_train_time:.1f}")

# VRAM
vram_our = torch.cuda.max_memory_allocated() / 1024**3
torch.cuda.reset_peak_memory_stats()
print(f"  Peak VRAM: {vram_our:.2f} GB")

# Cleanup
del model, gp
torch.cuda.empty_cache()
# Keep `renderer` for Part 3

# ========================
# Part 2: gsplat CUDA-accelerated renderer
# ========================
print("\n" + "="*60)
print("Part 2: gsplat CUDA Rasterizer (equivalent to official 3DGS)")
print("="*60)

# Initialize model same way for fair comparison
model2 = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)

# Build viewmats for gsplat (world-to-camera 4x4 matrices)
# COLMAP format: cam_pt = R @ world_pt + t
# gsplat expects viewmats as 4x4 matrices
R_np = sample['R'].numpy()
t_np = sample['t'].numpy().reshape(3)
viewmat = np.eye(4, dtype=np.float32)
viewmat[:3, :3] = R_np
viewmat[:3, 3] = t_np
viewmat_t = torch.from_numpy(viewmat).to(device).unsqueeze(0)  # (1, 4, 4)

K_np = sample['K'].numpy()
K_t = torch.from_numpy(K_np).to(device).unsqueeze(0)  # (1, 3, 3)

# Get gaussian params for gsplat
with torch.no_grad():
    params = model2.get_gaussian_params()

means = params.positions  # (N, 3)
quats = params.rotations  # (N, 4) - wxyz
scales = params.scales    # (N, 3) - exp space already
opacities = params.opacities.squeeze(-1)  # (N,)
colors = params.colors    # (N, 3)

# Warmup gsplat
with torch.no_grad():
    _ = gsplat.rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmat_t, Ks=K_t,
        width=W, height=H,
        packed=True, tile_size=16
    )

torch.cuda.synchronize()

# Measure gsplat rendering speed
times = []
with torch.no_grad():
    for i in range(N_warmup + N_measure):
        # Recompute params each time like our impl does
        params2 = model2.get_gaussian_params()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rendered, alpha, info = gsplat.rasterization(
            means=params2.positions,
            quats=params2.rotations,
            scales=params2.scales,
            opacities=params2.opacities.squeeze(-1),
            colors=params2.colors,
            viewmats=viewmat_t, Ks=K_t,
            width=W, height=H,
            packed=True, tile_size=16
        )
        torch.cuda.synchronize()
        if i >= N_warmup:
            times.append(time.perf_counter() - t0)

gsplat_fwd_time = np.mean(times)
print(f"  Forward render time: {gsplat_fwd_time*1000:.1f} ms")
print(f"  Render FPS: {1.0/gsplat_fwd_time:.1f}")
print(f"  Speedup vs ours: {our_fwd_time/gsplat_fwd_time:.1f}x")

# Measure gsplat training step
times = []
gt_image = sample['image'].to(device)  # (H, W, 3)
for i in range(N_warmup + N_measure):
    params2 = model2.get_gaussian_params()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rendered, alpha, info = gsplat.rasterization(
        means=params2.positions,
        quats=params2.rotations,
        scales=params2.scales,
        opacities=params2.opacities.squeeze(-1),
        colors=params2.colors,
        viewmats=viewmat_t, Ks=K_t,
        width=W, height=H,
        packed=True, tile_size=16
    )
    loss = torch.abs(rendered.squeeze(0) - gt_image).mean()
    loss.backward()
    torch.cuda.synchronize()
    if i >= N_warmup:
        times.append(time.perf_counter() - t0)
    model2.zero_grad()

gsplat_train_time = np.mean(times)
print(f"  Training step time (fwd+bwd): {gsplat_train_time*1000:.1f} ms")
print(f"  Training it/s: {1.0/gsplat_train_time:.1f}")
print(f"  Speedup vs ours: {our_train_time/gsplat_train_time:.1f}x")

# VRAM
vram_gsplat = torch.cuda.max_memory_allocated() / 1024**3
torch.cuda.reset_peak_memory_stats()
print(f"  Peak VRAM: {vram_gsplat:.2f} GB")
print(f"  VRAM saving: {vram_our - vram_gsplat:.2f} GB ({(1 - vram_gsplat/vram_our)*100:.0f}%)")

# ========================
# Part 3: Render quality comparison (PSNR)
# ========================
print("\n" + "="*60)
print("Part 3: Render Quality Comparison (using trained checkpoint)")
print("="*60)

# Load trained model
ckpt_path = "data/chair/checkpoints/checkpoint_000040.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model2.load_state_dict(ckpt['model_state_dict'])

    # Compute PSNR for both renderers on a few views
    test_indices = [0, 10, 20, 30, 40]
    our_psnrs = []
    gsplat_psnrs = []

    for idx in test_indices:
        sample = dataset[idx]
        gt = sample['image'].to(device)
        K = sample['K'].to(device)
        R = sample['R'].to(device)
        t_val = sample['t'].to(device)

        # Build viewmat for gsplat
        R_np = R.cpu().numpy()
        t_np = t_val.cpu().numpy().reshape(3)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R_np
        viewmat[:3, 3] = t_np
        viewmat_t = torch.from_numpy(viewmat).to(device).unsqueeze(0)
        K_t = K.unsqueeze(0).to(device)

        with torch.no_grad():
            params2 = model2.get_gaussian_params()

            # Our renderer
            our_render = renderer(
                means3D=params2.positions,
                covs3d=params2.covariance,
                colors=params2.colors,
                opacities=params2.opacities,
                K=K, R=R, t=t_val.reshape(-1)
            )
            mse_our = torch.mean((our_render - gt) ** 2).item()
            psnr_our = 10 * np.log10(1.0 / max(mse_our, 1e-10))
            our_psnrs.append(psnr_our)

            # gsplat renderer
            gsplat_render, _, _ = gsplat.rasterization(
                means=params2.positions,
                quats=params2.rotations,
                scales=params2.scales,
                opacities=params2.opacities.squeeze(-1),
                colors=params2.colors,
                viewmats=viewmat_t, Ks=K_t,
                width=W, height=H,
                packed=True, tile_size=16
            )
            gsplat_img = gsplat_render.squeeze(0)  # (1, H, W, 3) -> (H, W, 3)
            mse_gsplat = torch.mean((gsplat_img - gt) ** 2).item()
            psnr_gsplat = 10 * np.log10(1.0 / max(mse_gsplat, 1e-10))
            gsplat_psnrs.append(psnr_gsplat)

    print(f"  View   | Our PSNR | gsplat PSNR")
    print(f"  -------|----------|------------")
    for i, idx in enumerate(test_indices):
        print(f"  {idx:3d}    | {our_psnrs[i]:7.2f}  | {gsplat_psnrs[i]:8.2f}")
    print(f"  Average| {np.mean(our_psnrs):7.2f}  | {np.mean(gsplat_psnrs):8.2f}")

# ========================
# Summary
# ========================
print("\n" + "="*60)
print("SUMMARY: Simplified PyTorch vs gsplat (Official 3DGS-equivalent)")
print("="*60)
print(f"  {'Metric':<25} {'Ours (PyTorch)':<20} {'gsplat (CUDA)':<20} {'Improvement':<15}")
print(f"  {'-'*80}")
print(f"  {'Render FPS':<25} {1/our_fwd_time:<20.1f} {1/gsplat_fwd_time:<20.1f} {our_fwd_time/gsplat_fwd_time:<15.1f}x")
print(f"  {'Train step time (ms)':<25} {our_train_time*1000:<20.1f} {gsplat_train_time*1000:<20.1f} {our_train_time/gsplat_train_time:<15.1f}x")
print(f"  {'Peak VRAM (GB)':<25} {vram_our:<20.2f} {vram_gsplat:<20.2f} {vram_our-vram_gsplat:<15.2f} saved")
print(f"  {'PSNR (dB)':<25} {np.mean(our_psnrs) if os.path.exists(ckpt_path) else 'N/A':<20} {'Same model':<20} --")
