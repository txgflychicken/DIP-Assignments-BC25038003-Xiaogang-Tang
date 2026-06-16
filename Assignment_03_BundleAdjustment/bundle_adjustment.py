"""Bundle Adjustment from scratch using PyTorch.

Recovers camera intrinsics (focal length), extrinsics (R, T for 50 views),
and 20,000 3D point coordinates from 2D observations via gradient-based optimization.

Approach:
  1. Initialize 3D points by back-projecting from the frontal view
  2. Initialize camera rotations spread across the viewing arc
  3. Three-stage optimization: cameras -> points -> joint, with cyclic LR restarts
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ============================================================
# Euler angles -> Rotation matrix (XYZ intrinsic convention)
# ============================================================
def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    rx, ry, rz = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    ca, sa = torch.cos(rx), torch.sin(rx)
    cb, sb = torch.cos(ry), torch.sin(ry)
    cc, sc = torch.cos(rz), torch.sin(rz)
    z0 = torch.zeros_like(ca)
    o0 = torch.ones_like(ca)

    Rz = torch.stack([
        torch.stack([cc, -sc, z0], dim=-1),
        torch.stack([sc,  cc, z0], dim=-1),
        torch.stack([z0,  z0, o0], dim=-1),
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cb, z0, sb], dim=-1),
        torch.stack([z0, o0, z0], dim=-1),
        torch.stack([-sb, z0, cb], dim=-1),
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([o0, z0,  z0], dim=-1),
        torch.stack([z0, ca, -sa], dim=-1),
        torch.stack([z0, sa,  ca], dim=-1),
    ], dim=-2)
    return Rz @ Ry @ Rx


# ============================================================
# Rotation matrix -> Euler angles (XYZ convention)
# ============================================================
def matrix_to_euler_angles(R, convention="XYZ"):
    """Convert rotation matrices back to Euler angles (XYZ)."""
    # R = Rz @ Ry @ Rx
    # Extract angles, assuming no gimbal lock
    sy = R[..., 0, 2]  # sin(beta)
    # Handle gimbal lock cases
    mask = (torch.abs(sy) > 0.99999).float()
    beta = torch.asin(torch.clamp(sy, -1.0, 1.0))
    alpha = torch.atan2(-R[..., 1, 2] * (1 - mask) + R[..., 1, 0] * mask,
                          R[..., 2, 2] * (1 - mask) + R[..., 1, 1] * mask)
    gamma = torch.atan2(-R[..., 0, 1] * (1 - mask) + R[..., 1, 0] * mask,
                         R[..., 0, 0] * (1 - mask) + R[..., 1, 1] * mask)
    return torch.stack([alpha, beta, gamma], dim=-1)


# ============================================================
# Configuration
# ============================================================
IMAGE_W, IMAGE_H = 1024, 1024
CX, CY = IMAGE_W / 2, IMAGE_H / 2
NUM_VIEWS = 50
NUM_POINTS = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Data Loading
# ============================================================
def load_data():
    data = np.load(f"{DATA_DIR}/points2d.npz")
    colors = np.load(f"{DATA_DIR}/points3d_colors.npy")
    all_obs = [data[f"view_{i:03d}"] for i in range(NUM_VIEWS)]
    observations = np.stack(all_obs, axis=0)
    return (
        torch.tensor(observations[:, :, :2], dtype=torch.float32, device=DEVICE),
        torch.tensor(observations[:, :, 2], dtype=torch.float32, device=DEVICE),
        torch.tensor(colors, dtype=torch.float32, device=DEVICE),
    )


# ============================================================
# Projection
# ============================================================
def project(points_3d, euler_angles, translations, f):
    R = euler_angles_to_matrix(euler_angles)
    Xc = torch.einsum("vij,jn->vin", R, points_3d.T) + translations.unsqueeze(-1)
    Xc = Xc.permute(0, 2, 1)
    X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    u = -f * X / Z + CX
    v =  f * Y / Z + CY
    return torch.stack([u, v], dim=-1)


# ============================================================
# Initialization
# ============================================================
def init_points_from_frontal(obs_2d, visibility, f_init, d_init):
    """Back-project points from view_000 (near-frontal)."""
    u0, v0 = obs_2d[0, :, 0], obs_2d[0, :, 1]
    X = (u0 - CX) * d_init / f_init
    Y = -(v0 - CY) * d_init / f_init
    torch.manual_seed(42)
    Z = 0.05 * torch.randn(NUM_POINTS, device=DEVICE)
    pts = torch.stack([X, Y, Z], dim=-1)
    invisible = ~(visibility[0] > 0.5)
    if invisible.any():
        torch.manual_seed(42)
        pts[invisible] = 0.05 * torch.randn(invisible.sum().item(), 3, device=DEVICE)
    return pts


def init_cameras_in_arc(span_deg=70.0, d_init=2.5):
    """Initialize 50 cameras in an arc spanning +/- span_deg degrees around Y axis."""
    angles = torch.linspace(-span_deg / 2, span_deg / 2, NUM_VIEWS, device=DEVICE)
    angles_rad = angles * np.pi / 180.0

    euler = torch.zeros(NUM_VIEWS, 3, device=DEVICE)
    euler[:, 1] = angles_rad  # Y rotation for looking left/right

    trans = torch.zeros(NUM_VIEWS, 3, device=DEVICE)
    trans[:, 0] = d_init * torch.sin(angles_rad)   # X offset
    trans[:, 2] = -d_init * torch.cos(angles_rad)   # Z offset (negative = in front)

    return euler, trans


# ============================================================
# Bundle Adjustment
# ============================================================
def run_ba(obs_2d, visibility,
           f_init=850.0, d_init=2.5, span_deg=70.0,
           batch_size=5000,
           epochs_stage1=4000, epochs_stage2=4000, epochs_stage3=6000):
    mask = visibility > 0.5
    all_idx = torch.arange(NUM_POINTS, device=DEVICE)

    # --- Parameter initialization ---
    f = nn.Parameter(torch.tensor(float(f_init), device=DEVICE))
    euler_init = torch.zeros(NUM_VIEWS, 3, device=DEVICE)
    trans_init = torch.tensor([[0.0, 0.0, -float(d_init)]], device=DEVICE).repeat(NUM_VIEWS, 1)
    euler = nn.Parameter(euler_init.clone())
    trans = nn.Parameter(trans_init.clone())
    points_3d = nn.Parameter(init_points_from_frontal(obs_2d, visibility, f_init, d_init))

    print(f"  f_init={f_init:.1f}, d_init={d_init:.1f}")

    loss_history = []

    def compute_batch_loss(pidx):
        pts = points_3d[pidx]
        uv_pred = project(pts, euler, trans, f)
        uv_obs = obs_2d[:, pidx, :]
        m = mask[:, pidx]
        diff = uv_pred - uv_obs
        if m.sum() == 0:
            return torch.tensor(0.0, device=DEVICE, requires_grad=True)
        return (diff[m] ** 2).sum() / m.sum()

    def full_loss():
        """Compute loss on all visible pairs (for evaluation)."""
        with torch.no_grad():
            uv_pred = project(points_3d, euler, trans, f)
            diff = uv_pred - obs_2d
            return (diff[mask] ** 2).sum() / mask.sum()

    # ============================================================
    # Stage 1: Cameras only
    # ============================================================
    print("Stage 1: Optimizing cameras (f, R, T) ...")
    opt1 = torch.optim.Adam([f, euler, trans], lr=3e-3)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt1, T_0=800, T_mult=2)

    for ep in (pbar := tqdm(range(epochs_stage1))):
        opt1.zero_grad()
        pidx = all_idx[torch.randperm(NUM_POINTS, device=DEVICE)[:batch_size]]
        loss = compute_batch_loss(pidx)
        loss.backward()
        opt1.step()
        sched1.step()
        loss_history.append(loss.item())
        if ep % 800 == 0 or ep == epochs_stage1 - 1:
            full = full_loss()
            pbar.set_description(f"S1 loss: {loss.item():.1f} | full: {full.item():.1f}")

    # ============================================================
    # Stage 2: Points only
    # ============================================================
    print("Stage 2: Optimizing 3D points ...")
    opt2 = torch.optim.Adam([points_3d], lr=5e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=1000, T_mult=2)

    for ep in (pbar := tqdm(range(epochs_stage2))):
        opt2.zero_grad()
        pidx = all_idx[torch.randperm(NUM_POINTS, device=DEVICE)[:batch_size]]
        loss = compute_batch_loss(pidx)
        loss.backward()
        opt2.step()
        sched2.step()
        loss_history.append(loss.item())
        if ep % 1000 == 0 or ep == epochs_stage2 - 1:
            full = full_loss()
            pbar.set_description(f"S2 loss: {loss.item():.1f} | full: {full.item():.1f}")

    # ============================================================
    # Stage 3: Joint optimization
    # ============================================================
    print("Stage 3: Joint optimization ...")
    opt3 = torch.optim.Adam([
        {"params": [f, euler, trans], "lr": 5e-4},
        {"params": [points_3d], "lr": 2e-3},
    ])
    sched3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt3, T_0=1500, T_mult=2)

    for ep in (pbar := tqdm(range(epochs_stage3))):
        opt3.zero_grad()
        pidx = all_idx[torch.randperm(NUM_POINTS, device=DEVICE)[:batch_size]]
        loss = compute_batch_loss(pidx)
        loss.backward()
        opt3.step()
        sched3.step()
        loss_history.append(loss.item())
        if ep % 1500 == 0 or ep == epochs_stage3 - 1:
            full = full_loss()
            pbar.set_description(f"S3 loss: {loss.item():.1f} | full: {full.item():.1f}")

    # ============================================================
    # Stage 4: Full batch, very low LR polish
    # ============================================================
    print("Stage 4: Full-batch fine-tuning ...")
    opt4 = torch.optim.Adam([
        {"params": [f, euler, trans], "lr": 1e-4},
        {"params": [points_3d], "lr": 5e-4},
    ])

    for ep in (pbar := tqdm(range(1000))):
        opt4.zero_grad()
        uv_pred = project(points_3d, euler, trans, f)
        diff = uv_pred - obs_2d
        loss = (diff[mask] ** 2).sum() / mask.sum()
        loss.backward()
        opt4.step()
        loss_history.append(loss.item())
        if ep % 250 == 0 or ep == 999:
            rmse = loss.item() ** 0.5
            pbar.set_description(f"S4 loss: {loss.item():.2f} | RMSE: {rmse:.2f} px")

    return {
        "f": f, "euler_angles": euler, "translations": trans,
        "points_3d": points_3d, "loss_history": loss_history,
    }


# ============================================================
# Visualization & Output
# ============================================================
def plot_loss(loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    total = len(loss_history)
    s1, s2, s3 = total * 4000 // 15000, total * 8000 // 15000, total * 14000 // 15000

    ax1.plot(loss_history, linewidth=0.5)
    for x, name in [(s1, "S1→2"), (s2, "S2→3"), (s3, "S3→4")]:
        ax1.axvline(x=x, color="red" if x == s1 else "orange" if x == s2 else "green",
                     linestyle="--", alpha=0.5)
        ax1.text(x, ax1.get_ylim()[1] * 0.9, name, fontsize=7, ha="center")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reprojection Error (MSE)")
    ax1.set_title("Bundle Adjustment — Loss Curve")
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_history, linewidth=0.5)
    ax2.set_yscale("log")
    for x, name in [(s1, "S1→2"), (s2, "S2→3"), (s3, "S3→4")]:
        ax2.axvline(x=x, color="red" if x == s1 else "orange" if x == s2 else "green",
                     linestyle="--", alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Reprojection Error (MSE, log)")
    ax2.set_title("Bundle Adjustment — Loss Curve (log scale)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
    plt.close()
    print(f"Loss curve saved to {OUTPUT_DIR}/loss_curve.png")


def save_colored_obj(points_np, colors_np, filepath):
    with open(filepath, "w") as f:
        f.write("# Reconstructed 3D point cloud from Bundle Adjustment\n")
        for i in range(len(points_np)):
            x, y, z = points_np[i]
            r, g, b = colors_np[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
    print(f"OBJ saved: {filepath} ({len(points_np)} vertices)")


def print_summary(f, euler, trans):
    f_val = f.item()
    fov = 2 * np.arctan(IMAGE_H / (2 * f_val)) * 180 / np.pi
    print(f"\n{'='*60}")
    print(f"  Optimized Results")
    print(f"  Focal length f = {f_val:.2f} px  ->  FoV ~ {fov:.2f} deg")
    print(f"  Image size      = {IMAGE_W} x {IMAGE_H},  cx=cy={CX:.0f}")
    print(f"  Num views = {NUM_VIEWS},  Num points = {NUM_POINTS}")
    print(f"{'='*60}")
    print("  Camera extrinsics (every 10th view):")
    for i in range(0, NUM_VIEWS, 10):
        e = euler[i].detach().cpu().numpy()
        t = trans[i].detach().cpu().numpy()
        ry_deg = e[1] * 180 / np.pi
        print(f"  View {i:03d}: Ry={ry_deg:+7.2f} deg  "
              f"T=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")


def compute_final_error(obs_2d, visibility, results):
    with torch.no_grad():
        uv_pred = project(results["points_3d"], results["euler_angles"],
                          results["translations"], results["f"])
        mask = visibility > 0.5
        diff = uv_pred - obs_2d
        rmse = torch.sqrt((diff[mask] ** 2).sum() / mask.sum())
    print(f"  Final reprojection RMSE: {rmse.item():.4f} px")


def main():
    import time
    print(f"Device: {DEVICE}")
    print("Loading data ...")
    obs_2d, visibility, colors = load_data()
    total_vis = (visibility > 0.5).sum().item()
    print(f"  obs_2d: {obs_2d.shape},  visible: {total_vis} "
          f"({total_vis / (NUM_VIEWS * NUM_POINTS) * 100:.1f}%)")

    t0 = time.time()
    results = run_ba(obs_2d, visibility)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    plot_loss(results["loss_history"])
    print_summary(results["f"], results["euler_angles"], results["translations"])
    compute_final_error(obs_2d, visibility, results)

    pts = results["points_3d"].detach().cpu().numpy()
    cols = colors.cpu().numpy()
    save_colored_obj(pts, cols, f"{OUTPUT_DIR}/reconstructed.obj")

    torch.save({
        "f": results["f"].detach().cpu(),
        "euler_angles": results["euler_angles"].detach().cpu(),
        "translations": results["translations"].detach().cpu(),
        "points_3d": results["points_3d"].detach().cpu(),
    }, f"{OUTPUT_DIR}/ba_parameters.pt")

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
