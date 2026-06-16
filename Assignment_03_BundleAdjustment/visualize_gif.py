"""Generate rotating GIF animations of reconstructed 3D point clouds.

Usage: python visualize_gif.py

Outputs:
  - output/task1_pytorch_ba.gif   (Task 1: 20,000 colored points, PyTorch BA)
  - output/task2_colmap.gif       (Task 2: 112,511 dense points, COLMAP)

Requirements: open3d, pillow, numpy, matplotlib
"""
import numpy as np
import open3d as o3d
import os
from PIL import Image

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMES = 72          # frames per full 360° rotation
DURATION = 40        # ms per frame (~25 fps)
SIZE = 800           # render resolution (square)


def load_obj_as_pointcloud(filepath):
    """Load a colored OBJ file as an Open3D PointCloud."""
    pts, colors = [], []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                pts.append([x, y, z])
                colors.append([r, g, b])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return pcd


def render_rotation_gif(pcd, output_path, elevation=15.0, dist=5.0):
    """Render a 360-degree rotating GIF of a point cloud."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=SIZE, height=SIZE)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.12, 0.12, 0.14])

    ctr = vis.get_view_control()
    el_rad = np.radians(elevation)
    frames = []

    print(f"  Rendering {FRAMES} frames ...")
    for i in range(FRAMES):
        azimuth = i * 360.0 / FRAMES
        az_rad = np.radians(azimuth)

        cam_x = dist * np.cos(el_rad) * np.sin(az_rad)
        cam_y = dist * np.sin(el_rad)
        cam_z = dist * np.cos(el_rad) * np.cos(az_rad)

        front = np.array([-cam_x, -cam_y, -cam_z])
        front = front / np.linalg.norm(front)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(front, up)) > 0.99:
            up = np.array([0.0, 0.0, 1.0])

        ctr.set_front(front)
        ctr.set_lookat(np.array([0.0, 0.0, 0.0]))
        ctr.set_up(up)
        ctr.set_zoom(1.0)

        vis.poll_events()
        vis.update_renderer()

        img_array = vis.capture_screen_float_buffer(do_render=True)
        img_uint8 = (np.asarray(img_array) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_uint8))

        if (i + 1) % 18 == 0:
            print(f"    {i + 1}/{FRAMES} frames")

    vis.destroy_window()

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,
    )
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  Saved: {output_path} ({size_kb} KB, {FRAMES} frames)")


def main():
    print("=" * 60)
    print("  Generating Rotating GIF Animations")
    print("=" * 60)

    # ---- Task 1: PyTorch BA ----
    obj_path = "output/reconstructed.obj"
    if os.path.exists(obj_path):
        print(f"\n[Task 1] Loading {obj_path} ...")
        pcd1 = load_obj_as_pointcloud(obj_path)
        center = np.mean(np.asarray(pcd1.points), axis=0)
        pcd1.translate(-center)
        print(f"  {len(pcd1.points):,} points")
        render_rotation_gif(pcd1, f"{OUTPUT_DIR}/task1_pytorch_ba.gif",
                            elevation=15.0)
    else:
        print(f"[Task 1] {obj_path} not found — run bundle_adjustment.py first")

    # ---- Task 2: COLMAP dense ----
    ply_path = "data/colmap/dense/fused.ply"
    if os.path.exists(ply_path):
        print(f"\n[Task 2] Loading {ply_path} ...")
        pcd2 = o3d.io.read_point_cloud(ply_path)
        center2 = np.mean(np.asarray(pcd2.points), axis=0)
        pcd2.translate(-center2)
        print(f"  {len(pcd2.points):,} points")

        # Apply height-based coloring if no color
        if not pcd2.has_colors() or len(np.asarray(pcd2.colors)) == 0:
            pts = np.asarray(pcd2.points)
            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            z_norm = (pts[:, 2] - z_min) / (z_max - z_min + 1e-8)
            import matplotlib.pyplot as plt
            colors = plt.cm.viridis(z_norm)[:, :3]
            pcd2.colors = o3d.utility.Vector3dVector(colors)

        render_rotation_gif(pcd2, f"{OUTPUT_DIR}/task2_colmap.gif",
                            elevation=15.0)
    else:
        print(f"[Task 2] {ply_path} not found")

    print(f"\nDone! GIFs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
