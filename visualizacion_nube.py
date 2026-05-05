import sys
import time
import numpy as np
import open3d as o3d

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from obtencion_nube_puntos import Custom


VOXEL_SIZE = 0.05          # 5 cm — equilibrio entre detalle y memoria
DOWNSAMPLE_EVERY = 5       # frames entre voxel_down_sample
MAX_POINTS = 2_000_000     # tope duro antes de forzar downsample
Z_RANGE = (-2.0, 5.0)      # rango fijo (m) para que el color sea coherente entre frames


def colorize_by_z(xyz, z_min=Z_RANGE[0], z_max=Z_RANGE[1]):
    z = xyz[:, 2].astype(np.float32, copy=False)
    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = t
    g = 0.5 * (1.0 - np.abs(2.0 * t - 1.0))
    b = 1.0 - t
    return np.stack([r, g, b], axis=1).astype(np.float64, copy=False)


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Go2 LiDAR (acumulado)", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.05, 0.05, 0.05])

    accumulated = o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axis)

    geometry_added = False
    frames_since_voxel = 0

    try:
        while True:
            data = custom.get_cloud()

            if data is not None and len(data["xyz"]) > 0:
                xyz = data["xyz"].astype(np.float64, copy=False)
                colors = colorize_by_z(xyz)

                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)

                accumulated += frame_pcd
                frames_since_voxel += 1

                if frames_since_voxel >= DOWNSAMPLE_EVERY or len(accumulated.points) > MAX_POINTS:
                    accumulated = accumulated.voxel_down_sample(VOXEL_SIZE)
                    frames_since_voxel = 0

                pcd.points = accumulated.points
                pcd.colors = accumulated.colors

                if not geometry_added:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    geometry_added = True
                else:
                    vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

            if data is None:
                time.sleep(0.01)
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    main()
