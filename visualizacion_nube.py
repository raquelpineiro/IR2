import sys
import time
import numpy as np
import open3d as o3d

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from obtencion_nube_puntos import Custom


def colorize(xyz, cloud):
    if "intensity" in cloud.dtype.names:
        vals = cloud["intensity"].astype(np.float32, copy=False)
    else:
        vals = xyz[:, 2].astype(np.float32, copy=False)

    vmin, vmax = np.percentile(vals, [2.0, 98.0])
    t = np.clip((vals - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.stack([t, 0.5 * t, 1.0 - t], axis=1).astype(np.float64, copy=False)


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Go2 LiDAR", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.05, 0.05, 0.05])

    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axis)

    geometry_added = False

    try:
        while True:
            data = custom.get_cloud()

            if data is not None and len(data["xyz"]) > 0:
                xyz = data["xyz"].astype(np.float64, copy=False)
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colorize(xyz, data["cloud"]))

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
