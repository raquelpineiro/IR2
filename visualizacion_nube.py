import sys
import time
import numpy as np
import open3d as o3d

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

from obtencion_nube_puntos import Custom


TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"
TOPIC_POSE = "rt/utlidar/robot_pose"

VOXEL_SIZE = 0.05            # 5 cm en el mapa final
DOWNSAMPLE_EVERY = 5
MAX_POINTS = 2_000_000
Z_RANGE = (-2.0, 5.0)

MESH_REBUILD_EVERY = 30      # frames (~3 s a 10 Hz)
MESH_VOXEL = 0.10            # voxel previo al meshing (más grueso = más rápido)
ALPHA = 0.20                 # parámetro alpha-shapes


def quat_to_rot(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),         2.0 * (xz + wy)],
        [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),         2.0 * (yz - wx)],
        [      2.0 * (xz - wy),       2.0 * (yz + wx),   1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


class OdomTracker:
    def __init__(self, topic=TOPIC_POSE):
        self.subscriber = ChannelSubscriber(topic, PoseStamped_)
        self.subscriber.Init()
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros(3, dtype=np.float64)
        self.has_pose = False

    def update(self):
        msg = self.subscriber.Read()
        if msg is None:
            return
        p = msg.pose.position
        q = msg.pose.orientation
        self.t = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.R = quat_to_rot(q.x, q.y, q.z, q.w)
        self.has_pose = True


def colorize_by_z(xyz, z_min=Z_RANGE[0], z_max=Z_RANGE[1]):
    z = xyz[:, 2].astype(np.float32, copy=False)
    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = t
    g = 0.5 * (1.0 - np.abs(2.0 * t - 1.0))
    b = 1.0 - t
    return np.stack([r, g, b], axis=1).astype(np.float64, copy=False)


def build_mesh(pcd):
    if len(pcd.points) < 200:
        return None
    coarse = pcd.voxel_down_sample(MESH_VOXEL)
    if len(coarse.points) < 100:
        return None
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(coarse, ALPHA)
    except Exception:
        return None
    if len(mesh.triangles) == 0:
        return None
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colorize_by_z(verts))
    return mesh


class State:
    def __init__(self):
        self.accumulated = o3d.geometry.PointCloud()
        self.pcd = o3d.geometry.PointCloud()
        self.current_mesh = None
        self.show_points = True
        self.show_mesh = True
        self.points_added = False
        self.frames_since_voxel = 0
        self.frames_since_mesh = 0


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(TOPIC_CLOUD)
    odom = OdomTracker()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Go2 LiDAR (mapa + malla)", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.mesh_show_back_face = True

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axis)

    s = State()

    def toggle_points(_v):
        s.show_points = not s.show_points
        if s.points_added:
            if s.show_points:
                vis.add_geometry(s.pcd, reset_bounding_box=False)
            else:
                vis.remove_geometry(s.pcd, reset_bounding_box=False)
        return False

    def toggle_mesh(_v):
        s.show_mesh = not s.show_mesh
        if s.current_mesh is not None:
            if s.show_mesh:
                vis.add_geometry(s.current_mesh, reset_bounding_box=False)
            else:
                vis.remove_geometry(s.current_mesh, reset_bounding_box=False)
        return False

    def clear_map(_v):
        s.accumulated.clear()
        s.pcd.clear()
        if s.current_mesh is not None:
            vis.remove_geometry(s.current_mesh, reset_bounding_box=False)
            s.current_mesh = None
        if s.points_added:
            vis.update_geometry(s.pcd)
        s.frames_since_voxel = 0
        s.frames_since_mesh = 0
        return False

    def save_map(_v):
        ts = int(time.time())
        if len(s.accumulated.points) > 0:
            o3d.io.write_point_cloud(f"mapa_{ts}.pcd", s.accumulated)
            print(f"[guardado] mapa_{ts}.pcd  ({len(s.accumulated.points)} pts)")
        if s.current_mesh is not None and len(s.current_mesh.triangles) > 0:
            o3d.io.write_triangle_mesh(f"malla_{ts}.ply", s.current_mesh)
            print(f"[guardado] malla_{ts}.ply ({len(s.current_mesh.triangles)} tri)")
        return False

    vis.register_key_callback(ord("P"), toggle_points)
    vis.register_key_callback(ord("M"), toggle_mesh)
    vis.register_key_callback(ord("C"), clear_map)
    vis.register_key_callback(ord("S"), save_map)

    print("[teclas]  P: puntos on/off   M: malla on/off   C: limpiar mapa   S: guardar")

    try:
        while True:
            odom.update()
            data = custom.get_cloud()

            if data is not None and len(data["xyz"]) > 0 and odom.has_pose:
                xyz_lidar = data["xyz"].astype(np.float64, copy=False)
                xyz_world = xyz_lidar @ odom.R.T + odom.t
                colors = colorize_by_z(xyz_world)

                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_world)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)

                s.accumulated += frame_pcd
                s.frames_since_voxel += 1
                s.frames_since_mesh += 1

                if s.frames_since_voxel >= DOWNSAMPLE_EVERY or len(s.accumulated.points) > MAX_POINTS:
                    s.accumulated = s.accumulated.voxel_down_sample(VOXEL_SIZE)
                    s.frames_since_voxel = 0

                s.pcd.points = s.accumulated.points
                s.pcd.colors = s.accumulated.colors

                if not s.points_added:
                    if s.show_points:
                        vis.add_geometry(s.pcd, reset_bounding_box=True)
                    s.points_added = True
                elif s.show_points:
                    vis.update_geometry(s.pcd)

                if s.frames_since_mesh >= MESH_REBUILD_EVERY:
                    new_mesh = build_mesh(s.accumulated)
                    if new_mesh is not None:
                        if s.current_mesh is not None and s.show_mesh:
                            vis.remove_geometry(s.current_mesh, reset_bounding_box=False)
                        s.current_mesh = new_mesh
                        if s.show_mesh:
                            vis.add_geometry(s.current_mesh, reset_bounding_box=False)
                    s.frames_since_mesh = 0

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
