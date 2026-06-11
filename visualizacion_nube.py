import sys
import time
import glob
import numpy as np
import open3d as o3d

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

from obtencion_nube_puntos import Custom

from go2_lidar.mapping.accumulator import square_subdivision, grid_lineset

GRID_HIT_THRESHOLD = 5       # conteo de puntos por celda para pintarla como ocupada


TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"
TOPIC_POSE = "rt/utlidar/robot_pose"

VOXEL_SIZE = 0.04            # 4 cm — algo más fino para suavizar mejor
DOWNSAMPLE_EVERY = 10        # ~1 s a 10 Hz: agrupa el coste del post-procesado
MAX_POINTS = 2_000_000
Z_RANGE = (-1.5, 4.5)

OUTLIER_NB = 20
OUTLIER_STD = 2.0
NORMAL_RADIUS = 0.20
NORMAL_MAX_NN = 30

TRAJ_MIN_STEP = 0.03         # 3 cm: paso mínimo para añadir vértice de trayectoria


# Aproximación viridis (5 puntos de control)
_VIRIDIS = np.array([
    [0.00, 0.267, 0.005, 0.329],
    [0.25, 0.130, 0.330, 0.550],
    [0.50, 0.139, 0.595, 0.532],
    [0.75, 0.479, 0.789, 0.314],
    [1.00, 0.992, 0.906, 0.144],
])


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
            return False
        p = msg.pose.position
        q = msg.pose.orientation
        self.t = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.R = quat_to_rot(q.x, q.y, q.z, q.w)
        self.has_pose = True
        return True

    @property
    def T(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


def colorize_by_z(xyz, z_min=Z_RANGE[0], z_max=Z_RANGE[1]):
    z = xyz[:, 2].astype(np.float32, copy=False)
    u = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 1])
    g = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 2])
    b = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 3])
    return np.stack([r, g, b], axis=1).astype(np.float64)


def convex_hull_2d(points):
    """Casco convexo en planta (monotone chain). Usa solo XY. Devuelve vértices CCW (M, 2)."""
    pts = np.unique(np.asarray(points, dtype=np.float64)[:, :2], axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def build(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross2d(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    lower = build(pts)
    upper = build(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def points_inside_hull_xy(xyz, hull):
    """Máscara booleana: True si la proyección XY del punto cae dentro del polígono convexo (CCW)."""
    xy = np.asarray(xyz)[:, :2]
    inside = np.ones(len(xy), dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        cross = (b[0] - a[0]) * (xy[:, 1] - a[1]) - (b[1] - a[1]) * (xy[:, 0] - a[0])
        inside &= cross >= 0.0
        if not inside.any():
            break
    return inside


def post_process(pcd, robot_pos):
    if len(pcd.points) < 50:
        return pcd
    cleaned, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
    cleaned.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    )
    cleaned.orient_normals_towards_camera_location(robot_pos)
    return cleaned


def _cell_quad(box, n_div, r, c, z=0.02, color=(0.6, 0.2, 0.2)):
    """Cuadrado relleno (TriangleMesh) sobre la celda (fila r, col c) del grid."""
    v = np.asarray(box, dtype=float)
    v0, e_s, e_t = v[0], v[1] - v[0], v[3] - v[0]
    s0, s1 = c / n_div, (c + 1) / n_div
    t0, t1 = r / n_div, (r + 1) / n_div

    def P(s, t):
        xy = v0 + e_s * s + e_t * t
        return [xy[0], xy[1], z]

    verts = np.array([P(s0, t0), P(s1, t0), P(s1, t1), P(s0, t1)], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    )
    m.paint_uniform_color(list(color))
    m.compute_vertex_normals()
    return m


def build_last_grid(hit_threshold=GRID_HIT_THRESHOLD, path=None):
    """Carga el último cellMap_*.npz y devuelve la lista de geometrías Open3D del
    grid: la malla (líneas) y un cuadrado por cada celda ocupada. Vacía si no hay."""
    if path is None:
        files = sorted(glob.glob("cellMap_*.npz"))
        if not files:
            print("[grid] No hay ningún cellMap_*.npz que mostrar")
            return []
        path = files[-1]
    d = np.load(path, allow_pickle=True)
    occ = np.asarray(d["occupancy"])
    box = np.asarray(d["vertices"], dtype=float)
    n_div = int(d["n_div"])
    print(f"[grid] Mostrando {path}  ({n_div}x{n_div})")

    geoms = []
    grid = square_subdivision(box, n_lado=n_div - 1)
    geoms.append(grid_lineset(grid, z=0.05, color=(0.4, 0.8, 1.0)))
    for r, c in zip(*np.where(occ > hit_threshold)):
        geoms.append(_cell_quad(box, n_div, r, c, z=0.02, color=(0.6, 0.2, 0.2)))
    return geoms


class State:
    def __init__(self):
        self.accumulated = o3d.geometry.PointCloud()
        self.pcd = o3d.geometry.PointCloud()
        self.show_points = True
        self.show_trajectory = True
        self.show_grid = True
        self.points_added = False
        self.frames_since_voxel = 0
        self.traj_points = []
        self.prev_robot_T = np.eye(4)


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(TOPIC_CLOUD)
    odom = OdomTracker()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Go2 LiDAR (mapa)", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.04, 0.04, 0.07])
    opt.light_on = True

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[odom.t[0], odom.t[1], odom.t[2]])
    vis.add_geometry(world_axis)

    robot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[odom.t[0], odom.t[1], odom.t[2]])
    vis.add_geometry(robot_axis)

    trajectory = o3d.geometry.LineSet()
    vis.add_geometry(trajectory)

    s = State()

    # Último grid extraído (malla + celdas ocupadas).
    grid_geoms = build_last_grid()
    for g in grid_geoms:
        vis.add_geometry(g)

    def toggle_points(_v):
        s.show_points = not s.show_points
        if s.points_added:
            (vis.add_geometry if s.show_points else vis.remove_geometry)(s.pcd, reset_bounding_box=False)
        return False

    def toggle_traj(_v):
        s.show_trajectory = not s.show_trajectory
        (vis.add_geometry if s.show_trajectory else vis.remove_geometry)(trajectory, reset_bounding_box=False)
        return False

    def clear_map(_v):
        s.accumulated.clear()
        s.pcd.clear()
        s.traj_points.clear()
        trajectory.clear()
        if s.points_added:
            vis.update_geometry(s.pcd)
        vis.update_geometry(trajectory)
        s.frames_since_voxel = 0
        return False

    def save_map(_v):
        ts = int(time.time())
        if len(s.accumulated.points) > 0:
            o3d.io.write_point_cloud(f"mapa_{ts}.pcd", s.accumulated)
            print(f"[guardado] mapa_{ts}.pcd  ({len(s.accumulated.points)} pts)")
        return False

    def toggle_grid(_v):
        s.show_grid = not s.show_grid
        for g in grid_geoms:
            (vis.add_geometry if s.show_grid else vis.remove_geometry)(g, reset_bounding_box=False)
        return False

    vis.register_key_callback(ord("P"), toggle_points)
    vis.register_key_callback(ord("T"), toggle_traj)
    vis.register_key_callback(ord("C"), clear_map)
    vis.register_key_callback(ord("S"), save_map)
    vis.register_key_callback(ord("G"), toggle_grid)

    print("[teclas]  P: puntos   T: trayectoria   C: limpiar mapa   S: guardar   G: grid")

    try:
        while True:
            pose_changed = odom.update()

            if pose_changed:
                T_now = odom.T
                delta = T_now @ np.linalg.inv(s.prev_robot_T)
                robot_axis.transform(delta)
                vis.update_geometry(robot_axis)
                s.prev_robot_T = T_now

                pos = odom.t.copy()
                if len(s.traj_points) == 0 or np.linalg.norm(pos - s.traj_points[-1]) > TRAJ_MIN_STEP:
                    s.traj_points.append(pos)
                    if len(s.traj_points) >= 2:
                        pts = np.asarray(s.traj_points)
                        n = len(pts)
                        lines = np.column_stack([np.arange(n - 1), np.arange(1, n)])
                        cols = np.tile([1.0, 0.85, 0.2], (n - 1, 1))
                        trajectory.points = o3d.utility.Vector3dVector(pts)
                        trajectory.lines = o3d.utility.Vector2iVector(lines)
                        trajectory.colors = o3d.utility.Vector3dVector(cols)
                        vis.update_geometry(trajectory)

            data = custom.get_cloud()
            if data is not None and len(data["xyz"]) > 0 and odom.has_pose:
                xyz_lidar = data["xyz"].astype(np.float64, copy=False)
                xyz_world = xyz_lidar @ odom.R.T + odom.t
                colors = colorize_by_z(xyz_lidar)

                frame_pcd = o3d.geometry.PointCloud()

                # Pasar los puntos al sistema de referencia del robot (pose inversa):
                # asi la caja de filtrado gira y se traslada con el robot.
                xyz_robot = (xyz_lidar - odom.t) @ odom.R
                x_rob, y_rob, z_rob = xyz_robot[:, 0], xyz_robot[:, 1], xyz_robot[:, 2]
                x_range = (0, 5)
                y_range = (-5, 0)
                z_min = 0.1
                z_max = 0.8
                mask = (
                    (z_rob > z_min) & (z_rob < z_max)
                    & (x_rob >= x_range[0]) & (x_rob < x_range[1])
                    & (y_rob >= y_range[0]) & (y_rob < y_range[1]))
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_lidar[mask])
                frame_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
                s.accumulated += frame_pcd
                s.frames_since_voxel += 1

                if s.frames_since_voxel >= DOWNSAMPLE_EVERY or len(s.accumulated.points) > MAX_POINTS:
                    s.accumulated = s.accumulated.voxel_down_sample(VOXEL_SIZE)
                    s.accumulated = post_process(s.accumulated, odom.t)
                    s.frames_since_voxel = 0

                s.pcd.points = s.accumulated.points
                s.pcd.colors = s.accumulated.colors
                if s.accumulated.has_normals():
                    s.pcd.normals = s.accumulated.normals

                if not s.points_added:
                    if s.show_points:
                        vis.add_geometry(s.pcd, reset_bounding_box=True)
                    s.points_added = True
                elif s.show_points:
                    vis.update_geometry(s.pcd)

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
