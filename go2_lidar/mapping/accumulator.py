import open3d as o3d
import numpy as np
import time

OUTLIER_NB = 20
OUTLIER_STD = 2.0
NORMAL_RADIUS = 0.20
NORMAL_MAX_NN = 30

TRAJ_MIN_STEP = 0.03         # 3 cm: paso mínimo para añadir vértice de trayectoria


VOXEL_SIZE = 0.04            # 4 cm — algo más fino para suavizar mejor
DOWNSAMPLE_EVERY = 10        # ~1 s a 10 Hz: agrupa el coste del post-procesado
MAX_POINTS = 2_000_000
Z_RANGE = (-1.5, 4.5)


# Aproximación viridis (5 puntos de control)
_VIRIDIS = np.array([
    [0.00, 0.267, 0.005, 0.329],
    [0.25, 0.130, 0.330, 0.550],
    [0.50, 0.139, 0.595, 0.532],
    [0.75, 0.479, 0.789, 0.314],
    [1.00, 0.992, 0.906, 0.144],
])

# [0.850, 0.150, 0.100]


def colorize_by_z(xyz, z_min=Z_RANGE[0], z_max=Z_RANGE[1]):
    z = xyz[:, 2].astype(np.float32, copy=False)
    u = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 1])
    g = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 2])
    b = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 3])
    return np.stack([r, g, b], axis=1).astype(np.float64)

def post_process(pcd, robot_pos):
    if len(pcd.points) < 50:
        return pcd
    cleaned, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
    cleaned.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    )
    cleaned.orient_normals_towards_camera_location(robot_pos)
    return cleaned

class State:
    def __init__(self):
        self.accumulated = o3d.geometry.PointCloud()
        self.pcd = o3d.geometry.PointCloud()
        self.show_points = True
        self.show_trajectory = True
        self.points_added = False
        self.frames_since_voxel = 0
        self.traj_points = []
        self.prev_robot_T = np.eye(4)


def visualizator_start(odom, custom):    

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Go2 LiDAR (mapa)", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.04, 0.04, 0.07])
    opt.light_on = True

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(world_axis)

    robot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(robot_axis)

    trajectory = o3d.geometry.LineSet()
    vis.add_geometry(trajectory)

    s = State()

    def toggle_points(_v):
        s.show_points = not s.show_points
        if s.points_added:
            (vis.add_geometry if s.show_points else vis.remove_geometry)(s.pcd, reset_bounding_box=False)
        return False

    def toggle_traj(_v):
        s.show_trajectory = not s.show_trajectory
        #print(f"La trayectoria es: {trajectory.points[0]}, {trajectory.points[1]}")
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

    vis.register_key_callback(ord("P"), toggle_points)
    vis.register_key_callback(ord("T"), toggle_traj)
    vis.register_key_callback(ord("C"), clear_map)
    vis.register_key_callback(ord("S"), save_map)

    print("[teclas]  P: puntos   T: trayectoria   C: limpiar mapa   S: guardar")
    started = False

    try:
        #while custom.end == False:
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
                        started = True
                        ini_point = s.traj_points[-2]
                        end_point = s.traj_points[-1]
                        pts = np.asarray(s.traj_points)
                        n = len(pts)
                        lines = np.column_stack([np.arange(n - 1), np.arange(1, n)])
                        cols = np.tile([1.0, 0.85, 0.2], (n - 1, 1))
                        trajectory.points = o3d.utility.Vector3dVector(pts)
                        trajectory.lines = o3d.utility.Vector2iVector(lines)
                        trajectory.colors = o3d.utility.Vector3dVector(cols)
                        vis.update_geometry(trajectory)

            data = custom.get_cloud()
            if data is not None and len(data["xyz"]) > 0 and odom.has_pose and started:
                xyz_lidar = data["xyz"].astype(np.float64, copy=False)
                """mask = (end_point[0]-ini_point[0])*(xyz_lidar[:,1] - ini_point[1]) - (end_point[1]-ini_point[1]) *(xyz_lidar[:,0] - ini_point[0])
                mask_izq = mask > 0
                xyz_lidar = xyz_lidar[mask_izq]"""
                colors = colorize_by_z(xyz_lidar)

                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_lidar)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)
                s.accumulated += frame_pcd

                if custom.occupancy != None:
                    frame_obstacle = o3d.geometry.PointCloud()
                    frame_obstacle.points = o3d.utility.Vector3dVector(custom.occupancy)
                    frame_obstacle.colors = o3d.utility.Vector3dVector(np.array(0.850, 0.150, 0.100))
                    s.accumulated += frame_obstacle
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
        try:
            while custom.end:
                # pendiente filtrar por poligono
                pass
        finally:
            vis.destroy_window()