import sys
import time
import math
import numpy as np
import open3d as o3d
import threading  # <-- NUEVO: Necesario para que el robot se mueva y pinte a la vez

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.go2.sport.sport_client import SportClient # <-- NUEVO: Necesario para mover al robot

from obtencion_nube_puntos import Custom


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


def  get_yaw_from_rot(R):
    """Extrae el ángulo Yaw (rotación en Z) de la matriz de rotación 3x3 de la odometría."""
    return math.atan2(R[1, 0], R[0, 0])


def _go_to_world_xy(target_x, target_y, client, odom, tolerance=0.3):
    """
    Bucle de control interno: navega a (target_x, target_y) en coordenadas
    del MUNDO (las que publica rt/utlidar/robot_pose).
    """
    Kp_v = 0.6  # Ganancia proporcional para la velocidad lineal (aceleración)
    Kp_w = 1.2  # Ganancia proporcional para la velocidad de giro
    max_v = 0.4 # m/s máximo por seguridad
    max_w = 0.8 # rad/s máximo por seguridad

    while True:
        # 1. Obtener la posición actual
        curr_x = odom.t[0]
        curr_y = odom.t[1]
        curr_yaw = get_yaw_from_rot(odom.R)

        # 2. Calcular los errores (diferencia espacial)
        dx = target_x - curr_x
        dy = target_y - curr_y
        distance = math.hypot(dx, dy)

        print(f"distancia: {distance}")

        # 3. Condición de parada (¡Hemos llegado!)
        if distance < tolerance:
            print(f"[NAV] ¡Destino ({target_x:+.2f}, {target_y:+.2f}) alcanzado!")
            # client.StopMove()
            return

        # 4. Calcular hacia dónde tenemos que mirar
        target_yaw = math.atan2(dy, dx)

        # Calcular cuánto tenemos que girar (normalizando el ángulo entre -pi y pi)
        yaw_error = target_yaw - curr_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # 5. Calcular velocidades
        vyaw = Kp_w * yaw_error
        #vyaw = max(-max_w, min(max_w, vyaw))

        vx = Kp_v * distance * math.cos(yaw_error)
        vx = max(-max_v, min(max_v, vx))

        if abs(yaw_error) > math.radians(30):
            vx = 0.0

        # 6. Enviar comando al perro
        client.Move(vx, 0.0, vyaw)

        time.sleep(0.05)


def _wait_for_pose(odom):
    while not odom.has_pose:
        time.sleep(0.05)


def _initial_frame(odom):
    """Captura el frame inicial del robot a partir de la odometría."""
    x0 = odom.t[0]
    y0 = odom.t[1]
    yaw0 = get_yaw_from_rot(odom.R)
    return x0, y0, yaw0


def _robot_to_world(rx, ry, x0, y0, yaw0):
    """Transforma (rx, ry) del frame del robot inicial al frame del mundo."""
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    return x0 + c0 * rx - s0 * ry, y0 + s0 * rx + c0 * ry


def go_to_waypoint(target_x_rel, target_y_rel, client, odom, tolerance=0.3):
    """
    Navega hacia una coordenada (X, Y) expresada en el frame del robot
    en el instante de arranque: +X hacia delante, +Y hacia la izquierda.
    """
    _wait_for_pose(odom)
    x0, y0, yaw0 = _initial_frame(odom)
    target_x, target_y = _robot_to_world(target_x_rel, target_y_rel, x0, y0, yaw0)

    print(f"[THREAD] Objetivo relativo ({target_x_rel:+.2f}, {target_y_rel:+.2f}) "
          f"-> mundo ({target_x:+.2f}, {target_y:+.2f})")

    _go_to_world_xy(target_x, target_y, client, odom, tolerance=tolerance)


def do_square(client, odom, step=0.65, stops_per_side=4, pause_s=1.0,
              clockwise=False, tolerance=0.35):
    """
    Recorre un cuadrado en el plano del suelo deteniéndose cada `step` metros.
    Cada lado tiene `stops_per_side` paradas (incluyendo la esquina final),
    así que la longitud de un lado es `step * stops_per_side`.

    Por defecto: 5 paradas/lado x 0.65 m = 3.25 m por lado, sentido antihorario
    (delante → izquierda → atrás → derecha).
    """
    _wait_for_pose(odom)
    x0, y0, yaw0 = _initial_frame(odom)
    side_len = step * stops_per_side

    # Dirección de cada lado en el frame inicial del robot (vector unitario)
    if clockwise:
        sides = [(1, 0), (0, -1), (-1, 0), (0, 1)]   # delante, derecha, atrás, izquierda
    else:
        sides = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # delante, izquierda, atrás, derecha

    # Construir lista de waypoints en frame inicial, partiendo del origen del robot
    waypoints_rel = []
    base_x, base_y = 0.0, 0.0
    for dir_x, dir_y in sides:
        for i in range(1, stops_per_side + 1):
            waypoints_rel.append((base_x + dir_x * step * i,
                                  base_y + dir_y * step * i))
        base_x += dir_x * side_len
        base_y += dir_y * side_len

    print(f"[SQUARE] Cuadrado de {side_len:.2f} m de lado, "
          f"{stops_per_side} paradas/lado, paso {step:.2f} m "
          f"({'horario' if clockwise else 'antihorario'})")

    for idx, (rx, ry) in enumerate(waypoints_rel, start=1):
        wx, wy = _robot_to_world(rx, ry, x0, y0, yaw0)
        print(f"[SQUARE] {idx:2d}/{len(waypoints_rel)} "
              f"rel=({rx:+.2f},{ry:+.2f}) -> mundo=({wx:+.2f},{wy:+.2f})")
        _go_to_world_xy(wx, wy, client, odom, tolerance=tolerance)
        pause_end = time.time() + pause_s
        while time.time() < pause_end:
            client.Move(0.0, 0.0, 0.0)
            time.sleep(0.05)

    client.StopMove()
    print("[SQUARE] Cuadrado completado")


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(TOPIC_CLOUD)
    odom = OdomTracker()

    # --- NUEVO: Inicializar cliente de movimiento y lanzar hilo ---
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    # Cuadrado: 4 paradas por lado (incluida la esquina) cada 0.65 m
    # -> lado = 2.60 m, sentido antihorario (delante, izquierda, atrás, derecha)
    nav_thread = threading.Thread(
        target=do_square,
        kwargs=dict(client=client, odom=odom,
                    step=0.6, stops_per_side=3, pause_s=1.0,
                    clockwise=True),
        daemon=True
    )
    nav_thread.start()
    # ---------------------------------------------------------------

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
                colors = colorize_by_z(xyz_world)

                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_world)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)
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