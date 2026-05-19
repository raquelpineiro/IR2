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


def get_yaw_from_rot(R):
    """Extrae el ángulo Yaw (rotación en Z) de la matriz de rotación 3x3 de la odometría."""
    return math.atan2(R[1, 0], R[0, 0])


def _wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def _apply_vyaw_deadband(vyaw, yaw_error, yaw_tolerance, min_useful=0.5):
    """
    El Go2 ignora vyaw muy pequeños (banda muerta del firmware/gait).
    Si pedimos un vyaw insuficiente pero seguimos fuera de tolerancia,
    forzar un mínimo útil que sí mueva las patas.
    """
    if abs(yaw_error) > yaw_tolerance and abs(vyaw) < min_useful:
        return math.copysign(min_useful, yaw_error)
    return vyaw


def _pivot_to_heading(target_yaw_world, client, odom,
                      yaw_tolerance=math.radians(5.0), tag="pivot"):
    """
    Pivota en el sitio hasta encarar un yaw absoluto del mundo, sin importar
    posición. Útil en esquinas: garantiza 90° aunque la odometría tenga
    deriva en XY.

    Al cruzar la tolerancia manda explícitamente Move(0,0,0) para cortar la
    rotación residual del gait — sin esto, el robot sigue girando 50-200 ms
    a la última vyaw comandada (= overshoot importante con la banda muerta).
    """
    Kp_w = 1.5
    max_w = 0.8
    last_log = 0.0

    while True:
        curr_yaw = get_yaw_from_rot(odom.R)
        yaw_error = _wrap_pi(target_yaw_world - curr_yaw)

        if abs(yaw_error) < yaw_tolerance:
            client.Move(0.0, 0.0, 0.0)
            return

        vyaw = max(-max_w, min(max_w, Kp_w * yaw_error))
        vyaw = _apply_vyaw_deadband(vyaw, yaw_error, yaw_tolerance)
        client.Move(0.0, 0.0, vyaw)

        now = time.time()
        if now - last_log > 0.5:
            print(f"  [{tag}] yaw_obj={math.degrees(target_yaw_world):+6.1f}°  "
                  f"yaw_now={math.degrees(curr_yaw):+6.1f}°  "
                  f"err={math.degrees(yaw_error):+6.1f}°  "
                  f"vyaw={vyaw:+.2f}")
            last_log = now

        time.sleep(0.05)


def _pivot_to_heading_precise(target_yaw_world, client, odom, tag="corner",
                              tolerances=(math.radians(5.0),
                                          math.radians(2.0),
                                          math.radians(1.0)),
                              settle_s=0.4):
    """
    Pivote iterativo con tolerancia decreciente:

      1. Pivota a tol gruesa (5°) — corrige el grueso del giro.
      2. Manda Move(0,0,0) durante `settle_s` para que el gait se asiente.
      3. Re-mide yaw. Si sigue fuera de la próxima tol (2°), repite.
      4. Idem para 1°.

    Mata el overshoot del deadband: lo que el primer pivote pasa de largo,
    los siguientes lo recortan con menos velocidad (más fino el control P).
    """
    for i, tol in enumerate(tolerances):
        curr_yaw = get_yaw_from_rot(odom.R)
        err = _wrap_pi(target_yaw_world - curr_yaw)

        if abs(err) < tol:
            # Ya estamos dentro de esta tolerancia; saltamos al siguiente nivel
            continue

        print(f"  [{tag}] paso {i + 1}/{len(tolerances)}  "
              f"tol={math.degrees(tol):.1f}°  err_inicial={math.degrees(err):+5.2f}°")

        _pivot_to_heading(target_yaw_world, client, odom,
                          yaw_tolerance=tol, tag=f"{tag}#{i + 1}")

        settle_end = time.time() + settle_s
        while time.time() < settle_end:
            client.Move(0.0, 0.0, 0.0)
            time.sleep(0.05)

    curr_yaw = get_yaw_from_rot(odom.R)
    err = _wrap_pi(target_yaw_world - curr_yaw)
    print(f"  [{tag}] final  obj={math.degrees(target_yaw_world):+6.1f}°  "
          f"yaw={math.degrees(curr_yaw):+6.1f}°  err={math.degrees(err):+5.2f}°")


def _pivot_to_face(target_x, target_y, client, odom,
                   yaw_tolerance=math.radians(5.0)):
    """
    Pivota en el sitio hasta encarar (target_x, target_y).
    Útil para correcciones finas; en esquinas usa _pivot_to_heading.
    """
    while True:
        curr_x = odom.t[0]
        curr_y = odom.t[1]
        dx = target_x - curr_x
        dy = target_y - curr_y

        if math.hypot(dx, dy) < 1e-3:
            return

        target_yaw = math.atan2(dy, dx)
        _pivot_to_heading(target_yaw, client, odom,
                          yaw_tolerance=yaw_tolerance, tag="pivot")
        return


def _walk_to(target_x, target_y, client, odom, tolerance=0.1):
    """
    Fase 2: avanza hacia (target_x, target_y) con correcciones suaves
    de yaw para mantener el rumbo. Si el error de yaw crece más allá del
    umbral, vuelve a pivotar y reintenta.
    """
    Kp_v = 0.6
    Kp_w = 1.0
    max_v = 0.4
    max_w = 0.4                       # más suave: solo correcciones finas
    yaw_redo_threshold = math.radians(20.0)

    last_log = 0.0

    while True:
        curr_x = odom.t[0]
        curr_y = odom.t[1]
        curr_yaw = get_yaw_from_rot(odom.R)

        dx = target_x - curr_x
        dy = target_y - curr_y
        distance = math.hypot(dx, dy)

        if distance < tolerance:
            print(f"[NAV] ¡Destino ({target_x:+.2f}, {target_y:+.2f}) alcanzado!")
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - curr_yaw),
                               math.cos(target_yaw - curr_yaw))

        # Si la desviación crece demasiado, re-pivotar antes de seguir
        if abs(yaw_error) > yaw_redo_threshold:
            return _pivot_then_walk(target_x, target_y, client, odom, tolerance)

        vx = max(-max_v, min(max_v, Kp_v * distance))
        vyaw = max(-max_w, min(max_w, Kp_w * yaw_error))
        client.Move(vx, 0.0, vyaw)

        now = time.time()
        if now - last_log > 0.5:
            print(f"  [walk]  dist={distance:.3f} m  "
                  f"yaw_err={math.degrees(yaw_error):+6.1f}°  "
                  f"vx={vx:+.2f}  vyaw={vyaw:+.2f}")
            last_log = now

        time.sleep(0.05)


def _pivot_then_walk(target_x, target_y, client, odom, tolerance):
    _pivot_to_face(target_x, target_y, client, odom)
    _walk_to(target_x, target_y, client, odom, tolerance=tolerance)


def _go_to_world_xy(target_x, target_y, client, odom, tolerance=0.1):
    """
    Bucle de control: pivota en el sitio para encarar y después
    camina recto. Si el robot se desvía durante la marcha, vuelve a
    pivotar.
    """
    _pivot_to_face(target_x, target_y, client, odom)
    _walk_to(target_x, target_y, client, odom, tolerance=tolerance)


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


def go_to_waypoint(target_x_rel, target_y_rel, client, odom, tolerance=0.1):
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


class OccupancyGrid:
    """
    Rejilla N x N binaria de ocupación (N = stops_per_side), en el frame
    relativo al inicio del cuadrado. Cada celda mide `step` x `step` metros.

    Convención: +X = delante, +Y = izquierda.
      - antihorario: cuadrado en rx ∈ [0, L], ry ∈ [0, L]
      - horario:     cuadrado en rx ∈ [0, L], ry ∈ [-L, 0]
    """

    def __init__(self, step, stops_per_side, clockwise, x0, y0, yaw0,
                 z_min=-0.10, z_max=0.50, hit_threshold=5):
        self.step = step
        self.n = stops_per_side
        self.side_len = step * stops_per_side
        self.x_range = (0.0, self.side_len)
        self.y_range = (-self.side_len, 0.0) if clockwise else (0.0, self.side_len)
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.z_min, self.z_max = z_min, z_max
        self.hit_threshold = hit_threshold
        self.counts = np.zeros((self.n, self.n), dtype=np.int64)
        self.n_captures = 0
        self._lock = threading.Lock()

    def capture(self, lidar, odom):
        data = lidar.get_cloud()
        if data is None or len(data["xyz"]) == 0:
            return 0

        xyz_lidar = data["xyz"].astype(np.float64, copy=False)
        xyz_world = xyz_lidar @ odom.R.T + odom.t

        # mundo -> frame inicial del cuadrado
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        dx = xyz_world[:, 0] - self.x0
        dy = xyz_world[:, 1] - self.y0
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        rz = xyz_world[:, 2]

        mask = (
            (rz > self.z_min) & (rz < self.z_max)
            & (rx >= self.x_range[0]) & (rx < self.x_range[1])
            & (ry >= self.y_range[0]) & (ry < self.y_range[1])
        )

        with self._lock:
            self.n_captures += 1
            if not mask.any():
                return 0
            hist, _, _ = np.histogram2d(
                rx[mask], ry[mask],
                bins=[self.n, self.n],
                range=[list(self.x_range), list(self.y_range)],
            )
            self.counts += hist.astype(np.int64)
            return int(mask.sum())

    def binary(self):
        with self._lock:
            return self.counts >= self.hit_threshold

    def print_map(self):
        binmap = self.binary()
        with self._lock:
            counts = self.counts.copy()
            ncap = self.n_captures

        print(f"\n[GRID] Mapa {self.n}x{self.n}  "
              f"(celda {self.step:.2f} m, hits>={self.hit_threshold}, "
              f"capturas={ncap})    +X=arriba   +Y=izquierda")
        for ix in reversed(range(self.n)):
            chars = ["#" if binmap[ix, iy] else "." for iy in reversed(range(self.n))]
            print("    " + "  ".join(chars))

        print("\n[GRID] Hits por celda:")
        for ix in reversed(range(self.n)):
            row = [f"{counts[ix, iy]:5d}" for iy in reversed(range(self.n))]
            print("    " + " ".join(row))

    def save(self, path):
        with self._lock:
            np.savez(
                path,
                counts=self.counts,
                binary=(self.counts >= self.hit_threshold).astype(np.uint8),
                step=self.step,
                stops_per_side=self.n,
                x_range=np.array(self.x_range),
                y_range=np.array(self.y_range),
            )


def do_square(client, odom, step=0.65, stops_per_side=5, pause_s=1.0,
              clockwise=False, tolerance=0.1, lidar=None, hit_threshold=5):
    """
    Recorre un cuadrado en el plano del suelo deteniéndose cada `step` metros.
    Cada lado tiene `stops_per_side` paradas (incluyendo la esquina final),
    así que la longitud de un lado es `step * stops_per_side`.

    Por defecto: 5 paradas/lado x 0.65 m = 3.25 m por lado, sentido antihorario
    (delante → izquierda → atrás → derecha).

    Si se pasa `lidar`, en cada parada se acumula un escaneo en una rejilla
    binaria de ocupación N x N (N = stops_per_side) que se imprime y se
    guarda como `mapa_ocupacion.npz` al terminar.
    """
    _wait_for_pose(odom)
    x0, y0, yaw0 = _initial_frame(odom)
    side_len = step * stops_per_side

    grid = None
    if lidar is not None:
        grid = OccupancyGrid(
            step=step, stops_per_side=stops_per_side, clockwise=clockwise,
            x0=x0, y0=y0, yaw0=yaw0, hit_threshold=hit_threshold,
        )

    # Dirección y heading ABSOLUTO (en el mundo) de cada lado.
    # El ángulo es relativo a yaw0 -> al sumar yaw0 obtenemos el yaw del mundo
    # que el robot debe mantener mientras camina ese lado.
    if clockwise:
        sides = [
            ((1, 0),  0.0),                 # delante
            ((0, -1), -math.pi / 2),        # derecha
            ((-1, 0), math.pi),             # atrás
            ((0, 1),  math.pi / 2),         # izquierda
        ]
    else:
        sides = [
            ((1, 0),  0.0),                 # delante
            ((0, 1),  math.pi / 2),         # izquierda
            ((-1, 0), math.pi),             # atrás
            ((0, -1), -math.pi / 2),        # derecha
        ]

    print(f"[SQUARE] Cuadrado de {side_len:.2f} m de lado, "
          f"{stops_per_side} paradas/lado, paso {step:.2f} m "
          f"({'horario' if clockwise else 'antihorario'})")

    # Asegurar que el robot está en un gait de marcha. Sin esto, Move(0,0,vyaw)
    # con vx=0 puede no rotar (BalanceStand mantiene las patas plantadas).
    print("[SQUARE] BalanceStand + ClassicWalk")
    client.BalanceStand()
    time.sleep(0.6)
    client.ClassicWalk(True)
    time.sleep(0.4)

    # Precalentar gait: las primeras décimas de segundo desde reposo no
    # producen desplazamiento porque las patas están entrando en cadencia.
    print("[SQUARE] Precalentando gait (1.2 s)...")
    warmup_end = time.time() + 1.2
    while time.time() < warmup_end:
        client.Move(0.0, 0.0, 0.0)
        time.sleep(0.05)

    base_x, base_y = 0.0, 0.0
    total_wp = stops_per_side * 4
    wp_idx = 0

    for side_num, ((dir_x, dir_y), angle_rel) in enumerate(sides, start=1):
        # Pivote a HEADING ABSOLUTO del lado (en el mundo), ignorando deriva en XY
        target_heading = _wrap_pi(yaw0 + angle_rel)
        print(f"[SQUARE] Lado {side_num}/4 -> heading mundo "
              f"{math.degrees(target_heading):+6.1f}°")
        _pivot_to_heading_precise(target_heading, client, odom, tag="corner")

        # Pausa breve tras la esquina para estabilizar
        pause_end = time.time() + 0.4
        while time.time() < pause_end:
            client.Move(0.0, 0.0, 0.0)
            time.sleep(0.05)

        # Caminar a cada waypoint del lado SIN volver a pivotar entre ellos
        for i in range(1, stops_per_side + 1):
            wp_idx += 1
            rx = base_x + dir_x * step * i
            ry = base_y + dir_y * step * i
            wx, wy = _robot_to_world(rx, ry, x0, y0, yaw0)
            print(f"[SQUARE] {wp_idx:2d}/{total_wp} "
                  f"rel=({rx:+.2f},{ry:+.2f}) -> mundo=({wx:+.2f},{wy:+.2f})")

            _walk_to(wx, wy, client, odom, tolerance=tolerance)

            # Pausa entre paradas dentro del mismo lado
            pause_end = time.time() + pause_s
            while time.time() < pause_end:
                client.Move(0.0, 0.0, 0.0)
                time.sleep(0.05)

            # Captura para el mapa de ocupación (robot detenido)
            if grid is not None:
                hits = grid.capture(lidar, odom)
                print(f"  [GRID] parada {wp_idx}/{total_wp}: {hits} hits dentro del cuadrado")

        base_x += dir_x * side_len
        base_y += dir_y * side_len

    client.StopMove()
    print("[SQUARE] Cuadrado completado")

    if grid is not None:
        grid.print_map()
        grid.save("mapa_ocupacion")
        print("[GRID] guardado en mapa_ocupacion.npz")


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
    # -> lado = 2.60 m, sentido horario (delante, derecha, atrás, izquierda)
    nav_thread = threading.Thread(
        target=do_square,
        kwargs=dict(client=client, odom=odom, lidar=custom,
                    step=0.65, stops_per_side=4, pause_s=1.0,
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
