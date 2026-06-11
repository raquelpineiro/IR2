"""
Búsqueda autónoma de una pelota verde sobre la malla de ocupación.

Usa la rejilla de ocupación generada por navegacion_manual.py (cellMap_*.npz)
para recorrer todas las casillas LIBRES buscando la pelota con la cámara:

  - El robot recorre las casillas libres de casilla en casilla (movimientos
    horizontales/verticales), usando el grid para saber en qué casilla está.
  - Antes de ENTRAR en cada casilla nueva, la encara, inclina el cuerpo (pitch)
    para que la cámara mire esa casilla y comprueba con la cámara (VideoClient +
    HSV) si está la pelota (verde).
  - Si la ve, se detiene y devuelve la posición (fila, col) de esa casilla.

La visualización (Open3D) muestra la malla, los ejes, las celdas ocupadas del
mapa base y, al encontrarla, pinta en verde la celda con la pelota. Además abre
una ventana con la imagen de la cámara (verde resaltado) refrescada cada cierto
tiempo.

Uso:
    python getBall.py [interfaz_red] [ruta_cellMap.npz]
"""

import sys
import glob
import time
import threading

import numpy as np
import cv2
import open3d as o3d

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient

from go2_lidar.odom import OdomTracker
from go2_lidar.transforms import _robot_to_world, _world_to_robot
from go2_lidar.mapping.get_cloudpoint import Custom
from go2_lidar.control.patterns import (
    autonomous_movement, _cell_centers_world, _world_to_cell,
)
from go2_lidar.mapping.accumulator import (
    colorize_by_z, square_subdivision, grid_lineset,
)

TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

HIT_THRESHOLD = 5            # conteo de puntos por celda para considerarla ocupada
GREEN_LO = np.array([35, 70, 50], dtype=np.uint8)    # verde en HSV (OpenCV: H 0-179)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)
GREEN_MIN_AREA = 500         # área mínima (px) del blob verde para darlo por válido

VOXEL_SIZE = 0.04
DOWNSAMPLE_EVERY = 10

CAMERA_PERIOD = 1.0          # cada cuántos segundos refrescar la ventana de cámara


class Search:
    """Estado compartido entre el hilo de movimiento y la visualización."""
    def __init__(self):
        self.end = False
        self.ball_cell = None      # (fila, col) donde se detecta la pelota
        self.ball_count = 0        # nº de puntos LiDAR en esa celda
        self.result = None


# --------------------------------------------------------------------------- #
# Carga del mapa base
# --------------------------------------------------------------------------- #
def load_baseline(path=None):
    if path is None:
        files = sorted(glob.glob("cellMap_*.npz"))
        if not files:
            raise FileNotFoundError(
                "No se encontró ningún cellMap_*.npz. Ejecuta antes navegacion_manual.py."
            )
        path = files[-1]
    d = np.load(path, allow_pickle=True)
    occ = np.asarray(d["occupancy"])
    box = np.asarray(d["vertices"], dtype=float)
    n_div = int(d["n_div"])
    z_range = tuple(float(z) for z in d["z_range"])
    origin = np.asarray(d["origin"], dtype=float) if "origin" in d.files else None
    print(f"[MAP] Cargado {path}  (rejilla {n_div}x{n_div}, z={z_range}, "
          f"origin={origin})")
    return occ, box, n_div, z_range, origin


def reanchor_box(box_abs, map_origin, cur_origin, n_div):
    """Re-ancla la malla al frame actual del robot.

    Lleva los vértices del frame absoluto de la sesión de mapeo a un frame
    relativo al arranque de aquella sesión (restando `map_origin`) y de ahí al
    frame del mundo de la sesión actual (con `cur_origin`).

    Además desplaza la malla media celda: la esquina donde arrancó el mapeo se
    sustituye por el CENTRO de esa primera casilla, de modo que el robot deba
    colocarse en el centro de la primera casilla (no sobre el vértice de la
    esquina) para que el mapa quede alineado.

    Devuelve (box_cur, start_cell).
    """
    mx, my, myaw = map_origin
    cx, cy, cyaw = cur_origin

    # 1) Vértices en el frame relativo al arranque del mapeo.
    box_rel = np.array([_world_to_robot(wx, wy, mx, my, myaw) for wx, wy in box_abs],
                       dtype=float)

    # 2) Casilla donde arrancó el robot (rel = (0,0)) y su centro -> al origen.
    r0, c0 = _world_to_cell(box_rel, n_div, (0.0, 0.0))
    r0 = int(np.clip(r0, 0, n_div - 1))
    c0 = int(np.clip(c0, 0, n_div - 1))
    centers_rel = _cell_centers_world(box_rel, n_div)
    box_rel = box_rel - centers_rel[r0, c0]

    # 3) Del frame relativo al mundo actual.
    box_cur = np.array([_robot_to_world(rx, ry, cx, cy, cyaw) for rx, ry in box_rel],
                       dtype=float)
    return box_cur, (r0, c0)


# --------------------------------------------------------------------------- #
# Detección por cámara (verde)
# --------------------------------------------------------------------------- #
def _sees_green(video):
    """True si la cámara frontal ve un blob verde suficientemente grande."""
    code, data = video.GetImageSample()
    if code != 0 or not data:
        return False
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    return max(cv2.contourArea(c) for c in cnts) >= GREEN_MIN_AREA


def make_look_for_ball(video, video_lock, n_frames=4, settle=0.12):
    """
    Devuelve look_for_ball() -> bool.

    Con el robot ya inclinado (pitch) y encarando la casilla, captura varios
    fotogramas y confirma si hay verde (la pelota) en alguno de ellos.
    """
    def look_for_ball():
        for _ in range(n_frames):
            with video_lock:
                green = _sees_green(video)
            if green:
                print("[DET] Verde detectado por la cámara")
                return True
            time.sleep(settle)
        return False

    return look_for_ball


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #
def _cell_quad(box, n_div, r, c, z=0.05, color=(0.1, 0.9, 0.1)):
    """Cuadrado relleno (TriangleMesh) sobre la celda (fila r, col c)."""
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


def _show_camera(video, video_lock):
    """Captura un fotograma y lo muestra en una ventana OpenCV, resaltando el
    verde detectado. No bloquea: si no hay imagen, simplemente no actualiza."""
    with video_lock:
        code, data = video.GetImageSample()
    if code != 0 or not data:
        return
    img = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    green = np.zeros_like(img)
    green[mask > 0] = (0, 255, 0)
    view = cv2.addWeighted(img, 0.75, green, 0.25, 0)
    cv2.imshow("Camara Go2 (verde resaltado)", view)
    cv2.waitKey(1)


def visualize(odom, lidar, search, box, n_div, baseline_occupied, cloud_lock,
              video, video_lock):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Go2 - Buscar pelota", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.04, 0.04, 0.07])
    opt.light_on = True

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(world_axis)
    robot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(robot_axis)

    # Malla de la rejilla (n_div celdas por lado -> n_lado = n_div - 1).
    grid = square_subdivision(box, n_lado=n_div - 1)
    malla = grid_lineset(grid, z=0.05, color=(0.5, 0.5, 0.5))
    vis.add_geometry(malla)

    # Celdas ocupadas del mapa base (rojo tenue).
    for (r, c) in zip(*np.where(baseline_occupied)):
        vis.add_geometry(_cell_quad(box, n_div, r, c, z=0.02, color=(0.5, 0.15, 0.15)))

    pcd = o3d.geometry.PointCloud()
    accumulated = o3d.geometry.PointCloud()
    prev_T = np.eye(4)
    points_added = False
    ball_added = False
    frames = 0
    last_cam = 0.0

    def paint_ball():
        nonlocal ball_added
        if search.ball_cell is not None and not ball_added:
            r, c = search.ball_cell
            vis.add_geometry(_cell_quad(box, n_div, r, c, z=0.08, color=(0.1, 0.9, 0.1)),
                             reset_bounding_box=False)
            ball_added = True
            print(f"[VIZ] Pelota pintada en la celda (fila={r}, col={c})")

    while not search.end:
        if odom.update():
            T = odom.T
            robot_axis.transform(T @ np.linalg.inv(prev_T))
            prev_T = T
            vis.update_geometry(robot_axis)

        with cloud_lock:
            data = lidar.get_cloud()
        if data is not None and len(data["xyz"]) > 0 and odom.has_pose:
            xyz = data["xyz"].astype(np.float64, copy=False)
            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd.points = o3d.utility.Vector3dVector(xyz)
            frame_pcd.colors = o3d.utility.Vector3dVector(colorize_by_z(xyz))
            accumulated += frame_pcd
            frames += 1
            if frames >= DOWNSAMPLE_EVERY:
                accumulated = accumulated.voxel_down_sample(VOXEL_SIZE)
                frames = 0
            pcd.points = accumulated.points
            pcd.colors = accumulated.colors
            if not points_added:
                vis.add_geometry(pcd, reset_bounding_box=True)
                points_added = True
            else:
                vis.update_geometry(pcd)

        paint_ball()

        # Refrescar la ventana de cámara cada CAMERA_PERIOD segundos.
        now = time.time()
        if now - last_cam > CAMERA_PERIOD:
            last_cam = now
            _show_camera(video, video_lock)

        if not vis.poll_events():
            break
        vis.update_renderer()
        if data is None:
            time.sleep(0.01)

    # Búsqueda terminada: asegurar que la pelota queda pintada y mantener ventana.
    paint_ball()
    while True:
        now = time.time()
        if now - last_cam > CAMERA_PERIOD:
            last_cam = now
            _show_camera(video, video_lock)
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.02)
    vis.destroy_window()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = [a for a in sys.argv[1:]]
    net_iface = None
    map_path = None
    for a in args:
        if a.endswith(".npz"):
            map_path = a
        else:
            net_iface = a

    if net_iface is not None:
        ChannelFactoryInitialize(0, net_iface)
    else:
        ChannelFactoryInitialize(0)

    occ, box_abs, n_div, z_range, map_origin = load_baseline(map_path)
    baseline_occupied = occ > HIT_THRESHOLD

    lidar = Custom(TOPIC_CLOUD)
    odom = OdomTracker()

    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()

    # Re-anclar la malla al frame actual: el robot debe arrancar físicamente en
    # el mismo punto/orientación que cuando se mapeó (el "(0,0)" del mapa).
    # OJO: _wait_for_pose() solo espera a has_pose; hay que bombear odom.update()
    # nosotros mismos porque aún no corre ningún otro hilo que lo haga.
    print("[INIT] Esperando pose para fijar el frame de arranque...")
    while not odom.has_pose:
        odom.update()
        time.sleep(0.02)
    cur_origin = odom._initial_frame()
    if map_origin is None:
        print("[INIT] AVISO: el mapa no contiene 'origin' (mapa antiguo). "
              "Se usa en frame absoluto; vuelve a ejecutar navegacion_manual.py "
              "para alinear correctamente.")
        box = box_abs
    else:
        box, start_cell = reanchor_box(box_abs, map_origin, cur_origin, n_div)
        print(f"[INIT] Malla re-anclada: primera casilla {start_cell} centrada "
              f"en la pose actual. origin_mapeo={map_origin} "
              f"origin_actual={tuple(round(v, 3) for v in cur_origin)}")

    search = Search()
    cloud_lock = threading.Lock()
    video_lock = threading.Lock()

    look_for_ball = make_look_for_ball(video, video_lock)

    def run_search():
        try:
            cell = autonomous_movement(
                client, odom, occ, box, n_div, look_for_ball,
                hit_threshold=HIT_THRESHOLD,
            )
            if cell is not None:
                search.ball_cell = cell
            search.result = cell
        finally:
            search.end = True

    nav_thread = threading.Thread(target=run_search, daemon=True)
    nav_thread.start()

    visualize(odom, lidar, search, box, n_div, baseline_occupied, cloud_lock,
              video, video_lock)

    if search.ball_cell is not None:
        r, c = search.ball_cell
        print(f"\n>>> PELOTA ENCONTRADA en la celda fila={r}, col={c}")
    else:
        print("\n>>> No se detectó la pelota durante el recorrido")


if __name__ == "__main__":
    print("WARNING: asegúrate de que no hay obstáculos alrededor del robot.")
    input("Pulsa Enter para continuar...")
    main()
