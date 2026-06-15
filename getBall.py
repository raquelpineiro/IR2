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
    colorize_by_z, square_subdivision, grid_lineset, constructCellMap,
)

TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

HIT_THRESHOLD = 20            # conteo de puntos por celda para considerarla ocupada
# Banda de altura para LOCALIZAR la pelota con el LiDAR (objeto bajo): por debajo
# del rango del mapeo (z=0.15-0.8, para paredes). Ajusta según el tamaño real.
BALL_Z_RANGE = (0.03, 0.35)
GREEN_LO = np.array([35, 70, 50], dtype=np.uint8)    # verde en HSV (OpenCV: H 0-179)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)
GREEN_MIN_AREA = 500         # área mínima (px) del blob verde para darlo por válido

VOXEL_SIZE = 0.04            # tamaño de voxel para reducir la nube en el visor
DOWNSAMPLE_EVERY = 10        # cada cuántos frames voxelizar la nube acumulada

CAMERA_PERIOD = 1.0          # cada cuántos segundos refrescar la ventana de cámara


class Search:
    """Estado compartido entre el hilo de movimiento y la visualización."""
    def __init__(self):
        self.end = False           # True cuando termina la búsqueda (corta el visor)
        self.ball_cell = None      # (fila, col) donde se detecta la pelota
        self.ball_count = 0        # nº de puntos LiDAR en esa celda
        self.result = None         # resultado final devuelto por la búsqueda


# --------------------------------------------------------------------------- #
# Carga del mapa base
# --------------------------------------------------------------------------- #
def load_baseline(path=None):
    """Carga un cellMap_*.npz (el más reciente si no se da `path`) y devuelve sus
    componentes: la matriz de ocupación, las 4 esquinas de la malla, el nº de
    celdas por lado, el rango de altura usado y el origen (frame de arranque del
    mapeo). Lanza FileNotFoundError si no hay ningún mapa que cargar."""
    # Sin ruta explícita -> coger el cellMap más reciente del directorio.
    if path is None:
        files = sorted(glob.glob("cellMap_*.npz"))
        if not files:
            raise FileNotFoundError(
                "No se encontró ningún cellMap_*.npz. Ejecuta antes navegacion_manual.py."
            )
        path = files[-1]
    d = np.load(path, allow_pickle=True)
    occ = np.asarray(d["occupancy"])                       # conteos por celda
    box = np.asarray(d["vertices"], dtype=float)           # esquinas de la malla
    n_div = int(d["n_div"])                                # celdas por lado
    z_range = tuple(float(z) for z in d["z_range"])        # franja z del mapeo
    # El origen puede no existir en mapas antiguos: se devuelve None en ese caso.
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
    mx, my, myaw = map_origin       # origen (x, y, yaw) de la sesión de mapeo
    cx, cy, cyaw = cur_origin        # origen (x, y, yaw) de la sesión actual

    # 1) Vértices en el frame relativo al arranque del mapeo.
    box_rel = np.array([_world_to_robot(wx, wy, mx, my, myaw) for wx, wy in box_abs],
                       dtype=float)

    # 2) Casilla donde arrancó el robot (rel = (0,0)) y su centro -> al origen.
    r0, c0 = _world_to_cell(box_rel, n_div, (0.0, 0.0))
    r0 = int(np.clip(r0, 0, n_div - 1))
    c0 = int(np.clip(c0, 0, n_div - 1))
    centers_rel = _cell_centers_world(box_rel, n_div)
    # Restar el centro de esa primera casilla -> el origen pasa a ser su centro.
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
    # 1) Capturar un fotograma JPEG del cliente de vídeo.
    code, data = video.GetImageSample()
    if code != 0 or not data:
        return False
    # 2) Decodificar el JPEG a imagen BGR.
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    # 3) Pasar a HSV y quedarnos con los píxeles dentro del rango verde.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    # 4) Buscar contornos del verde; válido solo si el mayor supera el área mín.
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
        # Intentar varias veces: basta con ver verde en un fotograma.
        for _ in range(n_frames):
            with video_lock:                 # acceso a la cámara protegido por lock
                green = _sees_green(video)
            if green:
                print("[DET] Verde detectado por la cámara")
                return True
            time.sleep(settle)               # esperar entre capturas
        return False

    return look_for_ball


def make_locate_ball(lidar, baseline_occupied, box, n_div, z_range, cloud_lock,
                     hit_threshold=HIT_THRESHOLD):
    """
    Devuelve locate_ball(from_cell, direction) -> celda (fila, col) | None.

    Cuando la cámara ya confirmó verde en una dirección, el LiDAR localiza la
    casilla real: construye la ocupación actual (con z baja, para objetos bajos),
    marca las celdas que estaban LIBRES y ahora están ocupadas (objeto nuevo) y
    recorre la línea del grid desde `from_cell` en `direction`, devolviendo la
    PRIMERA celda nueva-ocupada. Así la pelota se atribuye a su casilla real
    aunque esté 2+ casillas por delante. None si el LiDAR no ve nada en esa línea.
    """
    z_min, z_max = z_range

    def locate_ball(from_cell, direction):
        # 1) Leer la nube actual (protegida por lock por el otro hilo).
        with cloud_lock:
            data = lidar.get_cloud()
        if data is None or len(data["xyz"]) == 0:
            return None
        xyz = data["xyz"].astype(np.float64, copy=False)
        # 2) Ocupación actual en la franja z baja (la pelota es un objeto bajo).
        cur = constructCellMap(xyz, box, n_div, z_min=z_min, z_max=z_max)
        # 3) Celdas que AHORA están ocupadas y antes estaban libres = objeto nuevo.
        new_occ = (cur > hit_threshold) & (~baseline_occupied)

        # 4) Avanzar por la línea del grid desde from_cell en `direction` y
        #    devolver la primera celda con objeto nuevo.
        r, c = from_cell
        dr, dc = direction
        r += dr
        c += dc
        while 0 <= r < n_div and 0 <= c < n_div:
            if new_occ[r, c]:
                print(f"[DET] LiDAR localiza objeto nuevo en {(int(r), int(c))}")
                return (int(r), int(c))
            r += dr
            c += dc
        return None

    return locate_ball


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #
def _cell_quad(box, n_div, r, c, z=0.05, color=(0.1, 0.9, 0.1)):
    """Cuadrado relleno (TriangleMesh) sobre la celda (fila r, col c)."""
    # v0 esquina origen; e_s recorre columnas (v1-v0), e_t filas (v3-v0).
    v = np.asarray(box, dtype=float)
    v0, e_s, e_t = v[0], v[1] - v[0], v[3] - v[0]
    s0, s1 = c / n_div, (c + 1) / n_div
    t0, t1 = r / n_div, (r + 1) / n_div

    def P(s, t):
        xy = v0 + e_s * s + e_t * t
        return [xy[0], xy[1], z]

    # 4 vértices + 4 triángulos (doble cara, para que se vea por ambos lados).
    verts = np.array([P(s0, t0), P(s1, t0), P(s1, t1), P(s0, t1)], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    )
    m.paint_uniform_color(list(color))
    m.compute_vertex_normals()
    return m


def _show_camera(video, video_lock):
    """Captura un fotograma y lo muestra en una ventana OpenCV, resaltando el
    verde detectado. No bloquea: si no hay imagen, simplemente no actualiza."""
    # 1) Capturar fotograma (con lock, compartido con la detección).
    with video_lock:
        code, data = video.GetImageSample()
    if code != 0 or not data:
        return
    img = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    # 2) Calcular la máscara de verde y crear una capa verde sobre esos píxeles.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    green = np.zeros_like(img)
    green[mask > 0] = (0, 255, 0)
    # 3) Mezclar imagen original + capa verde y mostrar.
    view = cv2.addWeighted(img, 0.75, green, 0.25, 0)
    cv2.imshow("Camara Go2 (verde resaltado)", view)
    cv2.waitKey(1)


def visualize(odom, lidar, search, box, n_div, baseline_occupied, cloud_lock,
              video, video_lock):
    """Bucle de visualización Open3D (hilo principal): dibuja malla, ejes, celdas
    ocupadas del mapa base y la nube LiDAR acumulada; refresca la ventana de
    cámara periódicamente y pinta en verde la celda de la pelota cuando se
    detecta. Mantiene la ventana abierta al terminar la búsqueda."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Go2 - Buscar pelota", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.04, 0.04, 0.07])
    opt.light_on = True

    # Ejes del mundo (fijos) y del robot (se mueven con la pose).
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
        vis.add_geometry(_cell_quad(box, n_div, r, c, z=0.049, color=(0.5, 0.15, 0.15)))

    pcd = o3d.geometry.PointCloud()                 # nube que se dibuja
    accumulated = o3d.geometry.PointCloud()         # nube acumulada
    prev_T = np.eye(4)                              # pose previa (delta de ejes)
    points_added = False                            # ¿ya se añadió pcd al visor?
    ball_added = False                              # ¿ya se pintó la pelota?
    frames = 0                                      # frames desde el último voxel
    last_cam = 0.0                                  # marca de tiempo del último refresco de cámara

    def paint_ball():
        # Pinta en verde la celda de la pelota una sola vez (cuando aparece).
        nonlocal ball_added
        if search.ball_cell is not None and not ball_added:
            r, c = search.ball_cell
            vis.add_geometry(_cell_quad(box, n_div, r, c, z=0.049, color=(0.1, 0.9, 0.1)),
                             reset_bounding_box=False)
            ball_added = True
            print(f"[VIZ] Pelota pintada en la celda (fila={r}, col={c})")

    # ----- Bucle mientras la búsqueda sigue en marcha -------------------------
    while not search.end:
        # a) Actualizar ejes del robot con el delta de pose.
        if odom.update():
            T = odom.T
            robot_axis.transform(T @ np.linalg.inv(prev_T))
            prev_T = T
            vis.update_geometry(robot_axis)

        # b) Leer y acumular la nube LiDAR (coloreada por altura).
        with cloud_lock:
            data = lidar.get_cloud()
        if data is not None and len(data["xyz"]) > 0 and odom.has_pose:
            xyz = data["xyz"].astype(np.float64, copy=False)
            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd.points = o3d.utility.Vector3dVector(xyz)
            frame_pcd.colors = o3d.utility.Vector3dVector(colorize_by_z(xyz))
            accumulated += frame_pcd
            frames += 1
            # Voxelizar periódicamente para no acumular demasiados puntos.
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

        # c) Pintar la pelota si ya se detectó.
        paint_ball()

        # Refrescar la ventana de cámara cada CAMERA_PERIOD segundos.
        now = time.time()
        if now - last_cam > CAMERA_PERIOD:
            last_cam = now
            _show_camera(video, video_lock)

        # d) Procesar eventos de ventana; salir si se cierra.
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
    """Punto de entrada: parsea argumentos, carga el mapa base, re-ancla la malla
    al frame actual del robot, lanza la búsqueda autónoma en un hilo y la
    visualización en el principal, y reporta dónde se encontró la pelota."""
    # 1) Parsear argumentos: el .npz es la ruta del mapa; lo demás, la interfaz.
    args = [a for a in sys.argv[1:]]
    net_iface = None
    map_path = None
    for a in args:
        if a.endswith(".npz"):
            map_path = a
        else:
            net_iface = a

    # 2) Inicializar la red DDS.
    if net_iface is not None:
        ChannelFactoryInitialize(0, net_iface)
    else:
        ChannelFactoryInitialize(0)

    # 3) Cargar el mapa base y derivar qué celdas están ocupadas (paredes).
    occ, box_abs, n_div, z_range, map_origin = load_baseline(map_path)
    baseline_occupied = occ > HIT_THRESHOLD

    # 4) Inicializar lector LiDAR, odometría, locomoción y vídeo.
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
    # 5) Re-anclar la malla del mapeo al frame actual (o usarla tal cual si el
    #    mapa es antiguo y no guardó 'origin').
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

    # 6) Estado compartido y locks (cámara y nube se comparten entre hilos).
    search = Search()
    cloud_lock = threading.Lock()
    video_lock = threading.Lock()

    # 7) Construir los callbacks: ver verde (cámara) y localizar la celda (LiDAR).
    look_for_ball = make_look_for_ball(video, video_lock)
    locate_ball = make_locate_ball(lidar, baseline_occupied, box, n_div,
                                   BALL_Z_RANGE, cloud_lock,
                                   hit_threshold=HIT_THRESHOLD)

    def run_search():
        # Hilo de búsqueda: recorre las celdas libres; al terminar marca end.
        try:
            cell = autonomous_movement(
                client, odom, occ, box, n_div, look_for_ball,
                locate_ball=locate_ball, hit_threshold=HIT_THRESHOLD,
            )
            if cell is not None:
                search.ball_cell = cell
            search.result = cell
        finally:
            search.end = True

    # 8) Lanzar la búsqueda en un hilo y la visualización en el principal.
    nav_thread = threading.Thread(target=run_search, daemon=True)
    nav_thread.start()

    visualize(odom, lidar, search, box, n_div, baseline_occupied, cloud_lock,
              video, video_lock)

    # 9) Reportar el resultado final.
    if search.ball_cell is not None:
        r, c = search.ball_cell
        print(f"\n>>> PELOTA ENCONTRADA en la celda fila={r}, col={c}")
    else:
        print("\n>>> No se detectó la pelota durante el recorrido")


if __name__ == "__main__":
    print("WARNING: asegúrate de que no hay obstáculos alrededor del robot.")
    input("Pulsa Enter para continuar...")
    main()
