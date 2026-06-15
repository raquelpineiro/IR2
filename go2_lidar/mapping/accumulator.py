"""Acumulación, visualización y rejilla de ocupación de la nube LiDAR.

Contiene:
  - Coloreado por altura (viridis) y post-procesado de nube (outliers/normales).
  - Construcción de la malla del suelo a partir de la trayectoria recorrida y su
    subdivisión en una rejilla N x N.
  - `constructCellMap` / `points_per_cell`: conteo de puntos por celda (rejilla
    de ocupación) y asignación de cada punto a su celda.
  - `visualizator_start`: bucle de visualización Open3D del mapeo manual que, al
    terminar el recorrido, recorta la nube a la zona barrida, construye la
    rejilla y guarda en disco el cellMap / cellPoints / malla."""

import open3d as o3d
import numpy as np
import time
import cv2
import matplotlib.path as mpath

from go2_lidar.transforms import get_yaw_from_rot, _robot_to_world, _world_to_robot

OUTLIER_NB = 20             # vecinos para el filtro estadístico de outliers
OUTLIER_STD = 2.0           # nº de desviaciones típicas para marcar outlier
NORMAL_RADIUS = 0.20        # radio de búsqueda al estimar normales
NORMAL_MAX_NN = 30          # máx. vecinos al estimar normales

TRAJ_MIN_STEP = 0.03         # 3 cm: paso mínimo para añadir vértice de trayectoria
BOX_MARGIN = 0.20            # 20 cm: amplía el cuadrado del grid hacia fuera para
                             # incluir las casillas por las que pasa el robot


VOXEL_SIZE = 0.04            # 4 cm — algo más fino para suavizar mejor
DOWNSAMPLE_EVERY = 10        # ~1 s a 10 Hz: agrupa el coste del post-procesado
MAX_POINTS = 2_000_000       # tope de puntos antes de forzar un voxel down-sample
Z_RANGE = (-1.5, 4.5)        # rango de altura para mapear color por z


# Aproximación viridis (5 puntos de control): [posición(0-1), R, G, B]
_VIRIDIS = np.array([
    [0.00, 0.267, 0.005, 0.329],
    [0.25, 0.130, 0.330, 0.550],
    [0.50, 0.139, 0.595, 0.532],
    [0.75, 0.479, 0.789, 0.314],
    [1.00, 0.992, 0.906, 0.144],
])

# [0.850, 0.150, 0.100]


def colorize_by_z(xyz, z_min=Z_RANGE[0], z_max=Z_RANGE[1]):
    """Asigna a cada punto un color viridis según su altura z (normalizada al
    rango [z_min, z_max]), para distinguir suelo/paredes/techo por color."""
    z = xyz[:, 2].astype(np.float32, copy=False)
    u = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 1])
    g = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 2])
    b = np.interp(u, _VIRIDIS[:, 0], _VIRIDIS[:, 3])
    return np.stack([r, g, b], axis=1).astype(np.float64)

def post_process(pcd, robot_pos):
    """Limpia la nube (quita outliers estadísticos), estima sus normales y las
    orienta hacia el robot para un sombreado coherente. No hace nada si hay muy
    pocos puntos."""
    if len(pcd.points) < 50:
        return pcd
    cleaned, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
    cleaned.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    )
    cleaned.orient_normals_towards_camera_location(robot_pos)
    return cleaned


def square_subdivision(vertices, n_lado):
    """Subdivide la caja de 4 esquinas en una rejilla de (n_lado+1)x(n_lado+1)
    vértices mediante interpolación bilineal. Devuelve un array [filas, cols, 2]
    con las posiciones XY de los nodos de la malla."""
    n_lado = n_lado + 1
    v = np.array(vertices, dtype=float)
    v0, v1, v2, v3 = v[0], v[1], v[2], v[3]
    # Parámetros s (entre v0-v1 y v3-v2) y t (entre borde superior e inferior).
    s = np.linspace(0, 1, n_lado + 1)
    t = np.linspace(0, 1, n_lado + 1)
    S, T = np.meshgrid(s, t)
    # Interpolar los bordes y luego entre ellos (bilineal) para cada nodo.
    top = v0[None, None, :]*(1-S)[..., None] + v1[None, None, :]*(S)[..., None]
    bottom = v3[None, None, :]*(1-S)[..., None] + v2[None, None, :]*(S)[..., None]
    grid = top*(1-T)[..., None] + bottom * T[..., None]
    return grid

def grid_lineset(grid, z=0.0, color=(1.0, 0.3, 0.3)):
    """Convierte la malla de nodos (de square_subdivision) en un LineSet de
    Open3D: líneas horizontales y verticales que unen nodos vecinos, dibujadas
    a la altura `z` y con el color dado."""
    nrows, ncols, _ = grid.shape
    # Añadir la coordenada z constante y aplanar a lista de puntos 3D.
    pts3 = np.dstack([grid, np.full((nrows, ncols), z)]).reshape(-1, 3)

    def idx(r, c):
        # Índice lineal del nodo (r, c) en la lista aplanada.
        return r * ncols + c

    # Generar un segmento al vecino derecho y otro al vecino inferior de cada nodo.
    lines = []
    for r in range(nrows):
        for c in range(ncols):
            if c + 1 < ncols:
                lines.append([idx(r, c), idx(r, c+1)])
            if r + 1 < nrows:
                lines.append([idx(r, c), idx(r+1, c)])


    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts3), o3d.utility.Vector2iVector(np.asarray(lines)),)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls

def clean_box_from_trajectory(traj_xy, origin, margin=BOX_MARGIN):
    """Caja de la malla ALINEADA con el frame de arranque del robot.

    En vez de un minAreaRect (cuyo orden de esquinas es arbitrario y puede salir
    girado), construye el rectángulo (AABB) de la trayectoria en el frame
    relativo al arranque —ejes = delante/izquierda del robot— y lo devuelve en
    coordenadas del mundo. Ordena las esquinas para que `v0` sea la esquina de
    ARRANQUE: así esa esquina es la celda (0,0) y los giros entre celdas salen de
    90°.

    `margin` (m) amplía el cuadrado hacia fuera por los 4 lados. No modela que el
    LiDAR vaya adelantado respecto al cuerpo: lo que se busca es centrar el ROBOT
    COMPLETO en sus casillas (vía su pose, tal cual). La ampliación solo da
    holgura para que el arranque quede holgadamente dentro de la celda (0,0) y la
    malla cubra un poco más allá del recorrido.

    Si no hay `origin`, cae al minAreaRect de siempre (también ampliado).
    """
    traj_xy = np.asarray(traj_xy, dtype=float)
    # Sin origen (o trayectoria muy corta): caja de área mínima de OpenCV.
    if origin is None or len(traj_xy) < 2:
        (cx, cy), (w, h), ang = cv2.minAreaRect(traj_xy.astype(np.float32))
        return cv2.boxPoints(((cx, cy), (w + 2 * margin, h + 2 * margin), ang))

    # 1) Pasar la trayectoria al frame relativo al arranque (ejes delante/izq).
    ox, oy, oyaw = origin
    rel = np.array([_world_to_robot(x, y, ox, oy, oyaw) for x, y in traj_xy])
    # 2) Rectángulo alineado a esos ejes (AABB) ampliado por `margin`.
    smin, smax = rel[:, 0].min() - margin, rel[:, 0].max() + margin
    tmin, tmax = rel[:, 1].min() - margin, rel[:, 1].max() + margin

    # Esquina de arranque = la más cercana al origen relativo (0,0).
    # Se elige el extremo (min/max) más cercano a 0 en cada eje como v0.
    s0 = smin if abs(smin) <= abs(smax) else smax
    s1 = smax if s0 == smin else smin
    t0 = tmin if abs(tmin) <= abs(tmax) else tmax
    t1 = tmax if t0 == tmin else tmin

    # 3) Ordenar esquinas (v0=arranque, e_s, e_t) y volver al mundo.
    box_rel = [(s0, t0), (s1, t0), (s1, t1), (s0, t1)]   # v0=arranque, e_s, e_t
    return np.array([_robot_to_world(x, y, ox, oy, oyaw) for x, y in box_rel],
                    dtype=float)


def constructCellMap(pts, vertices, n_div, z_min = 0.15, z_max=0.8):
    """Rejilla de ocupación: cuenta cuántos puntos caen en cada celda de la
    malla n_div x n_div, considerando solo los puntos en la franja de altura
    [z_min, z_max] y dentro de la caja. Devuelve una matriz (n_div, n_div) de
    conteos."""
    # Base de la caja: v0 origen, e_s eje columnas, e_t eje filas.
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    e_s = v1 - v0
    e_t = v3 - v0

    # 1) Filtrar por altura (quedarse con la franja relevante, p.ej. paredes).
    z = pts[:,2]
    mask_z = (z>=z_min) & (z <= z_max)
    p = pts[mask_z, :2] - v0

    # 2) Coordenadas (s, t) de cada punto en la base de la caja.
    M = np.column_stack([e_s, e_t])
    st = p @ np.linalg.inv(M).T
    s_coord, t_coord = st[:,0], st[:,1]

    # 3) Quedarse solo con los puntos dentro de la caja (s,t en [0,1)).
    dentro = (s_coord >= 0) & (s_coord < 1) & (t_coord >= 0) & (t_coord < 1)
    s_coord, t_coord = s_coord[dentro], t_coord[dentro]

    # 4) Índices de celda (columna ci, fila cj) y acumular conteos.
    ci = np.floor(s_coord * n_div).astype(int)
    cj = np.floor(t_coord * n_div).astype(int)

    occupancy = np.zeros((n_div, n_div), dtype=int)
    np.add.at(occupancy, (cj, ci), 1)     # suma 1 por punto en su celda
    return occupancy

def points_per_cell(pts, vertices, n_div, z_min=0.15, z_max=0.8):
    """Asigna cada punto a su celda (i, j) de la rejilla n_div x n_div.

    Misma lógica que constructCellMap, pero en lugar de acumular el conteo
    devuelve, para los puntos que caen dentro del cuadrado y del rango z,
    su índice original en `pts` y los índices de celda (ci, cj).
    """
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    e_s = v1 - v0
    e_t = v3 - v0

    # Filtrar por altura, guardando el índice original de los puntos válidos.
    z = pts[:, 2]
    mask_z = (z >= z_min) & (z <= z_max)
    idx_z = np.flatnonzero(mask_z)
    p = pts[mask_z, :2] - v0

    # Coordenadas (s, t) y filtro de "dentro de la caja".
    M = np.column_stack([e_s, e_t])
    st = p @ np.linalg.inv(M).T
    s_coord, t_coord = st[:, 0], st[:, 1]

    dentro = (s_coord >= 0) & (s_coord < 1) & (t_coord >= 0) & (t_coord < 1)
    s_coord, t_coord = s_coord[dentro], t_coord[dentro]

    ci = np.floor(s_coord * n_div).astype(int)
    cj = np.floor(t_coord * n_div).astype(int)

    point_idx = idx_z[dentro]   # índice (en pts) de cada punto asignado
    return point_idx, ci, cj

class State:
    """Estado mutable de la sesión de mapeo (nube, trayectoria, banderas y el
    origen del frame de arranque), agrupado para usarlo desde los callbacks."""
    def __init__(self):
        self.accumulated = o3d.geometry.PointCloud()  # nube acumulada completa
        self.pcd = o3d.geometry.PointCloud()          # nube que se dibuja
        self.show_points = True
        self.show_trajectory = True
        self.points_added = False                     # ¿ya se añadió pcd al visor?
        self.frames_since_voxel = 0                   # frames desde el último down-sample
        self.traj_points = []                         # vértices de la trayectoria
        self.prev_robot_T = np.eye(4)                 # pose previa (delta de ejes)
        self.origin = None          # (x0, y0, yaw0) del robot al iniciar el mapeo

def visualizator_start(odom, custom):
    """Bucle de visualización del mapeo manual (hilo principal).

    Mientras `custom.end` sea False: actualiza los ejes del robot, traza la
    trayectoria y acumula/voxeliza la nube LiDAR. Cuando el recorrido termina
    (custom.end=True, lo pone do_square), recorta la nube a la zona barrida,
    construye la rejilla de ocupación y guarda en disco cellMap / cellPoints /
    malla / ejes. Devuelve un diccionario con todas las geometrías y datos."""

    # --- Configuración de la ventana y geometrías base -----------------------
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

    # --- Callbacks de teclado -------------------------------------------------
    def toggle_points(_v):
        # P: muestra/oculta la nube de puntos.
        s.show_points = not s.show_points
        if s.points_added:
            (vis.add_geometry if s.show_points else vis.remove_geometry)(s.pcd, reset_bounding_box=False)
        return False

    def toggle_traj(_v):
        # T: muestra/oculta la trayectoria.
        s.show_trajectory = not s.show_trajectory
        #print(f"La trayectoria es: {trajectory.points[0]}, {trajectory.points[1]}")
        (vis.add_geometry if s.show_trajectory else vis.remove_geometry)(trajectory, reset_bounding_box=False)
        return False

    def clear_map(_v):
        # C: vacía nube y trayectoria.
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
        # S: guarda la nube acumulada en .pcd.
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
    started = False     # se activa cuando hay ≥2 vértices de trayectoria

    try:
        # ===== Bucle de mapeo: corre hasta que do_square ponga custom.end =====
        while custom.end == False:
        #while True:
            pose_changed = odom.update()

            if pose_changed:
                # a) Mover los ejes del robot con el delta de pose.
                T_now = odom.T
                delta = T_now @ np.linalg.inv(s.prev_robot_T)
                robot_axis.transform(delta)
                vis.update_geometry(robot_axis)
                s.prev_robot_T = T_now

                pos = odom.t.copy()
                if s.origin is None:
                    # Frame inicial del robot: ancla del mapa (el "(0,0)" físico)
                    s.origin = np.array([pos[0], pos[1], get_yaw_from_rot(odom.R)], dtype=float)
                # b) Añadir vértice a la trayectoria si avanzó lo suficiente.
                if len(s.traj_points) == 0 or np.linalg.norm(pos - s.traj_points[-1]) > TRAJ_MIN_STEP:
                    s.traj_points.append(pos)
                    if len(s.traj_points) >= 2:
                        started = True
                        pts = np.asarray(s.traj_points)
                        n = len(pts)
                        lines = np.column_stack([np.arange(n - 1), np.arange(1, n)])
                        cols = np.tile([1.0, 0.85, 0.2], (n - 1, 1))   # amarillo
                        trajectory.points = o3d.utility.Vector3dVector(pts)
                        trajectory.lines = o3d.utility.Vector2iVector(lines)
                        trajectory.colors = o3d.utility.Vector3dVector(cols)
                        vis.update_geometry(trajectory)

            # c) Leer y acumular la nube (solo una vez ya en movimiento).
            data = custom.get_cloud()
            if data is not None and len(data["xyz"]) > 0 and odom.has_pose and started:
                xyz_lidar = data["xyz"].astype(np.float64, copy=False)
                colors = colorize_by_z(xyz_lidar)

                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_lidar)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)
                s.accumulated += frame_pcd
                s.frames_since_voxel += 1

                # Voxelizar + post-procesar periódicamente para limitar coste.
                if s.frames_since_voxel >= DOWNSAMPLE_EVERY or len(s.accumulated.points) > MAX_POINTS:
                    s.accumulated = s.accumulated.voxel_down_sample(VOXEL_SIZE)
                    s.accumulated = post_process(s.accumulated, odom.t)
                    s.frames_since_voxel = 0

                # Volcar a la nube que se dibuja.
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

            # d) Procesar eventos; salir si se cierra la ventana.
            if not vis.poll_events():
                break
            vis.update_renderer()

            if data is None:
                time.sleep(0.01)


    finally:
        # ===== Recorrido terminado: construir y guardar la rejilla ==========
        malla = None
        box = None
        cell_points = None

        try:
            # 1) Caja del suelo a partir de la trayectoria (alineada al arranque).
            pts = np.asarray(trajectory.points)[:, :2]
            # Caja alineada al arranque del robot: la esquina de arranque es v0
            # (-> celda (0,0)) y los ejes van delante/izquierda. Más estable y
            # navegable que el minAreaRect.
            box = clean_box_from_trajectory(pts, s.origin)
            print(box)

            # 2) Polígono de la caja para recortar la nube a la zona barrida.
            poly = mpath.Path(box)

            # 3) Malla visual de la rejilla.
            grid = square_subdivision(box, n_lado=custom.stops_per_side)
            malla = grid_lineset(grid, z=0.15)


            vis.add_geometry(malla)
            # 4) Recortar la nube acumulada a los puntos dentro de la caja.
            new_points = np.asarray(s.accumulated.points)
            new_colors = np.asarray(s.accumulated.colors)
            mask = poly.contains_points(new_points[:, :2])

            new_points = new_points[mask]
            new_colors = new_colors[mask]
            s.pcd.points = o3d.utility.Vector3dVector(new_points)
            s.pcd.colors = o3d.utility.Vector3dVector(new_colors)

            # 5) Rejilla de ocupación (conteos) y máscara de celdas ocupadas.
            n_div = custom.stops_per_side + 1
            occ = constructCellMap(new_points, box, n_div=n_div)
            ocupado = occ > 5

            # Para cada punto dentro del cuadrado, a qué celda (i, j) pertenece
            point_idx, cell_i, cell_j = points_per_cell(new_points, box, n_div=n_div)
            cell_points = {
                "points": new_points,
                "colors": new_colors,
                "point_idx": point_idx,
                "cell_i": cell_i,
                "cell_j": cell_j,
            }

            # 6) Guardar el cellMap (ocupación + caja + origen) que usará getBall.
            origin = s.origin if s.origin is not None else np.zeros(3)
            np.savez(f"cellMap_{int(time.time())}.npz", occupancy=occ, vertices=box, n_div=n_div,z_range=(0.15, 0.8), origin=origin)

            # d = np.load("cellMap_tiempo.npz")
            # occ = d["occupancy"]

            print(f"Occ = {occ} \n ocupado = {ocupado}")
            # Mantener la ventana refrescándose mientras custom.end siga activo.
            while custom.end:

                vis.update_geometry(s.pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()

            # Devolver todas las geometrías y datos del mapeo.
            return {
                "point_cloud": s.pcd,
                "accumulated": s.accumulated,
                "world_axis": world_axis,
                "malla": malla,
                "vertices": box,
                "n_div": n_div,
                "occupancy": occ,
                "occupied": ocupado,
                "cell_points": cell_points,
            }

        finally:
            # 7) Cerrar ventana y guardar también ejes, malla y puntos por celda.
            vis.destroy_window()
            # guardar world axis, la malla y la info de los puntos de cada celda
            ts = int(time.time())
            o3d.io.write_triangle_mesh(f"world_axis_{ts}.ply", world_axis)
            if malla is not None:
                o3d.io.write_line_set(f"malla_{ts}.ply", malla)
            if box is not None and cell_points is not None:
                np.savez(
                    f"cellPoints_{ts}.npz",
                    points=cell_points["points"],
                    colors=cell_points["colors"],
                    point_idx=cell_points["point_idx"],
                    cell_i=cell_points["cell_i"],
                    cell_j=cell_points["cell_j"],
                    vertices=box,
                    n_div=custom.stops_per_side + 1,
                )
            print(f"[guardado] world_axis_{ts}.ply  malla_{ts}.ply  cellPoints_{ts}.npz")
