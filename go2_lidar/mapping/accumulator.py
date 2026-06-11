import open3d as o3d
import numpy as np
import time
import cv2
import matplotlib.path as mpath

from go2_lidar.transforms import get_yaw_from_rot, _robot_to_world, _world_to_robot

OUTLIER_NB = 20
OUTLIER_STD = 2.0
NORMAL_RADIUS = 0.20
NORMAL_MAX_NN = 30

TRAJ_MIN_STEP = 0.03         # 3 cm: paso mínimo para añadir vértice de trayectoria
BOX_MARGIN = 0.20            # 20 cm: amplía el cuadrado del grid hacia fuera para
                             # incluir las casillas por las que pasa el robot


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


def square_subdivision(vertices, n_lado):
    n_lado = n_lado + 1
    v = np.array(vertices, dtype=float)
    v0, v1, v2, v3 = v[0], v[1], v[2], v[3]
    s = np.linspace(0, 1, n_lado + 1)
    t = np.linspace(0, 1, n_lado + 1)
    S, T = np.meshgrid(s, t)
    top = v0[None, None, :]*(1-S)[..., None] + v1[None, None, :]*(S)[..., None]
    bottom = v3[None, None, :]*(1-S)[..., None] + v2[None, None, :]*(S)[..., None]
    grid = top*(1-T)[..., None] + bottom * T[..., None]
    return grid

def grid_lineset(grid, z=0.0, color=(1.0, 0.3, 0.3)):
    nrows, ncols, _ = grid.shape
    pts3 = np.dstack([grid, np.full((nrows, ncols), z)]).reshape(-1, 3)

    def idx(r, c):
        return r * ncols + c
    
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
    if origin is None or len(traj_xy) < 2:
        (cx, cy), (w, h), ang = cv2.minAreaRect(traj_xy.astype(np.float32))
        return cv2.boxPoints(((cx, cy), (w + 2 * margin, h + 2 * margin), ang))

    ox, oy, oyaw = origin
    rel = np.array([_world_to_robot(x, y, ox, oy, oyaw) for x, y in traj_xy])
    smin, smax = rel[:, 0].min() - margin, rel[:, 0].max() + margin
    tmin, tmax = rel[:, 1].min() - margin, rel[:, 1].max() + margin

    # Esquina de arranque = la más cercana al origen relativo (0,0).
    s0 = smin if abs(smin) <= abs(smax) else smax
    s1 = smax if s0 == smin else smin
    t0 = tmin if abs(tmin) <= abs(tmax) else tmax
    t1 = tmax if t0 == tmin else tmin

    box_rel = [(s0, t0), (s1, t0), (s1, t1), (s0, t1)]   # v0=arranque, e_s, e_t
    return np.array([_robot_to_world(x, y, ox, oy, oyaw) for x, y in box_rel],
                    dtype=float)


def constructCellMap(pts, vertices, n_div, z_min = 0.15, z_max=0.8):
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    e_s = v1 - v0
    e_t = v3 - v0

    z = pts[:,2]
    mask_z = (z>=z_min) & (z <= z_max)
    p = pts[mask_z, :2] - v0
    
    M = np.column_stack([e_s, e_t])
    st = p @ np.linalg.inv(M).T
    s_coord, t_coord = st[:,0], st[:,1]

    dentro = (s_coord >= 0) & (s_coord < 1) & (t_coord >= 0) & (t_coord < 1)
    s_coord, t_coord = s_coord[dentro], t_coord[dentro]

    ci = np.floor(s_coord * n_div).astype(int)
    cj = np.floor(t_coord * n_div).astype(int)

    occupancy = np.zeros((n_div, n_div), dtype=int)
    np.add.at(occupancy, (cj, ci), 1)
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

    z = pts[:, 2]
    mask_z = (z >= z_min) & (z <= z_max)
    idx_z = np.flatnonzero(mask_z)
    p = pts[mask_z, :2] - v0

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
    def __init__(self):
        self.accumulated = o3d.geometry.PointCloud()
        self.pcd = o3d.geometry.PointCloud()
        self.show_points = True
        self.show_trajectory = True
        self.points_added = False
        self.frames_since_voxel = 0
        self.traj_points = []
        self.prev_robot_T = np.eye(4)
        self.origin = None          # (x0, y0, yaw0) del robot al iniciar el mapeo

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
        while custom.end == False:
        #while True:
            pose_changed = odom.update()

            if pose_changed:
                T_now = odom.T
                delta = T_now @ np.linalg.inv(s.prev_robot_T)
                robot_axis.transform(delta)
                vis.update_geometry(robot_axis)
                s.prev_robot_T = T_now

                pos = odom.t.copy()
                if s.origin is None:
                    # Frame inicial del robot: ancla del mapa (el "(0,0)" físico)
                    s.origin = np.array([pos[0], pos[1], get_yaw_from_rot(odom.R)], dtype=float)
                if len(s.traj_points) == 0 or np.linalg.norm(pos - s.traj_points[-1]) > TRAJ_MIN_STEP:
                    s.traj_points.append(pos)
                    if len(s.traj_points) >= 2:
                        started = True
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
                colors = colorize_by_z(xyz_lidar)
                
                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(xyz_lidar)
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

        malla = None
        box = None
        cell_points = None

        try:
            pts = np.asarray(trajectory.points)[:, :2]
            # Caja alineada al arranque del robot: la esquina de arranque es v0
            # (-> celda (0,0)) y los ejes van delante/izquierda. Más estable y
            # navegable que el minAreaRect.
            box = clean_box_from_trajectory(pts, s.origin)
            print(box)

            poly = mpath.Path(box)

            grid = square_subdivision(box, n_lado=custom.stops_per_side)
            malla = grid_lineset(grid, z=0.15)


            vis.add_geometry(malla)
            new_points = np.asarray(s.accumulated.points)
            new_colors = np.asarray(s.accumulated.colors)
            mask = poly.contains_points(new_points[:, :2])

            new_points = new_points[mask]
            new_colors = new_colors[mask]
            s.pcd.points = o3d.utility.Vector3dVector(new_points)
            s.pcd.colors = o3d.utility.Vector3dVector(new_colors)

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

            origin = s.origin if s.origin is not None else np.zeros(3)
            np.savez(f"cellMap_{int(time.time())}.npz", occupancy=occ, vertices=box, n_div=n_div,z_range=(0.15, 0.8), origin=origin)

            # d = np.load("cellMap_tiempo.npz")
            # occ = d["occupancy"]

            print(f"Occ = {occ} \n ocupado = {ocupado}")
            while custom.end:

                vis.update_geometry(s.pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()

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