import time
import math

import numpy as np

from go2_lidar.transforms import _robot_to_world, _wrap_pi

from go2_lidar.control.primitives import (
    _pivot_to_heading_precise, _walk_to, _go_to_world_xy,
)

def do_square(client, odom, step=0.65, stops_per_side=5, pause_s=1.0,
              clockwise=False, tolerance=0.165, lidar=None, hit_threshold=5):
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
    odom._wait_for_pose()
    x0, y0, yaw0 = odom._initial_frame()
    side_len = step * stops_per_side

    print(f"STARTING YAW IN THE FOLLOWING FORMAT: {yaw0}")
    print(f"YAW TO DEGREES: {math.degrees(yaw0)}")

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
        if side_num == 1:
            base_x, base_y = 0.0, 0.0
        elif side_num == 2:
            base_x, base_y = step*(stops_per_side) - 0.2, -0.2
        elif side_num == 3:
            base_x, base_y = step*(stops_per_side) - 0.4, -step*(stops_per_side)
        elif side_num == 4:
            base_x, base_y = -0.20, -step*(stops_per_side) + 0.2
        # Pivote a HEADING ABSOLUTO del lado (en el mundo), ignorando deriva en XY
        target_heading = _wrap_pi(yaw0 + angle_rel)
        print(f"[SQUARE] Lado {side_num}/4 -> heading mundo "
              f"{math.degrees(target_heading):+6.1f}°")
        
        # print(f"CURRENT ODOMETRY: {odom.t.copy()}")
        _pivot_to_heading_precise(target_heading, client, odom, tag="corner")
        # print(f"RESULTING ODOMETRY: {odom.t.copy()}")

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


    client.StopMove()
    lidar.end = True
    print("[SQUARE] Cuadrado completado")


# ---------------------------------------------------------------------------
# Búsqueda autónoma sobre la rejilla de ocupación
# ---------------------------------------------------------------------------

def _cell_centers_world(vertices, n_div):
    """Centro (x, y) en el mundo de cada celda de la rejilla n_div x n_div.

    Usa la misma parametrización que `constructCellMap`: v0 es el origen,
    e_s = v1 - v0 (dirección de columnas, índice ci) y e_t = v3 - v0
    (dirección de filas, índice cj). Devuelve un array [fila, col, 2].
    """
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    e_s = v1 - v0
    e_t = v3 - v0

    s = (np.arange(n_div) + 0.5) / n_div     # columnas (ci)
    t = (np.arange(n_div) + 0.5) / n_div     # filas (cj)
    S, T = np.meshgrid(s, t)                 # [fila, col]
    cx = v0[0] + e_s[0] * S + e_t[0] * T
    cy = v0[1] + e_s[1] * S + e_t[1] * T
    return np.stack([cx, cy], axis=-1)


def _world_to_cell(vertices, n_div, xy):
    """Celda (fila, col) en la que cae un punto (x, y) del mundo."""
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    M = np.column_stack([v1 - v0, v3 - v0])
    s, t = (np.asarray(xy, dtype=float) - v0) @ np.linalg.inv(M).T
    return int(np.floor(t * n_div)), int(np.floor(s * n_div))


def _nearest_free(free, centers, xy):
    """Celda libre cuyo centro está más cerca de (x, y)."""
    rs, cs = np.where(free)
    if len(rs) == 0:
        raise ValueError("No hay celdas libres en la rejilla de ocupación")
    d = np.linalg.norm(centers[rs, cs] - np.asarray(xy, dtype=float), axis=1)
    k = int(np.argmin(d))
    return int(rs[k]), int(cs[k])


def _coverage_walk(free, start):
    """Recorrido 4-conexo de todas las celdas libres alcanzables desde `start`.

    DFS con backtracking explícito: cada par consecutivo del recorrido son
    celdas adyacentes (comparten lado), de modo que el robot solo se mueve en
    horizontal o vertical, de casilla en casilla.
    """
    n_rows, n_cols = free.shape

    def neighbors(r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and free[nr, nc]:
                yield nr, nc

    visited = {start}
    walk = [start]
    stack = [(start, list(neighbors(*start)))]
    while stack:
        cell, nbrs = stack[-1]
        advanced = False
        while nbrs:
            nb = nbrs.pop()
            if nb not in visited:
                visited.add(nb)
                walk.append(nb)                  # paso adelante (adyacente)
                stack.append((nb, list(neighbors(*nb))))
                advanced = True
                break
        if not advanced:
            stack.pop()
            if stack:
                walk.append(stack[-1][0])         # backtrack (adyacente)
    return walk


def _hold(client, secs):
    """Mantiene al robot quieto (Move 0) durante `secs` segundos."""
    end = time.time() + secs
    while time.time() < end:
        client.Move(0.0, 0.0, 0.0)
        time.sleep(0.05)


def autonomous_movement(client, odom, occupancy, vertices, n_div, check_ball,
                        hit_threshold=5, pause_s=1.0, tolerance=0.165,
                        scan_headings=None, settle_s=0.5):
    """
    Recorre las celdas LIBRES de una rejilla de ocupación buscando la pelota.

    `occupancy` es la rejilla n_div x n_div obtenida con navegacion_manual.py;
    las celdas con conteo > `hit_threshold` se consideran ocupadas y no se
    pisan. El robot va de casilla en casilla moviéndose solo en horizontal o
    vertical (recorrido 4-conexo) y, en cada casilla nueva, gira encarando los
    `scan_headings` (por defecto N, E, S, O relativos al yaw inicial) y llama a
    `check_ball()`.

    `check_ball()` debe devolver la celda (fila, col) donde está la pelota si la
    detecta, o None. En cuanto devuelve una celda, el robot se detiene y esta
    función retorna esa celda. Si recorre todo sin éxito, devuelve None.
    """
    occupancy = np.asarray(occupancy)
    occupied = occupancy > hit_threshold
    free = ~occupied
    centers = _cell_centers_world(vertices, n_div)

    odom._wait_for_pose()
    x0, y0, yaw0 = odom._initial_frame()

    if scan_headings is None:
        scan_headings = [0.0, math.pi / 2, math.pi, -math.pi / 2]

    # Celda de partida: la del robot si es libre, si no la libre más cercana.
    r0, c0 = _world_to_cell(vertices, n_div, odom.t[:2])
    if not (0 <= r0 < n_div and 0 <= c0 < n_div and free[r0, c0]):
        r0, c0 = _nearest_free(free, centers, odom.t[:2])

    walk = _coverage_walk(free, (r0, c0))
    print(f"[AUTO] {int(free.sum())} celdas libres; recorrido de {len(walk)} "
          f"pasos desde la celda {(r0, c0)}")

    # Asegurar gait de marcha y precalentar (igual que do_square).
    print("[AUTO] BalanceStand + ClassicWalk")
    client.BalanceStand()
    time.sleep(0.6)
    client.ClassicWalk(True)
    time.sleep(0.4)
    warmup_end = time.time() + 1.2
    while time.time() < warmup_end:
        client.Move(0.0, 0.0, 0.0)
        time.sleep(0.05)

    scanned = set()
    for (r, c) in walk:
        wx, wy = centers[r, c]
        print(f"[AUTO] -> celda ({r},{c})  mundo=({wx:+.2f},{wy:+.2f})")
        _go_to_world_xy(wx, wy, client, odom, tolerance=tolerance)
        _hold(client, pause_s)

        if (r, c) in scanned:
            continue
        scanned.add((r, c))

        # Escaneo: gira encarando cada heading y mira si hay pelota.
        for ang in scan_headings:
            _pivot_to_heading_precise(_wrap_pi(yaw0 + ang), client, odom, tag="scan")
            _hold(client, settle_s)
            found = check_ball()
            if found is not None:
                client.StopMove()
                print(f"[AUTO] ¡Pelota detectada en la celda {found}!")
                return found

    client.StopMove()
    print("[AUTO] Recorrido completado: no se encontró la pelota")
    return None