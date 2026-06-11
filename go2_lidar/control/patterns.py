import time
import math

from collections import deque

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


def _free_neighbors(free, cell):
    """Vecinos 4-conexos transitables (True en `free`) de una celda."""
    r, c = cell
    n_rows, n_cols = free.shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols and free[nr, nc]:
            out.append((nr, nc))
    return out


def _bfs_path(free, start, goals):
    """Camino 4-conexo más corto (BFS) por celdas transitables desde `start`
    hasta cualquier celda de `goals`. Devuelve la lista de celdas (incluido
    `start`) o None si no hay ruta."""
    goals = set(goals)
    if start in goals:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in _free_neighbors(free, cur):
            if nb not in prev:
                prev[nb] = cur
                if nb in goals:
                    path = [nb]
                    p = cur
                    while p is not None:
                        path.append(p)
                        p = prev[p]
                    return path[::-1]
                q.append(nb)
    return None


def _ensure_walk(client, warmup_s=0.6):
    """Pone al robot en gait de marcha (tras un BalanceStand/Euler hay que
    reactivarlo) y precalienta brevemente para que las patas entren en cadencia."""
    client.BalanceStand()
    time.sleep(0.3)
    client.ClassicWalk(True)
    time.sleep(0.3)
    end = time.time() + warmup_s
    while time.time() < end:
        client.Move(0.0, 0.0, 0.0)
        time.sleep(0.05)


def _look_down(client, pitch, settle_s=0.6):
    """Inclina el cuerpo (morro abajo) para que la cámara apunte al suelo/objeto.
    La cámara del Go2 es fija; la única forma de bajar su mirada es el pitch del
    cuerpo vía Euler() en modo BalanceStand."""
    client.StopMove()
    client.BalanceStand()
    time.sleep(0.3)
    if hasattr(client, "Euler"):
        client.Euler(0.0, float(pitch), 0.0)
    else:
        print("[AUTO] AVISO: SportClient no expone Euler(); no puedo inclinar la cámara")
    time.sleep(settle_s)


def _look_level(client, settle_s=0.3):
    """Devuelve el cuerpo a la horizontal."""
    if hasattr(client, "Euler"):
        client.Euler(0.0, 0.0, 0.0)
    client.BalanceStand()
    time.sleep(settle_s)


def autonomous_movement(client, odom, occupancy, vertices, n_div,
                        detect_new_cells, confirm_ball,
                        hit_threshold=5, pause_s=0.5, tolerance=0.165,
                        camera_pitch=0.5, settle_s=0.6, explore=True):
    """
    Busca un objeto nuevo (la pelota) sobre la rejilla de ocupación.

    Estrategia:
      1. `detect_new_cells()` (LiDAR) devuelve las celdas que estaban LIBRES en
         el mapa base y ahora aparecen ocupadas: candidatas a objeto nuevo.
      2. El robot navega hasta una casilla libre CONTIGUA al candidato más
         cercano (BFS 4-conexo), evitando tanto las celdas ocupadas del mapa
         base como las nuevas (no choca con el objeto).
      3. Encara el candidato, baja la cámara con un `pitch` del cuerpo y llama a
         `confirm_ball()` (cámara) para ver si es verde.
      4. Si lo confirma, se detiene y devuelve la celda (fila, col). Si no, la
         descarta y sigue. Si no hay candidatos visibles y `explore=True`, se
         mueve por celdas libres para ganar visión.

    `detect_new_cells()` -> lista de celdas (fila, col).
    `confirm_ball()` -> bool.
    Devuelve la celda de la pelota, o None si no la encuentra.
    """
    occupancy = np.asarray(occupancy)
    free = ~(occupancy > hit_threshold)
    centers = _cell_centers_world(vertices, n_div)

    odom._wait_for_pose()
    odom._initial_frame()

    def cur_cell():
        r, c = _world_to_cell(vertices, n_div, odom.t[:2])
        r = int(np.clip(r, 0, n_div - 1))
        c = int(np.clip(c, 0, n_div - 1))
        if not free[r, c]:
            r, c = _nearest_free(free, centers, odom.t[:2])
        return (r, c)

    def walk_path(path):
        """Camina por las celdas de `path` (la primera es la actual)."""
        for (r, c) in path[1:]:
            wx, wy = centers[r, c]
            print(f"[AUTO] -> celda ({r},{c})  mundo=({wx:+.2f},{wy:+.2f})")
            _go_to_world_xy(wx, wy, client, odom, tolerance=tolerance)
            _hold(client, pause_s)

    # Orden de exploración (fallback cuando no se ve ningún objeto nuevo).
    explore_order = []
    if explore:
        seen = set()
        for cell in _coverage_walk(free, cur_cell()):
            if cell not in seen:
                seen.add(cell)
                explore_order.append(cell)
    ei = 0

    print(f"[AUTO] {int(free.sum())} celdas libres. BalanceStand + ClassicWalk")
    _ensure_walk(client, warmup_s=1.2)

    checked = set()     # candidatos ya inspeccionados que NO eran la pelota

    while True:
        new_cells = [tuple(int(v) for v in c) for c in detect_new_cells()]
        new_set = {c for c in new_cells if 0 <= c[0] < n_div and 0 <= c[1] < n_div}

        # Rejilla transitable: libres del mapa base MENOS los objetos nuevos.
        nav_free = free.copy()
        for (r, c) in new_set:
            nav_free[r, c] = False
        cur = cur_cell()
        nav_free[cur] = True    # nunca bloquear la celda donde está el robot

        cands = [c for c in new_set if c not in checked]
        if cands:
            best, best_path = None, None
            for cand in cands:
                goals = _free_neighbors(nav_free, cand)
                if not goals:
                    checked.add(cand)       # rodeado de ocupadas: inalcanzable
                    continue
                p = _bfs_path(nav_free, cur, goals)
                if p is not None and (best_path is None or len(p) < len(best_path)):
                    best, best_path = cand, p

            if best is not None:
                print(f"[AUTO] Investigando celda nueva {best} "
                      f"({len(best_path) - 1} pasos)")
                _ensure_walk(client)
                walk_path(best_path)

                # Encarar el candidato desde la casilla contigua.
                stand = best_path[-1]
                sx, sy = centers[stand]
                tx, ty = centers[best]
                heading = math.atan2(ty - sy, tx - sx)
                _pivot_to_heading_precise(_wrap_pi(heading), client, odom, tag="face")
                _hold(client, 0.3)

                # Bajar la cámara (pitch) y confirmar con la cámara.
                _look_down(client, camera_pitch, settle_s)
                found = confirm_ball()
                _look_level(client)

                if found:
                    client.StopMove()
                    print(f"[AUTO] ¡Pelota confirmada en la celda {best}!")
                    return best

                print(f"[AUTO] La celda {best} no es la pelota; descartada")
                checked.add(best)
                continue        # re-detectar desde la nueva posición

        # Sin candidatos accionables: explorar para ganar visión.
        moved = False
        while ei < len(explore_order):
            tgt = explore_order[ei]
            ei += 1
            if tgt == cur or not nav_free[tgt]:
                continue
            p = _bfs_path(nav_free, cur, {tgt})
            if p is not None and len(p) > 1:
                print(f"[AUTO] Explorando hacia {tgt}")
                _ensure_walk(client)
                walk_path(p)
                moved = True
                break
        if not moved:
            client.StopMove()
            print("[AUTO] No hay objetos nuevos por confirmar ni zonas por explorar")
            return None