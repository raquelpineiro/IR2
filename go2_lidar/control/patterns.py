"""Patrones de movimiento de alto nivel del Go2.

Dos recorridos completos construidos sobre las primitivas de `primitives.py`:

  - `do_square`: recorre un cuadrado parando cada `step` metros (usado en el
    mapeo manual para barrer el entorno con el LiDAR).
  - `autonomous_movement`: recorre todas las casillas LIBRES de una rejilla de
    ocupación buscando la pelota con la cámara (usado en getBall.py).

Incluye además las utilidades de rejilla (centros de celda, mundo<->celda,
recorrido de cobertura, BFS) que ambos comparten."""

import time
import math

from collections import deque

import numpy as np

from go2_lidar.transforms import _robot_to_world, _wrap_pi

from go2_lidar.control.primitives import (
    _pivot_to_heading_precise, _walk_to, _go_to_world_xy,
)

def do_square(client, odom, step=0.65, stops_per_side=5, pause_s=1.0,
              clockwise=False, tolerance=0.165, lidar=None, hit_threshold=20):
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
    # 1) Esperar la primera pose y fijar el frame de arranque (origen y yaw0).
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
    total_wp = stops_per_side * 4       # nº total de waypoints (4 lados)
    wp_idx = 0

    # 2) Recorrer los 4 lados del cuadrado.
    for side_num, ((dir_x, dir_y), angle_rel) in enumerate(sides, start=1):
        # Punto base (esquina) de cada lado, en coordenadas relativas al
        # arranque. Los pequeños offsets (±0.2) compensan deriva acumulada.
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
        # Girar a la orientación del lado con el pivote fino (multitolerancia).
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
            # Waypoint relativo al arranque -> coordenadas del mundo.
            rx = base_x + dir_x * step * i
            ry = base_y + dir_y * step * i
            wx, wy = _robot_to_world(rx, ry, x0, y0, yaw0)
            print(f"[SQUARE] {wp_idx:2d}/{total_wp} "
                  f"rel=({rx:+.2f},{ry:+.2f}) -> mundo=({wx:+.2f},{wy:+.2f})")

            # Caminar hasta el waypoint (el LiDAR va acumulando en el otro hilo).
            _walk_to(wx, wy, client, odom, tolerance=tolerance)

            # Pausa entre paradas dentro del mismo lado
            pause_end = time.time() + pause_s
            while time.time() < pause_end:
                client.Move(0.0, 0.0, 0.0)
                time.sleep(0.05)


    # 3) Fin del recorrido: parar y avisar al hilo del visor (lidar.end) para
    #    que cierre el mapeo y guarde el cellMap.
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

    # Fracciones centradas en cada celda: (i + 0.5)/n_div.
    s = (np.arange(n_div) + 0.5) / n_div     # columnas (ci)
    t = (np.arange(n_div) + 0.5) / n_div     # filas (cj)
    S, T = np.meshgrid(s, t)                 # [fila, col]
    # Interpolación bilineal sobre la caja para obtener el centro de cada celda.
    cx = v0[0] + e_s[0] * S + e_t[0] * T
    cy = v0[1] + e_s[1] * S + e_t[1] * T
    return np.stack([cx, cy], axis=-1)


def _world_to_cell(vertices, n_div, xy):
    """Celda (fila, col) en la que cae un punto (x, y) del mundo."""
    # Resolver (s, t) en [0,1) proyectando (xy - v0) sobre la base (e_s, e_t).
    v = np.asarray(vertices, dtype=float)
    v0, v1, v3 = v[0], v[1], v[3]
    M = np.column_stack([v1 - v0, v3 - v0])
    s, t = (np.asarray(xy, dtype=float) - v0) @ np.linalg.inv(M).T
    # Multiplicar por n_div y truncar -> índices (fila=t, col=s).
    return int(np.floor(t * n_div)), int(np.floor(s * n_div))


def _nearest_free(free, centers, xy):
    """Celda libre cuyo centro está más cerca de (x, y)."""
    rs, cs = np.where(free)
    if len(rs) == 0:
        raise ValueError("No hay celdas libres en la rejilla de ocupación")
    # Distancia del punto a cada centro libre y elegir el mínimo.
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
        # Vecinos 4-conexos (arriba/abajo/izq/der) que sean libres y válidos.
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and free[nr, nc]:
                yield nr, nc

    # DFS iterativo con pila: en cada nodo se guarda su lista de vecinos pendientes.
    visited = {start}
    walk = [start]
    stack = [(start, list(neighbors(*start)))]
    while stack:
        cell, nbrs = stack[-1]
        advanced = False
        # Intentar bajar a un vecino no visitado.
        while nbrs:
            nb = nbrs.pop()
            if nb not in visited:
                visited.add(nb)
                walk.append(nb)                  # paso adelante (adyacente)
                stack.append((nb, list(neighbors(*nb))))
                advanced = True
                break
        # Sin vecinos nuevos: retroceder y registrar el paso de backtracking
        # (también es a una celda adyacente, así el robot solo da pasos de celda).
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
    # BFS estándar guardando el predecesor de cada celda para reconstruir ruta.
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in _free_neighbors(free, cur):
            if nb not in prev:
                prev[nb] = cur
                # Al alcanzar un objetivo, reconstruir el camino hacia atrás.
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
    # Precalentar mandando Move(0) para que el gait arranque su cadencia.
    end = time.time() + warmup_s
    while time.time() < end:
        client.Move(0.0, 0.0, 0.0)
        time.sleep(0.05)


def _look_down_check(client, look_for_ball, settle_s=1.0):
    """Agacha al robot (StandDown) para bajar la cámara hacia el suelo, mira con
    la cámara si está la pelota y se vuelve a levantar (StandUp).

    La cámara del Go2 es fija en la cabeza; al agacharse, su punto de vista baja
    y se puede ver mejor un objeto bajo (la pelota) en la casilla de delante.
    Devuelve True si `look_for_ball()` ve la pelota. Es defensivo: si el cliente
    no tiene StandDown/StandUp, simplemente mira desde de pie.
    """
    client.StopMove()
    # Comprobar de forma defensiva si el cliente soporta agacharse/levantarse.
    can_down = hasattr(client, "StandDown")
    can_up = hasattr(client, "StandUp")
    if not can_down:
        print("[AUTO] AVISO: SportClient no expone StandDown(); miro desde de pie")

    # Agacharse para bajar la cámara hacia la casilla de delante.
    if can_down:
        client.StandDown()
        time.sleep(settle_s)        # agacharse y estabilizar

    # Mirar con la cámara si hay verde.
    found = look_for_ball()

    # Volver a levantarse y quedar en BalanceStand.
    if can_down and can_up:
        client.StandUp()
        time.sleep(settle_s)        # levantarse antes de seguir
    client.BalanceStand()
    time.sleep(0.3)
    return found


def autonomous_movement(client, odom, occupancy, vertices, n_div,
                        look_for_ball, locate_ball=None, hit_threshold=5,
                        pause_s=0.4, cell_tolerance=0.07, settle_s=1.0,
                        min_v=0.12):
    """
    Recorre todas las casillas LIBRES de la rejilla buscando la pelota con la
    cámara.

    Estrategia:
      1. Planifica un recorrido 4-conexo que entra en cada casilla libre (DFS
         con backtracking: cada salto es a una casilla adyacente).
      2. Antes de ENTRAR en cada casilla nueva, la encara (rumbo exacto del eje
         de la rejilla -> giro de 90°), baja la cámara con un `pitch` del cuerpo
         y llama a `look_for_ball()` para ver si la pelota está en esa casilla.
      3. Si la ve, se detiene y devuelve esa casilla (su posición en el grid).
         Si no, entra en la casilla y sigue.

    Movimiento centrado en celdas (sin LiDAR todavía, ver más abajo):
      - Cada paso a la casilla adyacente es un avance RELATIVO de exactamente
        un "paso de rejilla" (la distancia entre centros de celda) hacia delante
        en el frame del robot, no una navegación a una coordenada absoluta del
        mundo. Así no arrastramos el error absoluto de la malla re-anclada.
      - Al llegar se hace un CENTRADO FINO: se camina al destino con tolerancia
        pequeña (`cell_tolerance`) y un `min_v` que evita que el robot se quede
        atascado por la banda muerta del gait. Queda bien centrado en la casilla
        antes de continuar.
      Esto asume que los errores por paso son pequeños; cuando se integre el
      LiDAR, la corrección de posición vendrá de ahí.

    `look_for_ball()` -> bool (la cámara ve verde, con el robot ya inclinado).
    Devuelve la casilla (fila, col) de la pelota, o None si no la encuentra.
    """
    # 1) Celdas libres = las que NO superan el umbral de ocupación.
    occupancy = np.asarray(occupancy)
    free = ~(occupancy > hit_threshold)
    centers = _cell_centers_world(vertices, n_div)

    # Rumbos ABSOLUTOS de los ejes de la rejilla: +columna sigue e_s = v1-v0 y
    # +fila sigue e_t = v3-v0. Como la malla está re-anclada al heading del
    # robot, ir entre casillas perpendiculares es un giro exacto de 90°.
    v = np.asarray(vertices, dtype=float)
    e_s, e_t = v[1] - v[0], v[3] - v[0]
    h_col = math.atan2(e_s[1], e_s[0])
    h_row = math.atan2(e_t[1], e_t[0])
    # Paso de rejilla: distancia entre centros de celdas vecinas en cada eje.
    pitch_col = float(np.linalg.norm(e_s)) / n_div
    pitch_row = float(np.linalg.norm(e_t)) / n_div

    # 2) Esperar pose y fijar frame de arranque.
    odom._wait_for_pose()
    odom._initial_frame()

    def cur_cell():
        # Casilla actual del robot según su pose (acotada al rango válido).
        r, c = _world_to_cell(vertices, n_div, odom.t[:2])
        return (int(np.clip(r, 0, n_div - 1)), int(np.clip(c, 0, n_div - 1)))

    def step_to(a, b):
        """Rumbo absoluto y distancia (un paso de celda) para ir de la casilla
        `a` a la `b` adyacente."""
        # Según el desplazamiento de fila/columna, devolver el rumbo del eje
        # correspondiente (y su sentido) y el paso de rejilla de ese eje.
        drow, dcol = b[0] - a[0], b[1] - a[1]
        if dcol == 1:
            return _wrap_pi(h_col), pitch_col
        if dcol == -1:
            return _wrap_pi(h_col + math.pi), pitch_col
        if drow == 1:
            return _wrap_pi(h_row), pitch_row
        return _wrap_pi(h_row + math.pi), pitch_row

    # 3) Casilla inicial y plan de recorrido que cubre todas las libres.
    # La casilla inicial es siempre la (0,0) (el robot arranca en su centro).
    start_cell = (0, 0) if free[0, 0] else _nearest_free(free, centers, odom.t[:2])
    walk = _coverage_walk(free, start_cell)
    print(f"[AUTO] {int(free.sum())} celdas libres; recorrido de {len(walk)} "
          f"pasos desde {start_cell}")

    # 4) Poner al robot en marcha.
    print("[AUTO] BalanceStand + ClassicWalk")
    _ensure_walk(client, warmup_s=1.2)

    # 5) Recorrer el plan paso a paso (cada paso = a una casilla adyacente).
    checked = {start_cell}      # casillas ya miradas con la cámara
    for i in range(1, len(walk)):
        a, b = walk[i - 1], walk[i]
        heading, pitch = step_to(a, b)
        print(f"[AUTO] {a} -> {b}  rumbo={math.degrees(heading):+.0f}°  "
              f"paso={pitch:.2f} m")
        # Encarar la casilla destino (giro de 90° o 0° según el eje).
        _pivot_to_heading_precise(heading, client, odom, tag="paso")

        if b not in checked:
            # Antes de entrar: agacharse (StandDown) y mirar si la pelota está
            # en esa casilla.
            print(f"[AUTO] Mirando la casilla {b} (StandDown)...")
            found = _look_down_check(client, look_for_ball, settle_s=settle_s)
            checked.add(b)
            if found:
                client.StopMove()
                # La cámara ve verde en esta dirección, pero puede estar a 2+
                # casillas: el LiDAR localiza la celda real recorriendo la línea
                # del grid hacia delante. Si no la ve, se queda con la de delante.
                ball = b
                if locate_ball is not None:
                    direction = (b[0] - a[0], b[1] - a[1])
                    loc = locate_ball(a, direction)
                    if loc is not None:
                        ball = loc
                print(f"[AUTO] ¡Pelota detectada en la casilla {ball}!")
                return ball
            _ensure_walk(client)        # tras el StandUp, volver a marcha
            # El StandDown/StandUp puede haber desviado el rumbo: re-encarar.
            _pivot_to_heading_precise(heading, client, odom, tag="reencarar")

        # Navegación ABSOLUTA al centro de la casilla destino. Es auto-correctiva
        # (no acumula deriva): aunque el robot llegue algo desviado, apunta al
        # centro real de la celda, así su casilla real coincide con la del grid
        # (no "cree" estar en la siguiente) y no se queda corto al final.
        wx, wy = centers[b]
        _walk_to(wx, wy, client, odom, tolerance=cell_tolerance, min_v=min_v)
        _hold(client, pause_s)
        # Comprobar en qué casilla quedó realmente (para depurar la deriva).
        here = cur_cell()
        print(f"[AUTO] En casilla {here}" +
              ("" if here == b else f"  (esperaba {b})"))

    # 6) Recorrido agotado sin encontrar la pelota.
    client.StopMove()
    print("[AUTO] Recorrido completado: no se encontró la pelota")
    return None
