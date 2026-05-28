import math
import time
from go2_lidar.transforms import _robot_to_world, get_yaw_from_rot, _wrap_pi


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
            print(f"ESTOY ENTRANDO")
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



def go_to_waypoint(target_x_rel, target_y_rel, client, odom, tolerance=0.1):
    """
    Navega hacia una coordenada (X, Y) expresada en el frame del robot
    en el instante de arranque: +X hacia delante, +Y hacia la izquierda.
    """
    odom._wait_for_pose()
    x0, y0, yaw0 = odom._initial_frame()
    target_x, target_y = _robot_to_world(target_x_rel, target_y_rel, x0, y0, yaw0)

    print(f"[THREAD] Objetivo relativo ({target_x_rel:+.2f}, {target_y_rel:+.2f}) "
          f"-> mundo ({target_x:+.2f}, {target_y:+.2f})")

    _go_to_world_xy(target_x, target_y, client, odom, tolerance=tolerance)



def _pivot_to_heading_precise(target_yaw_world, client, odom, tag="corner",
                              tolerances=(math.radians(5.0), math.radians(2.0), math.radians(1.0)),
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