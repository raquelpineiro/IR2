import time
import math

from go2_lidar.transforms import _robot_to_world, _wrap_pi

from go2_lidar.control.primitives import _pivot_to_heading_precise, _walk_to

from go2_lidar.mapping.occupancy import OccupancyGrid

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

    grid = None
    if lidar is not None:
        grid = OccupancyGrid(
            step=step, stops_per_side=stops_per_side, clockwise=clockwise,
            x0=x0, y0=y0, yaw0=yaw0, hit_threshold=hit_threshold,
        )

    lidar.occupancy = grid

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
            grid.x_range = (0.0, step)
            grid.y_range = (-step/2, step/2)
        elif side_num == 2:
            base_x, base_y = step*(stops_per_side) - 0.2, -0.2
            grid.x_range = (-step/2, step/2)
            grid.y_range = (-step, 0.0)
        elif side_num == 3:
            base_x, base_y = step*(stops_per_side) - 0.4, -step*(stops_per_side)
            grid.x_range = (-step, 0.0)
            grid.y_range = (-step/2, step/2) 
        elif side_num == 4:
            base_x, base_y = -0.20, -step*(stops_per_side) + 0.2
            grid.x_range = (-step/2, step/2)
            grid.y_range = (0.0, step)
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

            # Captura para el mapa de ocupación (robot detenido)
            if grid is not None:
                if side_num == 1:
                    coords = (stops_per_side - i - 1, 0)
                elif side_num == 2:
                    coords = (0, i - 1)
                elif side_num == 3:
                    coords = (i - 1, stops_per_side)
                elif side_num == 4:
                    coords = (stops_per_side, stops_per_side - i - 1)
                hits = grid.capture(lidar, odom, i, coords)
                print(f"  [GRID] parada {wp_idx}/{total_wp}: {hits} hits dentro del cuadrado")
        #base_x += dir_x * side_len
        #base_y += dir_y * side_len

    client.StopMove()
    lidar.end = True
    print("[SQUARE] Cuadrado completado")

    if grid is not None:
        grid.print_map()
        grid.save("mapa_ocupacion")
        print("[GRID] guardado en mapa_ocupacion.npz")