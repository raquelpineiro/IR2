import sys
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from obtencion_nube_puntos import Custom


TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

# Geometría / banda de altura del cuerpo (frame del LiDAR, z=0 ~ altura del LiDAR)
ROBOT_Z_MIN = 0          # ignora suelo y patas
ROBOT_Z_MAX = 0.50           # ignora techo

# Zona de búsqueda en planta (XY)
LOOK_AHEAD = 2.5             # m hasta donde miramos
SELF_RADIUS = 0.20           # m: ignora puntos del propio cuerpo
FOV_DEG = 160.0              # arco frontal usado para decidir
N_SECTORS = 9                # bines angulares dentro del FOV

# Distancias críticas
STOP_DIST = 2             # m: por debajo, parar y reorientarse
SLOW_DIST = 1.5             # m: por debajo, frenar linealmente

# Velocidades (conservador)
V_MAX = 0.30                 # m/s lineal
W_MAX = 0.60                 # rad/s angular
W_TURN_IN_PLACE = 0.50       # rad/s al girar parado

# Lazo
CONTROL_HZ = 10.0
DATA_TIMEOUT = 0.4           # s sin LiDAR -> watchdog
PRINT_EVERY = 2              # imprimir 1 de cada N ciclos


def sector_min_distances(xyz):
    z = xyz[:, 2]
    band = (z > ROBOT_Z_MIN) & (z < ROBOT_Z_MAX)
    pts = xyz[band, :2]
    if len(pts) == 0:
        return None, None

    r = np.hypot(pts[:, 0], pts[:, 1])
    a = np.arctan2(pts[:, 1], pts[:, 0])

    half_fov = np.deg2rad(FOV_DEG) * 0.5
    keep = (r > SELF_RADIUS) & (r < LOOK_AHEAD) & (np.abs(a) < half_fov)
    if not keep.any():
        edges = np.linspace(-half_fov, half_fov, N_SECTORS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.full(N_SECTORS, LOOK_AHEAD), centers

    r, a = r[keep], a[keep]
    edges = np.linspace(-half_fov, half_fov, N_SECTORS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, N_SECTORS - 1)
    distances = np.full(N_SECTORS, LOOK_AHEAD)
    np.minimum.at(distances, bin_idx, r)
    return distances, centers


def decide_command(distances, centers):
    if distances is None:
        return V_MAX * 0.5, 0.0, "sin puntos en banda"

    front = N_SECTORS // 2
    central = distances[max(0, front - 1): front + 2]
    front_min = central.min()

    if front_min > STOP_DIST:
        best = int(np.argmax(distances))
        if distances[best] > SLOW_DIST:
            target = centers[best]
            return 0.0, np.sign(target) * W_TURN_IN_PLACE, f"OBST {front_min:.2f}m → giro {np.rad2deg(target):+.0f}°"
        return 0.0, W_TURN_IN_PLACE, f"RODEADO {front_min:.2f}m, busco salida"

    speed = V_MAX * np.clip(
        (front_min - STOP_DIST) / (SLOW_DIST - STOP_DIST), 0.0, 1.0
    )

    weights = np.maximum(distances - STOP_DIST, 0.0)
    if weights.sum() > 1e-3:
        target_angle = float(np.average(centers, weights=weights))
    else:
        target_angle = 0.0
    yaw = float(np.clip(1.5 * target_angle, -W_MAX, W_MAX))

    return speed, yaw, f"d_front={front_min:.2f}  v={speed:+.2f}  w={yaw:+.2f}"


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    lidar = Custom(TOPIC_CLOUD)

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    print("[init] activando BalanceStand...")
    try:
        sport.BalanceStand()
    except Exception as e:
        print(f"[aviso] BalanceStand falló: {e}. Asegúrate de que el robot está de pie.")
    time.sleep(1.0)

    period = 1.0 / CONTROL_HZ
    last_data_t = time.monotonic()
    tick = 0

    print("[run] iniciando navegación. Ctrl-C para parar.")
    try:
        while True:
            t0 = time.monotonic()
            data = lidar.get_cloud()

            if data is None or len(data["xyz"]) == 0:
                if t0 - last_data_t > DATA_TIMEOUT:
                    sport.StopMove()
                    print("\r[watchdog] sin LiDAR → STOP" + " " * 30, end="", flush=True)
                time.sleep(0.02)
                continue

            last_data_t = t0
            distances, centers = sector_min_distances(data["xyz"])
            vx, wz, msg = decide_command(distances, centers)
            sport.Move(vx, 0.0, wz)

            tick += 1
            if tick % PRINT_EVERY == 0:
                print(f"\r{msg:<60s}", end="", flush=True)

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[interrumpido] usuario")
    finally:
        try:
            sport.StopMove()
        except Exception:
            pass
        time.sleep(0.2)
        print("[fin] robot detenido.")


if __name__ == "__main__":
    print("=" * 64)
    print("  AVISO: el robot se va a mover de forma autónoma.")
    print("  - Asegúrate de que está en un espacio AMPLIO y DESPEJADO.")
    print("  - Para la PRIMERA prueba, ten el mando o suspéndelo.")
    print("  - Ctrl-C detiene el robot inmediatamente.")
    print("=" * 64)
    input("Pulsa Enter para confirmar y arrancar...")
    main()
