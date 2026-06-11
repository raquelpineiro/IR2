import numpy as np
import math
import threading


"""

LA X Y LA Y ESTAN INTERCAMBIADAS, HAY QUE CAMBIAR LOS RANGOS


"""


class OccupancyGrid:
    """
    Rejilla N x N binaria de ocupación (N = stops_per_side), en el frame
    relativo al inicio del cuadrado. Cada celda mide `step` x `step` metros.

    Convención: +X = delante, +Y = izquierda.
      - antihorario: cuadrado en rx ∈ [0, L], ry ∈ [0, L]
      - horario:     cuadrado en rx ∈ [0, L], ry ∈ [-L, 0]
    """

    def __init__(self, step, stops_per_side, clockwise, x0, y0, yaw0,
                 z_min=0.1, z_max=0.5, hit_threshold=5):
        self.step = step
        self.n = stops_per_side + 1
        self.side_len = step * (stops_per_side + 1)
        self.x_range = (0.0, self.step)
        self.y_range = (-self.step/2, self.step/2) 
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.z_min, self.z_max = z_min, z_max
        self.hit_threshold = hit_threshold
        self.counts = np.zeros((self.n, self.n), dtype=np.int64)
        self.n_captures = 0
        self._lock = threading.Lock()

    def capture(self, lidar, odom, current_stop, coords):
        data = lidar.get_cloud()
        if data is None or len(data["xyz"]) == 0:
            return 0

        xyz_lidar = data["xyz"].astype(np.float64, copy=False)
        xyz_rob = (xyz_lidar - odom.t) @ odom.R
        x_rob, y_rob, z_rob = xyz_rob[:, 0], xyz_rob[:,1], xyz_rob[:, 2]

        x_lidar, y_lidar, z_lidar = xyz_lidar[:, 0], xyz_lidar[:,1], xyz_lidar[:, 2]
        if current_stop == self.n:
            return 0
        else:
            x_range = (0, 5)
            y_range = (-5, 0)
            z_min = 0.05
            z_max = 0.8
            mask = (
                (z_lidar > z_min) & (z_lidar < z_max)
                & (x_rob >= x_range[0]) & (x_rob < x_range[1]) & (y_rob >= y_range[0]) & 
                (y_rob < y_range[1]))
            
            lidar.occupancy = xyz_lidar[mask]


            with self._lock:
                self.n_captures += 1
                print(f"[MASK VALUE] {xyz_lidar[mask]}")
                if not mask.any():
                    return 0
                self.counts[coords[0], coords[1]] += len(xyz_lidar[mask])
                return int(mask.sum())

    def binary(self):
        with self._lock:
            return self.counts >= self.hit_threshold

    def print_map(self):
        binmap = self.binary()
        with self._lock:
            counts = self.counts.copy()
            ncap = self.n_captures

        print(f"\n[GRID] Mapa {self.n}x{self.n}  "
              f"(celda {self.step:.2f} m, hits>={self.hit_threshold}, "
              f"capturas={ncap})    +X=arriba   +Y=izquierda")
        for ix in reversed(range(self.n)):
            chars = ["#" if binmap[ix, iy] else "." for iy in reversed(range(self.n))]
            print("    " + "  ".join(chars))

        print("\n[GRID] Hits por celda:")
        for ix in reversed(range(self.n)):
            row = [f"{counts[ix, iy]:5d}" for iy in reversed(range(self.n))]
            print("    " + " ".join(row))

    def save(self, path):
        with self._lock:
            np.savez(
                path,
                counts=self.counts,
                binary=(self.counts >= self.hit_threshold).astype(np.uint8),
                step=self.step,
                stops_per_side=self.n,
                x_range=np.array(self.x_range),
                y_range=np.array(self.y_range),
            )
