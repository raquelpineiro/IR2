import numpy as np
import math
import threading


class OccupancyGrid:
    """
    Rejilla N x N binaria de ocupación (N = stops_per_side), en el frame
    relativo al inicio del cuadrado. Cada celda mide `step` x `step` metros.

    Convención: +X = delante, +Y = izquierda.
      - antihorario: cuadrado en rx ∈ [0, L], ry ∈ [0, L]
      - horario:     cuadrado en rx ∈ [0, L], ry ∈ [-L, 0]
    """

    def __init__(self, step, stops_per_side, clockwise, x0, y0, yaw0,
                 z_min=0.25, z_max=0.5, hit_threshold=5):
        self.step = step
        self.n = stops_per_side + 1
        self.side_len = step * (stops_per_side + 1)
        self.x_range = (0.0, self.side_len / (stops_per_side + 1))
        self.y_range = (-self.side_len / (stops_per_side + 1), 0.0) if clockwise else (0.0, self.side_len / (stops_per_side + 1))
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.z_min, self.z_max = z_min, z_max
        self.hit_threshold = hit_threshold
        self.counts = np.zeros((self.n, self.n), dtype=np.int64)
        self.n_captures = 0
        self._lock = threading.Lock()

    def capture(self, lidar, odom):
        data = lidar.get_cloud()
        if data is None or len(data["xyz"]) == 0:
            return 0

        xyz_lidar = data["xyz"].astype(np.float64, copy=False)
        xyz_world = xyz_lidar @ odom.R.T + odom.t

        # mundo -> frame inicial del cuadrado
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        dx = xyz_world[:, 0] - self.x0
        dy = xyz_world[:, 1] - self.y0
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        rz = xyz_world[:, 2]

        mask = (
            (rz > self.z_min) & (rz < self.z_max)
            & (rx >= self.x_range[0]) & (rx < self.x_range[1])
            & (ry >= self.y_range[0]) & (ry < self.y_range[1])
        )

        with self._lock:
            self.n_captures += 1
            if not mask.any():
                return 0
            hist, _, _ = np.histogram2d(
                rx[mask], ry[mask],
                bins=[self.n, self.n],
                range=[list(self.x_range), list(self.y_range)],
            )
            self.counts += hist.astype(np.int64)
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
