"""Utilidades geométricas comunes.

Conversión de cuaterniones a matriz de rotación, extracción del yaw, normalizado
de ángulos y transformaciones de coordenadas entre el frame del mundo y el frame
de arranque del robot (origen en (x0, y0), eje +X según yaw0)."""

import numpy as np
import math

def quat_to_rot(x, y, z, w):
    """Convierte un cuaternión (x, y, z, w) en su matriz de rotación 3x3.

    La pose del robot llega como cuaternión; los cálculos geométricos y Open3D
    trabajan con matrices, de ahí la conversión."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),         2.0 * (xz + wy)],
        [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),         2.0 * (yz - wx)],
        [      2.0 * (xz - wy),       2.0 * (yz + wx),   1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def get_yaw_from_rot(R):
    """Extrae el ángulo Yaw (rotación en Z) de la matriz de rotación 3x3 de la odometría."""
    # atan2 de los elementos de la primera columna -> ángulo en el plano XY.
    return math.atan2(R[1, 0], R[0, 0])


def _wrap_pi(a):
    """Normaliza un ángulo al rango (-pi, pi]. Evita saltos al comparar rumbos."""
    return math.atan2(math.sin(a), math.cos(a))


def _robot_to_world(rx, ry, x0, y0, yaw0):
    """Transforma (rx, ry) del frame del robot inicial al frame del mundo."""
    # Rotar por yaw0 y trasladar al origen de arranque (x0, y0).
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    return x0 + c0 * rx - s0 * ry, y0 + s0 * rx + c0 * ry


def _world_to_robot(wx, wy, x0, y0, yaw0):
    """Inversa de _robot_to_world: lleva (wx, wy) del mundo al frame inicial
    del robot (origen en (x0, y0), eje +X según yaw0)."""
    # Trasladar al origen de arranque y rotar por -yaw0 (rotación transpuesta).
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    dx, dy = wx - x0, wy - y0
    return c0 * dx + s0 * dy, -s0 * dx + c0 * dy
