import numpy as np
import math

def quat_to_rot(x, y, z, w):
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
    return math.atan2(R[1, 0], R[0, 0])


def _wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def _robot_to_world(rx, ry, x0, y0, yaw0):
    """Transforma (rx, ry) del frame del robot inicial al frame del mundo."""
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    return x0 + c0 * rx - s0 * ry, y0 + s0 * rx + c0 * ry