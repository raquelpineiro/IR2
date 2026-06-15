"""Paquete go2_lidar: utilidades de odometría, geometría, control de locomoción
y mapeo LiDAR para el robot Unitree Go2.

Re-exporta lo más usado para poder importarlo directamente como
`from go2_lidar import OdomTracker, transforms`."""

from .odom import OdomTracker          # rastreador de pose (posición + orientación)
from . import transforms               # funciones geométricas (cuaterniones, frames)


__version__ = "0.1.0"

# Símbolos públicos del paquete (lo que expone `from go2_lidar import *`).
__all__ = ["OdomTracker",
           "transforms"
           ]