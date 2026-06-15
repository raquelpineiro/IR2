"""Lectura de la nube de puntos del LiDAR del Go2.

Define `Custom`, que se suscribe a un tópico PointCloud2 de DDS y entrega cada
escaneo como un array (N, 3) de coordenadas XYZ, interpretando dinámicamente el
formato binario del mensaje a partir de sus campos."""

import sys
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

import open3d as o3d

# source ../../../../go2/bin/activate

# ROS sensor_msgs/PointField.datatype  ->  numpy dtype
# Mapea el código de tipo de cada campo del mensaje a su dtype de numpy.
_PF_TO_NP = {
    1: np.int8,   2: np.uint8,
    3: np.int16,  4: np.uint16,
    5: np.int32,  6: np.uint32,
    7: np.float32, 8: np.float64,
}

class Custom:
    """Suscriptor de la nube de puntos LiDAR. Construye, la primera vez, el dtype
    estructurado que describe cada punto (según los campos del mensaje) y lo
    reutiliza para interpretar el buffer binario de los siguientes escaneos."""
    def __init__(self, topic="rt/utlidar/cloud"):
        self.subscriber = ChannelSubscriber(topic, PointCloud2_)
        self.subscriber.Init()
        self._dtype = None          # dtype estructurado de un punto (cacheado)
        self._point_step = None     # bytes por punto (para detectar cambios)

    @staticmethod
    def _build_dtype(fields, point_step):
        """Construye un dtype estructurado de numpy a partir de los campos del
        mensaje (nombre, tipo y offset de cada uno), con el tamaño total
        `point_step` por punto. Permite leer el buffer crudo sin copiar."""
        names, formats, offsets = [], [], []
        for f in fields:
            np_type = _PF_TO_NP[f.datatype]
            # count>1 -> campo vectorial (p.ej. un array dentro del punto).
            fmt = np.dtype(np_type) if f.count == 1 else np.dtype((np_type, f.count))
            names.append(f.name)
            formats.append(fmt)
            offsets.append(f.offset)
        return np.dtype({
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        })

    def get_cloud(self):
        """Lee el último mensaje de nube (no bloqueante). Devuelve None si no hay;
        si lo hay, interpreta el buffer y devuelve un dict con:
          - xyz: array (N, 3) de coordenadas,
          - cloud: array estructurado con todos los campos,
          - frame_id y stamp de la cabecera.
        Si la nube no es densa, filtra los puntos no finitos (NaN/inf)."""
        msg = self.subscriber.Read()
        if msg is None:
            return None

        # (Re)construir el dtype si es el primer mensaje o cambió el formato.
        if self._dtype is None or self._point_step != msg.point_step:
            self._dtype = self._build_dtype(msg.fields, msg.point_step)
            self._point_step = msg.point_step

        # Interpretar el buffer crudo como array estructurado de N puntos.
        n = msg.width * msg.height
        raw = np.asarray(msg.data, dtype=np.uint8)
        cloud = raw[: n * msg.point_step].view(self._dtype)

        # Extraer las coordenadas XYZ a un array (N, 3).
        xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))

        # Nube no densa -> descartar puntos con NaN/inf.
        if not msg.is_dense:
            mask = np.isfinite(xyz).all(axis=1)
            cloud = cloud[mask]
            xyz = xyz[mask]

        return {
            "xyz": xyz,
            "cloud": cloud,
            "frame_id": msg.header.frame_id,
            "stamp": msg.header.stamp,
        }
