import sys
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

import open3d as o3d

# source ../../../../go2/bin/activate

# ROS sensor_msgs/PointField.datatype  ->  numpy dtype
_PF_TO_NP = {
    1: np.int8,   2: np.uint8,
    3: np.int16,  4: np.uint16,
    5: np.int32,  6: np.uint32,
    7: np.float32, 8: np.float64,
}

class Custom:
    def __init__(self, topic="rt/utlidar/cloud"):
        self.subscriber = ChannelSubscriber(topic, PointCloud2_)
        self.subscriber.Init()
        self._dtype = None
        self._point_step = None

    @staticmethod
    def _build_dtype(fields, point_step):
        names, formats, offsets = [], [], []
        for f in fields:
            np_type = _PF_TO_NP[f.datatype]
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
        msg = self.subscriber.Read()
        if msg is None:
            return None

        if self._dtype is None or self._point_step != msg.point_step:
            self._dtype = self._build_dtype(msg.fields, msg.point_step)
            self._point_step = msg.point_step

        n = msg.width * msg.height
        raw = np.asarray(msg.data, dtype=np.uint8)
        cloud = raw[: n * msg.point_step].view(self._dtype)

        xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))

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
