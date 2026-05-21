import numpy as np
from .transforms import quat_to_rot
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
import time
from .transforms import get_yaw_from_rot

class OdomTracker:
    def __init__(self, topic="rt/utlidar/robot_pose"):
        self.subscriber = ChannelSubscriber(topic, PoseStamped_)
        self.subscriber.Init()
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros(3, dtype=np.float64)
        self.has_pose = False

    def update(self):
        msg = self.subscriber.Read()
        if msg is None:
            return False
        p = msg.pose.position
        q = msg.pose.orientation
        self.t = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.R = quat_to_rot(q.x, q.y, q.z, q.w)
        self.has_pose = True
        return True

    @property
    def T(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T
    
    def _wait_for_pose(self):
        while not self.has_pose:
            time.sleep(0.05)


    def _initial_frame(self):
        """Captura el frame inicial del robot a partir de la odometría."""
        x0 = self.t[0]
        y0 = self.t[1]
        print(f"HEIGHT AT THE BEGINNING: {self.t[2]}")
        yaw0 = get_yaw_from_rot(self.R)
        return x0, y0, yaw0