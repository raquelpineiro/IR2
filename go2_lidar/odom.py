"""Rastreo de la odometría del robot Go2.

Define `OdomTracker`, que se suscribe al tópico de pose del robot y mantiene en
todo momento su última posición (t) y orientación (R), además de utilidades para
esperar la primera pose y capturar el frame de arranque."""

import numpy as np
from .transforms import quat_to_rot
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
import time
from .transforms import get_yaw_from_rot

class OdomTracker:
    """Se suscribe al tópico de pose y guarda la última pose recibida del robot:
    R (matriz de rotación 3x3) y t (vector de traslación). `has_pose` indica si
    ya ha llegado al menos una pose."""
    def __init__(self, topic="rt/utlidar/robot_pose"):
        self.subscriber = ChannelSubscriber(topic, PoseStamped_)
        self.subscriber.Init()
        self.R = np.eye(3, dtype=np.float64)    # rotación actual (mundo<-robot)
        self.t = np.zeros(3, dtype=np.float64)  # posición actual en el mundo
        self.has_pose = False                   # ¿se recibió ya alguna pose?

    def update(self):
        """Lee el último mensaje de pose (no bloqueante). Si llega uno nuevo,
        actualiza t y R (convirtiendo el cuaternión a matriz) y devuelve True;
        si no hay mensaje, devuelve False sin tocar el estado."""
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
        """Matriz homogénea 4x4 (rotación + traslación) de la pose actual."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def _wait_for_pose(self):
        """Bloquea hasta que se reciba la primera pose. OJO: solo espera; asume
        que algún otro hilo está llamando a update() para refrescar el estado."""
        while not self.has_pose:
            time.sleep(0.05)


    def _initial_frame(self):
        """Captura el frame inicial del robot a partir de la odometría."""
        # Posición XY de arranque y yaw (orientación en el plano) iniciales.
        x0 = self.t[0]
        y0 = self.t[1]
        print(f"HEIGHT AT THE BEGINNING: {self.t[2]}")
        yaw0 = get_yaw_from_rot(self.R)
        return x0, y0, yaw0
