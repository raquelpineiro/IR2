import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

import numpy as np

# rt/utlidar/cloud  (obtener nube de puntos)
# utlidar _lidar   (coordenar la nube)
# rt/utlidar/cloud desckewed   (eliminar movimiento)
# odom
# rt/portmodestate   (transformar en timestamp)

# source ../../../../go2/bin/activate

class Custom:
    def __init__(self):
        # create publisher #
        self.publisher = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
        self.publisher.Init() 

    def go2_utlidar_switch(self):
        nube = self.publisher.Read()
        
        num_div = int(np.ceil(len(nube.data) / 32))
        puntos = np.array_split(nube.data, num_div)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.go2_utlidar_switch()
