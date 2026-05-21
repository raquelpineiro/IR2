import sys
import threading  # <-- NUEVO: Necesario para que el robot se mueva y pinte a la vez

from go2_lidar.odom import OdomTracker

from go2_lidar.mapping.accumulator import visualizator_start

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient # <-- NUEVO: Necesario para mover al robot

from go2_lidar.mapping.get_cloudpoint import Custom

from go2_lidar.control.patterns import do_square


TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(TOPIC_CLOUD)
    odom = OdomTracker()

    # --- NUEVO: Inicializar cliente de movimiento y lanzar hilo ---
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    # Cuadrado: 4 paradas por lado (incluida la esquina) cada 0.65 m
    # -> lado = 2.60 m, sentido horario (delante, derecha, atrás, izquierda)
    nav_thread = threading.Thread(
        target=do_square,
        kwargs=dict(client=client, odom=odom, lidar=custom,
                    step=0.60, stops_per_side=3, pause_s=1.0,
                    clockwise=True),
        daemon=True
    )
    nav_thread.start()
    # ---------------------------------------------------------------
    visualizator_start(odom, custom)


if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    main()