"""
Mapeo manual del entorno mediante un recorrido en cuadrado.

Este es el primer script de la entrega: el robot recorre automáticamente un
cuadrado (patrón `do_square`) mientras el LiDAR va acumulando puntos. Al
terminar, la visualización Open3D recorta la nube a la zona recorrida, construye
una rejilla de ocupación N x N y la guarda en disco como `cellMap_*.npz`. Ese
fichero es el que luego usa `getBall.py` para buscar la pelota.

Arquitectura de hilos:
  - Hilo secundario (daemon): `do_square` -> mueve el robot por el cuadrado.
  - Hilo principal: `visualizator_start` -> dibuja la nube y guarda el mapa.
Se usan dos hilos para que el robot se mueva y se pinte la nube a la vez.

Uso:
    python navegacion_manual.py [interfaz_red]
"""

import sys
import threading  # <-- NUEVO: Necesario para que el robot se mueva y pinte a la vez

from go2_lidar.odom import OdomTracker

from go2_lidar.mapping.accumulator import visualizator_start

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient # <-- NUEVO: Necesario para mover al robot

from go2_lidar.mapping.get_cloudpoint import Custom

from go2_lidar.control.patterns import do_square


# Tópico DDS del LiDAR ya "desalabeado" (deskewed): nube corregida del barrido.
TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

class Custom2(Custom):
    """Extiende la clase Custom (lectura de nube LiDAR) con los campos de estado
    que necesita la sesión de mapeo:
      - end: lo pone a True `do_square` al terminar -> corta el bucle del visor.
      - side / stops_per_side: nº de paradas por lado del cuadrado (define el
        tamaño N de la rejilla de ocupación = stops_per_side + 1)."""
    def __init__(self, topic="rt/utlidar/cloud"):
        super().__init__(topic)
        self.end = False                 # bandera de "recorrido terminado"
        self.side = 0
        self.stops_per_side = 1          # se sobrescribe en main()
def main():
    """Arranca DDS, lanza el recorrido en cuadrado en un hilo y la visualización
    (que además guarda el mapa) en el hilo principal."""
    # 1) Inicializar la red DDS de Unitree: con argumento -> interfaz concreta;
    #    sin argumento -> autodetección.
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 2) Lector de la nube LiDAR + rastreador de odometría (pose del robot).
    custom = Custom2(TOPIC_CLOUD)
    num_stops = 2
    custom.stops_per_side = num_stops    # rejilla resultante será 3x3 (num_stops+1)
    odom = OdomTracker()

    # --- NUEVO: Inicializar cliente de movimiento y lanzar hilo ---
    # 3) Cliente de locomoción del Go2 (SportClient): comandos de marcha/giro.
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    # Cuadrado: 4 paradas por lado (incluida la esquina) cada 0.65 m
    # -> lado = 2.60 m, sentido horario (delante, derecha, atrás, izquierda)
    # 4) Lanzar el recorrido en cuadrado en un hilo aparte (daemon, para que
    #    muera con el programa). Caminará el cuadrado mientras el visor pinta.
    nav_thread = threading.Thread(
        target=do_square,
        kwargs=dict(client=client, odom=odom, lidar=custom,
                    step=0.60, stops_per_side=num_stops, pause_s=1.0,
                    clockwise=True),
        daemon=True
    )
    nav_thread.start()
    # ---------------------------------------------------------------
    # 5) Bucle de visualización en el hilo principal: acumula la nube, y al
    #    detectar custom.end=True recorta la zona, construye y guarda el cellMap.
    visualizator_start(odom, custom)


if __name__ == "__main__":
    # Aviso de seguridad: el robot se va a mover solo, hay que despejar la zona.
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    main()
