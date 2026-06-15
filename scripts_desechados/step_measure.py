import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

def delante(client):
    duracion = 1.0
    t0 = time.time()
    while time.time() - t0 < duracion:
        client.Move(0.3, 0.0, 0.0)
        time.sleep(0.02)
    client.StopMove()



def main():
    # 1. Inicializar la red (igual que en el Lidar)
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    print("=== TEST DE MOVIMIENTO BÁSICO ===")
    
    # 2. Crear e inicializar el cliente de deportes (SportClient)
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    delante(client)

if __name__ == '__main__':
    input("Pulsa Enter cuando estés listo para arrancar...")
    main()