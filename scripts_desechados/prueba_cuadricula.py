import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from obtencion_nube_puntos import Custom

def giro(client):
    client.Move(0.0, 0.0, -(np.pi / 2))
    time.sleep(2.0)
    client.StopMove()

def delante(client):
    duracion = 3.0
    t0 = time.time()
    while time.time() - t0 < duracion:
        client.Move(0.3, 0.0, 0.0)
        time.sleep(0.2)
    client.StopMove()

def comprobacion():
    return 0

def exploration(size, lista, client, lidar):
    x = 0
    y = 0
    giro_var = 0

    while True:
        if y == 0 and x == 0 and giro_var == 3:
            giro(client)
            giro_var += 1
            break
        elif y < size-1 and x == 0:
            # comprobación
            lista[x,y+1] = comprobacion()
            delante(client)
            y += 1
            time.sleep(1.0)
        elif y == size-1 and x == 0 and giro_var == 0:
            giro(client)
            giro_var += 1
        elif y == size-1 and x < size-1:
            lista[x+1,y] = comprobacion()
            delante(client)
            x += 1
            time.sleep(1.0)
        elif y == size-1 and x == size-1 and giro_var == 1:
            giro(client)
            giro_var += 1
        elif y > 0 and x == size-1:
            lista[x, y-1] = comprobacion()
            delante(client)
            y -= 1
            time.sleep(1.0)
        elif y == 0 and x == size-1 and giro_var == 2:
            giro(client)
            giro_var += 1
        elif y == 0 and x > 0:
            lista[x-1,y] = comprobacion()
            delante(client)
            x -= 1
            time.sleep(1.0)

def main():
    # 1. Inicializar la red (igual que en el Lidar)
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    print("=== TEST DE MOVIMIENTO BÁSICO ===")
    TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

    lidar = Custom(TOPIC_CLOUD)
    
    # 2. Crear e inicializar el cliente de deportes (SportClient)
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    size = 4
    lista = np.zeros((size, size))

    exploration(size, lista, client, lidar)

    print('EXPLORATION FINISHED')


if __name__ == '__main__':
    input("Pulsa Enter cuando estés listo para arrancar...")
    main()

    '''
    # La función Move recibe 3 parámetros:
    # Move(vx, vy, vyaw)
    #   vx   : Velocidad hacia adelante/atrás en m/s (positivo es adelante)
    #   vy   : Velocidad lateral en m/s (positivo es izquierda)
    #   vyaw : Velocidad de giro en rad/s (positivo es giro a la izquierda)
    #client.Move(0.65, 0.0, 0.0) # aprox 65 cm
    client.Move(0.0, 0.0, -(np.pi / 2))
    # 4. Mantener la orden durante 1 segundo
    time.sleep(1.0)

    client.Move(0.0, 0.0, 0.0)

    time.sleep(1.0)

    # 5. Detener el robot
    #print("Deteniendo...")
    client.Move(0.0, 0.0, (np.pi / 2))
    #client.Move(-0.7, 0.0, 0.0)

    #time.sleep(1.0)

    #client.Move(0.0, 0.0, 0.0)
    
    # Un pequeño sleep para asegurar que el comando de parada se envía por la red
    # antes de que Python cierre el script.
    #time.sleep(0.5) 
    #print("Fin del test.")'''