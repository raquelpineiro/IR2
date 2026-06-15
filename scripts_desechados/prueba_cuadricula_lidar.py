import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from obtencion_nube_puntos import Custom

# --- PARÁMETROS DE SEGUIMIENTO DE PARED ---
TARGET_DIST = 0.60       # Distancia deseada a la pared izquierda en metros
MAX_DIST_BUSQUEDA = 1.5  # Si la pared está a más de 1.5m, la ignoramos
KP_ANGULO = 1.2          # Fuerza con la que corrige el ángulo
KP_DISTANCIA = 0.8       # Fuerza con la que corrige la distancia
WZ_MAX = 0.4             # Velocidad máxima de giro correctivo (rad/s)

def calcular_correccion_pared_izquierda(xyz):
    """
    Analiza la nube de puntos para encontrar la pared izquierda
    y devuelve la velocidad angular (wz) para mantenerse paralelo.
    """
    # 1. Filtrar por altura (ignorar suelo y techo)
    z = xyz[:, 2]
    band = (z > -0.10) & (z < 0.50)
    pts = xyz[band, :2]

    if len(pts) == 0:
        return 0.0  # Si no hay puntos, no corregimos (vamos recto)

    # Pasar a coordenadas polares (distancia y ángulo)
    r = np.hypot(pts[:, 0], pts[:, 1])
    a = np.arctan2(pts[:, 1], pts[:, 0])

    # 2. Quedarnos solo con el lado izquierdo (entre 45° y 135°)
    # En radianes: np.pi/4 (45°) a 3*np.pi/4 (135°)
    min_angle = np.pi / 4
    max_angle = 3 * np.pi / 4

    keep = (a > min_angle) & (a < max_angle) & (r < MAX_DIST_BUSQUEDA)
    pts_left = pts[keep]
    r_left = r[keep]
    a_left = a[keep]

    # Si hay muy pocos puntos, asumimos que no hay pared clara
    if len(r_left) < 10:
        return 0.0

    # 3. Buscar el punto más cercano en ese sector
    idx_min = np.argmin(r_left)
    min_r = r_left[idx_min]
    min_a = a_left[idx_min]

    # 4. Control PD para la corrección
    # Ángulo ideal de la pared es exactamente a la izquierda (90 grados o np.pi/2)
    error_angulo = min_a - (np.pi / 2)
    error_distancia = TARGET_DIST - min_r

    # Calcular la corrección de giro (wz)
    wz_correccion = (KP_ANGULO * error_angulo) + (KP_DISTANCIA * error_distancia)

    # Limitar el giro para que no haga movimientos bruscos
    return float(np.clip(wz_correccion, -WZ_MAX, WZ_MAX))

def giro(client):
    client.Move(0.0, 0.0, -(np.pi / 2))
    time.sleep(2.0)
    client.StopMove()

def delante_con_pared(client, lidar):
    """
    Se mueve hacia adelante aplicando correcciones en base al LiDAR.
    Sustituye a la antigua función delante() ciega.
    """
    duracion = 3.0
    t0 = time.time()
    
    while time.time() - t0 < duracion:
        wz = 0.0
        data = lidar.get_cloud()
        
        # Si tenemos datos, calculamos la corrección
        if data is not None and len(data["xyz"]) > 0:
            wz = calcular_correccion_pared_izquierda(data["xyz"])
        
        # Avanzar con velocidad constante (0.3) y la corrección de giro (wz)
        client.Move(0.3, 0.0, wz)
        
        # Dormimos un poco para simular un lazo de control a ~10Hz
        time.sleep(0.1) 
        
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
            lista[x,y+1] = comprobacion()
            delante_con_pared(client, lidar) # Usamos la nueva función
            y += 1
            time.sleep(1.0)
        elif y == size-1 and x == 0 and giro_var == 0:
            giro(client)
            giro_var += 1
        elif y == size-1 and x < size-1:
            lista[x+1,y] = comprobacion()
            delante_con_pared(client, lidar)
            x += 1
            time.sleep(1.0)
        elif y == size-1 and x == size-1 and giro_var == 1:
            giro(client)
            giro_var += 1
        elif y > 0 and x == size-1:
            lista[x, y-1] = comprobacion()
            delante_con_pared(client, lidar)
            y -= 1
            time.sleep(1.0)
        elif y == 0 and x == size-1 and giro_var == 2:
            giro(client)
            giro_var += 1
        elif y == 0 and x > 0:
            lista[x-1,y] = comprobacion()
            delante_con_pared(client, lidar)
            x -= 1
            time.sleep(1.0)

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    print("=== NAVEGACIÓN EN CUADRÍCULA CON SEGUIMIENTO DE PARED ===")
    TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

    lidar = Custom(TOPIC_CLOUD)
    
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    size = 4
    lista = np.zeros((size, size))

    exploration(size, lista, client, lidar)

    print('EXPLORACIÓN TERMINADA')

if __name__ == '__main__':
    print("  AVISO: Asegúrate de colocar el robot con una pared a su IZQUIERDA")
    input("Pulsa Enter cuando estés listo para arrancar...")
    main()
