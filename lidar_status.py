import sys
import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

# Recuerda cambiar esto entre "cloud" y "cloud_deskewed" según lo que haya funcionado
TOPIC_CLOUD = "rt/utlidar/cloud" 

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    print(f"=== INICIANDO DIAGNÓSTICO DE LIDAR ===")
    print(f"Escuchando en el topic : {TOPIC_CLOUD}")
    print("Pulsa Ctrl+C para salir.\n")

    subscriber = ChannelSubscriber(TOPIC_CLOUD, PointCloud2_)
    subscriber.Init()

    msg_count = 0
    empty_count = 0
    last_time = time.time()

    try:
        while True:
            # Lectura cruda. Al no haber interfaz gráfica, el bucle girará 
            # a la máxima velocidad posible sin saturar la cola DDS.
            msg = subscriber.Read()
            
            if msg is not None:
                msg_count += 1
                puntos = msg.width * msg.height
                
                if puntos == 0:
                    empty_count += 1

                current_time = time.time()
                elapsed = current_time - last_time

                # Imprimir un resumen cada 1 segundo exacto
                if elapsed >= 1.0:
                    hz = msg_count / elapsed
                    estado = "🟢 ESTABLE" if hz > 5.0 else ("🟡 LENTO" if hz > 0.0 else "🔴 MUERTO")
                    
                    print(f"[{estado}] Frecuencia: {hz:.1f} Hz | Puntos/paquete: {puntos} | Paquetes vacíos: {empty_count}")
                    
                    # Resetear contadores para el siguiente segundo
                    msg_count = 0
                    empty_count = 0
                    last_time = current_time
            else:
                # Una pequeña pausa para no poner la CPU al 100% si no hay datos
                time.sleep(0.005)
                
    except KeyboardInterrupt:
        print("\nDiagnóstico finalizado por el usuario.")

if __name__ == "__main__":
    main()