#!/usr/bin/env python3
"""
go2_straight_walk.py
====================
Etapa 5 — Caminar en línea recta con el Unitree Go2 usando un controlador
proporcional (P) en lazo cerrado sobre yaw, leyendo SportModeState_ por DDS.

Requisitos:
    pip install unitree_sdk2py
    (y tener CycloneDDS configurado contra el Go2: red 192.168.123.x)

Uso:
    python3 go2_straight_walk.py <interfaz_red> [--vx 0.3] [--dist 2.0]
                                                [--gait classic] [--kp 1.2]
                                                [--duration 0]

Ejemplos:
    # Caminar 2 metros recto a 0.3 m/s con ClassicWalk
    python3 go2_straight_walk.py eth0 --vx 0.3 --dist 2.0

    # Caminar 5 segundos a 0.2 m/s con StaticWalk (más estable, más lento)
    python3 go2_straight_walk.py enp2s0 --vx 0.2 --duration 5.0 --gait static

    # Trote rápido durante 3 segundos
    python3 go2_straight_walk.py eth0 --vx 0.6 --duration 3.0 --gait trot

Presiona Ctrl+C en cualquier momento para parada limpia.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


# =============================================================================
# Configuración de control (ajustar si hace falta afinar)
# =============================================================================
CONTROL_HZ        = 50.0     # frecuencia de envío de Move() — Unitree recomienda 20-50 Hz
KP_YAW            = 1.2      # ganancia P del controlador de rumbo (rad/s por rad de error)
KD_YAW            = 0.1      # amortiguamiento sobre yaw_speed (opcional, súbelo si oscila)
VYAW_LIMIT        = 0.6      # límite absoluto del comando de yaw (rad/s)
LOST_CONTACT_THR  = 25       # umbral foot_force: por debajo se considera pata "en el aire"
STATE_TIMEOUT_S   = 0.3      # si no llega SportModeState_ en este tiempo, parada
SLOW_FACTOR       = 0.5      # si 2+ patas en el aire, reducir vx por este factor


# =============================================================================
# Estado compartido entre el callback DDS y el bucle de control
# =============================================================================
class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.yaw = None              # rad, rumbo del cuerpo (imu_state.rpy[2])
        self.yaw_speed = 0.0         # rad/s, velocidad angular en yaw
        self.foot_force = [0, 0, 0, 0]   # FR, FL, RR, RL — cuentas brutas
        self.position = [0.0, 0.0, 0.0]  # x, y, z (odometría por cinemática)
        self.mode = 0
        self.gait_type = 0
        self.last_stamp = 0.0

    def update_from(self, msg: SportModeState_):
        with self.lock:
            self.yaw = msg.imu_state.rpy[2]
            self.yaw_speed = msg.yaw_speed
            self.foot_force = list(msg.foot_force)
            self.position = list(msg.position)
            self.mode = msg.mode
            self.gait_type = msg.gait_type
            self.last_stamp = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "yaw":         self.yaw,
                "yaw_speed":   self.yaw_speed,
                "foot_force":  list(self.foot_force),
                "position":    list(self.position),
                "mode":        self.mode,
                "gait_type":   self.gait_type,
                "age":         time.time() - self.last_stamp,
            }


def wrap_to_pi(angle: float) -> float:
    """Envolver un ángulo al rango (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def select_gait(sport: SportClient, gait_name: str):
    """Activar el gait deseado. ClassicWalk = recomendado para línea recta."""
    gait_name = gait_name.lower()
    if gait_name == "static":
        print("[gait] StaticWalk (lento, 3 patas en suelo — máxima estabilidad)")
        sport.StaticWalk()
    elif gait_name == "classic":
        print("[gait] ClassicWalk(True) (trote conservador pre-AI — recomendado)")
        sport.ClassicWalk(True)
    elif gait_name == "trot":
        print("[gait] TrotRun (trote rápido — solo si vx > 0.5 m/s)")
        sport.TrotRun()
    else:
        raise ValueError(f"Gait desconocido: {gait_name}. Usa static, classic o trot.")


# =============================================================================
# Bucle principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Caminar recto con Unitree Go2 + controlador P de yaw"
    )
    parser.add_argument("interface",
                        help="Interfaz de red conectada al Go2 (ej. eth0, enp2s0)")
    parser.add_argument("--vx", type=float, default=0.3,
                        help="Velocidad lineal hacia delante en m/s (default 0.3)")
    parser.add_argument("--dist", type=float, default=0.0,
                        help="Distancia objetivo en m. Si > 0, ignora --duration.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Duración del movimiento en s. Ignorado si --dist > 0.")
    parser.add_argument("--gait", choices=["static", "classic", "trot"],
                        default="classic", help="Gait a usar (default classic)")
    parser.add_argument("--kp", type=float, default=KP_YAW,
                        help=f"Ganancia P del controlador de yaw (default {KP_YAW})")
    parser.add_argument("--kd", type=float, default=KD_YAW,
                        help=f"Ganancia D sobre yaw_speed (default {KD_YAW})")
    parser.add_argument("--no-state", action="store_true",
                        help="Desactivar lazo cerrado (sólo Move open-loop, para debug)")
    args = parser.parse_args()

    if args.dist <= 0 and args.duration <= 0:
        print("ERROR: especifica --dist o --duration (>0).", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Inicializar DDS
    # -------------------------------------------------------------------------
    print(f"[init] DDS en interfaz '{args.interface}'")
    ChannelFactoryInitialize(0, args.interface)

    # -------------------------------------------------------------------------
    # 2. Inicializar SportClient
    # -------------------------------------------------------------------------
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    print("[init] SportClient listo")

    # -------------------------------------------------------------------------
    # 3. Suscriptor a SportModeState_
    # -------------------------------------------------------------------------
    state = RobotState()
    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(state.update_from, 10)

    print("[init] Esperando primer SportModeState_...")
    t0 = time.time()
    while state.snapshot()["yaw"] is None:
        if time.time() - t0 > 5.0:
            print("ERROR: no llega rt/sportmodestate. ¿Interfaz/red correcta?",
                  file=sys.stderr)
            sys.exit(2)
        time.sleep(0.05)
    print("[init] Estado recibido")

    # -------------------------------------------------------------------------
    # 4. Manejador Ctrl+C: parada limpia
    # -------------------------------------------------------------------------
    stop_flag = {"stop": False}

    def on_sigint(signum, frame):
        print("\n[stop] Ctrl+C — parando limpiamente...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, on_sigint)

    # -------------------------------------------------------------------------
    # 5. Preparar postura y gait
    # -------------------------------------------------------------------------
    print("[start] BalanceStand()")
    sport.BalanceStand()
    time.sleep(0.6)

    select_gait(sport, args.gait)
    time.sleep(0.3)

    # Capturar rumbo y posición de referencia
    snap0 = state.snapshot()
    yaw_ref = snap0["yaw"]
    pos0 = snap0["position"]
    print(f"[start] yaw_ref = {math.degrees(yaw_ref):+.2f}°, "
          f"pos0 = ({pos0[0]:+.2f}, {pos0[1]:+.2f})")

    # -------------------------------------------------------------------------
    # 6. Bucle de control 50 Hz
    # -------------------------------------------------------------------------
    dt = 1.0 / CONTROL_HZ
    next_t = time.perf_counter()
    t_start = time.time()

    use_closed_loop = not args.no_state
    print(f"[loop] vx={args.vx} m/s, "
          f"objetivo={'dist=%.2f m' % args.dist if args.dist > 0 else 'dur=%.2f s' % args.duration}, "
          f"lazo cerrado={'sí' if use_closed_loop else 'no'}")

    try:
        while not stop_flag["stop"]:
            snap = state.snapshot()

            # Watchdog: si se pierde la conexión con el estado, parar
            if snap["age"] > STATE_TIMEOUT_S:
                print(f"[watchdog] Sin estado durante {snap['age']:.2f}s — parando")
                break

            # ---- Condición de parada por distancia o tiempo ----
            elapsed = time.time() - t_start
            dx = snap["position"][0] - pos0[0]
            dy = snap["position"][1] - pos0[1]
            travelled = math.hypot(dx, dy)

            if args.dist > 0 and travelled >= args.dist:
                print(f"[done] Distancia alcanzada: {travelled:.2f} m")
                break
            if args.duration > 0 and elapsed >= args.duration:
                print(f"[done] Tiempo cumplido: {elapsed:.2f} s")
                break

            # ---- Calcular vyaw con controlador P (+ D opcional) ----
            if use_closed_loop:
                err = wrap_to_pi(yaw_ref - snap["yaw"])
                vyaw = args.kp * err - args.kd * snap["yaw_speed"]
                vyaw = max(-VYAW_LIMIT, min(VYAW_LIMIT, vyaw))
            else:
                vyaw = 0.0

            # ---- Seguridad por contacto de pies ----
            airborne = sum(1 for f in snap["foot_force"] if f < LOST_CONTACT_THR)
            vx_cmd = args.vx * (SLOW_FACTOR if airborne >= 2 else 1.0)

            # ---- Enviar comando ----
            sport.Move(vx_cmd, 0.0, vyaw)

            # ---- Telemetría cada ~0.5 s ----
            if int(elapsed * 2) != int((elapsed - dt) * 2):
                print(f"  t={elapsed:5.2f}s "
                      f"d={travelled:5.2f}m "
                      f"yaw_err={math.degrees(wrap_to_pi(yaw_ref - snap['yaw'])):+6.2f}° "
                      f"vyaw={vyaw:+.3f} "
                      f"vx={vx_cmd:.2f} "
                      f"pies_aire={airborne}")

            # ---- Mantener cadencia 50 Hz ----
            next_t += dt
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Ya vamos retrasados — resetear para no acumular jitter
                next_t = time.perf_counter()

    finally:
        # ---------------------------------------------------------------------
        # 7. Parada limpia
        # ---------------------------------------------------------------------
        print("[cleanup] StopMove()")
        sport.StopMove()
        time.sleep(0.5)
        print("[cleanup] BalanceStand()")
        sport.BalanceStand()
        time.sleep(0.5)

        snap = state.snapshot()
        dx = snap["position"][0] - pos0[0]
        dy = snap["position"][1] - pos0[1]
        print(f"[summary] distancia recorrida: {math.hypot(dx, dy):.2f} m "
              f"(dx={dx:+.2f}, dy={dy:+.2f}), "
              f"yaw final = {math.degrees(snap['yaw']):+.2f}°")

        # Forzar salida para que los hilos DDS no dejen colgado el proceso
        os._exit(0)


if __name__ == "__main__":
    main()
