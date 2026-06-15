"""Subpaquete de control de locomoción del Go2.

Re-exporta las primitivas de movimiento (pivotar, caminar a un punto, ir a un
waypoint) y expone el módulo `patterns` con los recorridos de alto nivel
(cuadrado de mapeo y búsqueda autónoma)."""

from .primitives import (_pivot_to_heading,
                         _pivot_to_heading_precise, 
                         _pivot_to_face, 
                         _walk_to, 
                         _go_to_world_xy, 
                         go_to_waypoint, 
                         _pivot_then_walk)
from . import patterns

__all__ = ["_pivot_to_heading", 
           "_pivot_to_heading_precise", 
           "_pivot_to_face", 
           "_walk_to", 
           "_go_to_world_xy", 
           "go_to_waypoint", 
           "_pivot_then_walk"]