# core/formations.py
"""
Handles troop spawn formations (multi-unit cards).
Returns offset positions so troops don't overlap.
"""

import math

def make_offsets(card_name: str, count: int):
    """
    Return list of (ox, oy) offsets for troop spawns.
    Offsets are centered around the deployment tile.
    """
    if count <= 1:
        return [(0.0, 0.0)]

    # Common formations
    if count == 2:
        return [(-0.3, 0.0), (0.3, 0.0)]
    if count == 3:
        return [(-0.4, 0.0), (0.4, 0.0), (0.0, 0.3)]
    if count == 4:
        return [(-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3)]
    if count == 5:
        return [(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3), (0.0, 0.0)]
    if count == 6:
        return [(-0.6, -0.3), (0.6, -0.3), (-0.6, 0.3), (0.6, 0.3), (-0.2, 0.0), (0.2, 0.0)]

    # Circle formation for hordes
    radius = 0.6
    offsets = []
    for i in range(count):
        angle = 2 * math.pi * i / count
        offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    return offsets
