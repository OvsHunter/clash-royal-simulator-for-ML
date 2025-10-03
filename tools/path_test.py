import sys
import os
import traceback

# Ensure repo root is on sys.path so 'core' package can be imported when this
# script is run directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.arena import Arena
from core.pathfinding import compute_path


def run():
    try:
        arena = Arena('data/arena.json')
        print('Arena size:', arena.width, 'x', arena.height)

        # choose a start tile roughly in player1 deploy area
        unit_ground = {'x': 8.5, 'y': 23.5, 'flying': False, 'owner': 1, 'aggro_range': 10, 'targeting': ['both']}
        unit_flying = {'x': 8.5, 'y': 23.5, 'flying': True}

        # find an enemy (p2) king/center tower
        tgt = None
        for t in arena.towers:
            lbl = t.get('label') if isinstance(t, dict) else getattr(t, 'label', None)
            if lbl and 'p2_king' in lbl:
                tgt = t
                break

        print('Found target:', tgt)

        p = compute_path(arena, unit_ground, tgt, allow_diagonal_cutting=False)
        print('Ground path length:', len(p))
        print('Ground path sample:', p[:12])

        p2 = compute_path(arena, unit_flying, tgt, allow_diagonal_cutting=False)
        print('Flying path length:', len(p2))
        print('Flying path sample:', p2[:12])

    except Exception as e:
        print('Exception during path test:')
        traceback.print_exc()


if __name__ == '__main__':
    run()
