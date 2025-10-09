import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.arena import Arena
from core.simulation import Engine, Unit
from core.pathfinding import find_target
from core.troop_data import list_cards


def run():
    arena = Arena('data/arena.json')
    engine = Engine(arena)
    card = 'Knight' if 'Knight' in list_cards() else list_cards()[0]
    engine.deploy(1, card, 8, 23)
    u = engine.units[0]
    print('Unit:', u.name, 'pos', u.x, u.y, 'owner', u.owner, 'targets', u.targeting)
    tgt = find_target(arena, u, [e for e in engine.units if e.owner != u.owner and e.alive])
    print('find_target (no enemies):', tgt)
    # now call with enemy tower fallback (pass empty enemy units list)
    tgt2 = find_target(arena, u, [])
    print('find_target (with towers fallback):', tgt2)

if __name__ == '__main__':
    run()
