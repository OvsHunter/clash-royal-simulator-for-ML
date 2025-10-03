import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.arena import Arena
from core.simulation import Engine
from core.troop_data import list_cards


def run():
    arena = Arena('data/arena.json')
    engine = Engine(arena)
    print('Initial P1 elixir:', engine.players[1].elixir)
    # deploy one Knight (if available) at a p1 deploy tile
    cards = list_cards()
    card = 'Knight' if 'Knight' in cards else cards[0]
    print('Using card:', card)
    ok = engine.deploy(1, card, 8, 23)
    print('Deployed?', ok)
    # step simulation for 2 seconds
    steps = int(2.0 / 0.05)
    for i in range(steps):
        engine.tick()
    print('After 2s P1 elixir:', engine.players[1].elixir)
    if engine.units:
        u = engine.units[0]
        print('Unit pos:', u.x, u.y, 'alive', u.alive)
    else:
        print('No units')

if __name__ == '__main__':
    run()
