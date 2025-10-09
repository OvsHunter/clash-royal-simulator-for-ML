from core.arena import Arena
from core.simulation import Engine

arena = Arena()
eng = Engine(arena)

# spawn two units if present in your data
eng.spawn_card(eng.p1, "Knight", 6, arena.grid_h - 3)
eng.spawn_card(eng.p2, "Knight", arena.grid_w - 7, 2)

for _ in range(200):
    eng.step()

print("P1 units:", len(eng.p1.units), "P2 units:", len(eng.p2.units))
print("P1 towers:", [(t.label, int(t.hp)) for t in eng.p1.towers])
print("P2 towers:", [(t.label, int(t.hp)) for t in eng.p2.towers])
