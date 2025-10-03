import tkinter as tk
from tkinter import Canvas, OptionMenu, StringVar, IntVar
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.arena import Arena
from core.simulation import Engine
from core.rules import TIME_STEP
from core.troop_data import list_cards

ARENA_FILE = "data/arena.json"

ZONE_COLORS = {
    "deploy_p1": "#4e845c",
    "deploy_p1_cond_left": "#beecca",
    "deploy_p1_cond_right": "#56a45f",
    "deploy_p2": "#bb9d9d",
    "deploy_p2_cond_left": "#ae7272",
    "deploy_p2_cond_right": "#d96969",
    "bridge": "#77ff00",
    "river": "#57e9ff",
    "unplayable": "#222222",
}


class ArenaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clash Royale Sim")

        self.arena = Arena(ARENA_FILE)
        self.engine = Engine(self.arena)

        self.tile_size = 24
        w = self.arena.width * self.tile_size
        h = self.arena.height * self.tile_size

        self.canvas = Canvas(root, width=w, height=h, bg="black")
        self.canvas.pack(side="left")

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        panel = tk.Frame(root, bg="#333333")
        panel.pack(side="right", fill="y")

        all_cards = list_cards() or ["Knight"]
        tk.Label(panel, text="P1 Card", fg="white", bg="#333333").pack(pady=5)
        self.card_var_p1 = StringVar(value="Knight" if "Knight" in all_cards else all_cards[0])
        OptionMenu(panel, self.card_var_p1, *all_cards).pack(pady=5)

        tk.Label(panel, text="P2 Card", fg="white", bg="#333333").pack(pady=5)
        self.card_var_p2 = StringVar(value="Knight" if "Knight" in all_cards else all_cards[0])
        OptionMenu(panel, self.card_var_p2, *all_cards).pack(pady=5)

        tk.Label(panel, text="P1 Troop Level", fg="white", bg="#333333").pack(pady=(15, 2))
        self.troop_level_p1 = IntVar(value=11)
        tk.Spinbox(panel, from_=1, to=15, width=5, textvariable=self.troop_level_p1).pack(pady=2)

        tk.Label(panel, text="P2 Troop Level", fg="white", bg="#333333").pack(pady=2)
        self.troop_level_p2 = IntVar(value=11)
        tk.Spinbox(panel, from_=1, to=15, width=5, textvariable=self.troop_level_p2).pack(pady=2)

        tk.Label(panel, text="P1 Tower Level", fg="white", bg="#333333").pack(pady=(15, 2))
        self.tower_level_p1 = IntVar(value=11)
        tk.Spinbox(panel, from_=1, to=15, width=5, textvariable=self.tower_level_p1).pack(pady=2)

        tk.Label(panel, text="P2 Tower Level", fg="white", bg="#333333").pack(pady=2)
        self.tower_level_p2 = IntVar(value=11)
        tk.Spinbox(panel, from_=1, to=15, width=5, textvariable=self.tower_level_p2).pack(pady=2)

        tk.Button(panel, text="Apply Levels", command=self.apply_levels).pack(pady=10, fill="x")

        self.last_time = time.time()
        self._accumulated_dt = 0.0
        self.apply_levels()
        self.update_loop()

    def on_left_click(self, event):
        tx = event.x // self.tile_size
        ty = event.y // self.tile_size
        level = getattr(self, "default_troop_level_p1", self.troop_level_p1.get())
        self.engine.deploy(1, self.card_var_p1.get(), tx, ty, level=level)

    def on_right_click(self, event):
        tx = event.x // self.tile_size
        ty = event.y // self.tile_size
        level = getattr(self, "default_troop_level_p2", self.troop_level_p2.get())
        self.engine.deploy(2, self.card_var_p2.get(), tx, ty, level=level)

    @staticmethod
    def _coerce_level(value, default: int = 11) -> int:
        try:
            lvl = int(value)
        except (TypeError, ValueError):
            lvl = default
        return max(1, min(15, lvl))

    def apply_levels(self):
        p1_tower = self._coerce_level(self.tower_level_p1.get())
        p2_tower = self._coerce_level(self.tower_level_p2.get())
        self.tower_level_p1.set(p1_tower)
        self.tower_level_p2.set(p2_tower)
        self.engine.set_tower_levels(p1_tower, p2_tower)
        self.default_troop_level_p1 = self._coerce_level(self.troop_level_p1.get())
        self.default_troop_level_p2 = self._coerce_level(self.troop_level_p2.get())
        self.troop_level_p1.set(self.default_troop_level_p1)
        self.troop_level_p2.set(self.default_troop_level_p2)
        self.engine.players[1].card_levels.clear()
        self.engine.players[2].card_levels.clear()

    def update_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self._accumulated_dt += max(0.0, dt)

        steps = int(self._accumulated_dt / TIME_STEP)
        if steps > 0:
            for _ in range(steps):
                self.engine.tick()
            self._accumulated_dt -= steps * TIME_STEP

        self.render()
        self.root.after(33, self.update_loop)  # ~30 FPS

    def render(self):
        self.canvas.delete("all")
        ts = self.tile_size

        # grid
        for y in range(self.arena.height):
            for x in range(self.arena.width):
                label = self.arena.tile(x, y)
                color = ZONE_COLORS.get(label, "#000000")
                self.canvas.create_rectangle(
                    x * ts, y * ts, (x + 1) * ts, (y + 1) * ts,
                    fill=color, outline="#111111"
                )

        # towers
        for t in self.arena.towers:
            if not t.get("alive", True):
                continue
            x0, y0 = t["x0"], t["y0"]
            w, h = t["width"], t["height"]
            color = "blue" if t["owner"] == 1 else "red"
            self.canvas.create_rectangle(
                x0 * ts, y0 * ts, (x0 + w) * ts, (y0 + h) * ts,
                fill=color, outline="yellow"
            )
            # HP bar
            hp = t.get("hp", 1.0)
            hp_max = t.get("hp_max", hp)
            frac = max(0.0, min(1.0, hp / hp_max))
            self.canvas.create_rectangle(
                x0 * ts, y0 * ts - 6,
                x0 * ts + (w * ts) * frac, y0 * ts - 2,
                fill="green", outline=""
            )
            level = t.get('level')
            if level:
                self.canvas.create_text(
                    (x0 + w / 2) * ts, y0 * ts - 12,
                    text="L{}".format(level), fill="white", font=("Arial", 8)
                )

        # units
        for u in self.engine.units:
            if not u.alive:
                continue
            cx, cy = u.x * ts, u.y * ts
            r = ts * 0.4
            color = "blue" if u.owner == 1 else "red"
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")
            # HP bar
            frac = max(0.0, min(1.0, u.hp / u.hp_max))
            self.canvas.create_rectangle(cx - r, cy - r - 6, cx - r + 2 * r * frac, cy - r - 2, fill="green", outline="")
            # name
            self.canvas.create_text(cx, cy - r - 10, text=f"{u.name} L{u.level}", fill="white", font=("Arial", 8))

        # projectiles
        for p in self.engine.projectiles:
            if not p.alive:
                continue
            cx, cy = p.x * ts, p.y * ts
            self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="yellow", outline="")

        # debug HUD
        self.canvas.create_text(8, 8, text=f"t={self.engine.time:.1f}", fill="white", anchor="nw")
        self.canvas.create_text(8, 24, text=f"P1 Elixir={self.engine.players[1].elixir:.1f}  Crowns={self.engine.players[1].crowns}", fill="white", anchor="nw")
        self.canvas.create_text(8, 40, text=f"P2 Elixir={self.engine.players[2].elixir:.1f}  Crowns={self.engine.players[2].crowns}", fill="white", anchor="nw")

        phase = "x1"
        if getattr(self.engine, 'triple_elixir', False):
            phase = "x3"
        elif getattr(self.engine, 'double_elixir', False):
            phase = "x2"
        status = f"Phase {phase}"
        if getattr(self.engine, 'overtime', False):
            status += " | OT"
        if getattr(self.engine, 'match_over', False):
            winner = getattr(self.engine, 'winner', None)
            if winner in (1, 2):
                status += f" | Winner: P{winner}"
            else:
                status += " | Winner: Draw"
        self.canvas.create_text(8, 56, text=status, fill="white", anchor="nw")


if __name__ == "__main__":
    root = tk.Tk()
    app = ArenaGUI(root)
    root.mainloop()
