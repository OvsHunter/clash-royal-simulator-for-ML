import tkinter as tk
import json
from pathlib import Path

# -------- Settings --------
GRID_W, GRID_H = 32, 18
TILE_SIZE = 24
ARENA_FILE = Path("data/arena.json")

ZONE_COLORS = {
    "p1_area": "#4e845c",
    "p1_unlock_left": "#beecca",
    "p1_unlock_right": "#56a45f",
    "p1_king": "#2141b5",
    "p1_princess_left": "#3552b9",
    "p1_princess_right": "#6b759b",
    "p2_area": "#bb9d9d",
    "p2_unlock_left": "#ae7272",
    "p2_unlock_right": "#d96969",
    "p2_king": "#fb0303",
    "p2_princess_left": "#b04949",
    "p2_princess_right": "#915d5d",
    "river": "#57e9ff",
    "bridge": "#77ff00",
    "unplayable": "#000000",
}

TOWER_SIZES = {
    "p1_king": (4, 4),
    "p1_princess_left": (3, 3),
    "p1_princess_right": (3, 3),
    "p2_king": (4, 4),
    "p2_princess_left": (3, 3),
    "p2_princess_right": (3, 3),
}

class ArenaEditor:
    def __init__(self, master):
        self.master = master
        master.title("Arena Editor")

        self.current_zone = tk.StringVar(value="p1_area")

        # Canvas
        self.canvas = tk.Canvas(master, width=GRID_W*TILE_SIZE, height=GRID_H*TILE_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.bind("<Button-1>", self.on_click)

        # Zone selector
        self.zone_menu = tk.OptionMenu(master, self.current_zone, *ZONE_COLORS.keys())
        self.zone_menu.grid(row=1, column=0)

        # Mode toggle
        self.mode = tk.StringVar(value="zone")
        self.zone_btn = tk.Radiobutton(master, text="Paint Zone", variable=self.mode, value="zone")
        self.tower_btn = tk.Radiobutton(master, text="Place Tower", variable=self.mode, value="tower")
        self.zone_btn.grid(row=1, column=1)
        self.tower_btn.grid(row=1, column=2)

        # Save button
        self.save_btn = tk.Button(master, text="Save", command=self.save_arena)
        self.save_btn.grid(row=2, column=1)

        # Grid + towers
        self.grid = [["unplayable" for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.towers = []

        # Try preload existing arena.json
        if ARENA_FILE.exists():
            try:
                data = json.loads(ARENA_FILE.read_text())
                self.grid = data.get("grid", self.grid)
                self.towers = data.get("towers", [])
                print(f"Loaded existing {ARENA_FILE}")
            except Exception as e:
                print(f"Could not load arena.json: {e}")

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        for y in range(GRID_H):
            for x in range(GRID_W):
                zone = self.grid[y][x]
                color = ZONE_COLORS.get(zone, "#cccccc")
                self.canvas.create_rectangle(
                    x*TILE_SIZE, y*TILE_SIZE,
                    (x+1)*TILE_SIZE, (y+1)*TILE_SIZE,
                    fill=color, outline="gray"
                )

        # draw towers
        for t in self.towers:
            x0, y0, w, h = t["x0"], t["y0"], t["w"], t["h"]
            color = ZONE_COLORS.get(t["label"], "#ff00ff")
            self.canvas.create_rectangle(
                x0*TILE_SIZE, y0*TILE_SIZE,
                (x0+w)*TILE_SIZE, (y0+h)*TILE_SIZE,
                fill=color, outline="white", width=2
            )
            self.canvas.create_text(
                (x0+w/2)*TILE_SIZE, (y0+h/2)*TILE_SIZE,
                text=t["label"], fill="white"
            )

    def on_click(self, event):
        gx, gy = event.x // TILE_SIZE, event.y // TILE_SIZE
        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            return

        if self.mode.get() == "zone":
            zone = self.current_zone.get()
            self.grid[gy][gx] = zone
        else:  # tower mode
            label = self.current_zone.get()
            if label not in TOWER_SIZES:
                print("⚠️ Not a tower label")
                return
            w, h = TOWER_SIZES[label]
            self.towers = [t for t in self.towers if t["label"] != label]  # replace existing
            self.towers.append({"label": label, "x0": gx, "y0": gy, "w": w, "h": h})
        self.draw_grid()

    def save_arena(self):
        data = {
            "grid_w": GRID_W,
            "grid_h": GRID_H,
            "grid": self.grid,
            "towers": self.towers,
            "river_rows": [GRID_H//2-1, GRID_H//2],  # defaults
            "bridge_cols": [7, 24],                  # defaults
        }
        ARENA_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARENA_FILE.write_text(json.dumps(data, indent=2))
        print(f"Saved arena → {ARENA_FILE}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ArenaEditor(root)
    root.mainloop()
