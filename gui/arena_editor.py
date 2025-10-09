import tkinter as tk
from tkinter import ttk, filedialog
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARENA_FILE = ROOT / "data" / "arena.json"

GRID_W, GRID_H = 18, 32
TILE_SIZE = 20

# -------------------
# Color map
# -------------------
COLOR_MAP = {
    # Player 2 towers
    (251, 3, 3): "p2_king",              # #fb0303
    (176, 73, 73): "p2_princess_left",   # #b04949
    (145, 93, 93): "p2_princess_right",  # #915d5d

    # Player 1 towers
    (33, 65, 181): "p1_king",            # #2141b5
    (53, 82, 185): "p1_princess_left",   # #3552b9
    (107, 117, 155): "p1_princess_right",# #6b759b

    # Deploy zones
    (187, 157, 157): "deploy_p2",            # #bb9d9d
    (174, 114, 114): "deploy_p2_cond_left",  # #ae7272
    (217, 105, 105): "deploy_p2_cond_right", # #d96969
    (78, 132, 92): "deploy_p1",              # #4e845c
    (190, 236, 202): "deploy_p1_cond_left",  # #beecca
    (86, 164, 95): "deploy_p1_cond_right",   # #56a45f

    # Terrain
    (87, 233, 255): "river",             # #57e9ff
    (119, 255, 0): "bridge",             # #77ff00
}

ZONE_COLORS = {v: "#" + "".join(f"{c:02x}" for c in rgb) for rgb, v in COLOR_MAP.items()}
ZONE_COLORS["unplayable"] = "#222222"

ZONE_OPTIONS = list(ZONE_COLORS.keys())

TOWER_SIZES = {
    "p1_king": (4, 4),
    "p2_king": (4, 4),
    "p1_princess_left": (3, 3),
    "p1_princess_right": (3, 3),
    "p2_princess_left": (3, 3),
    "p2_princess_right": (3, 3),
}

# -------------------
# Tower grouping helper
# -------------------
def detect_towers_from_tilemap(tilemap):
    towers = []
    seen = set()

    for y in range(GRID_H):
        for x in range(GRID_W):
            label = tilemap[y][x]
            if label not in TOWER_SIZES:
                continue
            if (x, y) in seen:
                continue

            w, h = TOWER_SIZES[label]

            # add tower bounding box
            towers.append({
                "label": label,
                "x0": x,
                "y0": y,
                "width": w,
                "height": h
            })

            # mark all cells in this footprint as visited
            for dy in range(h):
                for dx in range(w):
                    seen.add((x + dx, y + dy))

    return towers

# -------------------
# Arena Editor
# -------------------
class ArenaEditor(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()

        self.canvas = tk.Canvas(
            self, width=GRID_W * TILE_SIZE, height=GRID_H * TILE_SIZE, bg="black"
        )
        self.canvas.pack()

        self.tilemap = [["unplayable" for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.towers = []

        controls = tk.Frame(self)
        controls.pack(fill="x")

        self.current_zone = tk.StringVar(value="unplayable")
        zone_menu = ttk.Combobox(
            controls, textvariable=self.current_zone, values=ZONE_OPTIONS, state="readonly"
        )
        zone_menu.pack(side="left")

        tk.Button(controls, text="Save", command=self.save).pack(side="left")
        tk.Button(controls, text="Load", command=self.load).pack(side="left")
        tk.Button(controls, text="Import PNG", command=self.import_png).pack(side="left")
        tk.Button(controls, text="Clear", command=self.clear).pack(side="left")

        self.canvas.bind("<Button-1>", self.paint)

        self.redraw()

    def paint(self, event):
        gx, gy = event.x // TILE_SIZE, event.y // TILE_SIZE
        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            return
        zone = self.current_zone.get()
        if zone in TOWER_SIZES:
            w, h = TOWER_SIZES[zone]
            for dy in range(h):
                for dx in range(w):
                    x, y = gx + dx, gy + dy
                    if 0 <= x < GRID_W and 0 <= y < GRID_H:
                        self.tilemap[y][x] = zone
            self.towers.append(
                {"label": zone, "x0": gx, "y0": gy, "width": w, "height": h}
            )
        else:
            self.tilemap[gy][gx] = zone
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        for y in range(GRID_H):
            for x in range(GRID_W):
                zone = self.tilemap[y][x]
                color = ZONE_COLORS.get(zone, "#000000")
                self.canvas.create_rectangle(
                    x * TILE_SIZE,
                    y * TILE_SIZE,
                    (x + 1) * TILE_SIZE,
                    (y + 1) * TILE_SIZE,
                    fill=color,
                    outline="black",
                )

        # draw outlines for towers
        for tower in self.towers:
            x0 = tower["x0"] * TILE_SIZE
            y0 = tower["y0"] * TILE_SIZE
            x1 = (tower["x0"] + tower["width"]) * TILE_SIZE
            y1 = (tower["y0"] + tower["height"]) * TILE_SIZE
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="yellow", width=2
            )

    def save(self):
        data = {
            "grid_size": {"width": GRID_W, "height": GRID_H},
            "tilemap": self.tilemap,
            "towers": self.towers,
        }
        ARENA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ARENA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved arena to {ARENA_FILE}")

    def load(self):
        if not ARENA_FILE.exists():
            print("No arena.json found.")
            return
        with open(ARENA_FILE, "r") as f:
            data = json.load(f)
        self.tilemap = data.get("tilemap", self.tilemap)
        self.towers = data.get("towers", [])
        self.redraw()
        print(f"Loaded arena from {ARENA_FILE}")

    def import_png(self):
        path = filedialog.askopenfilename(
            title="Select Arena PNG", filetypes=[("PNG Images", "*.png")]
        )
        if not path:
            return
        img = Image.open(path).convert("RGB")
        img = img.resize((GRID_W, GRID_H), Image.NEAREST)
        pixels = img.load()

        new_tilemap = []
        for y in range(GRID_H):
            row = []
            for x in range(GRID_W):
                rgb = pixels[x, y]
                zone = COLOR_MAP.get(rgb, "unplayable")
                row.append(zone)
            new_tilemap.append(row)

        self.tilemap = new_tilemap
        self.towers = detect_towers_from_tilemap(self.tilemap)

        self.redraw()
        print(f"Imported arena from {path} with {len(self.towers)} towers detected")

    def clear(self):
        self.tilemap = [["unplayable" for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.towers = []
        self.redraw()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Clash Royale Arena Editor")
    app = ArenaEditor(root)
    app.mainloop()
